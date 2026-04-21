#!/usr/bin/env python
# coding: utf-8

import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
import numpy as np
import torch
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import supervision as sv
from supervision.draw.color import ColorPalette
from collections import deque
import queue
import gc

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tare_planner.msg import ViewpointRep, ObjectType, TargetObjectInstruction


from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
# Gain the pose of the robot
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String
# Use this library to create Time message and use for subscription
from builtin_interfaces.msg import Time as TimeMsg
# Realize the function of Time message
from std_msgs.msg import Header

import threading
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation
import open3d as o3d

from .utils import find_closest_stamp, find_neighbouring_stamps
from .semantic_map_new import ObjMapper
from .tools import ros2_bag_utils
from .cloud_image_fusion import CloudImageFusion

import yaml
import time
from line_profiler import profile

from tare_planner.msg import ObjectNode, ObjectNodeList, DetectionResult

captioner_not_found = False
try:
    from captioner.captioning_backend import Captioner
except ModuleNotFoundError:
    captioner_not_found = True
    print(f"Captioner not found. Fall back to no captioning version.")

def load_models():
    checkpoint = "src/semantic_mapping/semantic_mapping/external/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    mask_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    return mask_predictor

class MappingNode(Node):
    def __init__(self, mask_predictor):
        super().__init__('semantic_mapping_node')

        self.declare_parameter("device", "auto")
        self.declare_parameter("target_object", "refrigerator")  # New parameter for target object
        self.declare_parameter("platform", "mecanum_sim")
        self.declare_parameter("detection_linear_state_time_bias", 0.0)
        self.declare_parameter("detection_angular_state_time_bias", 0.0)
        self.declare_parameter("annotate_image", True)
        self.declare_parameter("grounding_score_thresh", 0.3)
        self.declare_parameter("object_file", "config/objects_rfdetr.yaml")
        self.declare_parameter("save_png", False)

        # class global containers
        self.cloud_stack = []
        self.cloud_stamps = []
        self.odom_stack = []
        self.odom_stamps = []
        self.detection_results_stack = []
        self.detection_results_stamps = []
        self.rgb_stack = []

        # class global last states
        self.new_rgb = False
        self.last_camera_odom = None
        self.last_vis_stamp = 0.0

        # demo freeze flag: when True, SAM segmentation still runs but its
        # results are not written into self.obj_mapper (toggled via /keyboard_input).
        self.demo_frozen = False

        self.cloud_cbk_lock = threading.Lock()
        self.odom_cbk_lock = threading.Lock()
        self.rgb_cbk_lock = threading.Lock()
        self.mapping_processing_lock = threading.Lock()
        self.detection_result_lock = threading.Lock()

        # parameters
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.target_object = self.get_parameter("target_object").get_parameter_value().string_value
        self.platform = self.get_parameter('platform').get_parameter_value().string_value
        # time compensation parameters
        self.detection_linear_state_time_bias = self.get_parameter('detection_linear_state_time_bias').get_parameter_value().double_value
        self.detection_angular_state_time_bias = self.get_parameter('detection_angular_state_time_bias').get_parameter_value().double_value
        self.ANNOTATE = self.get_parameter('annotate_image').get_parameter_value().bool_value
        self.grounding_score_thresh = self.get_parameter('grounding_score_thresh').get_parameter_value().double_value
        self.object_file_path = self.get_parameter('object_file').get_parameter_value().string_value
        self.save_png = self.get_parameter('save_png').get_parameter_value().bool_value

        print(
            f'Platform: {self.platform}\n,\
                Detection linear state time bias: {self.detection_linear_state_time_bias}\n,\
                Detection angular state time bias: {self.detection_angular_state_time_bias}\n,\
                Annotate image: {self.ANNOTATE}\n,\
                Grounding score threshold: {self.grounding_score_thresh}'
        )

        self.mask_predictor = mask_predictor

        self.total_mapping_calls = 0
        self.mapping_over_3s = 0

        with open(self.object_file_path, "r") as file:
            self.object_config = yaml.safe_load(file)
        self.label_template = self.object_config['prompts']
        self.text_prompt_list = []
        for value in self.label_template.values():
            self.text_prompt_list += value['prompts']
        self.text_prompt = " . ".join(self.text_prompt_list) + " ."
        self.text_prompt_list = np.array(self.text_prompt_list)
        print(f"Text prompt: {self.text_prompt}")

        self.pos_change_threshold = 0.05
        self.viewpoint_queue = deque()
        self.viewpoint_queue_lock = threading.Lock()
        self.processed_viewpoints = set()
        self.viewpoint_id = -1
        self.timestamp = -1.0

        self.cloud_img_fusion = CloudImageFusion(platform=self.platform)

        if self.ANNOTATE:
            self.box_annotator = sv.BoxAnnotator(color=ColorPalette.DEFAULT)
            self.label_annotator = sv.LabelAnnotator(
                color=ColorPalette.DEFAULT,
                text_padding=4,
                text_scale=0.5,
                text_position=sv.Position.TOP_LEFT,
                color_lookup=sv.ColorLookup.INDEX,
                smart_position=True,
            )
            self.mask_annotator = sv.MaskAnnotator(color=ColorPalette.DEFAULT)
            time_stamp = time.time()
            self.log_dir = 'logs'
            self.ANNOTATE_OUT_DIR = os.path.join('output/debug_mapper', 'segmentation_results')
            self.VIEWPOINT_IMAGE_DIR = os.path.join('output/viewpoint_images')

            os.system(f"rm -rf {self.ANNOTATE_OUT_DIR}")  # Clear previous results
            os.system(f"rm -rf {self.VIEWPOINT_IMAGE_DIR}")  # Clear previous results
            
            os.makedirs(self.ANNOTATE_OUT_DIR, exist_ok=True)
            os.makedirs(self.VIEWPOINT_IMAGE_DIR, exist_ok=True)

            self.base_image_dir= os.path.join('output/object_images')
            os.system(f"rm -rf {self.base_image_dir}")  # Clear previous images
            os.makedirs(self.base_image_dir, exist_ok=True)  # Create base directory

        self.anchor_object = ""
        
        self.bridge = CvBridge()

        # ROS2 subscriptions and publishers
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            '/registered_scan',
            self.cloud_callback,
            10,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        if self.platform == 'mecanum_bagfile':
            self.odom_sub = self.create_subscription(
                Odometry,
                '/aft_mapped_to_init_incremental',
                self.odom_callback,
                50,
                callback_group=MutuallyExclusiveCallbackGroup()
            )
        else:
            self.odom_sub = self.create_subscription(
                Odometry,
                '/state_estimation',
                self.odom_callback,
                50,
                callback_group=MutuallyExclusiveCallbackGroup()
            )

        # Here I create a subscription to the viewpoint topic to get the robot's position
        self.viewpoint_sub = self.create_subscription(
            ViewpointRep,
            '/viewpoint_rep_header',
            self.viewpoint_timestamp_callback,
            5,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.detection_result_sub = self.create_subscription(
            DetectionResult,
            '/detection_result',
            self.detection_result_callback,
            50,
            callback_group=MutuallyExclusiveCallbackGroup()
        )
        self.object_type_answer_sub = self.create_subscription(
            ObjectType,
            '/object_type_answer',
            self.object_type_answer_callback,
            50,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.target_object_instruction_sub = self.create_subscription(
            TargetObjectInstruction,
            '/target_object_instruction',
            self.target_object_instruction_callback,
            10,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.keyboard_input_sub = self.create_subscription(
            String,
            '/keyboard_input',
            self.keyboard_input_callback,
            10,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.object_type_query_pub = self.create_publisher(
            ObjectType,
            '/object_type_query',
            50
        )

        self.detection_counter = 0
        self.mapping_timer = self.create_timer(0.5, self.mapping_callback)

        self.obj_cloud_pub = self.create_publisher(PointCloud2, '/obj_points', 10)
        self.obj_box_pub = self.create_publisher(MarkerArray, '/obj_boxes', 10)
        self.obj_text_pub = self.create_publisher(MarkerArray, '/obj_labels', 10)
        self.annotated_image_pub = self.create_publisher(Image, '/annotated_image', 10)
        self.cloud_image_pub = self.create_publisher(Image, '/cloud_image', 10)
        self.object_node_pub = self.create_publisher(ObjectNodeList, '/object_nodes_list', 200)

        self.obj_mapper = ObjMapper(
                                    cloud_image_fusion=self.cloud_img_fusion, 
                                    label_template=self.label_template, 
                                    log_info=self.log_info,
                                    object_node_pub = self.object_node_pub,
                                    target_object = self.target_object,
                                    )
        
        threading.Thread(target=self.save_worker, daemon=True).start()

        self.log_info('Semantic mapping node has been started.')
        self.log_info(f"Save png: {self.save_png}")

    def log_info(self, msg):
        self.get_logger().info(msg)

    def save_worker(self):
        while rclpy.ok():
            try:
                flag, imgpath, maskpath, img, mask = self.obj_mapper.save_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if flag == 0:  # delete
                if os.path.exists(imgpath):
                    os.remove(imgpath)
                if os.path.exists(maskpath):
                    os.remove(maskpath)
            elif flag == 1:
                np.save(imgpath, img)  # 也可以用 cv2.imwrite
                np.save(maskpath, mask)
            
            if self.save_png:
                iamg_png_path = imgpath.replace('.npy', '.png')
                mask_png_path = maskpath.replace('.npy', '.png')
                if flag == 0: # delete
                    if os.path.exists(iamg_png_path):
                        os.remove(iamg_png_path)
                    if os.path.exists(mask_png_path):
                        os.remove(mask_png_path)
                elif flag == 1:
                    cv2.imwrite(iamg_png_path, img)
                    cv2.imwrite(mask_png_path, (mask * 255).astype(np.uint8))

            self.obj_mapper.save_queue.task_done()

    def cloud_callback(self, msg):
        with self.cloud_cbk_lock:
            points_numpy = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"))
            stamp_seconds = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            self.cloud_stack.append(points_numpy)
            self.cloud_stamps.append(stamp_seconds)

            # keep only the data after self.timestamp
            while self.cloud_stamps and self.cloud_stamps[0] < self.timestamp:
                self.cloud_stack.pop(0)
                self.cloud_stamps.pop(0)

    def viewpoint_timestamp_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        viewpoint_id = msg.viewpoint_id
        with self.viewpoint_queue_lock:
            # Check if this timestamp is already queued or processed
            self.get_logger().info(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Received viewpoint id: {msg.viewpoint_id}, timestamp: {timestamp:.3f}')
            if timestamp not in self.viewpoint_queue and timestamp not in self.processed_viewpoints:
                self.viewpoint_queue.append((timestamp, viewpoint_id))
                self.get_logger().info(f'Added viewpoint timestamp to queue: {msg.viewpoint_id}, queue size: {len(self.viewpoint_queue)}')
            else:
                self.get_logger().info(f'Skipping duplicate viewpoint timestamp: {msg.viewpoint_id}')

    def detection_result_callback(self, msg):
        self.detection_counter = (self.detection_counter + 1) % 5
        with self.detection_result_lock:
            det_tracked = {'bboxes': [], 'confidences': [], 'labels': [], 'ids': []}
            bboxes = []
            confidences = []
            labels = []
            obj_ids = []
            for i in range(len(msg.track_id)):
                bboxes.append([msg.x1[i], msg.y1[i], msg.x2[i], msg.y2[i]])
                confidences.append(msg.confidence[i])
                labels.append(msg.label[i])
                obj_ids.append(msg.track_id[i])
            det_tracked['bboxes'] = np.array(bboxes)
            det_tracked['confidences'] = np.array(confidences)
            det_tracked['labels'] = np.array(labels)
            det_tracked['ids'] = np.array(obj_ids)

            cv_image = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding='bgr8')

            det_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            self.detection_results_stack.append(det_tracked)
            self.detection_results_stamps.append(det_stamp)
            self.rgb_stack.append(cv_image)

            while self.detection_results_stamps and self.detection_results_stamps[0] < self.timestamp:
                self.detection_results_stack.pop(0)
                self.detection_results_stamps.pop(0)
                self.rgb_stack.pop(0)

    def odom_callback(self, msg):
        with self.odom_cbk_lock:
            odom = {}
            odom['position'] = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
            odom['orientation'] = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
            odom['linear_velocity'] = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
            odom['angular_velocity'] = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]

            self.odom_stack.append(odom)
            self.odom_stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9)

            while self.odom_stamps and self.odom_stamps[0] < self.timestamp:
                self.odom_stack.pop(0)
                self.odom_stamps.pop(0)

    def target_object_instruction_callback(self, msg):
        previous_target = self.target_object
        self.target_object = msg.target_object
        self.anchor_object = msg.anchor_object
        self.obj_mapper.target_object = msg.target_object
        self.obj_mapper.anchor_object = msg.anchor_object
        for single_obj in self.obj_mapper.single_obj_list:
            label = single_obj.get_dominant_label()
            if label == self.target_object or label == previous_target:
                single_obj.updated = True  # Force re-publish

    def keyboard_input_callback(self, msg):
        if msg.data == "demo" and not self.demo_frozen:
            self.demo_frozen = True
            self.log_info(
                "Demo mode enabled: SAM segmentation still runs, "
                "but results are no longer written into obj_mapper."
            )
        elif msg.data == "resume" and self.demo_frozen:
            self.demo_frozen = False
            self.log_info(
                "Demo mode disabled: obj_mapper updates resumed."
            )

    @profile
    def mapping_processing(self, image, camera_odom, detections_tracked, detection_stamp, neighboring_cloud, viewpoint_stamp_to_process=None):
        with self.mapping_processing_lock:
            start_time = time.time()

            det_labels = detections_tracked['labels']
            det_bboxes = detections_tracked['bboxes']
            det_confidences = detections_tracked['confidences']
            # ================== Infer Masks ==================
            # sam2
            sam2_start = time.time()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                self.mask_predictor.set_image(image)

                if len(detections_tracked['bboxes']) > 0:
                    masks, _, _ = self.mask_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=np.array(detections_tracked['bboxes']),
                        multimask_output=False,
                    )

                    if masks.ndim == 4:
                        masks = masks.squeeze(1)
                    
                    detections_tracked['masks'] = masks
                else: # no information need to add to map
                    detections_tracked['masks'] = []
                    # return
            sam2_time = time.time() - sam2_start

            annotate_start = time.time()
            if self.ANNOTATE:
                image_anno = image.copy()
                image_verbose = image_anno.copy()

                bboxes = detections_tracked['bboxes']
                masks = detections_tracked['masks']
                labels = detections_tracked['labels']
                obj_ids = detections_tracked['ids']
                confidences = detections_tracked['confidences']

                if len(bboxes) > 0:
                    image_anno = cv2.cvtColor(image_anno, cv2.COLOR_BGR2RGB)
                    class_ids = np.array(list(range(len(labels))))
                    annotation_labels = [
                        f"{class_name} {id_} {confidence:.2f}"
                        for class_name, id_, confidence in zip(
                            labels, obj_ids, confidences
                        )
                    ]
                    # To keep it simple in video rendering, only show class names
                    # annotation_labels = [
                    #     f"{class_name}"
                    #     for class_name in labels
                    # ]
                    detections = sv.Detections(
                        xyxy=np.array(bboxes),
                        mask=np.array(masks).astype(bool),
                        class_id=class_ids,
                    )
                    image_anno = self.box_annotator.annotate(scene=image_anno, detections=detections)
                    image_anno = self.label_annotator.annotate(scene=image_anno, detections=detections, labels=annotation_labels)
                    image_anno = self.mask_annotator.annotate(scene=image_anno, detections=detections)
                    image_anno = cv2.cvtColor(image_anno, cv2.COLOR_RGB2BGR)

                if len(det_bboxes) > 0:
                    image_verbose = cv2.cvtColor(image_verbose, cv2.COLOR_BGR2RGB)
                    class_ids = np.array(list(range(len(det_labels))))
                    annotation_labels = [
                        f"{class_name} {confidence:.2f}"
                        for class_name, confidence in zip(
                            det_labels, det_confidences
                        )
                    ]
                    detections = sv.Detections(
                        xyxy=np.array(det_bboxes),
                        class_id=class_ids,
                    )
                    image_verbose = self.box_annotator.annotate(scene=image_verbose, detections=detections)
                    image_verbose = self.label_annotator.annotate(scene=image_verbose, detections=detections, labels=annotation_labels)
                    image_verbose = cv2.cvtColor(image_verbose, cv2.COLOR_RGB2BGR)
                    image_verbose = np.vstack((image_verbose, image_anno))

                # draw pcd
                R_b2w = Rotation.from_quat(camera_odom['orientation']).as_matrix()
                t_b2w = np.array(camera_odom['position'])
                R_w2b = R_b2w.T
                t_w2b = -R_w2b @ t_b2w
                cloud_body = neighboring_cloud @ R_w2b.T + t_w2b

                # cv2.imwrite(os.path.join(self.IMAGE_DIR, f"{detection_stamp}.png"), image)
                # Here we save the image for the viewpoint timestamp if it is set
                if viewpoint_stamp_to_process is not None:
                    timestamp_for_filename = viewpoint_stamp_to_process
                    cv2.imwrite(os.path.join(self.VIEWPOINT_IMAGE_DIR, f"viewpoint_{self.viewpoint_id}_anno.png"), image_anno)
                    self.get_logger().info(f"Saved viewpoint images for timestamp: {timestamp_for_filename}")
                # else:
                    # self.get_logger().info("No need to save image for viewpoint timestamp.")
                ros_image = self.bridge.cv2_to_imgmsg(image_anno, encoding='bgr8')
                seconds = int(detection_stamp)
                nanoseconds = int((detection_stamp - seconds) * 1e9)
                # ros_image.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
                stamp = TimeMsg()
                stamp.sec = seconds
                stamp.nanosec = nanoseconds
                ros_image.header.stamp = stamp
                self.annotated_image_pub.publish(ros_image)

                # cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                # cv2.waitKey(1)
            
            annotate_time = time.time() - annotate_start

            # if len(detections_tracked['bboxes']) == 0:
            #     self.get_logger().info("No valid detections found. Skipping map update.")
            #     return

            # ================== Update the map ==================

            map_update_start = time.time()
            if not self.demo_frozen:
                self.obj_mapper.update_map(detections_tracked, detection_stamp, camera_odom, neighboring_cloud, image, viewpoint_stamp_to_process)
            map_update_time = time.time() - map_update_start

            # ================== Publish the map ==================
            publish_start = time.time()
            if viewpoint_stamp_to_process is not None:
                self.publish_map(viewpoint_stamp_to_process)
            else:
                self.publish_map(detection_stamp)
            publish_time = time.time() - publish_start
        
            target_objs = self.obj_mapper.check_target_objects()
            if len(target_objs) > 0:
                self.get_logger().info(f"Target objects {self.target_object} found: {target_objs}")
                self.publish_object_type_query(target_objs)

            total_time = time.time() - start_time
            # self.log_info(f"🚨🚨 Map update time: {map_update_time}, sam2 time: {sam2_time}, annotate time: {annotate_time}, publish time: {publish_time}, total time: {total_time}")
            self.total_mapping_calls += 1
            if total_time > 3.0:
                self.mapping_over_3s += 1
                self.log_info(f"Mapping processing took over 3 seconds! Total calls: {self.total_mapping_calls}, Over 3s calls: {self.mapping_over_3s}")
            
            # 定期强制垃圾回收
            if self.total_mapping_calls % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # Add timing into a txt file and calculate the percent of over 3s
            # with open('mapping_timing_gates.txt', 'a') as f:
            #     f.write(f"{detection_stamp}, {inference_time}, {map_update_time}, {sam2_time}, {annotate_time}, {time.time() - start_time}\n")

    def mapping_callback(self):
        # if self.detection_counter == 2:
        start = time.time()

        
        # 定期强制垃圾回收
        if self.total_mapping_calls % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if self.mapping_processing_lock.locked():
            print("Mapping processing is still ongoing. Skipping this cycle.")
            return
            
        with self.detection_result_lock:
            if len(self.detection_results_stamps) < 2:
                print("No detection found. Waiting for detection...")
                return
            detections = None
            viewpoint_stamp_to_process = None
            
            self.viewpoint_id = -1
            with self.viewpoint_queue_lock:
                if self.viewpoint_queue:
                    # check if the timestamp is in the range of detection timestamps
                    candidate_viewpoint_stamp = self.viewpoint_queue[0][0]
                    if candidate_viewpoint_stamp > self.detection_results_stamps[0] and candidate_viewpoint_stamp < self.detection_results_stamps[-1]:
                        viewpoint_stamp_to_process, viewpoint_id = self.viewpoint_queue.popleft()
                        self.viewpoint_id = viewpoint_id
                        self.get_logger().info(f'!!!!!!!!!!!!!!!Processing viewpoint from queue: {viewpoint_id}, remaining in queue: {len(self.viewpoint_queue)}')

            # Now we use the logic to control the freerun and image publishing
            # default: use the second newest frame (free-run)
            target_index = -2
            # if viewpoint timestamp is set, use closest frame
            if viewpoint_stamp_to_process is not None:
                min_diff = float('inf')
                closest_stamp = None
                for i, stamp in enumerate(self.detection_results_stamps):
                    diff = abs(stamp - viewpoint_stamp_to_process)
                    if diff < min_diff:
                        min_diff = diff
                        target_index = i
                        closest_stamp = stamp
                self.get_logger().info(f'✅ Viewpoint timestamp: {viewpoint_stamp_to_process:.3f}, '
                                    f'closest detection: {closest_stamp:.3f}, '
                                    f'time difference: {min_diff:.3f}s')
                if min_diff > 1.0:
                    self.get_logger().info(f'⚠️⚠️⚠️⚠️⚠️⚠️ Closest detection too far from viewpoint timestamp. Skipping this viewpoint.')

                self.processed_viewpoints.add(viewpoint_stamp_to_process)
            # else:
            #     self.get_logger().info(f'Not viewpoint timestamp. Running in free mode.')

            # image = self.rgb_stack[target_index].copy()
            # self.get_logger().info(f'Using detection index: {target_index}')
            detection_stamp = self.detection_results_stamps[target_index]
            detections = self.detection_results_stack[target_index] 
            image = self.rgb_stack[target_index].copy()


        # ================== Time synchronization ==================
        with self.odom_cbk_lock:
            det_linear_state_stamp = detection_stamp + self.detection_linear_state_time_bias
            det_angular_state_stamp = detection_stamp + self.detection_angular_state_time_bias

            # 使用引用（避免重复代码，但注意它们指向同一个对象）
            linear_state_stamps = self.odom_stamps
            angular_state_stamps = self.odom_stamps
            linear_states = self.odom_stack
            angular_states = self.odom_stack
            
            if len(linear_state_stamps) == 0:
                self.log_info("⚠️⚠️⚠️⚠️⚠️⚠️No odometry found. Waiting for odometry...")
                return

            target_left_odom_stamp, target_right_odom_stamp = find_neighbouring_stamps(linear_state_stamps, det_linear_state_stamp)
            if target_left_odom_stamp > det_linear_state_stamp: # wait for next detection
                self.log_info("⚠️⚠️⚠️⚠️⚠️⚠️Detection older than oldest odom. Waiting for next detection...")
                return
            if target_right_odom_stamp < det_linear_state_stamp: # wait for odometry
                self.log_info(f"⚠️⚠️⚠️⚠️⚠️⚠️Odom older than detection. Right odom: {target_right_odom_stamp}, det linear: {det_linear_state_stamp}. Waiting for odometry...")
                return

            # target_angular_odom_stamp = find_closest_stamp(angular_state_stamps, det_angular_state_stamp)
            # if abs(target_angular_odom_stamp - det_angular_state_stamp) > 0.1:
            #     print(f"No close angular state found. Angular odom found: {target_angular_odom_stamp}, det angular: {det_angular_state_stamp}. Waiting for odometry...")
            #     return
            # angular_odom = angular_states[angular_state_stamps.index(target_angular_odom_stamp)]

            left_linear_odom = linear_states[linear_state_stamps.index(target_left_odom_stamp)]
            right_linear_odom = linear_states[linear_state_stamps.index(target_right_odom_stamp)]

            linear_left_ratio = (det_linear_state_stamp - target_left_odom_stamp) / (target_right_odom_stamp - target_left_odom_stamp) if target_right_odom_stamp != target_left_odom_stamp else 0.5

            assert linear_left_ratio <= 1.0 and linear_left_ratio >= 0.0

            # interpolate for the camera odometry
            camera_odom = {}
            camera_odom['position'] = np.array(right_linear_odom['position']) * linear_left_ratio + np.array(left_linear_odom['position']) * (1 - linear_left_ratio)
            camera_odom['linear_velocity'] = np.array(right_linear_odom['linear_velocity']) * linear_left_ratio + np.array(left_linear_odom['linear_velocity']) * (1 - linear_left_ratio)
            camera_odom['angular_velocity'] = np.array(right_linear_odom['angular_velocity']) * linear_left_ratio + np.array(left_linear_odom['angular_velocity']) * (1 - linear_left_ratio)
            
            # SLERP for orientation interpolation
            rotations = Rotation.from_quat([left_linear_odom['orientation'], right_linear_odom['orientation']])
            slerp = Slerp([0, 1], rotations)
            camera_odom['orientation'] = slerp(linear_left_ratio).as_quat()
            # camera_odom['angular_velocity'] = angular_odom['angular_velocity']
            camera_odom['angular_velocity'] = np.array(right_linear_odom['angular_velocity']) * linear_left_ratio + np.array(left_linear_odom['angular_velocity']) * (1 - linear_left_ratio)
        
        # save the image for viewpoint timestamp
        if viewpoint_stamp_to_process is not None:
            # create a 3*4 matrix to save the position and orientation
            R_b2w = Rotation.from_quat(camera_odom['orientation']).as_matrix()
            t_b2w = np.array(camera_odom['position'])
            transform_matrix = np.eye(4)
            transform_matrix[0:3, 0:3] = R_b2w
            transform_matrix[0:3, 3] = t_b2w
            np.save(os.path.join(self.VIEWPOINT_IMAGE_DIR, f"viewpoint_{self.viewpoint_id}_transform.npy"), transform_matrix)
            cv2.imwrite(os.path.join(self.VIEWPOINT_IMAGE_DIR, f"viewpoint_{self.viewpoint_id}.png"), image)

        # ================== Find the cloud collected around rgb timestamp ==================
        with self.cloud_cbk_lock:
            if len(self.cloud_stamps) == 0:
                self.log_info("⚠️⚠️⚠️⚠️⚠️⚠️No cloud found. Waiting for cloud...")
                return
            # while len(self.cloud_stamps) > 0 and self.cloud_stamps[0] < (detection_stamp - 1.0):
            #     self.cloud_stack.pop(0)
            #     self.cloud_stamps.pop(0)
            #     if len(self.cloud_stack) == 0:
            #         self.log_info("⚠️⚠️⚠️⚠️⚠️⚠️No cloud found. Waiting for cloud...")
            #         return

            neighboring_cloud = []
            for i in range(len(self.cloud_stamps)):
                if self.cloud_stamps[i] >= (detection_stamp - 0.5) and self.cloud_stamps[i] <= (detection_stamp + 0.1):
                    neighboring_cloud.append(self.cloud_stack[i])
            if len(neighboring_cloud) == 0:
                self.log_info("⚠️⚠️⚠️⚠️⚠️⚠️No neighboring cloud found. Waiting for cloud...")
                return
            else:
                neighboring_cloud = np.concatenate(neighboring_cloud, axis=0)

        # if self.last_camera_odom is not None:
        #     if np.linalg.norm(self.last_camera_odom['position'] - camera_odom['position']) < 0.05:
        #         return
        
        self.last_camera_odom = camera_odom
        self.timestamp = detection_stamp - 0.5

        # threading.Thread(target=self.mapping_processing, args=(image, camera_odom, detections, detection_stamp, neighboring_cloud, viewpoint_stamp_to_process)).start()
        self.mapping_processing(image, camera_odom, detections, detection_stamp, neighboring_cloud, viewpoint_stamp_to_process)

    def publish_map(self, stamp):
        seconds = int(stamp)
        nanoseconds = int((stamp - seconds) * 1e9)

        bbox_marker_array_msg = MarkerArray()
        text_marker_array_msg = MarkerArray()
        bbox_marker_deleted_array_msg = MarkerArray()
        text_marker_deleted_array_msg = MarkerArray()
        bbox_marker_array_list = []
        text_marker_array_list = []

        bbox_msg_list_deleted, text_msg_list_deleted = self.obj_mapper.to_ros2_msgs_deleted(stamp)
        if len(bbox_msg_list_deleted) > 0:
            bbox_marker_deleted_array_msg.markers = bbox_msg_list_deleted
            self.obj_box_pub.publish(bbox_marker_deleted_array_msg)
        if len(text_msg_list_deleted) > 0:
            text_marker_deleted_array_msg.markers = text_msg_list_deleted
            self.obj_text_pub.publish(text_marker_deleted_array_msg)
        
        bbox_msg_list, text_msg_list, ros_pcd = self.obj_mapper.to_ros2_msgs(stamp, self.viewpoint_id)
        
        for msg in bbox_msg_list:
            bbox_marker_array_list.append(msg)
        for msg in text_msg_list:
            text_marker_array_list.append(msg)

        if ros_pcd is not None:
            self.obj_cloud_pub.publish(ros_pcd)
        if len(bbox_marker_array_list) > 1:
            bbox_marker_array_msg.markers = bbox_marker_array_list
            self.obj_box_pub.publish(bbox_marker_array_msg)
        if len(text_marker_array_list) > 1:
            text_marker_array_msg.markers = text_marker_array_list
            self.obj_text_pub.publish(text_marker_array_msg)
    
    def publish_object_type_query(self, target_objs):
        if len(target_objs) == 0:
            return
        for obj in target_objs:
            object_type_query = ObjectType()
            object_type_query.object_id = obj['object_id']
            object_type_query.img_path = obj['img_path']
            object_type_query.labels = obj['labels']
            self.object_type_query_pub.publish(object_type_query)
    
    def object_type_answer_callback(self, msg: ObjectType):
        self.log_info(f"Received target object answer: {msg.final_label}, id: {msg.object_id}, img_path: {msg.img_path}")
        self.obj_mapper.update_target_object(msg.object_id, msg.final_label)


def main(args=None):
    rclpy.init(args=args)

    mask_predictor = load_models()

    # Initialize your main MappingNode
    semantic_node = MappingNode(mask_predictor)
    executor = SingleThreadedExecutor()
    executor.add_node(semantic_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        semantic_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()