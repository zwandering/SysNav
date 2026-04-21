#!/usr/bin/env python
# coding: utf-8

# ========== Environment Setup ==========
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

# ========== Standard Library ==========
import time
from collections import deque
from pathlib import Path

# ========== Third-party Libraries ==========
import cv2
import numpy as np
import yaml

# ========== Computer Vision Libraries ==========
import supervision as sv
from supervision.draw.color import ColorPalette
from ultralytics import YOLO, YOLOE, YOLOWorld
from ultralytics.utils import LOGGER
LOGGER.setLevel("ERROR")

# ========== ROS2 Core ==========
import rclpy
from rclpy.node import Node
from rclpy.time import Time

# ========== ROS2 Messages ==========
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# ========== Custom Messages ==========
from tare_planner.msg import DetectionResult

class DetectNode(Node):
    def __init__(self, device='cuda'):
        super().__init__('semantic_mapping_node')
        self.CONFIG_DIR = Path(__file__).resolve().parent

        self.detection_stamps = deque(maxlen=10)
        self.rgb_stack = deque(maxlen=10)

        # parameters
        self.declare_parameter('platform', 'mecanum_sim')
        self.declare_parameter('grounding_score_thresh', 0.3)
        self.declare_parameter('device', device)
        self.declare_parameter('annotate_image', True)
        self.declare_parameter('object_file', str(self.CONFIG_DIR / 'config' / 'objects.yaml'))

        self.platform = self.get_parameter('platform').get_parameter_value().string_value
        self.ANNOTATE = self.get_parameter('annotate_image').get_parameter_value().bool_value
        self.grounding_score_thresh = self.get_parameter('grounding_score_thresh').get_parameter_value().double_value
        object_file_path = self.get_parameter('object_file').get_parameter_value().string_value

        with open(object_file_path, "r") as file:
            self.object_config = yaml.safe_load(file)
        self.label_template = self.object_config['prompts']
        self.text_prompt_list = []
        for value in self.label_template.values():
            self.text_prompt_list += value['prompts']
        self.text_prompt = " . ".join(self.text_prompt_list) + " ."
        self.text_prompt_list = np.array(self.text_prompt_list)
        print(f"Text prompt: {self.text_prompt}")

        self.grounding_model = YOLO(self.CONFIG_DIR / "external/yolov8x-worldv2_cus.engine", task='detect')
        # self.grounding_model = YOLOE(self.CONFIG_DIR / "external/yoloe-11l-seg.engine", task="segment")
        self.grounding_model = YOLOE(self.CONFIG_DIR / "external/yoloe-26x-seg.engine", task="segment")

        self.device = device

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
            self.ANNOTATE_OUT_DIR = os.path.join('output/debug_mapper', 'annotated_3d_in_loop_detection')
            self.IMAGE_DIR = os.path.join('output/debug_mapper', 'image_gates')
            self.VIEWPOINT_IMAGE_DIR = os.path.join(os.path.dirname(__file__), 'output/viewpoint_images')
            if os.path.exists(self.ANNOTATE_OUT_DIR):
                os.system(f"rm -r {self.ANNOTATE_OUT_DIR}")
            os.makedirs(self.ANNOTATE_OUT_DIR, exist_ok=True)

        self.bridge = CvBridge()

        # ROS2 subscriptions and publishers
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10,
            # qos_profile=QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

        self.annotated_image_pub = self.create_publisher(Image, '/annotated_image_detection', 10)
        self.detection_result_pub = self.create_publisher(DetectionResult, '/detection_result', 50)

        self.call_back_time_stamp = time.time()

        self.log_info('Detection node has been started.')

    def log_info(self, msg):
        self.get_logger().info(msg)

    def inference(self, cv_image):
        """
        Perform open-vocabulary semantic inference on the input image.

        cv_image: np.ndarray, shape (H, W, 3), BGR format
        """
        image = cv_image[:, :, ::-1]  # BGR to RGB
        # image = image.copy()
        start_time = time.time()
        results = self.grounding_model.track(image, imgsz=(640, 1920), half=True, conf=self.grounding_score_thresh, persist=True, tracker=self.CONFIG_DIR / "config/botsort.yaml")
        time1 = time.time()
        boxes = results[0].boxes  # Boxes 对象

        # 如果没有ID信息，直接返回空结果
        if boxes.id is None:
            self.log_info("No track IDs found in the results.")
            return {
                "bboxes": np.empty((0, 4), dtype=float),
                "labels": np.array([], dtype=str),
                "confidences": np.array([], dtype=float),
                "ids": np.array([], dtype=int)
            }
        
        bboxes = boxes.xyxy.cpu().numpy()        # [N, 4] x1, y1, x2, y2
        confidences = boxes.conf.cpu().numpy()   # [N,]
        class_names = boxes.cls.cpu().numpy()    # [N,]
        class_names = self.text_prompt_list[class_names.astype(int)]
        ids = boxes.id.int().cpu().numpy()  # [N,] Track IDs

        det_result = {
            "bboxes": bboxes,
            "labels": class_names,
            "confidences": confidences,
            'ids': ids
        }
        time2 = time.time()
        time_taken1 = time1 - start_time
        time_taken2 = time2 - time1
        # self.log_info(f"Bounding boxes size: {len(bboxes)}")
        # self.log_info(f"🚨🚨Detection time: {time_taken1*1000:.1f} ms, Data transfer time: {time_taken2*1000:.1f} ms.")

        return det_result

    def image_callback(self, msg):
        # self.log_info(f'1111111111111111Processed an image at {time.time()}')

        start_time = time.time()

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        det_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.detection_processing(cv_image, det_stamp)

    def detection_processing(self, image, detection_stamp):
        start_time = time.time()
        # self.log_info(f"------------------------------------------")
        # self.log_info(f"Callback interval: {start_time - self.call_back_time_stamp:.2f} seconds")
        self.call_back_time_stamp = start_time

        # ================== Process detection and tracking ==================
        detections = self.inference(image)
        detection_time = time.time()

        if self.ANNOTATE:
            image_anno = image.copy()

            bboxes = detections['bboxes']
            labels = detections['labels']
            obj_ids = detections['ids']

            if len(bboxes) > 0:
                # image_anno = cv2.cvtColor(image_anno, cv2.COLOR_BGR2RGB)
                class_ids = np.array(list(range(len(labels))))
                annotation_labels = [
                    f"{class_name} {id_}"
                    for class_name, id_ in zip(
                        labels, obj_ids
                    )
                ]
                detections_ = sv.Detections(xyxy=bboxes, class_id=class_ids)
                self.box_annotator.annotate(scene=image_anno, detections=detections_)
                self.label_annotator.annotate(scene=image_anno, detections=detections_, labels=annotation_labels) 

        anotate_time = time.time()
        self.publish_detection_results(detections, detection_stamp, image, image_anno)
        publish_time = time.time()
        # self.log_info(f"🚨🚨🚨🚨 Detection time: {time.time() - start_time:.2f} seconds, detection time: {detection_time - start_time:.2f}, annotate time: {anotate_time - detection_time:.2f}, publish time: {publish_time - anotate_time:.2f}")

    def publish_detection_results(self, detections_tracked, detection_stamp, image, image_anno):
        """
        Publish the detection results as a DetectionResult message.
        """
        # get the time stamp for the detection_stamp
        seconds = int(detection_stamp)
        nanoseconds = int((detection_stamp - seconds) * 1e9)

        detection_result_msg = DetectionResult()
        detection_result_msg.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
        detection_result_msg.header.frame_id = 'map'

        for i in range(len(detections_tracked['ids'])):
            detection_result_msg.track_id.append(detections_tracked['ids'][i])
            detection_result_msg.x1.append(detections_tracked['bboxes'][i][0])
            detection_result_msg.y1.append(detections_tracked['bboxes'][i][1])
            detection_result_msg.x2.append(detections_tracked['bboxes'][i][2])
            detection_result_msg.y2.append(detections_tracked['bboxes'][i][3])
            detection_result_msg.label.append(detections_tracked['labels'][i])
            detection_result_msg.confidence.append(detections_tracked['confidences'][i])
        # convert the image to ROS Image message
        detection_result_msg.image = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        # self.log_info(f'Publishing {len(detection_result_msg.track_id)} detections at {detection_stamp:.2f} seconds.')
        self.detection_result_pub.publish(detection_result_msg)

        annotated_image_msg = self.bridge.cv2_to_imgmsg(image_anno, encoding='bgr8')
        annotated_image_msg.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
        annotated_image_msg.header.frame_id = 'map'
        self.annotated_image_pub.publish(annotated_image_msg)



def main(args=None):
    rclpy.init(args=args)

    detection_node = DetectNode()

    rclpy.spin(detection_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()