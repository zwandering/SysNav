import cv2
import numpy as np
import open3d as o3d
from pytorch3d.ops import box3d_overlap
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation
import torch
from rclpy.time import Time
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import PointCloud2
from rclpy.serialization import serialize_message
from shapely.geometry import Polygon

from .cloud_image_fusion import CloudImageFusion
from .single_object_new import SingleObject
from .tools import ros2_bag_utils as ros2_bag_utils
from .utils import generate_colors, extract_meta_class, get_corners_from_box3d_torch, find_nearby_points, map_label_to_color
from .visualizer import VisualizerRerun

from tare_planner.msg import ObjectNode

from time import time
import os
import queue
captioner_not_found = False
try:
    from captioner.captioning_backend import Captioner
except ModuleNotFoundError:
    captioner_not_found = True

from line_profiler import profile

import copy
import time


def serialize_objs_to_bag(writer, obj_mapper, stamp: float, raw_cloud=None, odom=None):
    seconds = int(stamp)
    nanoseconds = int((stamp - seconds) * 1e9)

    marker_array = []
    delete_marker_time = stamp - 1e-4
    delete_marker_seconds = int(delete_marker_time)
    delete_marker_nanoseconds = int((delete_marker_time - delete_marker_seconds) * 1e9)
    clear_marker = Marker()
    clear_marker.header.frame_id = 'map'
    clear_marker.header.stamp = Time(seconds=delete_marker_seconds, nanoseconds=delete_marker_nanoseconds).to_msg()
    clear_marker.action = Marker.DELETEALL

    marker_array.append(clear_marker)

    map_vis_msgs = obj_mapper.to_ros2_msgs(stamp)
    
    for msg in map_vis_msgs:
        if isinstance(msg, PointCloud2):
            writer.write('obj_points', serialize_message(msg), int(stamp * 1e9))
        elif isinstance(msg, Marker):
            marker_array.append(msg)

    if len(marker_array) > 1:
        marker_array_msg = MarkerArray()
        marker_array_msg.markers = marker_array
        writer.write('obj_boxes', serialize_message(marker_array_msg), int(stamp * 1e9))

    if raw_cloud is not None:
        if raw_cloud.shape[0] > 1e5:
            downsampled_cloud = raw_cloud[np.random.choice(raw_cloud.shape[0], int(1e5), replace=False)]
        else:
            downsampled_cloud = raw_cloud

        ros_raw_pcd = ros2_bag_utils.create_point_cloud(downsampled_cloud, seconds, nanoseconds, frame_id='map')
        writer.write('registered_scan', serialize_message(ros_raw_pcd), int(stamp * 1e9))
    
    if odom is not None:
        odom_msg = ros2_bag_utils.create_odom_msg(odom, seconds, nanoseconds)
        tf_transform = ros2_bag_utils.create_tf_msg(odom, seconds, nanoseconds, 'map', 'sensor')

        writer.write('state_estimation', serialize_message(odom_msg), int(stamp * 1e9))
        writer.write('tf', serialize_message(tf_transform), int(stamp * 1e9))

INSTANCE_LEVEL_OBJECTS = []

OMIT_OBJECTS = [
    "window",
    "door",
]

BACKGROUND_OBJECTS = [
]

VERTICAL_OBJECTS = [
    "door", 'painting', 'picture'
]

BIG_OBJECTS = [
    'sofa', 'table', 'cabinet', 'refrigerator', 'chair', 'screen', 'painting', 'human', 'plant'
]

SMALL_OBJECTS = [
    'books', 'bottle', 'cup', 'bag'
]

DYNAMIC_CLEARING_VOTE_THRESH = {
    'default': 4, # for tracking
    # 'default': 1,

    # highly dynamic
    'person': 1,
    'car': 2,
    'bag': 1,
    # 'chair': 1,
}

MERGE_PRIMITIVE_GROUPS = [
    ['chair', 'sofa'],
    ['table', 'desk'],
]

class ObjMapper():
    def __init__(
        self,
        cloud_image_fusion: CloudImageFusion,
        label_template,
        captioner = None,
        visualize=False,
        log_info=print,
        object_node_pub = None,
        target_object = None,
    ):
        self.single_obj_list: list[SingleObject] = []
        self.background_obj_list = []

        self.cloud_stack = []
        self.stamp_stack = []
        self.valid_cnt = 1

        self.cloud_image_fusion = cloud_image_fusion

        self.label_template = label_template
        self.do_visualize = visualize
        self.object_node_pub = object_node_pub

        self.delete_bbox_list = []
        self.delete_text_list = []

        self.save_queue = queue.Queue()

        if visualize:
            self.rerun_visualizer = VisualizerRerun()
        else:
            self.rerun_visualizer = None

        self.log_info = log_info

        # params
        self.voxel_size = 0.03
        self.confidence_thres = 0.30
        self.cloud_to_odom_dist_thres = 8.0
        self.ground_height = -0.5
        self.num_angle_bin = 20
        self.percentile_thresh = 0.85
        self.clear_outliers_cycle = 1
        self.save_object_image = True # save object images
        self.image_save_interval = 1 # save image every n frames, right now we save every frame
        self.frame_count = 0

        # # possibly useful params
        # self.odom_move_dist_thres = 0.1

        self.target_object = target_object
        self.anchor_object = ""

        for label, val in self.label_template.items():
            if val["is_instance"] and label not in INSTANCE_LEVEL_OBJECTS:
                INSTANCE_LEVEL_OBJECTS.append(label)
            self.label_template[label] = val['prompts']
        
        self.log_info(f"Instance level objects: {INSTANCE_LEVEL_OBJECTS}")
        self.log_info(f"label template: {self.label_template}")

    def is_merge_allowed(self, obj1, obj2):
        """
            Check if the two objects can be merged.
            
        """
        l1, l2 = obj1.get_dominant_label(), obj2.get_dominant_label()
        if l1 == l2:
            return True
        elif [l1, l2] in MERGE_PRIMITIVE_GROUPS or [l2, l1] in MERGE_PRIMITIVE_GROUPS:
            return True
        else:
            return False
    
    def IoU_3D_Bbox(self, bbox_3d_object, bbox_3d_target, extent_object, extent_target):
        # calculating the 3D IoU
        rect_obj = bbox_3d_object[:4, :2]
        rect_target = bbox_3d_target[:4, :2]
        poly_obj = Polygon(rect_obj)
        poly_target = Polygon(rect_target)
        intersection_2d = poly_obj.intersection(poly_target).area

        zmin_obj, zmax_obj = float(bbox_3d_object[0, 2]), float(bbox_3d_object[4, 2])
        zmin_tgt, zmax_tgt = float(bbox_3d_target[0, 2]), float(bbox_3d_target[4, 2])
        height_z = max(0.0, min(zmax_obj, zmax_tgt) - max(zmin_obj, zmin_tgt))
        intersection_3d = intersection_2d * height_z
        bbox3d_object_vol = extent_object[0] * extent_object[1] * extent_object[2]
        bbox3d_target_vol = extent_target[0] * extent_target[1] * extent_target[2]
        total_vol = bbox3d_object_vol + bbox3d_target_vol - intersection_3d
        iou_3d = intersection_3d / total_vol if total_vol > 0 else 0
        ratio_obj = intersection_3d / bbox3d_object_vol if bbox3d_object_vol > 0 else 0
        ratio_target = intersection_3d / bbox3d_target_vol if bbox3d_target_vol > 0 else 0

        if iou_3d == 0 and (ratio_obj > 0 or ratio_target > 0):
            self.log_info(f"🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨Warning: iou_3d is 0 but ratio_obj {ratio_obj:.2f} ratio_target {ratio_target:.2f}")

        return iou_3d, ratio_obj, ratio_target

    # @memory_profiler.profile
    @profile
    def update_map(self, detections, detection_stamp, detection_odom, cloud, image=None, viewpoint_stamp_to_process=None):
        R_b2w = Rotation.from_quat(detection_odom['orientation']).as_matrix()
        t_b2w = np.array(detection_odom['position'])
        R_w2b = R_b2w.T
        t_w2b = -R_w2b @ t_b2w
        cloud_body = cloud @ R_w2b.T + t_w2b

        confidences = np.array(detections['confidences'])
        confidences_mask = (confidences >= self.confidence_thres)
        confidences = confidences[confidences_mask]

        masks = [mask for mask, confidence in zip(detections['masks'], detections['confidences']) if confidence >= self.confidence_thres]
        labels = [label for label, confidence in zip(detections['labels'], detections['confidences']) if confidence >= self.confidence_thres]
        obj_ids = [obj_id for obj_id, confidence in zip(detections['ids'], detections['confidences']) if confidence >= self.confidence_thres]
        bboxes = [bbox for bbox, confidence in zip(detections['bboxes'], detections['confidences']) if confidence >= self.confidence_thres]

        # if viewpoint_stamp_to_process is not None:
        #     self.log_info(f"🎯 [Publish1] {len(masks)}")
        
        t0 = time.time()
        t1 = time.time()
        # self.log_info(f"Mask erode time: {t1 - t0:.2f} seconds, {len(masks)} masks")

        # maintain adjacency graph
        if len(obj_ids) == 0:
            return
        
        t0 = time.time()
        # if viewpoint_stamp_to_process is not None:
        #     obj_clouds_world = self.cloud_image_fusion.generate_seg_cloud(cloud_body, masks, labels, confidences, R_b2w, t_b2w, image)
        # else:
        #     obj_clouds_world = self.cloud_image_fusion.generate_seg_cloud(cloud_body, masks, labels, confidences, R_b2w, t_b2w)
        obj_clouds_world = self.cloud_image_fusion.generate_seg_cloud(cloud_body, masks, labels, confidences, R_b2w, t_b2w)
        
        self.frame_count += 1
        t1 = time.time()
        # self.log_info(f"🚨🚨 Projection time: {t1 - t0:.2f} seconds.")

        # if viewpoint_stamp_to_process is not None:
        #     self.log_info(f"🎯 [Publish2] {len(obj_clouds_world)}")
        
        part1_time = 0
        part2_time = 0
        part3_time = 0
        t0 = time.time()
        # ===================== accociate objects =====================
        for cloud_cnt, cloud in enumerate(obj_clouds_world):
            part1_start = time.time()
            cloud_to_odom_dist = np.linalg.norm(cloud[:, :3] - t_b2w, axis=1)
            dist_mask = (cloud_to_odom_dist < self.cloud_to_odom_dist_thres)
            # dist_mask = dist_mask & (cloud[:, 2] > self.ground_height)
            cloud = cloud[dist_mask]
            
            # if cloud.shape[0] < 5 and viewpoint_stamp_to_process is None:
            if cloud.shape[0] < 2 and viewpoint_stamp_to_process is None:
                # self.log_info(f"Skipping 1")
                part1_end = time.time()
                part1_time += (part1_end - part1_start)
                continue
            
            class_id = labels[cloud_cnt]
            obj_id = obj_ids[cloud_cnt]
            
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])     
            # # pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            # # For each object cloud, we will find the main cluster through DBSCAN clustering
            # pts = np.asarray(pcd.points)
            # # parameters for dbscan, here we should ensure that at most 3 clusters are found, so we use a lenient eps
            # eps,min_cluster_size = 0.3, 10
            # db_labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_cluster_size, print_progress=False))
            # unique_labels = np.unique(db_labels)

            # valid_clusters = []
            # for label in unique_labels:
            #     if label == -1:
            #         continue
            #     cluster_mask = db_labels == label
            #     cluster_pts = pts[cluster_mask]
            #     avg_dist = np.mean(np.linalg.norm(cluster_pts - t_b2w, axis=1))
            #     valid_clusters.append((avg_dist, cluster_pts))

            # if len(valid_clusters) == 0:
            #     self.log_info(f"Skipping 3")
            #     continue

            # valid_clusters.sort(key=lambda x: x[0])
            # # Check the number of clusters found
            # self.log_info(f"Found {len(valid_clusters)} clusters for {class_id} with obj_id {obj_id} at {detection_stamp}")

            # dists = np.array([v[0] for v in valid_clusters])
            # self.log_info(f"!Cluster Size: {dists.size}")
            # pruned = False
            # if dists.size >= 3:
            #     q1 = np.percentile(dists, 25)
            #     q3 = np.percentile(dists, 75)
            #     iqr = q3 - q1
            #     if iqr > 0:
            #         upper = q3 + 1.5 * iqr
            #         if dists[-1] > upper:
            #             self.log_info(f"PRUNE far cluster: {dists[-1]:.2f} > {upper:.2f} (IQR rule)")
            #             pruned = True

            # elif dists.size == 2:
            #     # Simple ratio fallback for 2 clusters
            #     if dists[-1] > dists[0] * 2.0:
            #         self.log_info(f"PRUNE far cluster (2-cluster ratio): {dists[-1]:.2f} vs {dists[0]:.2f}")
            #         pruned = True
                    

            # if len(valid_clusters) == 0:
            #     continue

            # chosen_cluster_pts = valid_clusters[0][1] # select the nearest cluster
            # 
            # chosen_cluster_pts = valid_clusters[0][1]
            # points_np = chosen_cluster_pts
            # if pcd.is_empty():
            #     self.log_info(f"Skipping 2")
            #     part1_end = time.time()
            #     part1_time += (part1_end - part1_start)
            #     continue
            # points_np = np.array(pcd.points)

            pcd_new = o3d.geometry.PointCloud()
            pcd_new.points = o3d.utility.Vector3dVector(cloud[:, :3])
            # pcd_downsampled = pcd_new
            pcd_downsampled = pcd_new.voxel_down_sample(voxel_size=self.voxel_size)
            # Here for testing, we do not remove the outliers
            # pcd_downsampled, _ = pcd_downsampled.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
            points_np = np.array(pcd_downsampled.points)
            clip_feat = detections['clip_feats'][cloud_cnt] if 'clip_feats' in detections else None
            part1_end = time.time()
            part1_time += (part1_end - part1_start)

            part2_start = time.time()
            # Match object cloud to existing object based on object id
            merged = False
            
            if obj_id < 0:
                # for background_obj in self.background_obj_list:
                #     if obj_id in background_obj.obj_id:
                #         background_obj.merge(np.array(pcd.points), R_b2w, t_b2w, class_id, detection_stamp)
                #         background_obj.inactive_frame = -1
                #         merged = True
                #         break
                pass
            else:
                for single_obj in self.single_obj_list:
                    # Add more secure check apart from obj_id appearance
                    if obj_id in single_obj.obj_id:
                        single_obj.update(points_np, R_b2w, t_b2w, class_id, detection_stamp, clip_feat=clip_feat, confidence=confidences[cloud_cnt])
                        single_obj.reproject_obs_angle(R_w2b, t_w2b, masks[cloud_cnt], projection_func=self.cloud_image_fusion.scan2pixels)
                        single_obj.inactive_frame = -1
                        # Save masked image if enabled and image is provided
                        if (self.save_object_image and image is not None and 
                            self.frame_count % self.image_save_interval == 0):
                            single_obj.save_best_image(
                                image, 
                                masks[cloud_cnt],
                                confidences[cloud_cnt],
                                self.save_queue
                            )
                        merged = True
                        break
            part2_end = time.time()
            part2_time += (part2_end - part2_start)

            part3_start = time.time()
            if not merged:
                if obj_id < 0:
                    self.background_obj_list.append(SingleObject(class_id, obj_id, points_np, \
                        self.voxel_size, R_b2w, t_b2w, masks[cloud_cnt], detection_stamp, num_angle_bin=self.num_angle_bin,confidence=confidences[cloud_cnt]))
                else:
                    new_obj = SingleObject(class_id, obj_id, points_np, \
                        self.voxel_size, R_b2w, t_b2w, masks[cloud_cnt], detection_stamp, num_angle_bin=self.num_angle_bin,confidence=confidences[cloud_cnt])
                    # Save initial masked image if enabled
                    if self.save_object_image and image is not None:
                        new_obj.save_best_image(
                            image, 
                            masks[cloud_cnt],
                            confidences[cloud_cnt],
                            self.save_queue
                        )
                    self.single_obj_list.append(new_obj)
            part3_time_end = time.time()
            part3_time += (part3_time_end - part3_start)
        
        t1 = time.time()
        # self.log_info(f"Updated objects in {t1 - t0:.2f} seconds.")
        # self.log_info(f"  Part1 time: {part1_time:.2f} seconds.")
        # self.log_info(f"  Part2 time: {part2_time:.2f} seconds.")
        # self.log_info(f"  Part3 time: {part3_time:.2f} seconds.")
        
        # ===================== optimize objects in world =====================
        t0 = time.time()
        i = 0
        while i < len(self.single_obj_list):
            if (i >= len(self.single_obj_list)):
                break
            single_obj = self.single_obj_list[i]
            single_obj.life += 1

            if single_obj.inactive_frame > 20:
                # self.log_info(f"Delete inactive object {single_obj.class_id}:{single_obj.obj_id}")
                # self.publish_deleted_object(single_obj, detection_stamp, i)
                # single_obj.cleanup_images()
                # self.single_obj_list.remove(single_obj)
                # self.log_info(f"Remove {single_obj.class_id}:{single_obj.obj_id}")
                i += 1
                continue

            merged_obj = False
            merged_diff_obj = False
            swapped = False
            # self.log_info(f"!!!! Obj {single_obj.class_id}:{single_obj.obj_id} status updated: {single_obj.updated}")
            if single_obj.life < 1200 and single_obj.life > 0: # TODO: adding single_obj.updated here is correct?
                # self.log_info(f"Computing obj {single_obj.class_id}:{single_obj.obj_id} with life {single_obj.life}")
                if not single_obj.updated:
                    single_obj.inactive_frame += 1
                    single_obj.regularize_shape_v2(self.percentile_thresh)
                    # self.log_info(f"Obj {single_obj.class_id}:{single_obj.obj_id} with points {single_obj.valid_indices_regularized.shape[0]}, inactive frame {single_obj.inactive_frame}")
                    if single_obj.valid_indices_regularized.shape[0] < 15 and single_obj.inactive_frame > 50 and single_obj.get_dominant_label!=self.target_object: # TODO: voxel count thresh should be related to the object class
                        self.publish_deleted_object(single_obj, detection_stamp)
                        single_obj.cleanup_images(self.save_queue)
                        self.single_obj_list.remove(single_obj)
                        continue
                    else:
                        i += 1
                        continue
                
                # this branch means single_obj.updated = True
                if single_obj.life == 1:
                    single_obj.compute_valid_indices(self.percentile_thresh)

                centroid = single_obj.infer_centroid(diversity_percentile=self.percentile_thresh, regularized=True)
                label = single_obj.get_dominant_label()
                if centroid is None:
                    i += 1
                    continue

                target_obj_same = None
                target_obj_diff = None
                target_index_same = -1
                target_index_diff = -1
                minimum_dist_same = np.inf
                minimum_dist_diff = np.inf
                same_class_obj = None
                diff_class_obj = None
                # j = i + 1
                j = 0
                while j < len(self.single_obj_list):
                    j += 1
                    curr_obj = self.single_obj_list[j-1]
                    target_centroid = curr_obj.infer_centroid(diversity_percentile=self.percentile_thresh, regularized=True)
                    if target_centroid is None or j - 1 == i:
                        continue

                    dist = np.linalg.norm(target_centroid - centroid)
                    if curr_obj.get_dominant_label() != label:
                        if dist < minimum_dist_diff:
                            minimum_dist_diff = dist
                            target_obj_diff = curr_obj
                            target_index_diff = j - 1
                    if curr_obj.get_dominant_label() == label:
                        if dist < minimum_dist_same:
                            minimum_dist_same = dist
                            target_obj_same = curr_obj
                            target_index_same = j - 1
                # self.log_info(f"Obj {single_obj.class_id}:{single_obj.obj_id} found nearest same-class obj {target_obj_same.class_id if target_obj_same is not None else None}:{target_obj_same.obj_id if target_obj_same is not None else None} at distance {minimum_dist_same:.2f}")
                if target_obj_same is not None:
                    (center_object, extent_object, q_object), bbox3d_object = single_obj.infer_bbox_oriented(diversity_percentile=self.percentile_thresh, regularized=True)
                    (center_target, extent_target, q_target), bbox3d_target = target_obj_same.infer_bbox_oriented(diversity_percentile=self.percentile_thresh, regularized=True)
                    if extent_object is None or extent_target is None:
                        i += 1
                        continue

                    dist_thresh = np.linalg.norm((extent_object/2 + extent_target/2)/2) * 0.5
                    # merge directly if the distance is small
                    if (minimum_dist_same < dist_thresh or minimum_dist_same < 0.25):
                        # self.log_info(f"Distance Merge {single_obj.class_id}:{single_obj.obj_id} to {target_obj_same.class_id}:{target_obj_same.obj_id} with dist thresh {minimum_dist_same}")
                        merged_obj = True

                    if not merged_obj:
                        iou_3d, ratio_obj, ratio_target = self.IoU_3D_Bbox(bbox3d_object, bbox3d_target, extent_object, extent_target)
                        # self.log_info(f"Same Class Object {single_obj.class_id}:{single_obj.obj_id} to {target_obj_same.class_id}:{target_obj_same.obj_id} IoU {iou_3d:.2f}, ratio_obj {ratio_obj:.2f}, ratio_target {ratio_target:.2f}")
                        if iou_3d > 0.20 or ratio_obj > 0.4 or ratio_target > 0.4:
                            self.log_info(f"Same Class IoU Merge {single_obj.class_id}:{single_obj.obj_id} to {target_obj_same.class_id}:{target_obj_same.obj_id} with IoU {iou_3d:.2f} ratio_obj {ratio_obj:.2f} ratio_target {ratio_target:.2f}")
                            merged_obj = True
                            # self.log_info(f"Vol of object is {bbox3d_object_vol:.2f}, the dimension is {extent_object}")
                            # self.log_info(f"Vol of target is {bbox3d_target_vol:.2f}, the dimension is {extent_target}")
                    if merged_obj:
                        if target_index_same < i and not swapped:
                                single_obj, target_obj_same = target_obj_same, single_obj
                                target_index_same = i
                        single_obj.merge_object(target_obj_same)
                        single_obj.inactive_frame = -1
                        self.publish_deleted_object(target_obj_same, detection_stamp)
                        target_obj_same.cleanup_images(self.save_queue)
                        self.single_obj_list.remove(target_obj_same)
                        i -= 1
                        del target_obj_same
                    
                # # whether to perform cross-class merging
                # if (not merged_obj) and (target_obj_diff is not None):
                #     (center_object, extent_object, q_object), bbox3d_object = single_obj.infer_bbox_oriented(diversity_percentile=self.percentile_thresh, regularized=True)
                #     (center_target, extent_target, q_target), bbox3d_target = target_obj_diff.infer_bbox_oriented(diversity_percentile=self.percentile_thresh, regularized=True)

                #     if center_object is None or center_target is None:
                #         i += 1
                #         continue

                #     # # merge directly if the distance is extremely small
                #     # if (minimum_dist_diff < 0.1):
                #     #     self.log_info(f"Distance Merge {single_obj.class_id}:{single_obj.obj_id} to {target_obj_diff.class_id}:{target_obj_diff.obj_id} with dist thresh {minimum_dist_diff}")
                #     #     merged_obj = True

                #     if not merged_obj:
                #         iou_3d, ratio_obj, ratio_target = self.IoU_3D_Bbox(bbox3d_object, bbox3d_target, extent_object, extent_target)
                #         # self.log_info(f"Diff Class Object {single_obj.class_id}:{single_obj.obj_id} to {target_obj_diff.class_id}:{target_obj_diff.obj_id} IoU {iou_3d:.2f}, ratio_obj {ratio_obj:.2f}, ratio_target {ratio_target:.2f}")
                #         if iou_3d > 0.7 or ratio_obj > 0.9 or ratio_target > 0.9:
                #             self.log_info(f"Diff Class IoU Merge {single_obj.class_id}:{single_obj.obj_id} to {target_obj_diff.class_id}:{target_obj_diff.obj_id} with IoU {iou_3d:.2f} ratio_obj {ratio_obj:.2f} ratio_target {ratio_target:.2f}")
                #             merged_obj = True
                #     if merged_obj:
                #         if target_index_diff < i and not swapped:
                #                 single_obj, target_obj_diff = target_obj_diff, single_obj
                #                 target_index_diff = i
                #         single_obj.merge_object(target_obj_diff)
                #         single_obj.inactive_frame = -1
                #         self.publish_deleted_object(target_obj_diff, detection_stamp)
                #         target_obj_diff.cleanup_images(self.save_queue)
                #         self.single_obj_list.remove(target_obj_diff)
                #         i -= 1
                #         del target_obj_diff

            if not merged_obj:
                single_obj.regularize_shape_v2(self.percentile_thresh)
                i += 1
        

        t1 = time.time()
        # self.log_info(f"Optimized objects in {t1 - t0:.2f} seconds")

        self.valid_cnt += 1


    def publish_deleted_object(self, single_obj, detection_stamp):
        if self.object_node_pub is None:
            return
        seconds = int(detection_stamp)
        nanoseconds = int((detection_stamp - seconds) * 1e9)

        delete_msg = ObjectNode()
        delete_msg.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
        delete_msg.header.frame_id = 'map'
        delete_msg.object_id = single_obj.obj_id
        delete_msg.label = single_obj.get_dominant_label()
        delete_msg.status = "deleted"

        centroid = single_obj.infer_centroid(diversity_percentile=self.percentile_thresh, regularized=True)
        if centroid is not None:
            delete_msg.position.x = float(centroid[0])
            delete_msg.position.y = float(centroid[1])
            delete_msg.position.z = float(centroid[2])

        self.object_node_pub.publish(delete_msg)

        for obj_id in single_obj.obj_id:
            marker_bbox = Marker()
            marker_bbox.header.frame_id = 'map'
            marker_bbox.type = Marker.LINE_LIST
            marker_bbox.action = Marker.DELETE
            marker_bbox.id = int(obj_id)
            marker_bbox.ns = 'bbox'
            marker_bbox.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
            self.delete_bbox_list.append(marker_bbox)

            marker_text = Marker()
            marker_text.header.frame_id = 'map'
            marker_text.type = Marker.TEXT_VIEW_FACING
            marker_text.action = Marker.DELETE
            marker_text.id = int(obj_id)
            marker_text.ns = 'text'
            marker_text.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
            self.delete_text_list.append(marker_text)

    
    # @profile
    # def publish_all_objects(self, stamp):
    #     if self.object_node_pub is None:
    #         return
            
    #     seconds = int(stamp)
    #     nanoseconds = int((stamp - seconds) * 1e9)
        
    #     valid_objects_count = 0
    #     for single_obj in self.single_obj_list:

    #         if not getattr(single_obj,'updated',True):
    #             # if the object is not updated in this cycle, we skip it
    #             continue

    #         obj_points = single_obj.retrieve_valid_voxels(
    #             diversity_percentile=self.percentile_thresh,
    #             regularized=True
    #         )
            
    #         # # check if the object has valid points
    #         if len(obj_points) == 0:
    #             centroid = np.array([0, 0, 0])
    #         else:
    #             # center of mass
    #             centroid = single_obj.infer_centroid(
    #                 diversity_percentile=self.percentile_thresh, 
    #                 regularized=True
    #             )
                
    #         # create ObjectNode message
    #         object_node_msg = ObjectNode()
    #         object_node_msg.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
    #         object_node_msg.header.frame_id = 'map'
            
    #         # Get the dominant id
    #         object_node_msg.object_id = single_obj.obj_id
    #         object_node_msg.label = single_obj.get_dominant_label()
    #         object_node_msg.status = getattr(single_obj, 'publish_status', 'new')

    #         object_node_msg.position.x = float(centroid[0])
    #         object_node_msg.position.y = float(centroid[1])
    #         object_node_msg.position.z = float(centroid[2])

    #         # create point cloud - use the simplest method
    #         ros_pcd = ros2_bag_utils.create_point_cloud(
    #             obj_points, seconds, nanoseconds, frame_id='map'
    #         )
    #         object_node_msg.cloud = ros_pcd
            
    #         self.object_node_pub.publish(object_node_msg)
    #         valid_objects_count += 1
    #         single_obj.last_published_stamp = stamp  # update last published stamp
    #         single_obj.updated = False  # reset updated flag after publishing
    #         single_obj.publish_status = 'unchanged'  # reset publish status after publishing
    #         # self.log_info(f"🎯 Published object {single_obj.obj_id[0]} ({single_obj.get_dominant_label()}) with {len(obj_points)} points")
                
    def to_ros2_msgs_deleted(self, stamp):
        bbox_msg_list = []
        text_msg_list = []
        for del_bbox in self.delete_bbox_list:
            bbox_msg_list.append(del_bbox)
        for del_text in self.delete_text_list:
            text_msg_list.append(del_text)
        
        self.delete_bbox_list = []
        self.delete_text_list = []

        return bbox_msg_list, text_msg_list

    def to_ros2_msgs(self, stamp, viewpoint_id):
        # colors_to_choose = generate_colors(len(self.single_obj_list), is_int=False)
        seconds = int(stamp)
        nanoseconds = int((stamp - seconds) * 1e9)

        points_list = []
        colors_list = []
        bbox_msg_list = []
        text_msg_list = []

        for single_obj in self.single_obj_list:
            obj_points = single_obj.retrieve_valid_voxels(
                diversity_percentile=self.percentile_thresh,
                regularized=True
            )
            label = single_obj.get_dominant_label()

            if len(obj_points) == 0:
                continue

            point = obj_points
            color = np.array([map_label_to_color(label)] * obj_points.shape[0]) 
            points_list.append(point)
            colors_list.append(color)

            if not getattr(single_obj,'updated',True):
                # if the object is not updated in this cycle, we skip it
                continue
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point)
            pcd.colors = o3d.utility.Vector3dVector(color)

            aabb = pcd.get_axis_aligned_bounding_box()
            center = aabb.get_center()
            extent = aabb.get_extent()
            # center[2] += extent[2] / 2  # adjust the center to the middle of the box height

            (center_object, extent_object, q_object), bbox3d_corners = single_obj.infer_bbox_oriented(diversity_percentile=self.percentile_thresh, regularized=True)

            # add the confidence here 
            confidence = single_obj.get_dominant_confidence()
            id = single_obj.obj_id[0]
            label = single_obj.get_dominant_label()
            label_with_id = f"{single_obj.get_dominant_label()}({id})"
            label_with_confidence = f"{single_obj.get_dominant_label()} ({confidence:.2f})"
            color = map_label_to_color(label) if label != self.target_object else [1.0, 0.0, 0.0]

            obj_marker = ros2_bag_utils.create_wireframe_marker(
                center=aabb.get_center(),
                extent=aabb.get_extent(),
                yaw=0.0,
                ns='bbox',
                box_id=single_obj.obj_id[0],
                color=color,
                seconds=seconds,
                nanoseconds=nanoseconds,
                frame_id='map'
            )

            # if bbox3d_corners is None:
            #     continue
            # obj_marker = ros2_bag_utils.create_wireframe_marker_from_corners(
            #     corners=bbox3d_corners,
            #     ns='bbox',
            #     box_id=single_obj.obj_id[0],
            #     color=color,
            #     seconds=seconds,
            #     nanoseconds=nanoseconds,
            #     frame_id='map'
            # )


            text_msg = ros2_bag_utils.create_text_marker(
                center=center,
                marker_id=single_obj.obj_id[0],
                # text = label,
                text = label_with_id,
                # text=label_with_confidence,
                color=color,
                text_height=0.4,
                seconds=seconds,
                nanoseconds=nanoseconds,
                frame_id='map'
            )

            bbox_msg_list.append(obj_marker)
            text_msg_list.append(text_msg)

            # # check if the object has valid points
            if len(obj_points) == 0:
                centroid = np.array([0, 0, 0])
            else:
                # center of mass
                centroid = single_obj.infer_centroid(
                    diversity_percentile=self.percentile_thresh, 
                    regularized=True
                )
                
            # create ObjectNode message
            object_node_msg = ObjectNode()
            object_node_msg.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
            object_node_msg.header.frame_id = 'map'
            
            # Get the dominant id
            object_node_msg.object_id = single_obj.obj_id

            # if this object is target object and haven't been checked by the vlm, we give it a fake label
            if label == self.target_object and not single_obj.is_asked_vlm:
                object_node_msg.label = "Potential Target"
            else:
                object_node_msg.label = label
            object_node_msg.status = getattr(single_obj, 'publish_status', 'new')

            object_node_msg.position.x = float(centroid[0])
            object_node_msg.position.y = float(centroid[1])
            object_node_msg.position.z = float(centroid[2])

            if bbox3d_corners is None:
                continue
            for bbox_id in range(8):
                object_node_msg.bbox3d[bbox_id].x = float(bbox3d_corners[bbox_id, 0])
                object_node_msg.bbox3d[bbox_id].y = float(bbox3d_corners[bbox_id, 1])
                object_node_msg.bbox3d[bbox_id].z = float(bbox3d_corners[bbox_id, 2])

            # create point cloud - use the simplest method
            ros_pcd = ros2_bag_utils.create_point_cloud(
                obj_points, seconds, nanoseconds, frame_id='map'
            )
            object_node_msg.cloud = ros_pcd
            object_node_msg.img_path = single_obj.best_image_path
            object_node_msg.is_asked_vlm = single_obj.is_asked_vlm
            object_node_msg.viewpoint_id = viewpoint_id if not single_obj.updated_by_vlm else -1
            
            self.object_node_pub.publish(object_node_msg)
            single_obj.last_published_stamp = stamp  # update last published stamp
            single_obj.updated = False  # reset updated flag after publishing
            single_obj.updated_by_vlm = False
            single_obj.publish_status = 'unchanged'  # reset publish status after publishing

        if len(points_list) != 0:
            points = np.concatenate(points_list, axis=0)
            colors = np.concatenate(colors_list, axis=0)
            ros_pcd = ros2_bag_utils.create_colored_point_cloud(
                points=points,
                colors=colors,
                seconds=seconds,
                nanoseconds=nanoseconds,
                frame_id='map'
            )
        else:
            ros_pcd = None

        return bbox_msg_list, text_msg_list, ros_pcd

    def rerun_vis(self, odom, regularized=True, show_bbox=False, debug=False, enforce=False):
        if self.do_visualize:
            if debug:
                self.rerun_visualizer.visualize_debug(self.single_obj_list, odom)
            else:
                self.rerun_visualizer.visualize(self.single_obj_list, odom, regularized=regularized, show_bbox=show_bbox)
        else:
            if enforce:
                self.rerun_visualizer = VisualizerRerun() if self.rerun_visualizer is None else self.rerun_visualizer
                self.rerun_visualizer.visualize(self.single_obj_list, odom, regularized=regularized, show_bbox=show_bbox)
            else:
                self.log_info("Visualizer is not enabled!!!")
    
    def print_obj_info(self):
        self.log_info('==== All Objects Info ====')
        for single_obj in self.single_obj_list:
            obj_str = single_obj.get_info_str()
            self.log_info(obj_str)
    
    def check_target_objects(self):
        # iterate through all objects to find the target object
        target_objects = []
        for single_obj in self.single_obj_list:
            class_id_length = len(single_obj.class_id)
            label = single_obj.get_dominant_label()
            if (label == self.target_object or label == self.anchor_object) and not single_obj.is_asked_vlm:
                single_target_obj = {
                    'object_id': single_obj.obj_id[0],
                    'labels': list(single_obj.class_id.keys()),
                    'img_path': single_obj.best_image_path,
                }
                target_objects.append(single_target_obj)
        return target_objects

    def update_target_object(self, object_id, new_label):
        for single_obj in self.single_obj_list:
            if object_id in single_obj.obj_id:
                old_label = single_obj.get_dominant_label()
                if old_label == new_label:
                    single_obj.class_id[new_label] += 50
                    single_obj.conf_list[new_label] == 50.0
                else:
                    single_obj.class_id[new_label] = single_obj.class_id.get(new_label, 0) + 50
                    single_obj.conf_list[new_label] = 50.0

                single_obj.is_asked_vlm = True
                single_obj.updated = True # force publish after VLM update
                single_obj.updated_by_vlm = True


