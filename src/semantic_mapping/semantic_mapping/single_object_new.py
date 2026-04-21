import cv2
import datetime
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .utils import normalize_angles_to_pi, R_to_yaw, discretize_angles
import open3d as o3d

from line_profiler import profile

import os
import time

# not using these
DIMENSION_PRIORS = {
'default': (4.0, 4.0, 2.0), 
'table': (3.0, 3.0, 2.0), 
'chair': (1.5, 1.5, 1.5),
'sofa': (2.0, 2.0, 1.0),
'tv_monitor': (1.0, 0.2, 0.6),
# 'pottedplant': (1.0, 1.0, 1.0),
# 'fireextinguisher': (0.5, 0.5, 1.0),
'door': (1.0, 0.2, 2.5),
'potted_plant': (0.8, 0.8, 1.5),
'garbage_bin': (1.0, 1.0, 1.0),
# 'person': (1.0, 1.0, 2.0),
# 'case': (1.0, 1.0, 1.0)
}

def percentile_index_search_binary(sorted_weights, percentile):
    total_weight = np.sum(sorted_weights)
    percentile_weight = total_weight * percentile
    current_weight = 0
    # left = 0
    # right = len(sorted_weights)
    # while left < right:
    #     mid = (left + right) // 2
    #     current_weight += sorted_weights[mid]
    #     if current_weight > percentile_weight:
    #         right = mid
    #     else:
    #         left = mid + 1
    i = 0
    while i < len(sorted_weights) and current_weight < percentile_weight:
        current_weight += sorted_weights[i]
        i += 1

    return i

def get_box_3d(points):
    min_xyz = points[:, :3].min(axis=0)
    max_xyz = points[:, :3].max(axis=0)
    center = (min_xyz + max_xyz) / 2
    extent = (max_xyz - min_xyz)
    q = [0.0, 0.0, 0.0, 1.0] # xyzw
    return center, extent, q

class VoxelFeatureManager:
    def __init__(self, voxels: np.array, voxel_size: int, odom_R, odom_t, num_angle_bin=15):
        num_points = voxels.shape[0]
        
        voxel_to_odom = voxels - odom_t
        voxel_to_odom = voxel_to_odom @ odom_R # transform to body frame. Note: odom_R = odom_R.T.T
        obs_angles = np.arctan2(voxel_to_odom[:, 1], voxel_to_odom[:, 0])
        obs_angles = normalize_angles_to_pi(obs_angles)
        obs_angles = discretize_angles(obs_angles, num_angle_bin)
        
        self.voxels = voxels
        self.voxel_size = voxel_size
        self.tree = cKDTree(voxels)
        self.vote = np.ones(num_points)
        self.observation_angles = np.zeros([num_points, num_angle_bin])
        self.observation_angles[np.arange(num_points), obs_angles] = 1
        
        self.regularized_voxel_mask = np.zeros(num_points, dtype=bool)

        self.remove_vote = np.zeros(num_points, dtype=int)
        
        self.num_angle_bin = num_angle_bin
    
    # @profile
    def update(self, voxels, odom_R, odom_t):
        num_points = voxels.shape[0]
        
        voxel_to_odom = voxels - odom_t
        # voxel_to_odom = voxel_to_odom @ odom_R
        obs_angles = np.arctan2(voxel_to_odom[:, 1], voxel_to_odom[:, 0])
        obs_angles = normalize_angles_to_pi(obs_angles)
        obs_angles = discretize_angles(obs_angles, self.num_angle_bin)
        
        distances, indices = self.tree.query(voxels) # indices: the index of the closest point in the original point cloud
        indices_to_merge = indices[distances < self.voxel_size]
        obs_angles_to_merge = obs_angles[distances < self.voxel_size]
        self.vote[indices_to_merge] += 1
        self.observation_angles[indices_to_merge, obs_angles_to_merge] = 1
        
        # process new voxels
        new_voxels = voxels[distances >= self.voxel_size]
        self.vote = np.concatenate([self.vote, np.ones(new_voxels.shape[0])])
        new_observation_angles = np.zeros([new_voxels.shape[0], self.num_angle_bin])
        new_observation_angles[np.arange(new_voxels.shape[0]), obs_angles[distances >= self.voxel_size]] = 1
        self.observation_angles = np.concatenate([self.observation_angles, new_observation_angles], axis=0)
        self.voxels = np.concatenate([self.voxels, new_voxels], axis=0)
        
        self.regularized_voxel_mask = np.concatenate([self.regularized_voxel_mask, np.zeros(new_voxels.shape[0], dtype=bool)])
        
        self.tree = cKDTree(self.voxels)

        self.remove_vote = np.concatenate([self.remove_vote, np.zeros(new_voxels.shape[0], dtype=int)])
        assert self.observation_angles.shape[0] == self.vote.shape[0] == self.voxels.shape[0] == self.regularized_voxel_mask.shape[0]
    
    def update_through_vote_stat(self, vote_stat):
        voxels = vote_stat.voxels
        votes = vote_stat.vote
        obs_angles = vote_stat.observation_angles
        
        distances, indices = self.tree.query(voxels)
        indices_to_merge = indices[distances < self.voxel_size]
        obs_angles_to_merge = obs_angles[distances < self.voxel_size]
        self.vote[indices_to_merge] += votes[distances < self.voxel_size]
        self.observation_angles[indices_to_merge] = np.logical_or(self.observation_angles[indices_to_merge], obs_angles_to_merge)
        
        new_voxels = voxels[distances >= self.voxel_size]
        new_votes = votes[distances >= self.voxel_size]
        new_observation_angles = obs_angles[distances >= self.voxel_size]
        new_regularized_voxel_mask = vote_stat.regularized_voxel_mask[distances >= self.voxel_size]
        self.voxels = np.concatenate([self.voxels, new_voxels])
        self.vote = np.concatenate([self.vote, new_votes])
        self.observation_angles = np.concatenate([self.observation_angles, new_observation_angles])
        self.tree = cKDTree(self.voxels)

        self.regularized_voxel_mask = np.concatenate([self.regularized_voxel_mask, new_regularized_voxel_mask])
        self.remove_vote = np.concatenate([self.remove_vote, np.zeros(new_voxels.shape[0], dtype=int)])

        assert self.observation_angles.shape[0] == self.vote.shape[0] == self.voxels.shape[0] == self.regularized_voxel_mask.shape[0]
    
    def increment_remove_vote(self, increment_mask):
        self.remove_vote[increment_mask] += 1
    
    def decrease_remove_vote(self, decrement_mask):
        self.remove_vote[decrement_mask] -= 1
        self.remove_vote[self.remove_vote < 0] = 0

    def reset_remove_vote(self,reset_mask):
        self.remove_vote[reset_mask] = 0

    def update_through_mask(self, mask):
        self.voxels = self.voxels[mask]
        self.vote = self.vote[mask]
        self.observation_angles = self.observation_angles[mask]
        self.tree = cKDTree(self.voxels)
        self.remove_vote = self.remove_vote[mask]
        
        assert self.observation_angles.shape[0] == self.vote.shape[0] == self.voxels.shape[0] == self.remove_vote.shape[0]
    
    def cal_distance(self, voxels):
        distances, indices = self.tree.query(voxels)
        return distances.mean()
    
    def reproject_depth_mask(self, R_w2b, t_w2b, depth_image, depth_projection_func, det_mask=None, rgb_projection_func=None, debug_reproj_indices=None):
        """
            Ray casting the object points given a depth mask.

            R_w2b: rotation from world to body frame
            t_w2b: translation from world to body frame
            depth_image: the depth image of the object, assume invalid depth is 0
            depth_projection_func: the function to project the points to the depth image
            det_mask (bool): the mask of the latest detection, used to filter out surface points that are ON the mask

            Returns:
                valid_voxels_mask: the mask of the voxels that are on or behind the depth image
                on_surface_oxels_mask: the mask of the voxels that are on the surface of the depth image
        """
        voxels = self.voxels @ R_w2b.T + t_w2b # convert to camera frame

        vox_on_depth_image = depth_projection_func(voxels)
        depth_shape = depth_image.shape
        out_of_bound_filter = (vox_on_depth_image[:, 0] >= 0) & \
                            (vox_on_depth_image[:, 0] < depth_shape[1]) & \
                            (vox_on_depth_image[:, 1] >= 0) & \
                            (vox_on_depth_image[:, 1] < depth_shape[0]) & \
                            (vox_on_depth_image[:, 2] > 0) # only keep the points that are in the depth image
        
        vox_on_depth_image = vox_on_depth_image[out_of_bound_filter]

        if debug_reproj_indices is not None and isinstance(debug_reproj_indices, list):
            debug_reproj_indices = vox_on_depth_image

        vox_on_depth_image_depth = vox_on_depth_image[:, 2]
        vox_on_depth_image_idx_hor, vox_on_depth_image_idx_vert = vox_on_depth_image[:, 0].astype(int), vox_on_depth_image[:, 1].astype(int)
        obj_depth_mask = depth_image[vox_on_depth_image_idx_vert, vox_on_depth_image_idx_hor]
        
        valid_voxels_mask = (vox_on_depth_image_depth - obj_depth_mask) > -0.05 # object point in front of the mask
        on_surface_voxels_mask = np.logical_and((vox_on_depth_image_depth - obj_depth_mask) < 0.1, valid_voxels_mask) # object point on the mask

        tlbr = np.array([0, 0, 0, 0])
        if rgb_projection_func is not None:
            # BUG: this can be more efficient with no extra projection
            vox_on_rgb = rgb_projection_func(voxels[out_of_bound_filter])
            rgb_shape = det_mask.shape
            rgb_out_of_bound_filter = (vox_on_rgb[:, 0] >= 0) & \
                                    (vox_on_rgb[:, 0] < rgb_shape[1]) & \
                                    (vox_on_rgb[:, 1] >= 0) & \
                                    (vox_on_rgb[:, 1] < rgb_shape[0]) & \
                                    (vox_on_rgb[:, 2] > 0) # only keep the points that are in the rgb image
            
            # calculate the on_surface_voxels tlbr on image frame
            on_surface_voxels_rgb_mask = on_surface_voxels_mask & rgb_out_of_bound_filter
            if on_surface_voxels_rgb_mask.sum() > 30:
                tlbr = np.array([0, 0, 0, 0])
            else:
                valid_voxels = voxels[self.retrieve_valid_voxel_indices(0.9, regularized=True)]
                full_vox_on_rgb = rgb_projection_func(valid_voxels)
                rgb_out_of_bound_filter_full = (full_vox_on_rgb[:, 0] >= 0) & \
                                        (full_vox_on_rgb[:, 0] < rgb_shape[1]) & \
                                        (full_vox_on_rgb[:, 1] >= 0) & \
                                        (full_vox_on_rgb[:, 1] < rgb_shape[0]) & \
                                        (full_vox_on_rgb[:, 2] > 0) # only keep the points that are in the rgb image
                
                full_vox_on_rgb = full_vox_on_rgb[rgb_out_of_bound_filter_full]

                if len(full_vox_on_rgb) < 30:
                    tlbr = np.array([0, 0, 0, 0])
                else:
                    vox_on_surface_rgb_idx_hor, vox_on_surface_rgb_idx_vert = full_vox_on_rgb[:, 0].astype(int), full_vox_on_rgb[:, 1].astype(int)
                    top, left = vox_on_surface_rgb_idx_vert.min(), vox_on_surface_rgb_idx_hor.min()
                    bottom, right = vox_on_surface_rgb_idx_vert.max(), vox_on_surface_rgb_idx_hor.max()
                    tlbr = np.array([left, top, right, bottom])

            vox_on_rgb = vox_on_rgb[rgb_out_of_bound_filter]
            vox_on_rgb_idx_hor, vox_on_rgb_idx_vert = vox_on_rgb[:, 0].astype(int), vox_on_rgb[:, 1].astype(int)

            det_mask_filter = det_mask[vox_on_rgb_idx_vert, vox_on_rgb_idx_hor].astype(bool)
            det_mask_filter_with_surface_vox_shape = np.zeros_like(on_surface_voxels_mask)
            det_mask_filter_with_surface_vox_shape[rgb_out_of_bound_filter] = det_mask_filter
            on_surface_voxels_mask = on_surface_voxels_mask & (~det_mask_filter_with_surface_vox_shape)

        # recover the mask to global indices
        valid_voxels_mask_global = np.ones(self.voxels.shape[0], dtype=bool)
        valid_voxels_mask_global[out_of_bound_filter] = valid_voxels_mask

        on_surface_voxels_mask_global = np.zeros(self.voxels.shape[0], dtype=bool)
        on_surface_voxels_mask_global[out_of_bound_filter] = on_surface_voxels_mask


        return valid_voxels_mask_global, on_surface_voxels_mask_global, tlbr
        
    def reproject_positive_mask(self, R_w2b, t_w2b, mask, projection_func):
        voxels = self.voxels
        voxels = voxels @ R_w2b.T + t_w2b
        vox_on_depth_image = projection_func(voxels).astype(int)
        voxels_mask = mask[vox_on_depth_image[:, 1], vox_on_depth_image[:, 0]].astype(bool)
        
        self.vote[~voxels_mask] -= 1

    def reproject_obs_angle(self, R_w2b, t_w2b, mask, projection_func, image_src=None):
        voxels = self.voxels
        voxels = voxels @ R_w2b.T + t_w2b
        vox_on_image = projection_func(voxels).astype(int)

        image_shape = mask.shape
        out_of_bound_filter = (vox_on_image[:, 0] >= 0) & \
                            (vox_on_image[:, 0] < image_shape[1]) & \
                            (vox_on_image[:, 1] >= 0) & \
                            (vox_on_image[:, 1] < image_shape[0])

        vox_on_image = vox_on_image[out_of_bound_filter]

        if mask.size == 4: # bbox
            xmin, ymin, xmax, ymax = mask
            voxels_mask = (vox_on_image[:, 0] >= xmin) & (vox_on_image[:, 0] <= xmax) & (vox_on_image[:, 1] >= ymin) & (vox_on_image[:, 1] <= ymax)
        else:
            voxels_mask = mask[vox_on_image[:, 1], vox_on_image[:, 0]].astype(bool)

        voxels_mask_global = np.zeros(self.voxels.shape[0], dtype=bool)
        voxels_mask_global[out_of_bound_filter] = voxels_mask
        # print(f"voxels on mask: {np.sum(voxels_mask)}")

        if np.sum(voxels_mask) == 0 and image_src is not None:
            image_src[vox_on_image[:, 1], vox_on_image[:, 0]] = [0, 0, 255]
        
        odom_t = -R_w2b.T @ t_w2b
        odom_R = R_w2b.T

        voxel_to_odom = voxels[voxels_mask_global] - odom_t
        # voxel_to_odom = voxel_to_odom @ odom_R
        obs_angles = np.arctan2(voxel_to_odom[:, 1], voxel_to_odom[:, 0])
        obs_angles = normalize_angles_to_pi(obs_angles)
        obs_angles = discretize_angles(obs_angles, self.num_angle_bin)
        self.observation_angles[voxels_mask_global, obs_angles] = 1
    
    def retrieve_valid_voxel_indices(self, diversity_percentile=0.3, regularized=True):
        if regularized:
            voxels = self.voxels[self.regularized_voxel_mask]
            obs_angles = self.observation_angles[self.regularized_voxel_mask]
            votes = self.vote[self.regularized_voxel_mask]
        else:
            voxels = self.voxels
            obs_angles = self.observation_angles
            votes = self.vote
        
        # Newer version, add votes as weight
        if len(obs_angles) > 0:
            angle_diversity = np.sum(obs_angles, axis=1)
            sorted_indices = np.argsort(angle_diversity) # smaller to larger
            sorted_diversity = angle_diversity[sorted_indices]
            sorted_weights = votes[sorted_indices]
            total_weight = np.sum(sorted_weights)

            # search for the index that surpasses the percentile
            sorted_weights = sorted_diversity * sorted_weights
            percentile_index = percentile_index_search_binary(sorted_weights, 1 - diversity_percentile)
            voxel_indices = sorted_indices[percentile_index:]
        else:
            voxel_indices = np.empty(0, dtype=int)
        
        return voxel_indices
    
    # def retrieve_valid_voxels_by_clustering(self, clustering_labels, diversity_percentile=0.3):
    #     voxels = self.voxels
    #     obs_angles = self.observation_angles

    #     assert len(clustering_labels) == len(voxels)

    #     added_cluster_labels = set()
    #     voxel_indices = []
    #     if len(obs_angles) > 0:
    #         angle_diversity = np.sum(obs_angles, axis=1)
    #         sorted_indices = np.argsort(angle_diversity) # smaller to larger
    #         sorted_diversity = angle_diversity[sorted_indices]
    #         sorted_weights = self.vote[sorted_indices]
    #         total_weight = np.sum(sorted_weights)

    #         current_weight = 0
    #         # search for the index that surpasses the percentile
    #         for ind in reversed(sorted_indices):
    #             if clustering_labels[ind] in added_cluster_labels:
    #                 continue
    #             else:
    #                 added_cluster_labels.add(clustering_labels[ind])
    #                 selected_cluster_mask = (clustering_labels == clustering_labels[ind])
    #                 cluster_weight = np.sum(angle_diversity[selected_cluster_mask])
    #                 current_weight += cluster_weight
    #                 voxel_indices.append(np.where(selected_cluster_mask != 0)[0])
    #                 if current_weight > diversity_percentile * total_weight:
    #                     break
    #         voxel_indices = np.concatenate(voxel_indices, axis=0)
    #     else:
    #         voxel_indices = np.empty(0, dtype=int)

    #     return voxel_indices

class SingleObject:
    def __init__(self, class_id, obj_id, voxels, voxel_size, odom_R, odom_t, mask, stamp, clip_feat=None, num_angle_bin=15, confidence=1.0, centroid=None):
        self.class_id = {class_id: 1}
        self.obj_id = [obj_id]
        self.voxel_manager = VoxelFeatureManager(voxels, voxel_size, odom_R, odom_t, num_angle_bin)

         # Add the feature of confidence to the object
        self.confidence = confidence
        self.conf_list = {class_id: confidence}
        self.points_count = {class_id: len(voxels)}
        self.weighted_class_scores = {class_id: confidence * self.points_count[class_id]}

        self.observation_records = {
            'class_name': [class_id],
            'n_points': [len(voxels)],
            'conf': [confidence]
        }

        self.robot_poses = [{'R': odom_R.copy(), 't': odom_t.copy()}]
        self.angle_threshold = np.deg2rad(5)  # 5 degrees
        self.distance_threshold = 0.3  # 0.3 meters
        
        self.life = 0
        self.inactive_frame = -1
        
        self.key_frames = [mask]
        self.key_pose = odom_t + [R_to_yaw(odom_R)]
        
        self.clip_feat = clip_feat
        self.fusion_weight = 0.2

        self.latest_stamp = stamp
        self.info_frames_cnt = 1

        self.valid_indices = None
        self.valid_indices_regularized = None
        self.clustering_labels = None

        # self.valid_indices_by_clustering = None

        self.req_clustering = True
        self.req_shape_regularization = True
        self.req_recompute_indices = True
        self.req_recompute_centroid = True
        self.req_recompute_bbox3d = True
        self.req_recompute_bbox3d_oriented = True

        self.movement_stack = []
        self.movement_stack_size = 15

        self.centroid = centroid
        self.bbox3d = None
        self.bbox3d_oriented = None
        self.bbox3d_oriented_corners = None
        self.merged_obj_ids = None

        self.is_asked_vlm = False

        self.status = 'new' # new, persistent, moving, disappeared
        self.updated = True # whether the object has been updated since the last time it was processed
        self.updated_by_vlm = False # whether the object is updated by VLM
        # add the status of the object
        self.last_published_stamp = None
        self.publish_status = 'new' # new, updated, unchanged
        self.spatial_relations = {
            'in': [],
            'contain': [],
            'on': [],
            'under': [],
            'beside': [],
            'above': [],
        }

        self.best_image_path = None
        self.best_image_score = 0
        self.base_image_dir= os.path.join('output/object_images')

    def reset_spatial_relations(self):
        self.spatial_relations = {
            'in': [],
            'contain': [],
            'on': [],
            'under': [],
            'beside': [],
            'above': [],
        }

    # Only save the best image based on confidence * mask area, otherwise skip saving
    def save_best_image(self, image, mask, confidence, save_queue=None):
        """
        Save image ONLY if it's better than the current best.
        Filename is just the object ID.
        """
        if self.obj_id[0] < 0:  # Skip background objects
            return False
        
        # Convert image to BGR if needed
        # if len(image.shape) == 3:
        #     image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # else:
        #     image_bgr = image
        
        mask_area = np.sum(mask)
        # score = confidence * mask_area
        score = mask_area  # prioritize larger masks
        
        # First image OR better than current best
        if self.best_image_path is None or score > self.best_image_score:
            # Delete old image if it exists
            if self.best_image_path and os.path.exists(self.best_image_path):
                os.remove(self.best_image_path)
                # print(f"Replaced image for object {self.obj_id[0]} (old score: {self.best_image_score:.0f})")
            
            cropped_img, cropped_mask = self._apply_mask_to_image(image, mask)
            if cropped_img is None:
                return False
            # filename = f"{self.get_dominant_label()}_{self.obj_id[0]}.npy"
            filename = f"{self.obj_id[0]}.npy"
            maskname = f"{self.obj_id[0]}_mask.npy"
            filepath = os.path.join(self.base_image_dir, filename)
            maskpath = os.path.join(self.base_image_dir, maskname)
            
            self.is_asked_vlm = False  # reset the flag to ask VLM again for the new image

            try:
                # cv2.imwrite(filepath, cropped_img)  # Save masked image directly
                save_queue.put((1, filepath, maskpath, cropped_img, cropped_mask))  # Indicate save operation
                
                # Update tracking variables
                self.best_image_path = filepath
                self.best_image_score = score
                
                # print(f"Saved best image for object {self.obj_id[0]}: "
                #       f"score={score:.0f} (conf={confidence:.2f} * area={mask_area}px)")
                return True
                
            except Exception as e:
                print(f"Error saving image for object {self.obj_id[0]}: {e}")
                return False
        else:
            # print(f"Skipped saving for object {self.obj_id[0]}: "
            #       f"score={score:.0f} < best={self.best_image_score:.0f}")
            return False
        
    def _apply_mask_to_image(self, image, mask):
        """Apply mask to image, setting non-mask areas to black and keep the image with just the size of the bounding box"""
        # Ensure mask is boolean
        if mask.dtype != bool:
            mask = mask.astype(bool)
        
        masked_image = image.copy()
        
        # if len(image.shape) == 3:  # Color image
        #     masked_image[~mask] = [0, 0, 0]
        # else:  # Grayscale image
        #     masked_image[~mask] = 0
            
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None, None  # No mask, return None or handle as needed
        
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1  # +1 since slicing is exclusive at the end

        # apply a small padding
        padding = max(int(1.0 * (y1 - y0)), int(1.0 * (x1 - x0)), 80)
        y0 = max(0, y0 - padding)
        x0 = max(0, x0 - padding)
        y1 = min(masked_image.shape[0], y1 + padding)
        x1 = min(masked_image.shape[1], x1 + padding)

        # Crop image to bounding box
        cropped_image = masked_image[y0:y1, x0:x1]
        cropped_mask = mask[y0:y1, x0:x1]

        return cropped_image, cropped_mask
    

    def cleanup_images(self, save_queue=None):
        """Delete the stored image"""
        if self.best_image_path and os.path.exists(self.best_image_path):
            try:
                maskpath = self.best_image_path.replace('.npy', '_mask.npy')
                save_queue.put((0, self.best_image_path, maskpath, None, None))  # Indicate deletion
                self.best_image_path = None
                self.best_image_score = 0
            except Exception as e:
                print(f"Error during cleanup for object {self.obj_id[0]}: {e}")

    def is_similar_pose(self, new_R, new_t):
        """check the similarity of the old and new poses"""
        obj_center = self.infer_centroid(diversity_percentile=0.8, regularized=True)
        if obj_center is None:
            obj_center = np.mean(self.voxel_manager.voxels, axis=0)
        for pose in self.robot_poses:
            old_R, old_t = pose['R'], pose['t']
            
            old_to_obj = obj_center - old_t
            new_to_obj = obj_center - new_t
            
            old_angle = np.arctan2(old_to_obj[1], old_to_obj[0])
            new_angle = np.arctan2(new_to_obj[1], new_to_obj[0])
            angle_diff = abs(normalize_angles_to_pi(new_angle - old_angle))
            
            old_dist = np.linalg.norm(old_to_obj)
            new_dist = np.linalg.norm(new_to_obj)
            dist_diff = abs(new_dist - old_dist)

            if angle_diff < self.angle_threshold and dist_diff < self.distance_threshold:
                return True
        return False
    
    def add_spatial_relation(self, relation, obj):
        self.spatial_relations[str(relation)].append(obj)

    def reset_flags(self):
        self.req_clustering = True
        self.req_shape_regularization = True
        self.req_recompute_indices = True
        self.req_recompute_centroid = True
        self.req_recompute_bbox3d = True
        self.req_recompute_bbox3d_oriented = True
        self.updated = True
    
    def add_key_frame(self, mask, odom_R, odom_t):
        self.key_frames.append(mask)
        self.key_pose.append(odom_t + [R_to_yaw(odom_R)])

    # @profile
    def update(self, voxels, odom_R, odom_t, label, stamp, clip_feat=None,confidence=1.0):
        """
            Merge the new object with the existing one.
            Update the voxel manager, class id, and other attributes.
        """
        new_points_count = len(voxels)
        self.voxel_manager.update(voxels, odom_R, odom_t)
        self.info_frames_cnt += 1
        self.latest_stamp = stamp

        self.clip_feat = clip_feat * self.fusion_weight + self.clip_feat * (1 - self.fusion_weight) if self.clip_feat is not None else clip_feat
        # check if the new pose is similar to the old poses
        is_similar = self.is_similar_pose(odom_R, odom_t)
        
        if not is_similar:

            self.conf_list[label] = max(self.conf_list.get(label, 0), confidence)
            self.points_count[label] = self.points_count.get(label, 0) + new_points_count
            
            self.class_id[label] = self.class_id.get(label, 0) + 1

            self.observation_records['class_name'].append(label)
            self.observation_records['n_points'].append(new_points_count)
            self.observation_records['conf'].append(confidence)

            self.robot_poses.append({'R': odom_R.copy(), 't': odom_t.copy()})
            self.weighted_class_scores[label] = self.points_count[label] * self.conf_list[label]

        self.reset_flags()
        self.updated = True
        if self.publish_status != 'new':
            self.publish_status = 'updated'
    
    def merge_object(self, single_obj):
        self.obj_id.extend(single_obj.obj_id)
        # self.obj_id.sort() # make the obj_id consistent

        self.voxel_manager.update_through_vote_stat(single_obj.voxel_manager)
        self.life = max(self.life, single_obj.life)
        self.info_frames_cnt += single_obj.info_frames_cnt
        self.latest_stamp = max(self.latest_stamp, single_obj.latest_stamp)

        for label, conf in single_obj.conf_list.items():
            self.conf_list[label] = max(self.conf_list.get(label, 0), conf)
            self.points_count[label] = self.points_count.get(label, 0) + single_obj.points_count.get(label, 0)
            self.class_id[label] = self.class_id.get(label, 0) + single_obj.class_id.get(label, 0)
            self.weighted_class_scores[label] = self.points_count[label] * self.conf_list[label]

        self.observation_records['class_name'].extend(single_obj.observation_records['class_name'])
        self.observation_records['n_points'].extend(single_obj.observation_records['n_points'])
        self.observation_records['conf'].extend(single_obj.observation_records['conf'])

        unique_poses_added = 0
        total_poses_to_merge = len(single_obj.robot_poses)
        
        for pose in single_obj.robot_poses:
            pose_R, pose_t = pose['R'], pose['t']
            if not self.is_similar_pose(pose_R, pose_t):
                self.robot_poses.append(pose)
                unique_poses_added += 1
            else:
                print(f"Skipping similar pose in merge_object for object {self.obj_id}")

        self.merged_obj_ids = single_obj.obj_id
        self.is_asked_vlm = self.is_asked_vlm or single_obj.is_asked_vlm

        # print(f'cosine similarity: {self.clip_feat @ single_obj.clip_feat / (np.linalg.norm(self.clip_feat) * np.linalg.norm(single_obj.clip_feat))}')
        # if self.clip_feat is not None and single_obj.clip_feat is not None:
        #     self.clip_feat = self.clip_feat * self.fusion_weight + single_obj.clip_feat * (1 - self.fusion_weight)
        # else:
        #     self.clip_feat = self.clip_feat if self.clip_feat is not None else single_obj.clip_feat

        self.reset_flags()
        if self.publish_status != 'new':
            self.publish_status = 'updated'

    # def reproject_depth_mask(self, R_w2b, t_w2b, depth_image, projection_func, det_mask=None, rgb_projection_func=None):
    #     """
    #         A proxy function for ray casting the object points given a depth mask
    #     """
    #     if det_mask is not None:
    #         assert rgb_projection_func is not None, "det_mask is on rgb is provided, but rgb_projection_func is not provided"
    #     valid_voxels_mask, on_surface_voxel_mask, tlbr = self.voxel_manager.reproject_depth_mask(R_w2b, t_w2b, depth_image, projection_func, det_mask, rgb_projection_func)
    #     # self.voxel_manager.update_through_mask(valid_voxels_mask)
    #     # self.reset_flags()
    #     self.dirty = True
    #     return valid_voxels_mask, on_surface_voxel_mask, tlbr

    # def reproject_positive_mask(self, R_w2b, t_w2b, mask, projection_func):
    #     """
    #         Reproject the object to the mask using odometry and update the vote by substracting 1 to points that are not on the mask.
    #         R_w2b: rotation from world to body frame
    #         t_w2b: translation from world to body frame
    #         mask: the mask of the object
    #     """
    #     self.dirty = True
    #     self.voxel_manager.reproject_positive_mask(R_w2b, t_w2b, mask, projection_func)

    def reproject_obs_angle(self, R_w2b, t_w2b, mask, projection_func, image_src=None):
        """
            Reproject the object to the mask and update the observation angle by adding 1 to the points that are on the mask.
            R_w2b: rotation from world to body frame
            t_w2b: translation from world to body frame
            mask: the mask of the object
            projection_func: the function to project the points to the mask
        """
        # if mask.size == 4:
        #     bbox = mask.astype(int)
        #     mask = np.zeros([640, 1920])
        #     mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        self.voxel_manager.reproject_obs_angle(R_w2b, t_w2b, mask, projection_func, image_src)
        # self.reset_flags()
        # self.dirty = True

    def get_dominant_label(self):
        max_label = max(self.weighted_class_scores, key=self.weighted_class_scores.get)
        return max_label
        
    def get_dominant_confidence(self):
        """Get the confidence of the dominant label"""
        if not self.conf_list:
            return self.confidence
        
        dominant_label = self.get_dominant_label()
        return self.conf_list.get(dominant_label, self.confidence)
        
    
    def dbscan_cluster_params(self):
        if self.info_frames_cnt < 3 and self.inactive_frame < 5:
            min_points = 5
        else:
            min_points = 10
        return self.voxel_manager.voxel_size * 2.5, min_points
    
    def cal_clusters(self):
        if self.req_clustering:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.voxel_manager.voxels)
            eps, min_points = self.dbscan_cluster_params()
            time_start = time.time()
            self.clustering_labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)) 
            time_end = time.time()
            # if time_end - time_start > 0.2:
            # print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!DBSCAN clustering time: {time_end - time_start:.4f} seconds, points: {self.voxel_manager.voxels.shape[0]}, eps: {eps}, min_points: {min_points}")
                # o3d.visualization.draw_plotly([pcd], window_name="DBSCAN Clustering", width=1920, height=1080)

    def regularize_shape(self, percentile=None):
        """
            Apply a regularization to the object shape.
            This function first do a DBSCAN clustering, then rank the object points from the most important to the least important.
            Then starting from the most important cluster, try adding the cluster. If it makes the object larger than the prior, remove the cluster.
        """
        if self.req_shape_regularization:
            self.cal_clusters()
            clustering_labels = self.clustering_labels
            unique_labels = np.unique(clustering_labels)
            # dim_prior = DIMENSION_PRIORS.get(self.get_dominant_label(), DIMENSION_PRIORS['default'])

            # if dim_prior[0] == DIMENSION_PRIORS['default'][0] and dim_prior[1] == DIMENSION_PRIORS['default'][1]:
            #     print(f"Warning: No dimension prior for class {self.get_dominant_label()}, {self.obj_id}")

            cluster_masks = []
            clustering_weight = []
            for label in unique_labels:
                if label == -1:
                    continue
                cluster_masks.append(clustering_labels == label)
                cluster_point_mask = (clustering_labels == label)
                cluster_obs_angles = self.voxel_manager.observation_angles[cluster_point_mask]
                cluster_weight = np.sum(cluster_obs_angles)
                clustering_weight.append(cluster_weight)
            
            clustering_weight_index = np.argsort(clustering_weight)
            valid_cluster_mask = np.zeros(self.voxel_manager.voxels.shape[0], dtype=bool)

            current_weight = 0
            current_attempt_weight = 0
            total_weight = np.sum(clustering_weight)
            
            for weight_index in reversed(clustering_weight_index):
                mask = cluster_masks[weight_index]

                # remove very small clusters. Useful when the camera odom isn't very accurate
                if clustering_weight[weight_index] < 10:
                    continue

                cluster_mask_attempt = np.logical_or(valid_cluster_mask, mask)
                center, dim, q = get_bbox_3d_oriented(self.voxel_manager.voxels[cluster_mask_attempt])

                if center is None:
                    continue

                # if dim[0] > dim_prior[0] or dim[1] > dim_prior[1] or dim[2] > dim_prior[2]:
                #     continue
                valid_cluster_mask = np.logical_or(valid_cluster_mask, mask)

                if percentile is not None:
                    assert percentile <= 1 and percentile > 0
                    current_weight += clustering_weight[weight_index]
                    if current_weight > percentile * total_weight:
                        break

            
            # self.voxel_manager.update_through_voxel_mask(self.voxel_manager.voxels, valid_cluster_mask)
            # self.voxel_manager.regularized_voxel_mask = np.ones(self.voxel_manager.voxels.shape[0], dtype=bool)
            # valid_cluster_mask = np.ones(self.voxel_manager.voxels.shape[0], dtype=bool)
            self.voxel_manager.regularized_voxel_mask = valid_cluster_mask
            self.req_recompute_indices = True
            self.req_shape_regularization = False
            self.req_recompute_centroid = True
            self.req_recompute_bbox3d = True
            self.req_recompute_bbox3d_oriented = True

    def regularize_shape_v2(self, percentile=None):
        if not self.req_shape_regularization:
            return
        self.cal_clusters()
        clustering_labels = self.clustering_labels
        unique_labels = np.unique(clustering_labels)

        cluster_masks, clustering_weight, cluster_centroids = [], [], []
        for label in unique_labels:
            if label == -1:
                continue
            mask = (clustering_labels == label)
            if not np.any(mask):
                continue
            w = float(np.sum(self.voxel_manager.observation_angles[mask]))
            # remove small clusters when necessary
            if w < 5:
                continue
            cluster_masks.append(mask)
            clustering_weight.append(w)
            pts = self.voxel_manager.voxels[mask]
            cluster_centroids.append(np.mean(pts, axis=0))

        if len(cluster_masks) == 0:
            self.voxel_manager.regularized_voxel_mask = np.zeros(
                self.voxel_manager.voxels.shape[0], dtype=bool
            )
        elif len(cluster_masks) == 1:
            self.voxel_manager.regularized_voxel_mask = cluster_masks[0]
        else:
            clustering_weight = np.array(clustering_weight, dtype=float, copy=False)
            cluster_centroids = np.array(cluster_centroids, dtype=float, copy=False)

            # 主簇
            main_idx = int(np.argmax(clustering_weight))
            main_centroid = cluster_centroids[main_idx]

            # 距离调整
            dists = np.linalg.norm(cluster_centroids - main_centroid, axis=1)
            # max_dist = np.max(dists)
            if (self.bbox3d_oriented is not None):
                _, extent, _ = self.bbox3d_oriented
                if extent is not None:
                    dist_base = np.linalg.norm(extent)  # 对角线长度
                else:
                    dist_base = np.max(dists)
            else:
                dist_base = np.max(dists)

            if dist_base == 0:  # 防止除零
                adjusted_weight = clustering_weight.copy()
            else:
                alpha = 3.0  # 控制衰减速度，越大衰减越快
                scale = np.exp(-alpha * dists / dist_base)   # 指数衰减
                adjusted_weight = clustering_weight * scale

            main_adjusted_idx = int(np.argmax(adjusted_weight))

            # 初始化选择 mask
            chosen = np.zeros(len(cluster_masks), dtype=bool)

            if percentile is not None:
                assert 0 < percentile <= 1
                target = percentile * np.sum(adjusted_weight)

                # 先选主簇
                acc = adjusted_weight[main_adjusted_idx]
                chosen[main_adjusted_idx] = True

                # 从大到小排序
                order = np.argsort(adjusted_weight)[::-1]
                for i in order:
                    if chosen[i]:
                        continue
                    if acc + adjusted_weight[i] <= target + 1e-8:  # 加上tol
                        chosen[i] = True
                        acc += adjusted_weight[i]
                    if acc >= target:  # 达到目标提前退出
                        break
            else:
                chosen[:] = True

            # Merge chosen clusters
            valid_cluster_mask = np.zeros(self.voxel_manager.voxels.shape[0], dtype=bool)
            for i, cmask in enumerate(cluster_masks):
                if chosen[i]:
                    valid_cluster_mask |= cmask

            self.voxel_manager.regularized_voxel_mask = valid_cluster_mask

        self.req_recompute_indices = True
        self.req_shape_regularization = False
        self.req_recompute_centroid = True
        self.req_recompute_bbox3d = True
        self.req_recompute_bbox3d_oriented = True
        # self.reset_flags()
        self.updated = True
        # if self.publish_status != 'new':
        #     self.publish_status = 'updated'




    
    # def regularize_shape_v2(self, dist_thresh=0.25, percentile=None, min_cluster_weight=10, xy_only=False):
    #     """
    #     Greedy accumulate clusters by descending 'importance' (weight),
    #     but reject a cluster if it's farther than dist_thresh from the
    #     already-accepted set (by nearest XY distance).

    #     Args:
    #         dist_thresh (float): 最大允许的最近邻距离（米），超过则不合并该簇。
    #         percentile (float|None): 和 v1 一样，可选的累计权重上限（0~1]。
    #         min_cluster_weight (float): 丢弃极小/噪声簇的权重阈值。
    #         xy_only (bool): True 表示仅用 XY 距离；False 用 3D 距离。
    #     """
    #     if not self.req_shape_regularization:
    #         return

    #     # 1) DBSCAN
    #     self.cal_clusters()
    #     clustering_labels = self.clustering_labels
    #     unique_labels = np.unique(clustering_labels)

    #     V = self.voxel_manager.voxels
    #     A = self.voxel_manager.observation_angles

    #     # 2) collect
    #     cluster_masks = []
    #     cluster_weights = []

    #     for lab in unique_labels:
    #         if lab == -1:
    #             continue  # DBSCAN 噪声
    #         mask = (clustering_labels == lab)
    #         w = float(np.sum(A[mask]))
    #         if w < min_cluster_weight:
    #             continue
    #         cluster_masks.append(mask)
    #         cluster_weights.append(w)

    #     if len(cluster_masks) == 0:
    #         self.voxel_manager.regularized_voxel_mask = np.zeros(V.shape[0], dtype=bool)
    #         self.req_recompute_indices = True
    #         self.req_shape_regularization = False
    #         self.req_recompute_centroid = True
    #         self.req_recompute_bbox3d = True
    #         self.req_recompute_bbox3d_oriented = True
    #         return

    #     # 3) sort from most important to least
    #     order = np.argsort(cluster_weights)[::-1]
    #     total_w = float(np.sum([cluster_weights[i] for i in order]))
    #     acc_w = 0.0

    #     valid_mask = np.zeros(V.shape[0], dtype=bool)
    #     accepted_pts = None
    #     kdtree = None

    #     # 4) 逐簇尝试加入；如果与已选集合的最近距离 > dist_thresh，则拒绝
    #     for idx in order:
    #         mask = cluster_masks[idx]
    #         pts = V[mask]
    #         if pts.shape[0] == 0:
    #             continue

    #         if accepted_pts is None or accepted_pts.shape[0] == 0:
    #             valid_mask |= mask
    #             accepted_pts = V[valid_mask]
    #             pts_for_tree = accepted_pts[:, :2] if xy_only else accepted_pts
    #             kdtree = cKDTree(pts_for_tree)
    #             acc_w += cluster_weights[idx]
    #             if percentile is not None:
    #                 assert 0 < percentile <= 1
    #                 if acc_w > percentile * total_w:
    #                     break
    #             continue

    #         # 计算与已选点集的最近邻距离（XY 或 3D）
    #         query_pts = pts[:, :2] if xy_only else pts
    #         dists, _ = kdtree.query(query_pts, k=1)
    #         min_dist = float(np.min(dists)) if dists.size else np.inf

    #         if min_dist <= dist_thresh:
    #             # 接受该簇，并更新 KDTree
    #             valid_mask |= mask
    #             accepted_pts = V[valid_mask]
    #             pts_for_tree = accepted_pts[:, :2] if xy_only else accepted_pts
    #             kdtree = cKDTree(pts_for_tree)
    #             acc_w += cluster_weights[idx]
    #             if percentile is not None:
    #                 assert 0 < percentile <= 1
    #                 if acc_w > percentile * total_w:
    #                     break
    #         else:
    #             continue

    #     self.voxel_manager.regularized_voxel_mask = valid_mask
    #     self.req_recompute_indices = True
    #     self.req_shape_regularization = False
    #     self.req_recompute_centroid = True
    #     self.req_recompute_bbox3d = True
    #     self.req_recompute_bbox3d_oriented = True

    # def regularize_shape_simple(self, percentile):
    #     """
    #         Apply a regularization to the object shape.
    #         This function first do a DBSCAN clustering, then rank the object points from the most important to the least important.
    #         Then starting from the most important cluster, try adding the cluster. If it makes the object larger than the prior, remove the cluster.
    #     """
    #     if self.req_shape_regularization:
    #         dim_prior = DIMENSION_PRIORS.get(self.get_dominant_label(), DIMENSION_PRIORS['default'])
    #         dim_prior = np.array(dim_prior)


    #         # avoid recursion
    #         valid_indices = self.voxel_manager.retrieve_valid_voxel_indices(diversity_percentile=percentile, regularized=False)
            
    #         if len(valid_indices) == 0:
    #             if self.voxel_manager.voxels.shape[0] > 0:
    #                 self.voxel_manager.regularized_voxel_mask = np.zeros(self.voxel_manager.voxels.shape[0], dtype=bool)
    #             return
            
    #         centroid_all = self.infer_centroid_by_indices(valid_indices)
    #         vec_to_centroid = self.voxel_manager.voxels - centroid_all
    #         dist_2d = np.linalg.norm(vec_to_centroid[:, :2], axis=1)
    #         dist_2d_thresh = np.linalg.norm(dim_prior[:2] / 2)
            
    #         regularize_shape_mask = dist_2d < dist_2d_thresh
    #         regularize_shape_mask = np.logical_and(regularize_shape_mask, np.abs(vec_to_centroid[:, 2]) < dim_prior[2] / 2)

    #         self.dirty = True
    #         self.voxel_manager.regularized_voxel_mask = regularize_shape_mask
    #         self.req_recompute_indices = True
    #         self.req_shape_regularization = False
    #         self.req_recompute_centroid = True
    #         self.req_recompute_bbox3d = True
    #         self.req_recompute_bbox3d_oriented = True

    #         assert self.voxel_manager.voxels.shape[0] == self.voxel_manager.regularized_voxel_mask.shape[0]


    # def cal_distance(self, voxels):
    #     return self.voxel_manager.cal_distance(voxels)
    

    # def pop(self, mask):
    #     """
    #     Pop the info that is not in the mask
    #     """
    #     voxels_pop = self.voxel_manager.voxels[~mask]
    #     obs_angles_pop = self.voxel_manager.observation_angles[~mask]
    #     votes_pop = self.voxel_manager.vote[~mask]
    #     self.voxel_manager.update_through_mask(mask)

    #     self.reset_flags()
    #     self.dirty = True

    #     return voxels_pop, obs_angles_pop, votes_pop
    
    # def add(self, voxels, obs_angles, votes):
    #     self.voxel_manager.voxels = np.concatenate([self.voxel_manager.voxels, voxels])
    #     self.voxel_manager.observation_angles = np.concatenate([self.voxel_manager.observation_angles, obs_angles])
    #     self.voxel_manager.vote = np.concatenate([self.voxel_manager.vote, votes])
        
    #     self.reset_flags()
    #     self.dirty = True
    def compute_valid_indices(self, diversity_percentile):
        if self.req_recompute_indices:
            self.valid_indices = self.voxel_manager.retrieve_valid_voxel_indices(diversity_percentile=diversity_percentile, regularized=False)
            if self.req_shape_regularization:
                # self.regularize_shape_simple(percentile=diversity_percentile)
                self.regularize_shape_v2(percentile=diversity_percentile)
                self.req_shape_regularization = False
            self.valid_indices_regularized = self.voxel_manager.retrieve_valid_voxel_indices(diversity_percentile=diversity_percentile, regularized=True)

            self.req_recompute_indices = False

    def retrieve_valid_voxels(self, diversity_percentile, regularized=True):
        self.compute_valid_indices(diversity_percentile)

        if self.obj_id[0] < 0: # background points aren't regularized
            regularized = False
        if regularized:
            return self.voxel_manager.voxels[self.voxel_manager.regularized_voxel_mask][self.valid_indices_regularized]
        else:
            return self.voxel_manager.voxels[self.valid_indices]

    # def retrieve_valid_voxels_with_weights(self, diversity_percentile, regularized=True):
    #     self.compute_valid_indices(diversity_percentile)

    #     if self.obj_id[0] < 0:
    #         regularized = False
    #     if regularized:
    #         valid_voxels = self.voxel_manager.voxels[self.voxel_manager.regularized_voxel_mask][self.valid_indices_regularized]
    #         valid_obs_angles = self.voxel_manager.observation_angles[self.voxel_manager.regularized_voxel_mask][self.valid_indices_regularized]
    #         valid_votes = self.voxel_manager.vote[self.voxel_manager.regularized_voxel_mask][self.valid_indices_regularized]
    #     else:
    #         valid_voxels = self.voxel_manager.voxels[self.valid_indices]
    #         valid_obs_angles = self.voxel_manager.observation_angles[self.valid_indices]
    #         valid_votes = self.voxel_manager.vote[self.valid_indices]
    #     return valid_voxels, valid_obs_angles.sum(axis=1) * valid_votes / 5
    
    # def retrieve_valid_voxels_clustered(self, diversity_percentile):
    #     self.compute_valid_indices(diversity_percentile)
    #     return self.voxel_manager.voxels[self.valid_indices_by_clustering]

    def infer_centroid(self, diversity_percentile, regularized=True):
        if self.req_recompute_centroid:
            self.compute_valid_indices(diversity_percentile)

            voxels = self.voxel_manager.voxels
            obs_angles = self.voxel_manager.observation_angles
            if regularized:
                regularized_mask = self.voxel_manager.regularized_voxel_mask
                valid_mask = self.valid_indices_regularized
                valid_voxels = voxels[regularized_mask][valid_mask]
                obs_angles = obs_angles[self.voxel_manager.regularized_voxel_mask]
                valid_obs_angles = obs_angles[self.valid_indices_regularized]
            else:
                valid_voxels = voxels[self.valid_indices]
                valid_obs_angles = obs_angles[self.valid_indices]
            weights = np.sum(valid_obs_angles, axis=1)
            if np.sum(weights) == 0:
                center = None
            else:
                center = np.average(valid_voxels, axis=0, weights=weights)
            
            # if center is not None:
            #     if len(self.movement_stack) >= self.movement_stack_size:
            #         self.movement_stack.pop(0)
            #     self.movement_stack.append(center)

            self.centroid = center
            self.req_recompute_centroid = False

        return self.centroid
    
    # def infer_centroid_by_indices(self, voxel_indices):
    #     if self.req_recompute_centroid:

    #         voxels = self.voxel_manager.voxels[voxel_indices]
    #         obs_angles = self.voxel_manager.observation_angles[voxel_indices]

    #         weights = np.sum(obs_angles, axis=1)
    #         if np.sum(weights) == 0:
    #             center = None
    #         else:
    #             center = np.average(voxels, axis=0, weights=weights)
            
    #         self.centroid = center

    #     return self.centroid
    def infer_bbox_oriented(self, diversity_percentile, regularized=True):
        if self.req_recompute_bbox3d_oriented:
            self.compute_valid_indices(diversity_percentile)

            voxels = self.voxel_manager.voxels
            if regularized:
                voxels = voxels[self.voxel_manager.regularized_voxel_mask]
                valid_voxels = voxels[self.valid_indices_regularized]
            else:
                valid_voxels = voxels[self.valid_indices]

            self.bbox3d_oriented, self.bbox3d_oriented_corners = get_bbox_3d_oriented(valid_voxels)

            self.req_recompute_bbox3d_oriented = False
        
        return self.bbox3d_oriented, self.bbox3d_oriented_corners
        
    def get_info_str(self):
        info_str = f"{self.obj_id[0]}, class: {self.get_dominant_label()}, all voxels: {len(self.valid_indices)}, regularized voxels: {len(self.valid_indices_regularized)}"
        return info_str

import math
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
def get_bbox_3d_oriented(points):
    if len(points) == 0:
        return None, None, None
    bbox2d, _ = minimum_bounding_rectangle(points[:, :2])
    if bbox2d is not None:
        center2d = np.mean(bbox2d, axis=0)
        edge1 = bbox2d[1] - bbox2d[0]
        edge2 = bbox2d[2] - bbox2d[1]
        edge1_length = np.linalg.norm(edge1)
        edge2_length = np.linalg.norm(edge2)
        longest_edge = edge1 if edge1_length > edge2_length else edge2
        orientation = math.atan2(longest_edge[1], longest_edge[0])
        q = Rotation.from_euler('z', orientation).as_quat()
        extent = np.array([edge1_length, edge2_length, points[:, 2].max() - points[:, 2].min()])
        z_center = points[:, 2].max() - extent[2] / 2
        center = np.array([center2d[0], center2d[1], z_center])
        bbox3d = np.hstack([bbox2d, np.ones((4, 1))*z_center])
        bbox3d_oriented_corners = np.array([
            bbox3d[0] - [0, 0, extent[2]/2],
            bbox3d[1] - [0, 0, extent[2]/2],
            bbox3d[2] - [0, 0, extent[2]/2],
            bbox3d[3] - [0, 0, extent[2]/2],
            bbox3d[0] + [0, 0, extent[2]/2],
            bbox3d[1] + [0, 0, extent[2]/2],
            bbox3d[2] + [0, 0, extent[2]/2],
            bbox3d[3] + [0, 0, extent[2]/2]
        ])
    else:
        center = None
        extent = None
        q = None
        bbox3d_oriented_corners = None
    return (center, extent, q), bbox3d_oriented_corners

def minimum_bounding_rectangle(points):
    try:
        # Compute the convex hull
        hull = ConvexHull(points, qhull_options="QJ")
        hull_points = points[hull.vertices]
        
        # Initialize variables
        min_area = float('inf')
        best_rectangle = None
        
        # Rotate calipers
        for i in range(len(hull_points)):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % len(hull_points)]
            edge = p2 - p1
            
            # Normalize edge vector
            edge_vector = edge / np.linalg.norm(edge)
            perpendicular_vector = np.array([-edge_vector[1], edge_vector[0]])
            
            # Project all points onto the edge and perpendicular vector
            projections_on_edge = points @ edge_vector
            projections_on_perpendicular = points @ perpendicular_vector
            
            # Find bounds
            min_proj_edge = projections_on_edge.min()
            max_proj_edge = projections_on_edge.max()
            min_proj_perp = projections_on_perpendicular.min()
            max_proj_perp = projections_on_perpendicular.max()
            
            # Compute width, height, and area
            width = max_proj_edge - min_proj_edge
            height = max_proj_perp - min_proj_perp
            area = width * height
            
            if area < min_area:
                min_area = area
                best_rectangle = (min_proj_edge, max_proj_edge, min_proj_perp, max_proj_perp, edge_vector, perpendicular_vector)
        
        # Recover rectangle corners
        min_proj_edge, max_proj_edge, min_proj_perp, max_proj_perp, edge_vector, perpendicular_vector = best_rectangle
        corner1 = min_proj_edge * edge_vector + min_proj_perp * perpendicular_vector
        corner2 = max_proj_edge * edge_vector + min_proj_perp * perpendicular_vector
        corner3 = max_proj_edge * edge_vector + max_proj_perp * perpendicular_vector
        corner4 = min_proj_edge * edge_vector + max_proj_perp * perpendicular_vector
        return np.array([corner1, corner2, corner3, corner4]), min_area
    except Exception as e:
        print(e)
        return None, None
    