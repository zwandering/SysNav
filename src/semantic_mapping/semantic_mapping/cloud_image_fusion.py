import numpy as np
import scipy.ndimage
from scipy.spatial.transform import Rotation
import cv2
import scipy
from scipy.stats import gaussian_kde, kurtosis, skew
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from scipy.ndimage import minimum_filter

import time

def scan2pixels(laserCloud, L2C_PARA, CAMERA_PARA, LIDAR_PARA):
    lidarX = L2C_PARA["x"] #   lidarXStack[imageIDPointer]
    lidarY = L2C_PARA["y"] # idarYStack[imageIDPointer]
    lidarZ = L2C_PARA["z"] # lidarZStack[imageIDPointer]
    lidarRoll = -L2C_PARA["roll"] #  lidarRollStack[imageIDPointer]
    lidarPitch = -L2C_PARA["pitch"] # lidarPitchStack[imageIDPointer]
    lidarYaw = -L2C_PARA["yaw"]# lidarYawStack[imageIDPointer]

    imageWidth = CAMERA_PARA["width"]
    imageHeight = CAMERA_PARA["height"]
    cameraOffsetZ = 0   #  additional pixel offset due to image cropping? 
    vertPixelOffset = 0  #  additional vertical pixel offset due to image cropping

    sinLidarRoll = np.sin(lidarRoll)
    cosLidarRoll = np.cos(lidarRoll)
    sinLidarPitch = np.sin(lidarPitch)
    cosLidarPitch = np.cos(lidarPitch)
    sinLidarYaw = np.sin(lidarYaw)
    cosLidarYaw = np.cos(lidarYaw)
    
    lidar_offset = np.array([lidarX, lidarY, lidarZ])
    camera_offset = np.array([0, 0, cameraOffsetZ])
    
    cloud = laserCloud[:, :3] - lidar_offset
    R_z = np.array([[cosLidarYaw, -sinLidarYaw, 0], [sinLidarYaw, cosLidarYaw, 0], [0, 0, 1]])
    R_y = np.array([[cosLidarPitch, 0, sinLidarPitch], [0, 1, 0], [-sinLidarPitch, 0, cosLidarPitch]])
    R_x = np.array([[1, 0, 0], [0, cosLidarRoll, -sinLidarRoll], [0, sinLidarRoll, cosLidarRoll]])
    cloud = cloud @ R_z @ R_y @ R_x
    cloud = cloud - camera_offset
    
    horiDis = np.sqrt(cloud[:, 0] ** 2 + cloud[:, 1] ** 2)
    horiPixelID = (-imageWidth / (2 * np.pi) * np.arctan2(cloud[:, 1], cloud[:, 0]) + imageWidth / 2 + 1).astype(int) - 1
    vertPixelID = (-imageWidth / (2 * np.pi) * np.arctan2(cloud[:, 2], horiDis) + imageHeight / 2 + 1 + vertPixelOffset).astype(int)
    PixelDepth = horiDis

    horiPixelID = np.clip(horiPixelID, 0, CAMERA_PARA["width"] - 1)
    vertPixelID = np.clip(vertPixelID, 0, CAMERA_PARA["height"] - 1)

    point_pixel_idx = np.array([horiPixelID, vertPixelID, PixelDepth]).T
    
    return point_pixel_idx.astype(int)

def scan2pixels_wheelchair(laserCloud):
    # project scan points to image pixels
    # https://github.com/jizhang-cmu/cmu_vla_challenge_unity/blob/noetic/src/semantic_scan_generation/src/semanticScanGeneration.cpp
    
    # Input: 
    # [#points, 3], x-y-z coordinates of lidar points
    
    # Output: 
    #    point_pixel_idx['horiPixelID'] : horizontal pixel index in the image coordinate
    #    point_pixel_idx['vertPixelID'] : vertical pixel index in the image coordinate

    # L2C_PARA= {"x": 0, "y": 0, "z": 0.235, "roll": -1.5707963, "pitch": 0, "yaw": -1.5707963} #  mapping from scan coordinate to camera coordinate(m) (degree), camera is  "z" higher than lidar
    L2C_PARA= {"x": 0, "y": 0, "z": 0.235, "roll": 0.0, "pitch": 0, "yaw": -0.0} #  mapping from scan coordinate to camera coordinate(m) (degree), camera is  "z" higher than lidar
    CAMERA_PARA= {"hfov": 360, "vfov": 120, "width": 1920, "height": 640}  # cropped 30 degree(160 pixels) in top and  30 degree(160 pixels) in bottom 
    LIDAR_PARA= {"hfov": 360, "vfov": 30}   
    
    return scan2pixels(laserCloud, L2C_PARA, CAMERA_PARA, LIDAR_PARA)

def scan2pixels_mecanum_sim(laserCloud):
    CAMERA_PARA= {"x": 0.0, "y": 0.0, "z": 0.1, "roll": -1.5707963, "pitch": 0, "yaw": -1.5707963, "hfov": 360, "vfov": 120, "width": 1920, "height": 640}  # cropped 30 degree(160 pixels) in top and  30 degree(160 pixels) in bottom 
    LIDAR_PARA= {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}

    lidar_offset = np.array([LIDAR_PARA["x"], LIDAR_PARA["y"], LIDAR_PARA["z"]])
    lidarRoll = LIDAR_PARA["roll"] #  lidarRollStack[imageIDPointer]
    lidarPitch = LIDAR_PARA["pitch"] # lidarPitchStack[imageIDPointer]
    lidarYaw = LIDAR_PARA["yaw"]# lidarYawStack[imageIDPointer]
    lidarR_z = np.array([[np.cos(lidarYaw), -np.sin(lidarYaw), 0], [np.sin(lidarYaw), np.cos(lidarYaw), 0], [0, 0, 1]])
    lidarR_y = np.array([[np.cos(lidarPitch), 0, np.sin(lidarPitch)], [0, 1, 0], [-np.sin(lidarPitch), 0, np.cos(lidarPitch)]])
    lidarR_x = np.array([[1, 0, 0], [0, np.cos(lidarRoll), -np.sin(lidarRoll)], [0, np.sin(lidarRoll), np.cos(lidarRoll)]])
    lidarR = lidarR_z @ lidarR_y @ lidarR_x

    cam_offset = np.array([CAMERA_PARA["x"], CAMERA_PARA["y"], CAMERA_PARA["z"]])
    camRoll = CAMERA_PARA["roll"]
    camPitch = CAMERA_PARA["pitch"]
    camYaw = CAMERA_PARA["yaw"]
    camR_z = np.array([[np.cos(camYaw), -np.sin(camYaw), 0], [np.sin(camYaw), np.cos(camYaw), 0], [0, 0, 1]])
    camR_y = np.array([[np.cos(camPitch), 0, np.sin(camPitch)], [0, 1, 0], [-np.sin(camPitch), 0, np.cos(camPitch)]])
    camR_x = np.array([[1, 0, 0], [0, np.cos(camRoll), -np.sin(camRoll)], [0, np.sin(camRoll), np.cos(camRoll)]])
    camR = camR_z @ camR_y @ camR_x

    xyz = laserCloud[:, :3] - lidar_offset
    xyz = xyz @ lidarR
    xyz = xyz - cam_offset
    xyz = xyz @ camR

    horiDis = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2)
    horiPixelID = (CAMERA_PARA["width"] / (2 * np.pi) * np.arctan2(xyz[:, 0], xyz[:, 2]) + CAMERA_PARA["width"] / 2 + 1).astype(int)
    vertPixelID = (CAMERA_PARA["width"] / (2 * np.pi) * np.arctan(xyz[:, 1] / horiDis) + CAMERA_PARA["height"] / 2 + 1).astype(int)
    pixelDepth = horiDis

    horiPixelID = np.clip(horiPixelID, 0, CAMERA_PARA["width"] - 1)
    vertPixelID = np.clip(vertPixelID, 0, CAMERA_PARA["height"] - 1)

    # --- Step1: 构建 depth_map ---
    H = CAMERA_PARA["height"]
    W = CAMERA_PARA["width"]
    depth_map = np.full((H, W), np.inf)
    idx = vertPixelID * W + horiPixelID
    np.minimum.at(depth_map.ravel(), idx, pixelDepth)

    neighborhood = 3
    # --- Step2: 邻域 Z-buffer ---
    if neighborhood > 0:
        depth_map = minimum_filter(depth_map, size=(2*neighborhood+1), mode='nearest')

    # --- Step3: 保留最近点 ---
    remove_mask = pixelDepth >= depth_map[vertPixelID, horiPixelID] + 0.15

    # 过滤后的结果
    horiPixelID[remove_mask] = -1
    vertPixelID[remove_mask] = -1
    point_pixel_idx = np.stack([horiPixelID, vertPixelID, pixelDepth], axis=-1)

    # 根据pixelDepth对于点云进行排序，近的在前面
    sort_idx = np.argsort(point_pixel_idx[:, 2])
    point_pixel_idx = point_pixel_idx[sort_idx]
    laserCloud[:] = laserCloud[sort_idx]

    return point_pixel_idx

def scan2pixels_mecanum(laserCloud):
    CAMERA_PARA= {"x": -0.0, "y": -0.0, "z": 0.272, "roll": -1.5707963, "pitch": 0, "yaw": -1.5707963, "hfov": 360, "vfov": 120, "width": 1920, "height": 640}  # cropped 30 degree(160 pixels) in top and  30 degree(160 pixels) in bottom 
    # CAMERA_PARA= {"x": -0.12, "y": -0.075, "z": 0.265, "roll": -1.5707963, "pitch": 0, "yaw": -1.5707963, "hfov": 360, "vfov": 120, "width": 1920, "height": 480}  # cropped 30 degree(160 pixels) in top and  30 degree(160 pixels) in bottom 
    LIDAR_PARA= {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}

    lidar_offset = np.array([LIDAR_PARA["x"], LIDAR_PARA["y"], LIDAR_PARA["z"]])
    lidarRoll = LIDAR_PARA["roll"] #  lidarRollStack[imageIDPointer]
    lidarPitch = LIDAR_PARA["pitch"] # lidarPitchStack[imageIDPointer]
    lidarYaw = LIDAR_PARA["yaw"]# lidarYawStack[imageIDPointer]
    lidarR_z = np.array([[np.cos(lidarYaw), -np.sin(lidarYaw), 0], [np.sin(lidarYaw), np.cos(lidarYaw), 0], [0, 0, 1]])
    lidarR_y = np.array([[np.cos(lidarPitch), 0, np.sin(lidarPitch)], [0, 1, 0], [-np.sin(lidarPitch), 0, np.cos(lidarPitch)]])
    lidarR_x = np.array([[1, 0, 0], [0, np.cos(lidarRoll), -np.sin(lidarRoll)], [0, np.sin(lidarRoll), np.cos(lidarRoll)]])
    lidarR = lidarR_z @ lidarR_y @ lidarR_x

    cam_offset = np.array([CAMERA_PARA["x"], CAMERA_PARA["y"], CAMERA_PARA["z"]])
    camRoll = CAMERA_PARA["roll"]
    camPitch = CAMERA_PARA["pitch"]
    camYaw = CAMERA_PARA["yaw"]
    camR_z = np.array([[np.cos(camYaw), -np.sin(camYaw), 0], [np.sin(camYaw), np.cos(camYaw), 0], [0, 0, 1]])
    camR_y = np.array([[np.cos(camPitch), 0, np.sin(camPitch)], [0, 1, 0], [-np.sin(camPitch), 0, np.cos(camPitch)]])
    camR_x = np.array([[1, 0, 0], [0, np.cos(camRoll), -np.sin(camRoll)], [0, np.sin(camRoll), np.cos(camRoll)]])
    camR = camR_z @ camR_y @ camR_x

    xyz = laserCloud[:, :3] - lidar_offset
    xyz = xyz @ lidarR
    xyz = xyz - cam_offset
    xyz = xyz @ camR

    horiDis = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2)
    horiPixelID = (CAMERA_PARA["width"] / (2 * np.pi) * np.arctan2(xyz[:, 0], xyz[:, 2]) + CAMERA_PARA["width"] / 2 + 1).astype(int)
    vertPixelID = (CAMERA_PARA["width"] / (2 * np.pi) * np.arctan(xyz[:, 1] / horiDis) + CAMERA_PARA["height"] / 2 + 1).astype(int)
    pixelDepth = horiDis

    horiPixelID = np.clip(horiPixelID, 0, CAMERA_PARA["width"] - 1)
    vertPixelID = np.clip(vertPixelID, 0, CAMERA_PARA["height"] - 1)

    # --- Step1: 构建 depth_map ---
    H = CAMERA_PARA["height"]
    W = CAMERA_PARA["width"]
    depth_map = np.full((H, W), np.inf)
    idx = vertPixelID * W + horiPixelID
    np.minimum.at(depth_map.ravel(), idx, pixelDepth)

    neighborhood = 3
    # --- Step2: 邻域 Z-buffer ---
    if neighborhood > 0:
        depth_map = minimum_filter(depth_map, size=(2*neighborhood+1), mode='nearest')

    # --- Step3: 保留最近点 ---
    remove_mask = pixelDepth >= depth_map[vertPixelID, horiPixelID] + 0.15

    # 过滤后的结果
    horiPixelID[remove_mask] = -1
    vertPixelID[remove_mask] = -1
    point_pixel_idx = np.stack([horiPixelID, vertPixelID, pixelDepth], axis=-1)

    # 根据pixelDepth对于点云进行排序，近的在前面
    sort_idx = np.argsort(point_pixel_idx[:, 2])
    point_pixel_idx = point_pixel_idx[sort_idx]
    laserCloud[:] = laserCloud[sort_idx]

    return point_pixel_idx

    # # --- Step3: 更新所有点的 depth ---
    # corrected_depth = depth_map[vertPixelID, horiPixelID]

    # # point_pixel_idx 对应的像素坐标 + 更新后的 depth
    # point_pixel_idx = np.stack([horiPixelID, vertPixelID, corrected_depth], axis=-1)

    # return point_pixel_idx

def scan2pixels_diablo(laserCloud):
    CAMERA_PARA= {"x": 0.0, "y": 0.0, "z": 0.185, "roll": -1.5707963, "pitch": 0, "yaw": -1.5707963, "hfov": 360, "vfov": 120, "width": 1920, "height": 640}  # cropped 30 degree(160 pixels) in top and  30 degree(160 pixels) in bottom 
    LIDAR_PARA= {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}

    lidar_offset = np.array([LIDAR_PARA["x"], LIDAR_PARA["y"], LIDAR_PARA["z"]])
    lidarRoll = LIDAR_PARA["roll"] #  lidarRollStack[imageIDPointer]
    lidarPitch = LIDAR_PARA["pitch"] # lidarPitchStack[imageIDPointer]
    lidarYaw = LIDAR_PARA["yaw"]# lidarYawStack[imageIDPointer]
    lidarR_z = np.array([[np.cos(lidarYaw), -np.sin(lidarYaw), 0], [np.sin(lidarYaw), np.cos(lidarYaw), 0], [0, 0, 1]])
    lidarR_y = np.array([[np.cos(lidarPitch), 0, np.sin(lidarPitch)], [0, 1, 0], [-np.sin(lidarPitch), 0, np.cos(lidarPitch)]])
    lidarR_x = np.array([[1, 0, 0], [0, np.cos(lidarRoll), -np.sin(lidarRoll)], [0, np.sin(lidarRoll), np.cos(lidarRoll)]])
    lidarR = lidarR_z @ lidarR_y @ lidarR_x

    cam_offset = np.array([CAMERA_PARA["x"], CAMERA_PARA["y"], CAMERA_PARA["z"]])
    camRoll = CAMERA_PARA["roll"]
    camPitch = CAMERA_PARA["pitch"]
    camYaw = CAMERA_PARA["yaw"]
    camR_z = np.array([[np.cos(camYaw), -np.sin(camYaw), 0], [np.sin(camYaw), np.cos(camYaw), 0], [0, 0, 1]])
    camR_y = np.array([[np.cos(camPitch), 0, np.sin(camPitch)], [0, 1, 0], [-np.sin(camPitch), 0, np.cos(camPitch)]])
    camR_x = np.array([[1, 0, 0], [0, np.cos(camRoll), -np.sin(camRoll)], [0, np.sin(camRoll), np.cos(camRoll)]])
    camR = camR_z @ camR_y @ camR_x

    xyz = laserCloud[:, :3] - lidar_offset
    xyz = xyz @ lidarR
    xyz = xyz - cam_offset
    xyz = xyz @ camR

    horiDis = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2)
    horiPixelID = (CAMERA_PARA["width"] / (2 * np.pi) * np.arctan2(xyz[:, 0], xyz[:, 2]) + CAMERA_PARA["width"] / 2 + 1).astype(int)
    vertPixelID = (CAMERA_PARA["width"] / (2 * np.pi) * np.arctan(xyz[:, 1] / horiDis) + CAMERA_PARA["height"] / 2 + 1).astype(int)
    pixelDepth = horiDis

    horiPixelID = np.clip(horiPixelID, 0, CAMERA_PARA["width"] - 1)
    vertPixelID = np.clip(vertPixelID, 0, CAMERA_PARA["height"] - 1)
    point_pixel_idx = np.array([horiPixelID, vertPixelID, pixelDepth]).T
    
    return point_pixel_idx

def scan2pixels_scannet(cloud):
    rgb_intrinsics = {
        'fx': 1169.621094,
        'fy': 1167.105103,
        'cx': 646.295044,
        'cy': 489.927032,
    }

    rgb_width = 1296
    rgb_height = 968

    x = cloud[:, 0]
    y = cloud[:, 1]
    x_rgb = x * rgb_intrinsics['fx'] / (cloud[:, 2] + 1e-6) + rgb_intrinsics['cx']
    y_rgb = y * rgb_intrinsics['fy'] / (cloud[:, 2] + 1e-6) + rgb_intrinsics['cy']

    point_pixel_idx = np.array([y_rgb, x_rgb, cloud[:, 2]]).T
    return point_pixel_idx

def grow_cluster_from_min(points, threshold=0.3):
    """
    从最小值点开始，区域生长聚类
    Args:
        points: numpy.ndarray, shape (N, D)，N个D维点
        threshold: float，最大距离阈值
    Returns:
        cluster_idx: numpy.ndarray, shape (N,) 的bool数组，True表示属于该簇
    """
    points = np.asarray(points)
    N = len(points)
    if N == 0:
        return np.array([], dtype=bool)

    # 找到最小值点（按第一维来找）
    min_idx = np.argmin(points[:, 0]) if points.ndim > 1 else np.argmin(points)
    cluster_idx = np.zeros(N, dtype=bool)
    cluster_idx[min_idx] = True

    changed = True
    while changed:
        changed = False
        # 当前簇里的点
        cluster_points = points[cluster_idx]
        # 簇外的点
        outside_idx = np.where(~cluster_idx)[0]
        if len(outside_idx) == 0:
            break

        # 计算簇外点到簇的最小距离
        dists = np.min(np.linalg.norm(points[outside_idx, None, :] - cluster_points[None, :, :], axis=2), axis=1)

        # 满足阈值的点加入簇
        new_points = outside_idx[dists < threshold]
        if len(new_points) > 0:
            cluster_idx[new_points] = True
            changed = True

    return cluster_idx

class CloudImageFusion:
    def __init__(self, platform):
        self.platform_list = ['wheelchair', 'mecanum', 'mecanum_sim', 'scannet', 'diablo']

        if platform not in self.platform_list:
            raise ValueError(f"Invalid platform: {platform}. Available platforms: {self.platform_list}")
        else:
            self.platform = platform
            self.scan2pixels = eval(f"scan2pixels_{platform}")
        
        if platform == 'wheelchair':
            self.scan2pixels = scan2pixels_wheelchair
        elif platform == 'mecanum':
            self.scan2pixels = scan2pixels_mecanum
        elif platform == 'mecanum_sim':
            self.scan2pixels = scan2pixels_mecanum_sim
        elif platform == 'scannet':
            self.scan2pixels = scan2pixels_scannet
        elif platform == 'diablo':
            self.scan2pixels = scan2pixels_diablo
        else:
            print(f"Invalid platform: {platform}. Available platforms: [wheelchair, mecanum, mecanum_sim, scannet, diablo]")
            raise ValueError
    
    def generate_seg_cloud(self, cloud: np.ndarray, masks, labels, confidences, R_b2w, t_b2w, image_src=None):
        # Project the cloud points to image pixels
        point_pixel_idx = self.scan2pixels(cloud) # [N, 3] array of pixel coordinates (x, y, depth)

        if masks is None or len(masks) == 0:
            return None, None
        
        image_shape = masks[0].shape
        
        out_of_bound_filter = (point_pixel_idx[:, 0] >= 0) & \
                            (point_pixel_idx[:, 0] < image_shape[1]) & \
                            (point_pixel_idx[:, 1] >= 0) & \
                            (point_pixel_idx[:, 1] < image_shape[0])

        point_pixel_idx = point_pixel_idx[out_of_bound_filter]
        cloud = cloud[out_of_bound_filter]
        
        horDis = point_pixel_idx[:, 2] 
        point_pixel_idx = point_pixel_idx.astype(int)

        all_obj_cloud_mask = np.zeros(cloud.shape[0], dtype=bool)
        all_obj_cloud_mask_ori = np.zeros(cloud.shape[0], dtype=bool)
        obj_cloud_world_list = []
        for i in range(len(labels)):
            obj_mask = masks[i]
            cloud_mask = obj_mask[point_pixel_idx[:, 1], point_pixel_idx[:, 0]].astype(bool)

            obj_depth = horDis[cloud_mask].reshape(-1, 1)
            obj_cloud = cloud[cloud_mask]

            if obj_depth.shape[0] <=1:
                obj_cloud_world = obj_cloud[:, :3] @ R_b2w.T + t_b2w
                obj_cloud_world_list.append(obj_cloud_world)
                continue
            # 错位相减obj_depth
            obj_depth_diff = (obj_depth[1:] - obj_depth[:-1]).squeeze()
            obj_depth_max = np.max(obj_depth_diff)
            # print(f"{obj_depth_diff}")
            # print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! {obj_depth_max}")
            # min_depth = np.min(obj_depth)
            # max_depth = np.max(obj_depth)
            # count = len(obj_depth > (min_depth + max_depth) / 2)
            # if obj_depth_max > 0.5 and len(obj_depth) > 5:
            if obj_depth_max > 0.3:
                # print(f"Object {i} has large depth variation: {obj_depth_max}, len: {len(obj_depth)}")
                # from sklearn.cluster import DBSCAN
                # db = DBSCAN(eps=0.2, min_samples=5).fit(obj_depth)
                # labels = db.labels_

                # # 找到最小值点（按第一维比较）
                # min_idx = np.argmin(obj_depth[:, 0]) if obj_depth.ndim > 1 else np.argmin(obj_depth)
                # min_label = labels[min_idx]
                # cluster_mask = (labels == min_label)
                # obj_cloud = obj_cloud[cluster_mask]

                # i = 0            
                # for j in range(len(cloud_mask)):
                #     if cloud_mask[j] == False:
                #         continue
                #     else:
                #         if cluster_mask[i] == False:
                #             cloud_mask[j] = False
                #         i += 1
                # all_obj_cloud_mask = np.logical_or(all_obj_cloud_mask, cloud_mask)

                # # ----------------------------------------------------------------------------
                # idx_tmp = grow_cluster_from_min(obj_depth.reshape(-1, 1), threshold=0.3)
                # obj_cloud = obj_cloud[idx_tmp]
                # # ----------------------------------------------------------------------------

                # ----------------------------------------------------------------------------
                # only keep the idx before the largest jump
                idx_tmp = np.ones(len(obj_depth), dtype=bool)
                jump_idx = np.argmax(obj_depth_diff)
                idx_tmp[jump_idx+1:] = False
                obj_cloud = obj_cloud[idx_tmp]

                # 用 idx_tmp 过滤 cloud_mask
                filtered_mask = cloud_mask.copy()
                filtered_mask[np.where(cloud_mask)[0][~idx_tmp]] = False
                # i = 0            
                # for j in range(len(cloud_mask)):
                #     if cloud_mask[j] == False:
                #         continue
                #     else:
                #         if idx_tmp[i] == False:
                #             cloud_mask[j] = False
                #         i += 1
                all_obj_cloud_mask = np.logical_or(all_obj_cloud_mask, filtered_mask)
                all_obj_cloud_mask_ori = np.logical_or(all_obj_cloud_mask_ori, cloud_mask)
            else:
                all_obj_cloud_mask = np.logical_or(all_obj_cloud_mask, cloud_mask)
                all_obj_cloud_mask_ori = np.logical_or(all_obj_cloud_mask_ori, cloud_mask)
                # obj_cloud_list.append(obj_cloud)
        
            obj_cloud_world = obj_cloud[:, :3] @ R_b2w.T + t_b2w
            obj_cloud_world_list.append(obj_cloud_world)

        # if image_src is not None:
        #     # all_obj_cloud = cloud
        #     # all_obj_point_pixel_idx = point_pixel_idx
        #     # horDis = horDis
        #     all_obj_cloud_mask = np.ones_like(all_obj_cloud_mask, dtype=bool)
        #     time1 = int(round(time.time() * 1000))

        #     all_obj_cloud = cloud[all_obj_cloud_mask]
        #     all_obj_point_pixel_idx = point_pixel_idx[all_obj_cloud_mask]
        #     horDis_tmp = horDis[all_obj_cloud_mask]
        #     maxRange = 6.0
        #     pixelVal = np.clip(255 * horDis_tmp / maxRange, 0, 255).astype(np.uint8)
        #     image_src[all_obj_point_pixel_idx[:, 1], all_obj_point_pixel_idx[:, 0]] = np.array([pixelVal, 255-pixelVal, np.zeros_like(pixelVal)]).T # assume RGB
        #     # image_src[all_obj_point_pixel_idx[:, 1], all_obj_point_pixel_idx[:, 0]] = np.array([np.zeros_like(horDis_tmp), horDis_tmp*10, np.zeros_like(horDis_tmp)]).T # assume RGB, gray image
        #     cv2.imwrite(f"debug_obj/debug_all_obj_points_{time1}_1.png", image_src)


        if image_src is not None:
            # Visualize ALL point cloud points projected on image (not just object points)
            all_obj_cloud = cloud
            all_obj_point_pixel_idx = point_pixel_idx
            horDis = horDis
            maxRange = 8.0
            pixelVal = np.clip(255 * horDis / maxRange, 0, 255).astype(np.uint8)
            
            # Create color map: close points = red, far points = blue
            colors = np.stack([pixelVal, 255 - pixelVal, np.zeros_like(pixelVal)], axis=1)
            
            # Draw each point as a small circle
            for coords, color in zip(point_pixel_idx, colors):
                x, y = coords[:2]
                cv2.circle(image_src, (int(x), int(y)), radius=1, 
                          color=tuple(int(c) for c in color), thickness=-1)
            time1 = int(round(time.time() * 1000))
            cv2.imwrite(f"debug_obj/debug_all_obj_points_{time1}_1.png", image_src)
        
        return obj_cloud_world_list

    # @profile
    def generate_seg_cloud_v2(self, cloud: np.ndarray, masks, labels, confidences, R_b2w, t_b2w, image_src=None):
        point_pixel_idx = self.scan2pixels(cloud)

        if masks is None:
            return None, None
        
        image_shape = masks[0].shape
        
        out_of_bound_filter = (point_pixel_idx[:, 0] >= 0) & \
                            (point_pixel_idx[:, 0] < image_shape[1]) & \
                            (point_pixel_idx[:, 1] >= 0) & \
                            (point_pixel_idx[:, 1] < image_shape[0])

        point_pixel_idx = point_pixel_idx[out_of_bound_filter]
        cloud = cloud[out_of_bound_filter]
        
        depths = point_pixel_idx[:, 2]
        point_pixel_idx = point_pixel_idx.astype(int)

        depth_image = np.full(image_shape, np.inf, dtype=np.float32)

        import time
        start_time = time.time()

        # pixel_indices, depths = min_depth_per_pixel(point_pixel_idx[:, :2], horDis)
        # pixel_indices = np.array(pixel_indices, dtype=int)
        # pixel_indices = pixel_indices[pixel_indices[:, 0] >= 0]
        # depths = np.array(depths)

        np.minimum.at(depth_image, (point_pixel_idx[:, 1], point_pixel_idx[:, 0]), depths)
        structure = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=np.uint8)
        inflated_depth_image = scipy.ndimage.grey_dilation(depth_image, footprint=structure, mode='nearest')

        inflated_depth_image = np.minimum(inflated_depth_image, depth_image)

        print(f'pixel conversion: {time.time() - start_time} for {point_pixel_idx.shape[0]} points')
        # for i, pixel_idx in enumerate(pixel_indices):
        #     depth_image[*pixel_idx[[1, 0]].tolist()] = depths[i]
            
        # depth_image[pixel_indices[:, 1], pixel_indices[:, 0]] = depths

        valid_mask = ~np.isinf(inflated_depth_image)  # Mask for valid depth values
        if valid_mask.any():
            min_depth = inflated_depth_image[valid_mask].min()
            max_depth = inflated_depth_image[valid_mask].max()

            print(f"Min depth: {min_depth}, Max depth: {max_depth}")

            # Normalize only valid depth values
            normalized_depth = np.zeros_like(inflated_depth_image, dtype=np.uint8)
            normalized_depth[valid_mask] = 255 * (1 - (inflated_depth_image[valid_mask] - min_depth) / (max_depth - min_depth + 1e-6))
        else:
            normalized_depth = np.zeros_like(inflated_depth_image, dtype=np.uint8)  # If all values are inf, return a blank image
        
        # cv2.imshow("Depth Image", normalized_depth)
        # cv2.waitKey(1)  # Wait for a key press to close the window

        all_obj_cloud_mask = np.zeros(cloud.shape[0], dtype=bool)
        obj_cloud_world_list = []
        for i in range(len(labels)):
            obj_mask = masks[i]
            cloud_mask = obj_mask[point_pixel_idx[:, 1], point_pixel_idx[:, 0]].astype(bool)
            all_obj_cloud_mask = np.logical_or(all_obj_cloud_mask, cloud_mask)
            obj_cloud = cloud[cloud_mask]
                    
            # obj_cloud_list.append(obj_cloud)
            
            obj_cloud_world = obj_cloud[:, :3] @ R_b2w.T + t_b2w
            obj_cloud_world_list.append(obj_cloud_world)

        if image_src is not None:
            all_obj_cloud = cloud
            all_obj_point_pixel_idx = point_pixel_idx
            horDis = horDis
            # all_obj_cloud = cloud[all_obj_cloud_mask]
            # all_obj_point_pixel_idx = point_pixel_idx[all_obj_cloud_mask]
            # horDis = horDis[all_obj_cloud_mask]
            maxRange = 6.0
            pixelVal = np.clip(255 * horDis / maxRange, 0, 255).astype(np.uint8)
            image_src[all_obj_point_pixel_idx[:, 1], all_obj_point_pixel_idx[:, 0]] = np.array([pixelVal, 255-pixelVal, np.zeros_like(pixelVal)]).T # assume RGB
        
        return obj_cloud_world_list, normalized_depth
