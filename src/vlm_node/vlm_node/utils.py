import numpy as np
from scipy.ndimage import minimum_filter
import cv2


def project_bbox3d(img, bbox3d, transform_matrix, platform='mecanum'):
    if platform == 'mecanum': # for mecanum wheel platform
        LIDAR_PARA= {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        CAMERA_PARA= {"x": -0.24, "y": -0.0, "z": 0.14, "roll": -1.5707963, "pitch": 0, "yaw": -1.5707963, "hfov": 360, "vfov": 120, "width": 1920, "height": 640}
    else: # for mecanum simulation platform
        LIDAR_PARA= {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        CAMERA_PARA= {"x": 0.0, "y": 0.0, "z": 0.1, "roll": -1.5707963, "pitch": 0, "yaw": -1.5707963, "hfov": 360, "vfov": 120, "width": 1920, "height": 640}

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

    # we have the transform matrix from lidar to world, I want to transform the bbox3d from world to lidar
    bbox3d = np.array(bbox3d)
    if bbox3d.shape[1] != 3:
        bbox3d = bbox3d.reshape(-1, 3)
    # get the center of the bbox3d and append it to the bbox3d
    center = np.mean(bbox3d, axis=0)
    bbox3d = np.vstack((bbox3d, center))

    Rb2w = transform_matrix[:3, :3]
    tb2w = transform_matrix[:3, 3]
    Rw2b = Rb2w.T
    tw2b = - Rw2b @ tb2w
    bbox3d = bbox3d @ Rw2b.T + tw2b

    xyz = bbox3d[:, :3] - lidar_offset
    xyz = xyz @ lidarR
    xyz = xyz - cam_offset
    xyz = xyz @ camR

    horiDis = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2)
    horiPixelID = (CAMERA_PARA["width"] / (2 * np.pi) * np.arctan2(xyz[:, 0], xyz[:, 2]) + CAMERA_PARA["width"] / 2 + 1).astype(int)
    vertPixelID = (CAMERA_PARA["width"] / (2 * np.pi) * np.arctan(xyz[:, 1] / horiDis) + CAMERA_PARA["height"] / 2 + 1).astype(int)
    pixelDepth = horiDis

    horiPixelID = np.clip(horiPixelID, 0, CAMERA_PARA["width"] - 1)
    vertPixelID = np.clip(vertPixelID, 0, CAMERA_PARA["height"] - 1)

    center_horiPixelID = horiPixelID[-1]
    center_vertPixelID = vertPixelID[-1]
    center_pixelDepth = pixelDepth[-1]

    horiPixelID = horiPixelID[:-1]
    vertPixelID = vertPixelID[:-1]
    pixelDepth = pixelDepth[:-1]

    # get the bbox in pixel
    min_x = np.min(horiPixelID)
    max_x = np.max(horiPixelID)
    min_y = np.min(vertPixelID)
    max_y = np.max(vertPixelID)

    # apply a margin
    margin = 5
    min_x = max(0, min_x - margin)
    max_x = min(CAMERA_PARA["width"] - 1, max_x + margin)
    min_y = max(0, min_y - margin)
    max_y = min(CAMERA_PARA["height"] - 1, max_y + margin)
    
    # 将全景图旋转，使得中心点在图像中间
    shift = center_horiPixelID - CAMERA_PARA["width"] // 2
    img = np.roll(img, -shift, axis=1)
    if shift > 0:
        min_x = (min_x - shift) % CAMERA_PARA["width"]
        max_x = (max_x - shift) % CAMERA_PARA["width"]
    else:
        min_x = (min_x - shift) % CAMERA_PARA["width"]
        max_x = (max_x - shift) % CAMERA_PARA["width"]

    # draw the bbox on the image
    img = cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    # # draw the poit cloud on the image
    # horiDis = horiDis
    # maxRange = 8.0
    # pixelVal = np.clip(255 * horiDis / maxRange, 0, 255).astype(np.uint8)
    # # Create color map: close points = red, far points = blue
    # colors = np.stack([pixelVal, 255 - pixelVal, np.zeros_like(pixelVal)], axis=1)
    # # Draw each point as a small circle
    # point_pixel_idx = np.stack([horiPixelID, vertPixelID, pixelDepth], axis=1)
    # for coords, color in zip(point_pixel_idx, colors):
    #     x, y = coords[:2]
    #     cv2.circle(img, (int(x), int(y)), radius=5, 
    #                 color=tuple(int(c) for c in color), thickness=-1)

    return img