import colorsys
import numpy as np
from bisect import bisect_left
import open3d as o3d
import torch
from scipy.spatial import cKDTree
import yaml


object_file_path = 'src/semantic_mapping/semantic_mapping/config/objects.yaml'
with open(object_file_path, "r") as file:
    object_config = yaml.safe_load(file)
label_template = object_config['prompts']
object_list = {}
for idx, value in enumerate(label_template.values()):
    labels = value['prompts']
    for label in labels:
        object_list[label] = idx

def R_to_yaw(R):
    return np.arctan2(R[1, 0], R[0, 0])

def normalize_angles_to_pi(angles):
    """
    Normalize angles to the range [-π, π).
    
    Parameters:
        angles (numpy array): Input array of angles in radians.
    
    Returns:
        numpy array: Angles normalized to the range [-π, π).
    """
    return (angles + np.pi) % (2 * np.pi) - np.pi

def discretize_angles(angles, num_bin=20):
    bin_width = 2 * np.pi / num_bin
    return np.floor((angles + np.pi) / bin_width).astype(int)

def generate_colors(n, is_int=False):
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.9
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        if is_int:
            rgb = [int(round(channel * 255)) for channel in rgb]
        colors.append(rgb)
    return colors

def map_label_to_color(label):
    if label in object_list:
        idx = object_list[label]
    else:
        print(f"Warning: Label '{label}' not found in object list. Assigning white.")
        return [1.0, 1.0, 1.0]  # 白色

    # 限制 hue 在蓝绿色区间 [160°, 240°]
    hue_min, hue_max = 160, 240
    hue_range = hue_max - hue_min
    hue = (idx * 37) % hue_range + hue_min   # 用不同步长避免过于接近

    saturation = 0.85  # 高饱和度
    value = 0.95       # 高亮度

    # 注意 colorsys.hsv_to_rgb 需要 0~1 范围的输入
    return colorsys.hsv_to_rgb(hue / 360.0, saturation, value)


def find_closest_stamp(stamp_list, q):
    i_stamp = bisect_left(stamp_list, q)
    if i_stamp == 0:
        return stamp_list[0]
    if i_stamp == len(stamp_list):
        return stamp_list[-1]

    if q - stamp_list[i_stamp - 1] < stamp_list[i_stamp] - q:
        return stamp_list[i_stamp - 1]
    else:
        return stamp_list[i_stamp]

def find_neighbouring_stamps(stamp_list, q):
    i_stamp = bisect_left(stamp_list, q)
    if i_stamp == 0:
        return stamp_list[0], stamp_list[0]
    if i_stamp == len(stamp_list):
        return stamp_list[-1], stamp_list[-1]
    return stamp_list[i_stamp - 1], stamp_list[i_stamp]

import spacy
nlp = spacy.load("en_core_web_sm")
def extract_meta_class(label, class_template):
    # doc = nlp(str(label))
    # Include tokens that are nouns or verbs acting as nouns (gerunds)
    # nouns_and_gerunds = [token.text for token in doc if token.pos_ in {"NOUN", "VERB"} and token.dep_ in {"ROOT", "nsubj", "dobj", "pobj"}]
    matched = False
    meta_class = None

    for meta_label, des in class_template.items():
        if label in des:
            matched = True
            meta_class = meta_label
            break
    
    # if not matched:
    #     for noun in nouns_and_gerunds:
    #         for meta_label, des in class_template.items():
    #             if noun in des:
    #                 matched = True
    #                 meta_class = meta_label
    #                 break
    #         if matched:
    #             break

    if not matched:
        label_words = label.split(" ")
        for word in label_words:
            for meta_label, des in class_template.items():
                if word == meta_label:
                    matched = True
                    meta_class = meta_label
                    break

    return meta_class


from bisect import bisect_left
def find_closest_stamp(stamp_list, q):
    i_stamp = bisect_left(stamp_list, q)
    if i_stamp == 0:
        return stamp_list[0]
    if i_stamp == len(stamp_list):
        return stamp_list[-1]

    if q - stamp_list[i_stamp - 1] < stamp_list[i_stamp] - q:
        return stamp_list[i_stamp - 1]
    else:
        return stamp_list[i_stamp]

def find_neighbouring_stamps(stamp_list, q):
    i_stamp = bisect_left(stamp_list, q)
    if i_stamp == 0:
        return stamp_list[0], stamp_list[0]
    if i_stamp == len(stamp_list):
        return stamp_list[-1], stamp_list[-1]
    return stamp_list[i_stamp - 1], stamp_list[i_stamp]

def get_corners_from_box3d_torch(center, half_extent, angle):
    if isinstance(center, np.ndarray):
        center = torch.tensor(center)
    # if isinstance(half_extent, np.ndarray):
        half_extent = torch.tensor(half_extent)
    # if isinstance(angle, np.ndarray or float or np.float64):
        angle = torch.tensor(angle)

    corners = torch.zeros(8, 3)

    # Get rotation matrix
    R = torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0],
        [torch.sin(angle), torch.cos(angle), 0],
        [0, 0, 1]
    ])

    # Get corners
    corners[0] = center + R @ torch.tensor([half_extent[0], half_extent[1], half_extent[2]])
    corners[1] = center + R @ torch.tensor([half_extent[0], half_extent[1], -half_extent[2]])
    corners[2] = center + R @ torch.tensor([half_extent[0], -half_extent[1], -half_extent[2]])
    corners[3] = center + R @ torch.tensor([half_extent[0], -half_extent[1], half_extent[2]])
    corners[4] = center + R @ torch.tensor([-half_extent[0], half_extent[1], half_extent[2]])
    corners[5] = center + R @ torch.tensor([-half_extent[0], half_extent[1], -half_extent[2]])
    corners[6] = center + R @ torch.tensor([-half_extent[0], -half_extent[1], -half_extent[2]])
    corners[7] = center + R @ torch.tensor([-half_extent[0], -half_extent[1], half_extent[2]])
    
    return corners

def convert_box3d_to_corners_batch(bboxes3d):
    if len(bboxes3d) == 0:
        return np.zeros((0, 8, 3))
    all_corners = []
    for i in range(len(bboxes3d)):
        center = bboxes3d[i, :3]
        extent = bboxes3d[i, 3:6]
        angle = bboxes3d[i, 6]
        corners = get_corners_from_box3d_torch(center, extent, angle)
        all_corners.append(corners)
    return torch.stack(all_corners)

def find_nearby_points(A, B, max_distance):
    """
    Finds points in B that have a nearest neighbor in A within max_distance.

    Parameters:
        A (ndarray): (N, d) array of reference points.
        B (ndarray): (M, d) array of query points.
        max_distance (float): Maximum allowable distance to consider a nearest neighbor.

    Returns:
        ndarray: Indices of points in B that satisfy the condition.
    """
    # Build a k-d tree for A
    tree = cKDTree(A)

    # Query the nearest neighbor distance for each point in B
    distances, _ = tree.query(B, k=1)

    # Find indices of B where the nearest neighbor in A is within max_distance
    valid_indices = np.where(distances <= max_distance)[0]

    return valid_indices