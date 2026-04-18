import os
import struct
from typing import List, Union

import numpy as np
import rclpy
import rosbag2_py
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Point, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker


def init_bag_writer(output_path: str):
    """Initialize a ROS2 bag writer."""
    storage_options = rosbag2_py.StorageOptions(
        uri=output_path,
        storage_id='sqlite3'
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )

    writer = rosbag2_py.SequentialWriter()
    writer.open(storage_options, converter_options)

    return writer


def add_topic(writer, topic_name: str, message_type: str):
    """Add a topic to a ROS2 bag writer."""
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            name=topic_name,
            type=message_type,
            serialization_format='cdr'
        )
    )


def create_odom_msg(odom: dict, seconds: int, nanoseconds: int, frame_id: str = "map") -> Odometry:
    """Create an Odometry message."""
    ros_odom = Odometry()
    ros_odom.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
    ros_odom.header.frame_id = frame_id
    ros_odom.pose.pose.position.x = odom['position'][0]
    ros_odom.pose.pose.position.y = odom['position'][1]
    ros_odom.pose.pose.position.z = odom['position'][2]
    ros_odom.pose.pose.orientation.x = odom['orientation'][0]
    ros_odom.pose.pose.orientation.y = odom['orientation'][1]
    ros_odom.pose.pose.orientation.z = odom['orientation'][2]
    ros_odom.pose.pose.orientation.w = odom['orientation'][3]
    return ros_odom


def create_tf_msg(odom: dict, seconds: int, nanoseconds: int,
                 frame_id: str = "map", child_frame_id: str = "sensor") -> TFMessage:
    """Create a TF message."""
    transform = TransformStamped()
    transform.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
    transform.header.frame_id = frame_id
    transform.child_frame_id = child_frame_id
    transform.transform.translation.x = odom['position'][0]
    transform.transform.translation.y = odom['position'][1]
    transform.transform.translation.z = odom['position'][2]
    transform.transform.rotation.x = odom['orientation'][0]
    transform.transform.rotation.y = odom['orientation'][1]
    transform.transform.rotation.z = odom['orientation'][2]
    transform.transform.rotation.w = odom['orientation'][3]

    tf_msg = TFMessage()
    tf_msg.transforms.append(transform)
    return tf_msg


def create_point_cloud(points: np.ndarray, seconds: int, nanoseconds: int,
                      frame_id: str = "map") -> PointCloud2:
    """Create a PointCloud2 message."""
    header = Header()
    header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
    header.frame_id = frame_id

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    cloud_data = points.astype(np.float32)
    data = cloud_data.tobytes()

    point_cloud = PointCloud2()
    point_cloud.header = header
    point_cloud.height = 1
    point_cloud.width = len(points)
    point_cloud.fields = fields
    point_cloud.is_bigendian = False
    point_cloud.point_step = 12
    point_cloud.row_step = point_cloud.point_step * len(points)
    point_cloud.is_dense = True
    point_cloud.data = data

    return point_cloud


def create_colored_point_cloud(points: np.ndarray, colors: np.ndarray,
                             seconds: int, nanoseconds: int,
                             frame_id: str = "map") -> PointCloud2:
    """Create a colored PointCloud2 message."""
    header = Header()
    header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
    header.frame_id = frame_id

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    if colors.max() <= 1:
        colors = colors * 255
    rgb_colors = colors.astype(np.uint32)
    rgb_colors = (rgb_colors[:, 0].astype(np.uint32) << 16) | \
                 (rgb_colors[:, 1].astype(np.uint32) << 8) | \
                 (rgb_colors[:, 2].astype(np.uint32))
    rgb_colors = rgb_colors.view(np.float32)
    cloud_data = np.concatenate((points, rgb_colors[:, None]), axis=1).astype(np.float32)

    data = cloud_data.tobytes()

    point_cloud = PointCloud2()
    point_cloud.header = header
    point_cloud.height = 1
    point_cloud.width = len(points)
    point_cloud.fields = fields
    point_cloud.is_bigendian = False
    point_cloud.point_step = 16
    point_cloud.row_step = point_cloud.point_step * len(points)
    point_cloud.is_dense = True
    point_cloud.data = data

    return point_cloud


def create_wireframe_marker_from_corners(corners: List[List[float]], ns: str, box_id: int,
                                       color: Union[list, np.ndarray], seconds: int,
                                       nanoseconds: int, frame_id: str = "map") -> Marker:
    """Create a wireframe marker from corners."""
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.id = int(box_id)
    marker.ns = ns
    marker.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3] if len(color) == 4 else 0.8

    marker.scale.x = 0.05

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for edge in edges:
        p1 = Point(x=corners[edge[0]][0], y=corners[edge[0]][1], z=corners[edge[0]][2])
        p2 = Point(x=corners[edge[1]][0], y=corners[edge[1]][1], z=corners[edge[1]][2])
        marker.points.append(p1)
        marker.points.append(p2)

    return marker


def get_3d_box(center: List[float], box_size: List[float], heading_angle: float) -> List[List[float]]:
    """Calculate 3D bounding box corners from its parameterization."""
    def rotz(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, s, 0],
                        [-s, c, 0],
                        [0, 0, 1]])

    rot = rotz(heading_angle)
    l, w, h = box_size
    x_corners = [-l/2, l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2]
    y_corners = [-w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2, w/2]
    z_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]
    corners_3d = np.dot(rot, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d.tolist()


def create_wireframe_marker(center: List[float], extent: List[float], yaw: float,
                          ns: str, box_id: int, color: List[float],
                          seconds: int, nanoseconds: int,
                          frame_id: str = "map") -> Marker:
    """Create a wireframe marker."""
    corners = get_3d_box(center, extent, yaw)
    return create_wireframe_marker_from_corners(corners, ns, box_id, color,
                                              seconds, nanoseconds, frame_id)


def create_point_marker(center: List[float], box_id: int = 0,
                       frame_id: str = "world",
                       color: tuple = (1.0, 0.0, 0.0, 0.8)) -> Marker:
    """Create a point marker."""
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.type = Marker.POINTS
    marker.action = Marker.ADD
    marker.id = int(box_id)
    marker.ns = "points"

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]

    marker.scale.x = 0.5
    marker.scale.y = 0.5
    marker.scale.z = 0.5

    marker.pose.position.x = 0
    marker.pose.position.y = 0
    marker.pose.position.z = 0
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1

    p = Point(*center)
    marker.points.append(p)

    return marker


def create_text_marker(center: List[float], marker_id: int, text: str,
                      color: List[float], text_height: float,
                      seconds: int, nanoseconds: int,
                      frame_id: str = "map") -> Marker:
    """Create a text marker."""
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
    marker.ns = "text"
    marker.id = int(marker_id)
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD

    marker.pose.position.x = center[0]
    marker.pose.position.y = center[1]
    marker.pose.position.z = center[2]

    marker.scale.z = text_height

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3] if len(color) == 4 else 1.0
    marker.text = text

    return marker


def create_box_marker(center: List[float], extent: List[float], yaw: float,
                     ns: str, box_id: int, color: List[float],
                     seconds: int, nanoseconds: int,
                     frame_id: str = "map") -> Marker:
    """Create a box marker."""
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
    marker.ns = ns
    marker.id = int(box_id)
    marker.type = Marker.CUBE
    marker.action = Marker.ADD

    marker.pose.position.x = center[0]
    marker.pose.position.y = center[1]
    marker.pose.position.z = center[2]

    quat = R.from_euler('xyz', [0, 0, yaw]).as_quat()
    marker.pose.orientation.x = quat[0]
    marker.pose.orientation.y = quat[1]
    marker.pose.orientation.z = quat[2]
    marker.pose.orientation.w = quat[3]

    marker.scale.x = extent[0]
    marker.scale.y = extent[1]
    marker.scale.z = extent[2]

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]

    return marker