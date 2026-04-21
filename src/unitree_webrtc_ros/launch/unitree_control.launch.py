import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from ament_index_python.packages import get_package_share_directory

def launch_detection_node(context):
    unitree_control_node = Node(
        package='unitree_webrtc_ros',
        executable='unitree_control_node',
        name='unitree_control_node',
        output='screen',
        parameters=[get_package_share_directory('unitree_webrtc_ros')+'/' + "unitree_params" + '.yaml']
    )
    return [unitree_control_node]


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        SetParameter(name='use_sim_time', value=use_sim_time),
        OpaqueFunction(function=launch_detection_node)
    ])
