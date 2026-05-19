"""Launch the map_fusion node on its own, with parameters from config/."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('map_fusion')
    default_params = os.path.join(pkg, 'config', 'map_fusion_params.yaml')

    params_arg = DeclareLaunchArgument(
        'params_file', default_value=default_params,
        description='Path to the map_fusion_node parameter file.')

    fusion_node = Node(
        package='map_fusion',
        executable='map_fusion_node',
        name='map_fusion_node',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
    )

    return LaunchDescription([params_arg, fusion_node])
