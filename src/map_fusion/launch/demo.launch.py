"""Demo launch: the mock arena publishers plus the fusion node.

No hardware required. With RViz set the fixed frame to ``world`` and add the
``/drone/map`` and ``/fusion/slam_reprojected`` OccupancyGrid displays to watch
the SLAM layer snap into alignment as confidence grows.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('map_fusion')
    params = os.path.join(pkg, 'config', 'map_fusion_params.yaml')

    mock = Node(
        package='map_fusion',
        executable='mock_arena',
        name='mock_arena',
        output='screen',
    )

    fusion_node = Node(
        package='map_fusion',
        executable='map_fusion_node',
        name='map_fusion_node',
        output='screen',
        parameters=[params],
    )

    return LaunchDescription([mock, fusion_node])
