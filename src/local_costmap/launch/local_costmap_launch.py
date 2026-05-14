import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    config = os.path.join(
        get_package_share_directory('local_costmap'),
        'config',
        'local_costmap_params.yaml'
    )

    costmap_node = Node(
        package='local_costmap',
        executable='local_costmap_node',
        name='local_costmap_node',
        output='screen',
        parameters=[config],
    )

    fake_map_node = Node(
        package='local_costmap',
        executable='fake_map_publisher',
        name='fake_map_publisher',
        output='screen',
        parameters=[{
            'scenario': 'room',      # room | corridor | lshape | dynamic
            'move_robot': True,
            'publish_rate': 1.0,
        }]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
    )

    return LaunchDescription([
        fake_map_node,    # publishes /map + TF map→base_link
        costmap_node,     # subscribes to /map, publishes local costmap
        rviz_node,
    ])