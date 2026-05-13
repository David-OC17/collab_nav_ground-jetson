from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cartographer_ros',
            executable='cartographer_node',
            name='cartographer_node',
            output='screen',
            arguments=[
                '-configuration_directory', '/home/jetson/collab_nav_ground-jetson/config',
                '-configuration_basename', 'cartographer.lua'
            ],
            remappings=[
                ('scan', 'scan_restamped'),
                ('odom', 'ekf/odom'),
            ]
        ),
        Node(
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            name='cartographer_occupancy_grid_node',
            output='screen',
            arguments=['-resolution', '0.05'],
        ),
    ])
