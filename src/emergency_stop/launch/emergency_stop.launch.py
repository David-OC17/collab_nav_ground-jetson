from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='emergency_stop',
            executable='emergency_stop',
            name='emergency_stop',
            parameters=['config/safety_params.yaml'],
            output='screen',
        )
    ])