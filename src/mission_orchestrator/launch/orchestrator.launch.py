import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('mission_orchestrator')
    default_cfg = os.path.join(pkg, 'config', 'orchestrator_params.yaml')

    cfg_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_cfg,
        description='Absolute path to orchestrator_params.yaml',
    )

    orchestrator = Node(
        package='mission_orchestrator',
        executable='orchestrator',
        name='mission_orchestrator',
        output='screen',
        emulate_tty=True,
        parameters=[{'config_file': LaunchConfiguration('config_file')}],
    )

    return LaunchDescription([cfg_arg, orchestrator])
