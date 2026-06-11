import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
_DEFAULT_CERTS = os.path.join(_PROJ_ROOT, 'certs')


def generate_launch_description():
    certs_dir_arg = DeclareLaunchArgument('certs_dir', default_value=_DEFAULT_CERTS)
    security_disabled_arg = DeclareLaunchArgument('security_disabled', default_value='0')
    set_kill_switch = SetEnvironmentVariable('ROS2_SECURITY_DISABLED', LaunchConfiguration('security_disabled'))

    return LaunchDescription([
        certs_dir_arg,
        security_disabled_arg,
        set_kill_switch,
        Node(
            package='emergency_stop',
            executable='emergency_stop',
            name='emergency_stop',
            parameters=[
                'config/safety_params.yaml',
                {'certs_dir': LaunchConfiguration('certs_dir')},
            ],
            output='screen',
        ),
    ])
