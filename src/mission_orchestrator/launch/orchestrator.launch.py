import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
_DEFAULT_CERTS = os.path.join(_PROJ_ROOT, 'certs')


def generate_launch_description():
    pkg = get_package_share_directory('mission_orchestrator')
    default_cfg = os.path.join(pkg, 'config', 'orchestrator_params.yaml')

    cfg_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_cfg,
        description='Absolute path to orchestrator_params.yaml',
    )
    certs_dir_arg = DeclareLaunchArgument('certs_dir', default_value=_DEFAULT_CERTS)
    security_disabled_arg = DeclareLaunchArgument('security_disabled', default_value='0')
    set_kill_switch = SetEnvironmentVariable('ROS2_SECURITY_DISABLED', LaunchConfiguration('security_disabled'))

    orchestrator = Node(
        package='mission_orchestrator',
        executable='orchestrator',
        name='mission_orchestrator',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'config_file': LaunchConfiguration('config_file'),
            'certs_dir':   LaunchConfiguration('certs_dir'),
        }],
    )

    return LaunchDescription([
        cfg_arg,
        certs_dir_arg,
        security_disabled_arg,
        set_kill_switch,
        orchestrator,
    ])
