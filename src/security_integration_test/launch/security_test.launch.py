"""
security_test.launch.py
───────────────────────────────────────────────────────────────────────
Launch stub hardware publishers alongside the relay processes to allow
manual / interactive verification of the relay pattern without real
sensors.

Usage:
  ros2 launch security_integration_test security_test.launch.py
  ros2 launch security_integration_test security_test.launch.py \
      certs_dir:=/abs/path/to/certs  security_disabled:=0

The stub publishers mimic:
  stub_scan_pub       — oradar LiDAR C++ driver  → /scan
  stub_optitrack_pub  — Optitrack C++ driver     → /optitrack/rigid_body
                                                 → /optitrack/marker

The relay processes re-sign native messages so secured Python nodes
can consume them.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
_DEFAULT_CERTS = os.path.join(_PROJ_ROOT, 'certs')


def generate_launch_description():
    certs_dir_arg = DeclareLaunchArgument('certs_dir', default_value=_DEFAULT_CERTS)
    security_disabled_arg = DeclareLaunchArgument('security_disabled', default_value='0')
    set_kill_switch = SetEnvironmentVariable(
        'ROS2_SECURITY_DISABLED', LaunchConfiguration('security_disabled'))

    certs = LaunchConfiguration('certs_dir')

    # ── Stub hardware publishers ─────────────────────────────────────────────
    stub_scan = Node(
        package='security_integration_test',
        executable='stub_scan_pub',
        name='stub_scan_pub',
        output='screen',
    )

    stub_optitrack = Node(
        package='security_integration_test',
        executable='stub_optitrack_pub',
        name='stub_optitrack_pub',
        output='screen',
    )

    # ── Legacy relay processes ───────────────────────────────────────────────
    scan_relay = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros2_security', 'legacy_relay',
            '--level', 'sign',
            '--certs-dir', certs,
            '--node-name', 'scan_relay',
            '--bridge', 'sensor_msgs/msg/LaserScan', '/scan', '/scan',
        ],
        output='screen',
        name='scan_relay',
    )

    optitrack_relay = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros2_security', 'legacy_relay',
            '--level', 'sign',
            '--certs-dir', certs,
            '--node-name', 'optitrack_relay',
            '--bridge', 'geometry_msgs/msg/PoseStamped',
                        '/optitrack/rigid_body', '/optitrack/rigid_body',
            '--bridge', 'geometry_msgs/msg/PointStamped',
                        '/optitrack/marker', '/optitrack/marker',
        ],
        output='screen',
        name='optitrack_relay',
    )

    return LaunchDescription([
        certs_dir_arg,
        security_disabled_arg,
        set_kill_switch,
        stub_scan,
        stub_optitrack,
        scan_relay,
        optitrack_relay,
    ])
