"""Launch the marker localizer service with parameters loaded from YAML."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("arena_marker_localizer")
    default_params = os.path.join(pkg_share, "config", "default.yaml")

    params_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to a parameter YAML for the service.",
    )

    return LaunchDescription([
        params_arg,
        Node(
            package="arena_marker_localizer",
            executable="marker_localizer_service",
            name="marker_localizer_service",
            output="screen",
            parameters=[LaunchConfiguration("params_file")],
        ),
    ])
