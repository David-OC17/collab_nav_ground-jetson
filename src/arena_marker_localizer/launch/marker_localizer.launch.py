"""Launch the marker localizer service with parameters loaded from YAML."""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Resolve relative to this file — works in source tree and after
    # installation (both use the same ../config/ layout), with no
    # dependency on the workspace being sourced.
    _config_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "config")
    )
    default_params = os.path.join(_config_dir, "default.yaml")
    default_intrinsics = os.path.join(_config_dir, "calibration.yaml")

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
            parameters=[
                LaunchConfiguration("params_file"),
                {"intrinsics_path": default_intrinsics},
            ],
        ),
    ])
