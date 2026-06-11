from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="world_mapper",
            executable="occupancy_mapper",
            name="occupancy_mapper",
            output="screen",
            parameters=[{
                "world_frame": "world",
                "laser_frame": "lidar",   # overridden per-scan by scan.header.frame_id
                "scan_topic": "/scan",
                "map_topic": "/amr/world_map",

                "resolution": 0.05,
                "width_m": 3.9,
                "height_m": 3.9,
                "origin_x": 0.0,               # set both to 0.0 if arena is in +x/+y quadrant
                "origin_y": 0.0,

                "l_occ": 0.65,
                "l_free": -0.60,
                "l_min": -5.0,
                "l_max": 5.0,

                "publish_rate": 1.0,
                "tf_timeout": 0.10,
            }],
        ),
    ])