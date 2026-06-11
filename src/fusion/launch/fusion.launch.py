from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        Node(
            package='fusion',
            executable='map_fusion_node',
            name='map_fusion_node',
            output='screen',
            parameters=[{
                # ── Topic names ─────────────────────────────────────────────
                # Must match arena_map_builder and world_mapper exactly.
                'drone_map_topic': '/drone/map',
                'amr_map_topic':   '/amr/world_map',
                'fused_map_topic': '/fused_map',
            }],
        ),
    ])
