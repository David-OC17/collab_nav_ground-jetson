#!/usr/bin/env python3
"""
Launch file: astar_planner + RViz
Assumes /map and TF are already being published (e.g. fake_map_publisher running).
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    pkg_share = get_package_share_directory('trajectory_planner')
    default_rviz_config = os.path.join(pkg_share, 'config', 'trajectory_planner.rviz')

    # ------------------------------------------------------------------
    # Launch arguments
    # ------------------------------------------------------------------
    map_topic_arg = DeclareLaunchArgument(
        'map_topic', default_value='/map',
        description='Topic for the occupancy grid')

    replan_on_map_arg = DeclareLaunchArgument(
        'replan_on_map', default_value='true',
        description='Replan when map updates')

    stuck_detection_arg = DeclareLaunchArgument(
        'stuck_detection', default_value='true',
        description='Enable stuck detection and replanning')

    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config', default_value=default_rviz_config,
        description='Path to RViz config file')

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    astar_node = Node(
        package='trajectory_planner',
        executable='astar_planner',
        name='astar_planner',
        output='screen',
        parameters=[{
            'map_topic':        LaunchConfiguration('map_topic'),
            'map_frame':        'map',
            'robot_base_frame': 'base_link',
            'replan_on_map':    LaunchConfiguration('replan_on_map'),
            'stuck_detection':  LaunchConfiguration('stuck_detection'),
            'stuck_check_rate': 1.0,
            'path_marker_z':    0.05,
        }]
    )

    # UNCOMMENT THIS FOR SIMULATION
    trajectory_follower_sim_node = Node(
        package='trajectory_planner',
        executable='trajectory_follower_sim',
        name='trajectory_follower_sim',
        output='screen',
        parameters=[{
        'linear_speed':     0.5,
        'update_rate':      20.0,
        'goal_tolerance':   0.10,
        'map_frame':        'map',
        'robot_base_frame': 'base_link',
        'path_topic':       '/trajectory_planner/path',
        }]
    )


    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')]
    )

    return LaunchDescription([
        map_topic_arg,
        replan_on_map_arg,
        stuck_detection_arg,
        rviz_config_arg,
        astar_node,
        trajectory_follower_sim_node,
        rviz_node,
    ])