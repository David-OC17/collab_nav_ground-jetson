#!/usr/bin/env python3
"""
Unified simulation launch file:
  - fake_map_publisher   → /drone/map, /fusion/slam_reprojected, /fusion/confidence
  - astar_planner        → /trajectory_planner/path, /trajectory_planner/path_raw
  - trajectory_follower  → /amr/reference, TF map→base_link
  - rviz2                → visualisation
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    pkg_planner  = get_package_share_directory('trajectory_planner')
    pkg_costmap  = get_package_share_directory('local_costmap')

    default_rviz = os.path.join(pkg_planner, 'config', 'trajectory_planner.rviz')

    # ------------------------------------------------------------------
    # Launch arguments
    # ------------------------------------------------------------------
    scenario_arg = DeclareLaunchArgument(
        'scenario', default_value='room2',
        description='Map scenario: room | room2 | corridor')

    replan_arg = DeclareLaunchArgument(
        'replan_on_map', default_value='true',
        description='Replan when map updates')

    stuck_arg = DeclareLaunchArgument(
        'stuck_detection', default_value='true',
        description='Enable stuck detection and replanning')

    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config', default_value=default_rviz,
        description='Path to RViz config file')

    confidence_ramp_arg = DeclareLaunchArgument(
        'confidence_ramp_sec', default_value='20.0',
        description='Seconds to ramp fusion confidence from 0 to 1')

    slam_reveal_arg = DeclareLaunchArgument(
        'slam_reveal_sec', default_value='120.0',
        description='Seconds to fully reveal the SLAM map')

    # ------------------------------------------------------------------
    # fake_map_publisher
    # Publishes: /drone/map, /fusion/slam_reprojected, /fusion/confidence
    # ------------------------------------------------------------------
    fake_map_node = Node(
        package='local_costmap',
        executable='fake_map_publisher',
        name='fake_map_publisher',
        output='screen',
        parameters=[{
            'scenario':            LaunchConfiguration('scenario'),
            'map_resolution':      0.05,
            'map_width_m':         12.0,
            'map_height_m':        12.0,
            'publish_rate':        1.0,
            'move_robot':          False,
            'confidence_ramp_sec': LaunchConfiguration('confidence_ramp_sec'),
            'slam_reveal_sec':     LaunchConfiguration('slam_reveal_sec'),
            'sensor_range_m':        2.0,
            'position_log_spacing':  0.3,
        }]
    )

    # ------------------------------------------------------------------
    # A* planner
    # Subscribes: /drone/map, /fusion/slam_reprojected, /fusion/confidence
    # Publishes:  /trajectory_planner/path, /trajectory_planner/path_raw
    # ------------------------------------------------------------------
    astar_node = Node(
        package='trajectory_planner',
        executable='astar_planner',
        name='astar_planner',
        output='screen',
        parameters=[{
            # Map topics — no longer /map; now split into drone + slam
            'map_topic':                  '/drone/map',       # drone map subscription
            'slam_reprojected_topic':     '/fusion/slam_reprojected',
            'fusion_confidence_topic':    '/fusion/confidence',
            'map_fusion_threshold':       0.4,

            # Frames
            'map_frame':                  'map',
            'robot_base_frame':           'base_link',

            # Planning behaviour
            'replan_on_map':              LaunchConfiguration('replan_on_map'),
            'stuck_detection':            LaunchConfiguration('stuck_detection'),
            'stuck_check_rate':           1.0,
            'min_replan_interval_sec':    3.0,

            # Inflation
            'inflation_radius':           0.2,
            'robot_radius':               0.20,
            'cost_scaling':               3.5,

            # Spline
            'spline_enabled':             True,
            'spline_decimation':          5,
            'spline_samples':             200,

            # Visualisation
            'path_marker_z':              0.05,
        }]
    )

    # ------------------------------------------------------------------
    # Trajectory follower (simulation)
    # Subscribes: /trajectory_planner/path
    # Publishes:  /amr/reference, TF map→base_link, /follower/robot_marker
    # ------------------------------------------------------------------
    follower_node = Node(
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

    # ------------------------------------------------------------------
    # RViz
    # ------------------------------------------------------------------
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')]
    )

    return LaunchDescription([
        # Arguments
        scenario_arg,
        replan_arg,
        stuck_arg,
        rviz_config_arg,
        confidence_ramp_arg,
        slam_reveal_arg,

        # Nodes — order matters: map first, then planner, then follower
        fake_map_node,
        astar_node,
        follower_node,
        rviz_node,
    ])