#!/usr/bin/env python3
"""
Real-robot launch — Frontier Exploration Fallback Stack
========================================================
Assumes the following are already running and providing a complete TF tree:
  • occupancy_mapper  → /amr/world_map  (frame_id='world')
  • EKF               → /amr/ekf/odom + TF odom→base_footprint
  • Localisation node → TF world→odom  (external)
  • D435i driver      → image + camera_info + camera TF subtree
  • trajectory_planner→ astar_planner2 + spline_follower
  • AMR base ctrl     → subscribes /amr/reference

TF chain expected:
    world → odom → base_footprint → camera_link → camera_color_optical_frame
                                  → lidar

frontier_explorer uses odom_topic (/amr/ekf/odom) and transforms it to
world frame via TF on every tick — so it automatically gets the correct
world-frame position once world→odom exists in the tree.

Usage
-----
  ros2 launch frontier_explorer explore_real_launch.py

  # Start mission after localisation is confirmed:
  ros2 topic pub /mission/start std_msgs/msg/Bool "{data: true}" --once

  # Verify TF tree is complete before starting:
  ros2 run tf2_ros tf2_echo world base_footprint
"""

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    pkg      = get_package_share_directory('frontier_explorer')
    rviz_cfg = os.path.join(pkg, 'rviz', 'frontier_exploration.rviz')

    args = [

        # ── Frames ───────────────────────────────────────────────────────
        DeclareLaunchArgument(
            'world_frame', default_value='world',
            description='Fixed world frame — must match occupancy_mapper world_frame'
        ),
        DeclareLaunchArgument(
            'camera_frame', default_value='camera_color_optical_frame',
        ),

        # ── Topics ───────────────────────────────────────────────────────
        DeclareLaunchArgument(
            'odom_topic', default_value='/amr/ekf/odom',
            description='EKF odometry — transformed to world frame via TF'
        ),
        DeclareLaunchArgument(
            'map_topic', default_value='/amr/world_map',
            description='Occupancy map published by occupancy_mapper'
        ),

        # ── Camera mount (base_footprint → camera_link) ──────────────────
        # roll=-π/2, yaw=-π/2 compensates for RealSense optical frame rotation.
        # Verified: optical Z aligns with robot X (forward) in world frame.
        DeclareLaunchArgument('camera_x',     default_value='0.1'),
        DeclareLaunchArgument('camera_y',     default_value='0.0'),
        DeclareLaunchArgument('camera_z',     default_value='0.2'),
        DeclareLaunchArgument('camera_roll',  default_value='-1.5708'),
        DeclareLaunchArgument('camera_pitch', default_value='0.0'),
        DeclareLaunchArgument('camera_yaw',   default_value='-1.5708'),

        # ── ArUco (mission goal marker) ──────────────────────────────────
        DeclareLaunchArgument('target_marker_id', default_value='4', description='ArUco ID of the mission goal marker'),
        DeclareLaunchArgument('marker_size_m',      default_value='0.13'),
        DeclareLaunchArgument('aruco_dict',         default_value='DICT_4X4_50'),
        DeclareLaunchArgument('min_detection_area', default_value='200'),

        # ── Frontier explorer ────────────────────────────────────────────
        DeclareLaunchArgument('w_dist',                     default_value='0.85'),
        DeclareLaunchArgument('w_size',                     default_value='0.15'),
        DeclareLaunchArgument('w_heading',                  default_value='0.40'),
        DeclareLaunchArgument('min_cluster_size',           default_value='5'),
        DeclareLaunchArgument('max_frontier_dist',          default_value='8.0'),
        DeclareLaunchArgument('frontier_update_rate',       default_value='1.0'),
        DeclareLaunchArgument('min_goal_dist',              default_value='0.40'),
        DeclareLaunchArgument('frontier_goal_reached_dist', default_value='0.20'),
        DeclareLaunchArgument('safe_goal_radius',           default_value='0.25'),
        DeclareLaunchArgument('require_camera_coverage',    default_value='false'),

        # ── explorer_controller ──────────────────────────────────────────
        DeclareLaunchArgument('goal_reached_dist',     default_value='0.35'),
        DeclareLaunchArgument('detection_timeout_sec', default_value='2.0'),

        # ── camera_fov_tracker ───────────────────────────────────────────
        DeclareLaunchArgument('fov_update_rate', default_value='5.0'),

        # ── Misc ─────────────────────────────────────────────────────────
        DeclareLaunchArgument('rviz', default_value='true'),
    ]

    # =========================================================================
    # Nodes
    # =========================================================================

    # ── Static TF: world → odom (bootstrap identity) ──────────────────────
    # Publishes a zero-offset world→odom transform so occupancy_mapper and
    # frontier_explorer can start immediately. Your external localisation
    # node will overwrite this with the true offset once it computes it.
    # If your localisation node publishes world→odom itself, remove this.
    world_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_odom_tf',
        output='screen',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.0',
            '--roll', '0.0', '--pitch', '0.0', '--yaw', '0.0',
            '--frame-id', 'world',
            '--child-frame-id', 'odom',
        ]
    )

    # ── Static TF: base_footprint → camera_link ───────────────────────────
    camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_link_tf',
        output='screen',
        arguments=[
            '--x',     LaunchConfiguration('camera_x'),
            '--y',     LaunchConfiguration('camera_y'),
            '--z',     LaunchConfiguration('camera_z'),
            '--roll',  LaunchConfiguration('camera_roll'),
            '--pitch', LaunchConfiguration('camera_pitch'),
            '--yaw',   LaunchConfiguration('camera_yaw'),
            '--frame-id',       'base_footprint',
            '--child-frame-id', 'camera_link',
        ]
    )

    # ── ArUco detector — detects goal marker ─────────────────────────────
    aruco_goal_detector = Node(
        package='frontier_explorer',
        executable='aruco_goal_detector',
        name='aruco_goal_detector',
        output='screen',
        parameters=[{
            'marker_size_m':       LaunchConfiguration('marker_size_m'),
            'camera_frame':        LaunchConfiguration('camera_frame'),
            'image_topic':         '/camera/camera/color/image_raw',
            'camera_info_topic':   '/camera/camera/color/camera_info',
            'aruco_dict':          LaunchConfiguration('aruco_dict'),
            'min_detection_area':  LaunchConfiguration('min_detection_area'),
            'publish_debug_image': True,
        }]
    )

    # ── ArUco world bridge — goal marker camera frame → world frame ───────
    aruco_world_bridge = Node(
        package='frontier_explorer',
        executable='aruco_world_bridge',
        name='aruco_world_bridge',
        output='screen',
        parameters=[{
            'target_marker_id': LaunchConfiguration('target_marker_id'),
            'world_frame':      LaunchConfiguration('world_frame'),
            'tf_timeout_sec':   0.10,
        }]
    )

    # ── Camera FOV tracker ────────────────────────────────────────────────
    # Only launched when require_camera_coverage=true AND the camera is
    # pitched downward enough for FOV rays to hit the floor plane.
    # With a horizontal camera (default) this node produces only
    # 'FOV rays do not intersect floor plane' warnings — disable it.
    camera_fov_tracker = Node(
        package='frontier_explorer',
        executable='camera_fov_tracker',
        name='camera_fov_tracker',
        output='screen',
        condition=IfCondition(LaunchConfiguration('require_camera_coverage')),
        parameters=[{
            'camera_frame':      LaunchConfiguration('camera_frame'),
            'world_frame':       LaunchConfiguration('world_frame'),
            'camera_info_topic': '/camera/camera/color/camera_info',
            'map_topic':         LaunchConfiguration('map_topic'),
            'odom_topic':        LaunchConfiguration('odom_topic'),
            'fov_map_topic':     '/camera/fov_map',
            'fov_marker_topic':  '/camera/fov_marker',
            'update_rate_hz':    LaunchConfiguration('fov_update_rate'),
        }]
    )

    # ── Frontier explorer ─────────────────────────────────────────────────
    frontier_explorer = Node(
        package='frontier_explorer',
        executable='frontier_explorer',
        name='frontier_explorer',
        output='screen',
        parameters=[{
            'map_topic':               LaunchConfiguration('map_topic'),
            'pose_topic':              '/follower/pose',   # unused — odom_topic set
            'goal_topic':              '/frontier/goal',
            'world_frame':             LaunchConfiguration('world_frame'),
            'w_dist':                  LaunchConfiguration('w_dist'),
            'w_size':                  LaunchConfiguration('w_size'),
            'w_heading':               LaunchConfiguration('w_heading'),
            'min_cluster_size':        LaunchConfiguration('min_cluster_size'),
            'max_frontier_dist':       LaunchConfiguration('max_frontier_dist'),
            'update_rate':             LaunchConfiguration('frontier_update_rate'),
            'active':                  False,
            'odom_topic':              LaunchConfiguration('odom_topic'),
            'min_goal_dist':           LaunchConfiguration('min_goal_dist'),
            'goal_reached_dist':       LaunchConfiguration('frontier_goal_reached_dist'),
            'safe_goal_radius':        LaunchConfiguration('safe_goal_radius'),
            'require_camera_coverage': LaunchConfiguration('require_camera_coverage'),
            'fov_map_topic':           '/camera/fov_map',
        }]
    )

    # ── ArUco visual servo ───────────────────────────────────────────────────
    # Takes over /amr/reference when explorer_controller detects the target
    # marker and sends /aruco_servo/enable = True. Drives purely from camera
    # image — no map, no TF, no EKF drift.
    aruco_visual_servo = Node(
        package='frontier_explorer',
        executable='aruco_visual_servo',
        name='aruco_visual_servo',
        output='screen',
        parameters=[{
            'target_marker_id':    LaunchConfiguration('target_marker_id'),
            'world_frame':         LaunchConfiguration('world_frame'),
            'stop_dist_m':         0.50,
            'Kw':                  0.60,
            'Kv':                  0.40,
            'max_linear':          0.25,
            'max_angular':         0.60,
            'centering_threshold': 0.10,
            'timeout_sec':         2.0,
        }]
    )

    # ── Mission controller ────────────────────────────────────────────────
    explorer_controller = Node(
        package='frontier_explorer',
        executable='explorer_controller',
        name='explorer_controller',
        output='screen',
        parameters=[{
            'target_marker_id':      LaunchConfiguration('target_marker_id'),
            'goal_reached_dist':     LaunchConfiguration('goal_reached_dist'),
            'detection_timeout_sec': LaunchConfiguration('detection_timeout_sec'),
            'world_frame':           LaunchConfiguration('world_frame'),
        }]
    )

    # ── RViz ─────────────────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_cfg],
        condition=IfCondition(LaunchConfiguration('rviz'))
    )

    return LaunchDescription(args + [
        # world_odom_tf,   # must be first — everything needs world frame
        camera_tf,
        aruco_goal_detector,
        aruco_world_bridge,
        camera_fov_tracker,
        frontier_explorer,
        explorer_controller,
        aruco_visual_servo,
        rviz_node,
    ])