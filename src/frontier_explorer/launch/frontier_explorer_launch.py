#!/usr/bin/env python3
"""
Real-robot launch — Frontier Exploration Fallback Stack
========================================================
Starts every node this package owns that is needed on real hardware.
Does NOT start SLAM, EKF, the low-level AMR controller, the D435i driver, or
the trajectory planner (astar_planner2 + spline_follower) — those must already
be running before this launch file is used. In the mission, the trajectory
planner is brought up by trajectory_planner/planner_launch.py; this fallback
stack only adds the exploration nodes that feed it via /aruco/goal/pose.

Architecture
------------
  /map ───────────────► frontier_explorer (frontiers, perceived-distance scoring)
  EKF odom  ──────────►   │ /frontier/goal
  TF odom→world            ▼
                      explorer_controller ◄── /aruco/detection
                        (IDLE/EXPLORING/HOMING/DONE)
                            │ /aruco/goal/pose
                            ▼
                      astar_planner2  ──► spline_follower ──► /amr/reference
                      (trajectory_planner — already running)

  D435i ──────────────► aruco_goal_detector ──► aruco_world_bridge
                                                    │ /aruco/detection
                        camera_fov_tracker ─────────► frontier_explorer
                            │ /camera/fov_map

Prerequisites (must be running BEFORE this launch)
--------------------------------------------------
  • SLAM (slam_toolbox)     — publishes /map (OccupancyGrid, TRANSIENT_LOCAL)
  • Alignment               — alignment_node publishes static TF: world → odom
  • EKF (robot_localization)— publishes /amr/ekf/odom (Odometry)
                              TF: odom → base_footprint
  • Static TFs              — base_footprint → laser
                              base_footprint → camera_color_optical_frame
  • D435i driver            — /camera/camera/color/image_raw + camera_info
  • trajectory_planner      — astar_planner2 + spline_follower already running
  • AMR base controller     — subscribes to /amr/reference

Usage
-----
  ros2 launch frontier_explorer explore_real_launch.py

  # With overrides:
  ros2 launch frontier_explorer explore_real_launch.py \\
      odom_topic:=/amr/ekf/odom \\
      marker_size_m:=0.15 \\
      rviz:=false

  # Start exploration once everything is running:
  ros2 topic pub /mission/start std_msgs/msg/Bool "{data: true}" --once

  # Stop / reset:
  ros2 topic pub /mission/start std_msgs/msg/Bool "{data: false}" --once
"""

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    pkg      = get_package_share_directory('frontier_explorer')
    rviz_cfg = os.path.join(pkg, 'rviz', 'frontier_exploration.rviz')

    # =========================================================================
    # Launch arguments
    # =========================================================================
    args = [

        # ── Odometry ─────────────────────────────────────────────────────────
        DeclareLaunchArgument(
            'odom_topic', default_value='/amr/ekf/odom',
            description='EKF odometry topic (nav_msgs/Odometry)'
        ),

        # ── TF / frame names ─────────────────────────────────────────────────
        DeclareLaunchArgument(
            'world_frame', default_value='world',
            description='Root TF frame — world (alignment_node publishes world→odom)'
        ),
        DeclareLaunchArgument(
            'camera_frame', default_value='camera_color_optical_frame',
            description='D435i optical TF frame'
        ),

        # ── Map topic ────────────────────────────────────────────────────────
        DeclareLaunchArgument(
            'slam_map_topic', default_value='/map',
            description='SLAM map topic — slam_toolbox publishes /map by default'
        ),

        # ── D435i topics ─────────────────────────────────────────────────────
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/camera/color/image_raw',
            description='D435i RGB image topic'
        ),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value='/camera/camera/color/camera_info',
            description='D435i CameraInfo topic'
        ),

        # ── ArUco ────────────────────────────────────────────────────────────
        DeclareLaunchArgument(
            'target_marker_id', default_value='0',
            description='ArUco marker ID the robot should home toward'
        ),
        DeclareLaunchArgument(
            'marker_size_m', default_value='0.13',
            description='Physical side length of the ArUco marker in metres'
        ),
        DeclareLaunchArgument(
            'aruco_dict', default_value='DICT_4X4_50',
            description='OpenCV ArUco dictionary'
        ),
        DeclareLaunchArgument(
            'min_detection_area', default_value='200',
            description='Minimum marker corner area in px² — filters noise'
        ),

        # ── Frontier explorer ─────────────────────────────────────────────────
        DeclareLaunchArgument(
            'w_dist', default_value='0.85',
            description='Proximity weight (perceived-distance scoring)'
        ),
        DeclareLaunchArgument(
            'w_size', default_value='0.15',
            description='Cluster area weight'
        ),
        DeclareLaunchArgument(
            'w_heading', default_value='0.40',
            description='Kept for compatibility — not used (baked into perceived_dist)'
        ),
        DeclareLaunchArgument(
            'min_cluster_size', default_value='5',
            description='Minimum frontier cluster size in cells'
        ),
        DeclareLaunchArgument(
            'max_frontier_dist', default_value='8.0',
            description='Ignore frontiers farther than this (m)'
        ),
        DeclareLaunchArgument(
            'frontier_update_rate', default_value='1.0',
            description='Frontier detection loop rate (Hz)'
        ),
        DeclareLaunchArgument(
            'min_goal_dist', default_value='0.40',
            description='Ignore frontier goals closer than this (m)'
        ),
        DeclareLaunchArgument(
            'frontier_goal_reached_dist', default_value='0.20',
            description='Frontier arrival threshold (m)'
        ),
        DeclareLaunchArgument(
            'safe_goal_radius', default_value='0.25',
            description='Safe goal walk radius — must be >= A* inflation_radius'
        ),
        DeclareLaunchArgument(
            'require_camera_coverage', default_value='false',
            description=(
                'Skip frontiers already seen by the camera. '
                'Requires camera pitched downward (camera_pitch < -0.38 rad) '
                'so FOV rays intersect the floor plane. '
                'Keep false for a horizontally-mounted camera.'
            )
        ),

        # ── explorer_controller ───────────────────────────────────────────────
        DeclareLaunchArgument(
            'goal_reached_dist', default_value='0.35',
            description='Distance to ArUco goal that triggers DONE (m)'
        ),
        DeclareLaunchArgument(
            'detection_timeout_sec', default_value='2.0',
            description='Seconds without ArUco detection before returning to EXPLORING'
        ),

        # ── camera_fov_tracker ────────────────────────────────────────────────
        DeclareLaunchArgument(
            'fov_update_rate', default_value='5.0',
            description='Camera FOV tracker update rate (Hz)'
        ),

        # ── Camera mount (base_footprint → camera_link) ──────────────────────
        # Camera is aligned with robot X axis (forward).
        # roll=-π/2, yaw=-π/2 compensates for the RealSense driver's
        # built-in optical frame rotation so the optical Z axis points
        # along the robot's forward (X) direction in the world frame.
        # Verified with: ros2 run tf2_ros tf2_echo world camera_color_optical_frame
        # Expected matrix first column ≈ [1, 0, 0] — optical axis = world X.
        DeclareLaunchArgument(
            'camera_x', default_value='0.1',
            description='Camera X offset from base_footprint (m, forward)'
        ),
        DeclareLaunchArgument(
            'camera_y', default_value='0.0',
            description='Camera Y offset from base_footprint (m, left)'
        ),
        DeclareLaunchArgument(
            'camera_z', default_value='0.2',
            description='Camera Z offset from base_footprint (m, up)'
        ),
        DeclareLaunchArgument(
            'camera_roll',  default_value='-1.5708',
            description='Camera roll  (rad) — -π/2 to compensate RealSense optical frame'
        ),
        DeclareLaunchArgument(
            'camera_pitch', default_value='0.0',
            description='Camera pitch (rad) — add downward tilt if using camera coverage'
        ),
        DeclareLaunchArgument(
            'camera_yaw',   default_value='-1.5708',
            description='Camera yaw   (rad) — -π/2 to compensate RealSense optical frame'
        ),

        # ── Misc ──────────────────────────────────────────────────────────────
        DeclareLaunchArgument(
            'rviz', default_value='true',
            description='Launch RViz (true/false)'
        ),
    ]

    # =========================================================================
    # Nodes
    # =========================================================================

    # ── ArUco detection (camera frame) ───────────────────────────────────────
    aruco_goal_detector = Node(
        package='frontier_explorer',
        executable='aruco_goal_detector',
        name='aruco_goal_detector',
        output='screen',
        parameters=[{
            'marker_size_m':       LaunchConfiguration('marker_size_m'),
            'camera_frame':        LaunchConfiguration('camera_frame'),
            'image_topic':         LaunchConfiguration('image_topic'),
            'camera_info_topic':   LaunchConfiguration('camera_info_topic'),
            'aruco_dict':          LaunchConfiguration('aruco_dict'),
            'min_detection_area':  LaunchConfiguration('min_detection_area'),
            'publish_debug_image': True,
        }]
    )

    # ── ArUco pose: camera frame → world frame ────────────────────────────────
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

    # ── Camera FOV tracker ────────────────────────────────────────────────────
    camera_fov_tracker = Node(
        package='frontier_explorer',
        executable='camera_fov_tracker',
        name='camera_fov_tracker',
        output='screen',
        parameters=[{
            'camera_frame':      LaunchConfiguration('camera_frame'),
            'world_frame':       LaunchConfiguration('world_frame'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'map_topic':         LaunchConfiguration('slam_map_topic'),
            'odom_topic':        LaunchConfiguration('odom_topic'),
            'fov_map_topic':     '/camera/fov_map',
            'fov_marker_topic':  '/camera/fov_marker',
            'update_rate_hz':    LaunchConfiguration('fov_update_rate'),
        }]
    )

    # ── Frontier explorer ─────────────────────────────────────────────────────
    frontier_explorer = Node(
        package='frontier_explorer',
        executable='frontier_explorer',
        name='frontier_explorer',
        output='screen',
        parameters=[{
            'map_topic':               LaunchConfiguration('slam_map_topic'),
            'pose_topic':              '/follower/pose',   # unused — odom_topic is set
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

    # ── Mission controller ────────────────────────────────────────────────────
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

    # ── A* planner + spline follower ──────────────────────────────────────────
    # NOT launched here — owned by trajectory_planner/planner_launch.py which
    # must be running before this launch. This stack feeds the existing planner
    # via /aruco/goal/pose only.

    # ── Static TF: base_footprint → camera_link ─────────────────────────────
    # Connects the D435i's internal TF tree (camera_link and its children)
    # to the robot body frame. Adjust camera_x/y/z/roll/pitch/yaw at launch
    # to match your physical mounting position.
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

    # ── RViz ─────────────────────────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_cfg],
        condition=IfCondition(LaunchConfiguration('rviz'))
    )

    # Delay camera_fov_tracker by 3 s so the TRANSIENT_LOCAL /map publisher
    # is fully discovered by DDS before the subscriber connects.
    # Without this delay the TRANSIENT_LOCAL replay race on ROS 2 Humble
    # causes the node to miss the latched map message at startup.
    camera_fov_tracker_delayed = TimerAction(
        period=3.0,
        actions=[camera_fov_tracker]
    )

    return LaunchDescription(args + [
        camera_tf,           # must be first — other nodes need this TF
        aruco_goal_detector,
        aruco_world_bridge,
        camera_fov_tracker_delayed,
        frontier_explorer,
        explorer_controller,
        rviz_node,
    ])