#!/usr/bin/env python3
"""
Launch file for the frontier exploration fallback stack.

Usage:
  ros2 launch frontier_explorer frontier_exploration.launch.py
  ros2 launch frontier_explorer frontier_exploration.launch.py scenario:=room2
  ros2 launch frontier_explorer frontier_exploration.launch.py rviz:=false

To trigger fallback exploration (once running):
  ros2 topic pub /mission/start std_msgs/msg/Bool "{data: true}" --once
"""

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess


def generate_launch_description():

    pkg        = get_package_share_directory('frontier_explorer')
    rviz_cfg   = os.path.join(pkg, 'rviz', 'frontier_exploration.rviz')

    # ------------------------------------------------------------------
    # Launch arguments
    # ------------------------------------------------------------------
    args = [
        DeclareLaunchArgument('scenario',          default_value='room3',
            description='Map scenario: room1 | room2 | room3'),
        DeclareLaunchArgument('map_resolution',    default_value='0.05',
            description='Map resolution in m/cell'),
        DeclareLaunchArgument('map_width_m',       default_value='4.0',
            description='Map width in metres'),
        DeclareLaunchArgument('map_height_m',      default_value='4.0',
            description='Map height in metres'),
        DeclareLaunchArgument('sensor_range_m',    default_value='1.0',
            description='Simulated SLAM sensor reveal range in metres'),
        DeclareLaunchArgument('publish_rate',      default_value='2.0',
            description='SLAM map publish rate in Hz'),

        DeclareLaunchArgument('target_marker_id',  default_value='0',
            description='ArUco marker ID that triggers HOMING'),
        DeclareLaunchArgument('goal_reached_dist', default_value='0.35',
            description='Distance to ArUco goal that triggers DONE (m)'),

        DeclareLaunchArgument('w_dist',            default_value='0.85',
            description='Frontier scoring weight for proximity'),
        DeclareLaunchArgument('w_size',            default_value='0.15',
            description='Frontier scoring weight for cluster area'),
        DeclareLaunchArgument('w_heading',         default_value='0.40',
            description='Frontier scoring weight for heading changes'),
        DeclareLaunchArgument('min_cluster_size',  default_value='5',
            description='Minimum frontier cluster size in cells'),
        DeclareLaunchArgument('max_frontier_dist', default_value='8.0',
            description='Maximum frontier distance in metres'),
        DeclareLaunchArgument('update_rate',       default_value='1.0',
            description='Frontier explorer update rate in Hz'),

        DeclareLaunchArgument('min_goal_dist',    default_value='0.40',
            description='Ignore frontier goals closer than this (m)'),
        DeclareLaunchArgument('goal_reached_dist', default_value='0.12',
            description='Distance at which robot is considered to have reached frontier (m)'),

        DeclareLaunchArgument('max_speed',         default_value='0.30',
            description='Spline follower max speed in m/s'),
        DeclareLaunchArgument('max_accel',         default_value='0.20',
            description='Spline follower max acceleration in m/s²'),

        DeclareLaunchArgument('world_frame',       default_value='map',
            description='World / map TF frame'),
        DeclareLaunchArgument('robot_base_frame',  default_value='base_footprint',
            description='Robot base TF frame'),

        DeclareLaunchArgument('rviz',              default_value='true',
            description='Launch RViz (true/false)'),
        DeclareLaunchArgument('odom_topic', default_value='',
            description='EKF odometry topic for real robot (empty = simulation mode)'),

        DeclareLaunchArgument('robot_start_x', default_value='1.7'),
        DeclareLaunchArgument('robot_start_y', default_value='1.7'),

        DeclareLaunchArgument('aruco_marker_x', default_value='-1.7',
            description='Simulated ArUco marker world X position (m)'),
        DeclareLaunchArgument('aruco_marker_y', default_value='-1.7',
            description='Simulated ArUco marker world Y position (m)'),
    ]

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    fake_map_publisher = Node(
        package='frontier_explorer',
        executable='fake_map_publisher',
        name='fake_map_publisher',
        output='screen',
        parameters=[{
            'scenario':             LaunchConfiguration('scenario'),
            'map_resolution':       LaunchConfiguration('map_resolution'),
            'map_width_m':          LaunchConfiguration('map_width_m'),
            'map_height_m':         LaunchConfiguration('map_height_m'),
            'sensor_range_m':       LaunchConfiguration('sensor_range_m'),
            'publish_rate':         LaunchConfiguration('publish_rate'),
            'max_camera_range_m':   1.5,   # camera sees ~1.5 m ahead, not entire room
            'camera_hfov_deg':      69.4,
            'camera_near_m':        0.15,
            'robot_start_x':        LaunchConfiguration('robot_start_x'),
            'robot_start_y':        LaunchConfiguration('robot_start_y'),
        }]
    )

    frontier_explorer = Node(
        package='frontier_explorer',
        executable='frontier_explorer',
        name='frontier_explorer',
        output='screen',
        parameters=[{
            'map_topic':                '/slam/map',
            'pose_topic':               '/follower/pose',
            'goal_topic':               '/frontier/goal',
            'world_frame':              LaunchConfiguration('world_frame'),
            'w_dist':                   LaunchConfiguration('w_dist'),
            'w_size':                   LaunchConfiguration('w_size'),
            'w_heading':                LaunchConfiguration('w_heading'),
            'min_cluster_size':         LaunchConfiguration('min_cluster_size'),
            'max_frontier_dist':        LaunchConfiguration('max_frontier_dist'),
            'update_rate':              LaunchConfiguration('update_rate'),
            'active':                   False,
            'odom_topic':               LaunchConfiguration('odom_topic'),
            'min_goal_dist':            LaunchConfiguration('min_goal_dist'),
            'goal_reached_dist':        LaunchConfiguration('goal_reached_dist'),
            'safe_goal_radius':         0.20,   # tighter for 4×4 m map
            'require_camera_coverage':  False,
            'fov_map_topic':            '/camera/fov_map',
        }]
    )

    explorer_controller = Node(
        package='frontier_explorer',
        executable='explorer_controller',
        name='explorer_controller',
        output='screen',
        parameters=[{
            'target_marker_id':     LaunchConfiguration('target_marker_id'),
            'goal_reached_dist':    LaunchConfiguration('goal_reached_dist'),
            'world_frame':          LaunchConfiguration('world_frame'),
        }]
    )

    astar_planner = Node(
        package='trajectory_planner',
        executable='astar_planner2',
        name='astar_planner2',
        output='screen',
        parameters=[{
            'map_topic':            '/drone/map',   # ← full map for safe planning
            'goal_topic':           '/aruco/goal/pose',
            'world_frame':          'map',
            'robot_base_frame':     'base_footprint',
        }]
    )

    spline_follower = Node(
        package='trajectory_planner',
        executable='spline_follower',
        name='spline_follower',
        output='screen',
        parameters=[{
            'world_frame':          'map',
            'robot_base_frame':     'base_footprint',
            'max_speed':            LaunchConfiguration('max_speed'),
            'max_accel':            LaunchConfiguration('max_accel'),
        }]
    )

    odom_to_pose = Node(
        package='frontier_explorer',
        executable='odom_to_pose',
        name='odom_to_pose',
        output='screen',
        parameters=[{
            'robot_start_x': LaunchConfiguration('robot_start_x'),
            'robot_start_y': LaunchConfiguration('robot_start_y'),
        }]
    )

    pose_to_tf = Node(
        package='frontier_explorer',
        executable='pose_to_tf',
        name='pose_to_tf',
        output='screen',
    )

    fake_aruco_detector = Node(
    package='frontier_explorer',
    executable='fake_aruco_detector',
    name='fake_aruco_detector',
    output='screen',
    parameters=[{
        'marker_x':        LaunchConfiguration('aruco_marker_x'),
        'marker_y':        LaunchConfiguration('aruco_marker_y'),
        'marker_id':       LaunchConfiguration('target_marker_id'),
        'camera_hfov_deg': 69.4,    # must match fake_map_publisher
        'camera_range_m':  4.0,     # must match fake_map_publisher
        'camera_near_m':   0.15,    # must match fake_map_publisher
        'publish_rate':    10.0,
        'world_frame':     LaunchConfiguration('world_frame'),
    }]
)

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_cfg],
        condition=IfCondition(LaunchConfiguration('rviz'))
    )

    return LaunchDescription(args + [
        fake_map_publisher,
        frontier_explorer,
        explorer_controller,
        astar_planner,
        spline_follower,
        fake_aruco_detector,
        # camera_fov_tracker — not used in sim; fake_map_publisher publishes
        # /camera/fov_map directly using the robot pose + yaw wedge geometry.
        # Re-enable for real robot deployment.
        odom_to_pose,
        pose_to_tf,
        rviz_node,
    ])