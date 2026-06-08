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
        DeclareLaunchArgument('scenario',          default_value='room2',
            description='Map scenario: room | room2 | corridor'),
        DeclareLaunchArgument('map_resolution',    default_value='0.05',
            description='Map resolution in m/cell'),
        DeclareLaunchArgument('map_width_m',       default_value='12.0',
            description='Map width in metres'),
        DeclareLaunchArgument('map_height_m',      default_value='12.0',
            description='Map height in metres'),
        DeclareLaunchArgument('sensor_range_m',    default_value='4.0',
            description='Simulated SLAM sensor reveal range in metres'),
        DeclareLaunchArgument('publish_rate',      default_value='2.0',
            description='SLAM map publish rate in Hz'),

        DeclareLaunchArgument('target_marker_id',  default_value='0',
            description='ArUco marker ID that triggers HOMING'),
        DeclareLaunchArgument('goal_reached_dist', default_value='0.35',
            description='Distance to ArUco goal that triggers DONE (m)'),

        DeclareLaunchArgument('w_dist',            default_value='0.7',
            description='Frontier scoring weight for proximity'),
        DeclareLaunchArgument('w_size',            default_value='0.3',
            description='Frontier scoring weight for cluster area'),
        DeclareLaunchArgument('min_cluster_size',  default_value='5',
            description='Minimum frontier cluster size in cells'),
        DeclareLaunchArgument('max_frontier_dist', default_value='8.0',
            description='Maximum frontier distance in metres'),
        DeclareLaunchArgument('update_rate',       default_value='1.0',
            description='Frontier explorer update rate in Hz'),

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
        }]
    )

    frontier_explorer = Node(
        package='frontier_explorer',
        executable='frontier_explorer',
        name='frontier_explorer',
        output='screen',
        parameters=[{
            'map_topic':            '/slam/map',   # ← partial SLAM map
            'pose_topic':           '/follower/pose',
            'goal_topic':           '/frontier/goal',
            'world_frame':          LaunchConfiguration('world_frame'),
            'w_dist':               LaunchConfiguration('w_dist'),
            'w_size':               LaunchConfiguration('w_size'),
            'min_cluster_size':     LaunchConfiguration('min_cluster_size'),
            'max_frontier_dist':    LaunchConfiguration('max_frontier_dist'),
            'update_rate':          LaunchConfiguration('update_rate'),
            'active':               False,          # starts silent, controller activates it
            'odom_topic':           LaunchConfiguration('odom_topic'), 
            'safe_goal_radius': 0.45,   # slightly larger than astar's inflation+robot radius
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
    )

    pose_to_tf = Node(
        package='frontier_explorer',
        executable='pose_to_tf',
        name='pose_to_tf',
        output='screen',
    )

    # aruco_detector = Node(
    #     package='frontier_explorer',
    #     executable='aruco_detector',
    #     name='aruco_detector',
    #     output='screen',
    #     parameters=[{
    #         'world_frame':          LaunchConfiguration('world_frame'),
    #         'robot_base_frame':     LaunchConfiguration('robot_base_frame'),
    #     }]
    # )

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
        # aruco_detector,
        odom_to_pose,
        pose_to_tf,
        rviz_node,
    ])