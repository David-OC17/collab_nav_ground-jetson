#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── AStarPlanner2 arguments ────────────────────────────────────────────
    args_astar = [
        DeclareLaunchArgument('map_topic',  default_value='/drone/map',
                              description='OccupancyGrid input topic'),
        DeclareLaunchArgument('goal_topic', default_value='/aruco/goal/pose',
                              description='Goal PoseWithCovarianceStamped topic'),

        DeclareLaunchArgument('inflation_radius',         default_value='0.20',
                              description='Obstacle inflation radius [m]'),
        DeclareLaunchArgument('robot_radius',             default_value='0.20',
                              description='Inscribed robot radius [m]'),
        DeclareLaunchArgument('cost_scaling',             default_value='3.5',
                              description='Inflation exponential decay factor'),

        DeclareLaunchArgument('goal_change_threshold',    default_value='0.30',
                              description='Min goal displacement to replan [m]'),
        DeclareLaunchArgument('collision_cost_threshold', default_value='80.0',
                              description='Inflated cost above which path is blocked'),
        DeclareLaunchArgument('global_change_threshold',  default_value='0.05',
                              description='Ratio of new obstacles that triggers replan [0-1]'),
        DeclareLaunchArgument('path_proximity_threshold', default_value='5.0',
                              description='Proximity-weighted obstacle score threshold'),
        DeclareLaunchArgument('path_proximity_radius',    default_value='2.0',
                              description='Decay radius for proximity weighting [m]'),
        DeclareLaunchArgument('min_replan_interval_sec',  default_value='0.5',
                              description='Minimum time between replans [s]'),
    ]

    # ── SplineFollower arguments ───────────────────────────────────────────
    args_spline = [
        DeclareLaunchArgument('max_speed',      default_value='0.30',
                              description='Maximum cruise speed [m/s]'),
        DeclareLaunchArgument('max_accel',      default_value='0.20',
                              description='Maximum acceleration/deceleration [m/s²]'),
        DeclareLaunchArgument('goal_tolerance', default_value='0.05',
                              description='Distance to goal considered reached [m]'),
        DeclareLaunchArgument('update_rate',    default_value='50.0',
                              description='Reference publish rate [Hz]'),
    ]

    # ── astar_planner2 ────────────────────────────────────────────────────
    astar_node = Node(
        package='trajectory_planner',
        executable='astar_planner2',
        name='astar_planner2',
        output='screen',
        parameters=[{
            'map_topic':                LaunchConfiguration('map_topic'),
            'goal_topic':               LaunchConfiguration('goal_topic'),
            'robot_base_frame':         'base_footprint',
            'inflation_radius':         LaunchConfiguration('inflation_radius'),
            'robot_radius':             LaunchConfiguration('robot_radius'),
            'cost_scaling':             LaunchConfiguration('cost_scaling'),
            'goal_change_threshold':    LaunchConfiguration('goal_change_threshold'),
            'collision_cost_threshold': LaunchConfiguration('collision_cost_threshold'),
            'global_change_threshold':  LaunchConfiguration('global_change_threshold'),
            'path_proximity_threshold': LaunchConfiguration('path_proximity_threshold'),
            'path_proximity_radius':    LaunchConfiguration('path_proximity_radius'),
            'min_replan_interval_sec':  LaunchConfiguration('min_replan_interval_sec'),
        }],
    )

    # ── spline_follower ───────────────────────────────────────────────────
    spline_node = Node(
        package='trajectory_planner',
        executable='spline_follower',
        name='spline_follower',
        output='screen',
        parameters=[{
            'path_topic':       '/trajectory_planner2/path',
            'robot_base_frame': 'base_footprint',
            'max_speed':        LaunchConfiguration('max_speed'),
            'max_accel':        LaunchConfiguration('max_accel'),
            'goal_tolerance':   LaunchConfiguration('goal_tolerance'),
            'update_rate':      LaunchConfiguration('update_rate'),
        }],
    )

    return LaunchDescription(
        args_astar + args_spline + [astar_node, spline_node]
    )
