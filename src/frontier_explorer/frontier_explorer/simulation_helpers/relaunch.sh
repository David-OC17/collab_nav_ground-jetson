#!/bin/bash
# relaunch.sh
echo "Killing all nodes from previous run..."
pkill -f fake_map_publisher
pkill -f frontier_explorer
pkill -f explorer_controller
pkill -f astar_planner2
pkill -f spline_follower
pkill -f fake_aruco_detector
pkill -f odom_to_pose
pkill -f pose_to_tf
sleep 1

echo "Flushing ROS 2 daemon..."
ros2 daemon stop
sleep 1
ros2 daemon start
sleep 1

echo "Launching..."
ros2 launch frontier_explorer explore_sim_launch.py "$@"