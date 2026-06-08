#!/usr/bin/env bash
#
# start_vslam.sh — bring up Isaac ROS Visual SLAM (cuVSLAM) for the RealSense.
#
# Owns the container internals so the orchestrator (stage 05) only has to run
# this script and then verify /visual_slam/tracking/odometry from outside.
#
# Steps:
#   1. docker start <container>                  (05.b — start container)
#   2. docker exec -d <container> ... ros2 launch (05.c — launch visual SLAM)
#
# The launch is run DETACHED (-d) inside an INTERACTIVE bash (-i) so the
# container's ~/.bashrc sources ROS + the Isaac ROS workspace before launching.
#
# Override via env vars:
#   VSLAM_CONTAINER   container name           (default: isaac_ros_vslam)
#   VSLAM_LAUNCH      ros2 launch command      (default: the realsense launch)
set -euo pipefail

CONTAINER="${VSLAM_CONTAINER:-isaac_ros_vslam}"
LAUNCH="${VSLAM_LAUNCH:-ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_realsense.launch.py}"

echo "[start_vslam] starting container '${CONTAINER}' …"
docker start "${CONTAINER}"

echo "[start_vslam] launching Visual SLAM inside '${CONTAINER}' (detached) …"
docker exec -d "${CONTAINER}" bash -ic "${LAUNCH}"

echo "[start_vslam] done — container up, VSLAM launching."
