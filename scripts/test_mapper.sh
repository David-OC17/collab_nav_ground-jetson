#!/bin/bash
# test_world_mapper.sh
# Run from the Jetson. Opens a tmux session with all nodes needed to test world_mapper.
# SSHes into the Raspberry Pi for amr_bringup and teleop.
#
# Usage: ./test_world_mapper.sh <rasp_password>

# ── Args ──────────────────────────────────────────────────────────────────────
if [ -z "$1" ]; then
    echo "Usage: $0 <rasp_password>"
    exit 1
fi
RASP_PASS="$1"

# ── Config ────────────────────────────────────────────────────────────────────
RASP_USER="root"
RASP_IP="10.42.0.50"          # ← fill in your Rasp's static IP
SSH="sshpass -p $RASP_PASS ssh -t $RASP_USER@$RASP_IP"

# Jetson runs Humble, Rasp runs Foxy
JETSON_ROS_SETUP="/opt/ros/humble/setup.bash"
RASP_ROS_SETUP="/opt/ros/foxy/setup.bash"
JETSON_WS="$HOME/collab_nav_ground-jetson"
RASP_WS="/root/collab_nav_ground-rasp"

# Initial pose of the AMR in the world frame (from OptiTrack/ArUco at startup)
INIT_X=3.0
INIT_Y=0.8
INIT_YAW=1.5708                # radians (~90°)

# Rosbag output directory (on Jetson)
BAG_DIR="$HOME/bags"
BAG_NAME="world_mapper_$(date +%Y%m%d_%H%M%S)"

SESSION="world_mapper_test"
# ─────────────────────────────────────────────────────────────────────────────

JSOURCE="source $JETSON_ROS_SETUP && source $JETSON_WS/install/setup.bash"
RSOURCE="source $RASP_ROS_SETUP && source $RASP_WS/install/setup.bash"

# Kill any existing session with the same name
tmux kill-session -t $SESSION 2>/dev/null

# ── Window 0: LiDAR ──────────────────────────────────────────────────────────
tmux new-session -d -s $SESSION -n "lidar"
tmux send-keys -t $SESSION:0 \
    "cd $JETSON_WS && $JSOURCE && ros2 launch oradar_lidar ms200_scan.launch.py" Enter

# ── Window 1: Static TF world->odom ──────────────────────────────────────────
tmux new-window -t $SESSION -n "static_tf"
tmux send-keys -t $SESSION:1 \
    "sleep 3 && $JSOURCE && \
    ros2 run tf2_ros static_transform_publisher \
        --x $INIT_X --y $INIT_Y --z 0.0 \
        --yaw $INIT_YAW --pitch 0.0 --roll 0.0 \
        --frame-id world --child-frame-id odom" Enter

# ── Window 2: world_mapper ────────────────────────────────────────────────────
tmux new-window -t $SESSION -n "mapper"
tmux send-keys -t $SESSION:2 \
    "sleep 5 && cd $JETSON_WS && $JSOURCE && ros2 launch world_mapper mapper.launch.py" Enter

# ── Window 3: Rasp bringup (SSH) ─────────────────────────────────────────────
tmux new-window -t $SESSION -n "rasp_bringup"
tmux send-keys -t $SESSION:3 \
    "sleep 4 && $SSH 'cd $RASP_WS && $RSOURCE && ros2 launch amr_bringup rasp_launch.py'" Enter

# ── Window 4: Teleop (SSH) ────────────────────────────────────────────────────
tmux new-window -t $SESSION -n "teleop"
tmux send-keys -t $SESSION:4 \
    "sleep 8 && $SSH '$RSOURCE && ros2 run amr_bringup ds3_teleop_node'" Enter

# ── Window 5: TF sanity check ─────────────────────────────────────────────────
tmux new-window -t $SESSION -n "tf_check"
tmux send-keys -t $SESSION:5 \
    "sleep 6 && $JSOURCE && ros2 run tf2_ros tf2_echo world lidar" Enter

# ── Window 6: Rosbag record ───────────────────────────────────────────────────
tmux new-window -t $SESSION -n "rosbag"
tmux send-keys -t $SESSION:6 \
    "sleep 6 && mkdir -p $BAG_DIR && $JSOURCE && \
    ros2 bag record \
        /scan \
        /tf \
        /tf_static \
        /world_map \
        /odom \
        -o $BAG_DIR/$BAG_NAME" Enter

# Focus on lidar window at startup
tmux select-window -t $SESSION:0
tmux attach-session -t $SESSION