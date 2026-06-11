#!/bin/bash
# ─────────────────────────────────────────────
#  AMR hardware-test bringup (single terminal)
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
ROS_DOMAIN_ID=113
JETSON_WS=~/collab_nav_ground-jetson

HW_TEST="$JETSON_WS/src/mission_orchestrator/scripts/run_hw_test_amr_nav.py"

RASP_USER="root"
RASP_IP="10.42.0.50"
RASP_PASS="root"
RASP_WS="~/collab_nav_ground-rasp"

SCAN_ID=20

### START POSITION + YAW ###
START_X=0.31
START_Y=0.31
START_YAW=90.0          # radians

### GOAL POSITION + YAW ###
GOAL_X=3.51
GOAL_Y=3.51
GOAL_YAW=0.0           # radians

GOAL_TOLERANCE=0.15

WORLD_ODOM_PKG=amr_drone_nav
WORLD_ODOM_EXE=world_odom_tf_node

LAUNCH_OPTITRACK=false

echo "==========================================="
echo "  AMR HW-test bringup"
echo "  scan-id=$SCAN_ID"
echo "  start=($START_X, $START_Y, yaw=$START_YAW rad)"
echo "  goal =($GOAL_X,  $GOAL_Y,  yaw=$GOAL_YAW rad)"
echo "  goal-tolerance=$GOAL_TOLERANCE m"
echo "  optitrack nodes: $LAUNCH_OPTITRACK"
echo "==========================================="

# ─────────────────────────────────────────────
# Emergency stop — publish zero cmd_vel on the Pi
# ─────────────────────────────────────────────
estop() {
    echo "[bringup] Sending emergency stop (zero cmd_vel) to Pi..."
    sshpass -p "$RASP_PASS" ssh "$RASP_USER@$RASP_IP" \
        "source /opt/ros/foxy/setup.bash && \
         export ROS_DOMAIN_ID=$ROS_DOMAIN_ID && \
         export ROS_LOCALHOST_ONLY=0 && \
         ros2 topic pub --times 5 /amr/cmd_vel geometry_msgs/msg/Twist \
             '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'" \
        2>/dev/null
    echo "[bringup] Emergency stop sent."
}

# ─────────────────────────────────────────────
# Cleanup on Ctrl+C
# ─────────────────────────────────────────────
_CLEANED=""
cleanup() {
    [ -n "$_CLEANED" ] && return
    _CLEANED=1
    echo ""
    echo "[bringup] Ctrl+C caught — stopping car before shutdown..."

    # 1. Kill the hw-test immediately so it stops sending commands
    kill $PID_HW_TEST 2>/dev/null
    sleep 0.5

    # 2. Send zero velocity to the car
    estop

    # 3. Stop local helper nodes
    echo "[bringup] Stopping local helper nodes..."
    kill $PID_WORLD_ODOM $PID_OPTITRACK_CLIENT $PID_OPTITRACK_POSE 2>/dev/null

    echo "[bringup] Done."
}
trap cleanup INT TERM EXIT

# ─────────────────────────────────────────────
# Source local ROS environment (Jetson)
# ─────────────────────────────────────────────
source /opt/ros/humble/setup.bash
source "$JETSON_WS/install/setup.bash"
export ROS_DOMAIN_ID=$ROS_DOMAIN_ID
export ROS_LOCALHOST_ONLY=0


# ─────────────────────────────────────────────
# Kill any leftover ROS processes from previous runs
# ─────────────────────────────────────────────
echo "[bringup] Cleaning up leftover ROS processes..."

# Kill stale ros2 topic pub (the one printing "publisher: beginning loop")
pkill -f "ros2 topic pub" 2>/dev/null

# Clear fastrtps shared memory ports
rm -f /dev/shm/fastrtps_* 2>/dev/null
rm -f /tmp/fastrtps_* 2>/dev/null

# Also clean up on the Pi
sshpass -p "$RASP_PASS" ssh "$RASP_USER@$RASP_IP" \
    "pkill -f 'ros2 topic pub' 2>/dev/null; \
     rm -f /dev/shm/fastrtps_* 2>/dev/null; \
     rm -f /tmp/fastrtps_* 2>/dev/null" \
    2>/dev/null

echo "[bringup] Cleanup done."
sleep 1

# ─────────────────────────────────────────────
# Launch local helper nodes (background)
# ─────────────────────────────────────────────
if [ "$LAUNCH_OPTITRACK" = "true" ]; then
    echo "[bringup] Starting optitrack_client..."
    ros2 run optitrack_client optitrack_client &
    PID_OPTITRACK_CLIENT=$!

    echo "[bringup] Starting optitrack_pose_node..."
    ros2 run amr_optitrack optitrack_pose_node &
    PID_OPTITRACK_POSE=$!
fi

echo "[bringup] Starting $WORLD_ODOM_EXE..."
ros2 run "$WORLD_ODOM_PKG" "$WORLD_ODOM_EXE" &
PID_WORLD_ODOM=$!

sleep 3

# ─────────────────────────────────────────────
# Launch the hardware-test script in the BACKGROUND
# so the trap can catch Ctrl+C before Python does
# ─────────────────────────────────────────────
echo "[bringup] Starting run_hw_test_amr_nav.py..."
echo "          Press Ctrl+C to stop everything safely."
echo "==========================================="

python3 "$HW_TEST" \
    --scan-id        "$SCAN_ID" \
    --start-x        "$START_X" \
    --start-y        "$START_Y" \
    --start-yaw      "$START_YAW" \
    --goal-x         "$GOAL_X" \
    --goal-y         "$GOAL_Y" \
    --goal-yaw       "$GOAL_YAW" \
    --goal-tolerance "$GOAL_TOLERANCE" \
    "$@" &
PID_HW_TEST=$!

# Wait for the hw-test to finish (or Ctrl+C to fire the trap)
wait $PID_HW_TEST