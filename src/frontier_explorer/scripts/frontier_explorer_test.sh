#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# frontier_explorer_test.sh
# Full bringup for frontier exploration fallback test
#
# Launch order:
#   1. Raspberry Pi — AMR base controller (via SSH)
#   2. Jetson — LiDAR driver
#   3. Jetson — RealSense D435i driver
#   4. Jetson — alignment_node  (publishes world→odom TF from known pose)
#   5. Jetson — initial pose    (published to /aruco/amr/pose for alignment_node)
#   6. Jetson — occupancy mapper
#   7. Jetson — frontier explorer stack
#   8. Wait for user confirmation, then send mission start
#
# Usage:
#   ./frontier_explorer_test.sh
#   ./frontier_explorer_test.sh x0:=3.0 y0:=0.8 theta0:=1.5708 rasp_ip:=10.42.0.50
#
# Initial pose arguments (x0, y0, theta0) control BOTH:
#   - rasp_launch.py   (odometry origin)
#   - /aruco/amr/pose  (world-frame pose for alignment_node)
# ─────────────────────────────────────────────────────────────────────────────

# ── Configuration ─────────────────────────────────────────────────────────────
RASP_USER="root"
RASP_IP="10.42.0.50"
RASP_PASS="root"
ROS_DOMAIN_ID=113

# Default initial pose — robot starts at (3.0, 0.8) facing 90° (z=0.7071, w=0.7071)
X0=3.0
Y0=0.8
THETA0=1.5708   # radians — 90°

RASP_WS="~/collab_nav_ground-rasp"
JETSON_WS="~/collab_nav_ground-jetson"

# ── Parse optional arguments ───────────────────────────────────────────────────
for arg in "$@"; do
    case $arg in
        x0:=*)      X0="${arg#*:=}" ;;
        y0:=*)      Y0="${arg#*:=}" ;;
        theta0:=*)  THETA0="${arg#*:=}" ;;
        rasp_ip:=*) RASP_IP="${arg#*:=}" ;;
    esac
done

# Compute quaternion z/w from theta (yaw only, 2D robot)
OZ=$(python3 -c "import math; print(math.sin(float('$THETA0')/2))")
OW=$(python3 -c "import math; print(math.cos(float('$THETA0')/2))")

echo "==========================================="
echo "  Frontier Explorer Test Bringup"
echo "  Rasp:         $RASP_USER@$RASP_IP"
echo "  Initial pose: x=$X0  y=$Y0  θ=$THETA0 rad"
echo "  Quaternion:   z=$OZ  w=$OW"
echo "==========================================="

# ── Track all PIDs for clean shutdown ─────────────────────────────────────────
PIDS=()

cleanup() {
    echo ""
    echo "[bringup] Shutting down all nodes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    sshpass -p "$RASP_PASS" ssh "$RASP_USER@$RASP_IP" \
        "pkill -f 'ros2'" 2>/dev/null
    echo "[bringup] Done."
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Source ROS environment ─────────────────────────────────────────────────────
source /opt/ros/humble/setup.bash
source "$JETSON_WS/install/setup.bash"
export ROS_DOMAIN_ID=$ROS_DOMAIN_ID
export ROS_LOCALHOST_ONLY=0

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Raspberry Pi: AMR base controller
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[bringup] ── Step 1: Launching AMR base on Raspberry Pi ──"
sshpass -p "$RASP_PASS" ssh -tt "$RASP_USER@$RASP_IP" \
    "source /opt/ros/foxy/setup.bash && \
     source $RASP_WS/install/setup.bash && \
     export ROS_DOMAIN_ID=$ROS_DOMAIN_ID && \
     export ROS_LOCALHOST_ONLY=0 && \
     ros2 launch amr_bringup rasp_launch.py \
         x0:=$X0 y0:=$Y0 theta0:=$THETA0" &
PIDS+=($!)
echo "[bringup] Raspberry Pi launch started (PID ${PIDS[-1]})"
echo "[bringup] Waiting 5s for Raspberry Pi nodes to initialise..."
sleep 5

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LiDAR driver (MS200)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[bringup] ── Step 2: Starting MS200 LiDAR ──"
ros2 launch oradar_lidar ms200_scan.launch.py &
PIDS+=($!)
echo "[bringup] LiDAR started (PID ${PIDS[-1]})"
sleep 3

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — RealSense D435i driver
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[bringup] ── Step 3: Starting RealSense D435i ──"
ros2 launch realsense2_camera rs_launch.py &
PIDS+=($!)
echo "[bringup] RealSense started (PID ${PIDS[-1]})"
sleep 3

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Alignment node
# Subscribes to /aruco/amr/pose and publishes the world→odom (or world→slam_map)
# TF so all downstream nodes can localise in the world frame.
# Must start BEFORE publishing the initial pose.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[bringup] ── Step 4: Starting alignment node ──"
ros2 run amr_drone_nav alignment_node &
PIDS+=($!)
echo "[bringup] Alignment node started (PID ${PIDS[-1]})"
echo "[bringup] Waiting 2s for alignment_node to come up..."
sleep 2

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Publish initial world-frame pose to alignment node
# alignment_node subscribes to /aruco/amr/pose and uses it to compute
# the world→odom TF offset. Published --once so it only fires at startup.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[bringup] ── Step 5: Publishing initial pose to alignment node ──"
echo "[bringup] Pose: world frame  x=$X0  y=$Y0  oz=$OZ  ow=$OW"
POSE_MSG="{header: {frame_id: 'world'}, pose: {pose: {position: {x: $X0, y: $Y0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: $OZ, w: $OW}}}}"
ros2 topic pub --once /aruco/amr/pose geometry_msgs/msg/PoseWithCovarianceStamped "$POSE_MSG"
echo "[bringup] Initial pose published"
sleep 2

# Verify world→odom TF now exists
echo "[bringup] Verifying world→odom TF from alignment node..."
timeout 5 ros2 run tf2_ros tf2_echo world odom 2>/dev/null | \
    grep -q "Translation" && \
    echo "[bringup] ✓ world→odom TF is live" || \
    echo "[bringup] ✗ WARNING: world→odom TF not yet available — alignment_node may need more time"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Occupancy mapper
# Needs world frame to exist (provided by alignment_node above).
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[bringup] ── Step 6: Starting occupancy mapper ──"
ros2 launch world_mapper mapper.launch.py \
    origin_x:=-1.95 origin_y:=-1.95 &
PIDS+=($!)
echo "[bringup] Occupancy mapper started (PID ${PIDS[-1]})"
sleep 2

echo "[bringup] Checking /amr/world_map is publishing..."
timeout 10 ros2 topic echo /amr/world_map --once --no-arr > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "[bringup] ✓ /amr/world_map is live"
else
    echo "[bringup] ✗ WARNING: /amr/world_map not yet received — check occupancy_mapper logs"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Frontier explorer stack
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[bringup] ── Step 7: Starting frontier explorer stack ──"
ros2 launch frontier_explorer frontier_explorer_launch.py rviz:=false &
PIDS+=($!)
echo "[bringup] Frontier explorer started (PID ${PIDS[-1]})"
sleep 3

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Verify full TF chain
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[bringup] ── Step 8: Verifying TF chain ──"

check_tf() {
    local from=$1
    local to=$2
    timeout 5 ros2 run tf2_ros tf2_echo "$from" "$to" 2>/dev/null | \
        grep -q "Translation" && \
        echo "[bringup] ✓ TF $from → $to" || \
        echo "[bringup] ✗ WARNING: TF $from → $to not available"
}

check_tf world odom
check_tf world base_footprint
check_tf world lidar
check_tf world camera_color_optical_frame

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Wait for operator confirmation, then send mission start
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "==========================================="
echo "  All nodes launched."
echo "  Verify the TF chain above is complete."
echo "  Press ENTER to start the mission."
echo "  (Ctrl+C to abort without starting)"
echo "==========================================="
read -r -p "[bringup] Press ENTER to start mission... "

echo ""
echo "[bringup] ── Sending /map_fail_fallback/start ──"
ros2 topic pub /map_fail_fallback/start std_msgs/msg/Bool \
    "{data: true}" --once

echo "[bringup] Mission started! Watching logs (Ctrl+C to stop all)..."
echo ""

# Keep alive until Ctrl+C
wait