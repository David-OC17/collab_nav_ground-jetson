#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# frontier_explorer_test.sh
# Full bringup for frontier exploration fallback test
#
# Opens Terminator with a vertical split:
#   LEFT  — full bringup sequence (steps 1-8)
#   RIGHT — trajectory planner
#
# Usage:
#   ./frontier_explorer_test.sh
#   ./frontier_explorer_test.sh x0:=3.0 y0:=0.8 theta0:=1.5708 rasp_ip:=10.42.0.50
# ─────────────────────────────────────────────────────────────────────────────

# ── Configuration ─────────────────────────────────────────────────────────────
RASP_USER="root"
RASP_IP="10.42.0.50"
RASP_PASS="root"
ROS_DOMAIN_ID=113

X0=0.0
Y0=0.0
THETA0=0.0   # radians — 90°

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

# ── If we are the LAUNCHER — write Terminator config and open the window ──────
if [ -z "$BRINGUP_PANE" ]; then

    TERM_CONF=$(mktemp /tmp/terminator_bringup_XXXX.conf)
    SCRIPT_PATH=$(realpath "$0")
    ARGS="$*"

    # Shared env exported into each pane
    ENV_EXPORTS="export X0=$X0; export Y0=$Y0; export THETA0=$THETA0; \
export OZ=$OZ; export OW=$OW; export RASP_IP=$RASP_IP; \
export RASP_USER=$RASP_USER; export RASP_PASS=$RASP_PASS; \
export ROS_DOMAIN_ID=$ROS_DOMAIN_ID; export JETSON_WS=$JETSON_WS; \
export RASP_WS=$RASP_WS"

    # Source ROS inside the pane shell BEFORE re-running this script.
    # Terminator opens a fresh shell so sourcing at the top of the script
    # is not enough — it must happen in the same shell that runs ros2.
    ROS_SOURCE="source /opt/ros/humble/setup.bash; \
source $JETSON_WS/install/setup.bash; \
export ROS_DOMAIN_ID=$ROS_DOMAIN_ID; \
export ROS_LOCALHOST_ONLY=0"

    CMD_BRINGUP="bash -c '$ROS_SOURCE; $ENV_EXPORTS; export BRINGUP_PANE=bringup; bash $SCRIPT_PATH $ARGS; exec bash'"
    CMD_PLANNER="bash -c '$ROS_SOURCE; $ENV_EXPORTS; export BRINGUP_PANE=planner; bash $SCRIPT_PATH $ARGS; exec bash'"

    # ── Terminator layout: simple HPaned (vertical split) ─────────────────────
    #   root → HPaned → term_bringup (left) | term_planner (right)
    cat > "$TERM_CONF" << EOF
[global_config]
  title_transmit_bg_color = "#d30102"

[keybindings]

[profiles]
  [[default]]
    scrollback_lines = 5000

[layouts]
  [[bringup]]
    [[[root]]]
      type = Window
      parent = ""
      order = 0
      position = 0:0
      size = 1800, 900
    [[[split]]]
      type = HPaned
      parent = root
      ratio = 0.5
    [[[term_bringup]]]
      type = Terminal
      parent = split
      order = 0
      title = "Frontier Explorer Bringup"
      command = $CMD_BRINGUP
    [[[term_planner]]]
      type = Terminal
      parent = split
      order = 1
      title = "Trajectory Planner"
      command = $CMD_PLANNER

[plugins]
EOF

    echo "Opening Terminator — left: bringup | right: trajectory planner"
    terminator -l bringup --config "$TERM_CONF"
    rm -f "$TERM_CONF"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# Inside a pane — $BRINGUP_PANE selects the role
# ─────────────────────────────────────────────────────────────────────────────

# ROS is already sourced — done in the Terminator pane command before
# this script was re-invoked. Nothing to do here.

PIDS=()
cleanup() {
    echo ""
    echo "[bringup] Shutting down..."
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null; done
    sshpass -p "$RASP_PASS" ssh "$RASP_USER@$RASP_IP" \
        "pkill -f 'ros2'" 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# ─────────────────────────────────────────────────────────────────────────────
case "$BRINGUP_PANE" in

# ── LEFT PANE — full bringup sequence ─────────────────────────────────────────
bringup)
    echo "==========================================="
    echo "  Frontier Explorer Test Bringup"
    echo "  Rasp:         $RASP_USER@$RASP_IP"
    echo "  Initial pose: x=$X0  y=$Y0  θ=$THETA0 rad"
    echo "  Quaternion:   z=$OZ  w=$OW"
    echo "==========================================="

    # ── Step 1: Raspberry Pi AMR base ─────────────────────────────────────────
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
    sleep 8

    # Verify EKF is up before continuing
    echo "[bringup] Waiting for odom→base_footprint TF..."
    until ros2 run tf2_ros tf2_echo odom base_footprint 2>/dev/null \
        | grep -q "Translation"; do
        sleep 0.5
    done
    echo "[bringup] ✓ EKF ready — odom→base_footprint confirmed"

    # ── Step 2: Alignment node ─────────────────────────────────────────────────
    echo ""
    echo "[bringup] ── Step 2: Starting alignment node ──"
    ros2 run amr_drone_nav alignment_node &
    PIDS+=($!)
    sleep 2

    # ── Step 3: Publish initial pose ──────────────────────────────────────────
    echo ""
    echo "[bringup] ── Step 3: Publishing initial pose to alignment node ──"
    echo "[bringup] Pose: world frame  x=$X0  y=$Y0  oz=$OZ  ow=$OW"
    POSE_MSG="{header: {frame_id: 'world'}, pose: {pose: {position: {x: $X0, y: $Y0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: $OZ, w: $OW}}}}"

    for i in $(seq 1 30); do
        ros2 topic pub --once /aruco/amr/pose \
            geometry_msgs/msg/PoseWithCovarianceStamped "$POSE_MSG" \
            > /dev/null 2>&1
        sleep 0.5
        if ros2 run tf2_ros tf2_echo world odom 2>/dev/null \
            | grep -q "Translation"; then
            echo "[bringup] ✓ world→odom TF confirmed after $i attempts"
            break
        fi
    done

    if ! ros2 run tf2_ros tf2_echo world odom 2>/dev/null | grep -q "Translation"; then
        echo "[bringup] ✗ FATAL: world→odom never appeared. Check alignment node."
        exit 1
    fi

    # ── Step 4: LiDAR ─────────────────────────────────────────────────────────
    echo ""
    echo "[bringup] ── Step 4: Starting MS200 LiDAR ──"
    ros2 launch oradar_lidar ms200_scan.launch.py &
    PIDS+=($!)
    sleep 3

    # ── Step 5: RealSense ─────────────────────────────────────────────────────
    echo ""
    echo "[bringup] ── Step 5: Starting RealSense D435i ──"
    ros2 launch realsense2_camera rs_launch.py &
    PIDS+=($!)
    sleep 3

    # ── Step 6: Occupancy mapper ───────────────────────────────────────────────
    echo ""
    echo "[bringup] ── Step 6: Starting occupancy mapper ──"
    ros2 run world_mapper occupancy_mapper --ros-args \
        -p origin_x:=-2.0 \
        -p origin_y:=-2.0 \
        -p width_m:=10.0 \
        -p height_m:=10.0 \
        -p map_topic:=/amr/world_map \
        -p world_frame:=world &
    PIDS+=($!)
    echo "[bringup] Occupancy mapper started (PID ${PIDS[-1]})"
    sleep 2

    # ── Step 7: Frontier explorer stack ───────────────────────────────────────
    echo ""
    echo "[bringup] ── Step 7: Starting frontier explorer stack ──"
    ros2 launch frontier_explorer frontier_explorer_launch.py rviz:=false &
    PIDS+=($!)
    echo "[bringup] Frontier explorer started (PID ${PIDS[-1]})"
    sleep 3

    # ── Step 8: TF chain verification + mission start ─────────────────────────
    echo ""
    echo "[bringup] ── Step 8: Verifying TF chain ──"
    check_tf() {
        timeout 5 ros2 run tf2_ros tf2_echo "$1" "$2" 2>/dev/null | \
            grep -q "Translation" && \
            echo "[bringup] ✓ TF $1 → $2" || \
            echo "[bringup] ✗ WARNING: TF $1 → $2 not available"
    }
    check_tf world odom
    check_tf world base_footprint
    check_tf world lidar
    check_tf world camera_color_optical_frame

    echo ""
    echo "==========================================="
    echo "  All nodes launched."
    echo "  Verify the TF chain above is complete."
    echo "  Press ENTER to start the mission."
    echo "==========================================="
    read -r -p "[bringup] Press ENTER to start mission... "

    echo ""
    echo "[bringup] ── Sending /map_fail_fallback/start ──"
    ros2 topic pub /map_fail_fallback/start std_msgs/msg/Bool \
        "{data: true}" --once

    echo "[bringup] Mission started! Watching logs (Ctrl+C to stop all)..."
    wait
    ;;

# ── RIGHT PANE — trajectory planner ───────────────────────────────────────────
planner)
    echo "==========================================="
    echo "  Trajectory Planner"
    echo "  Waiting for bringup to complete..."
    echo "==========================================="
    echo ""
    ros2 launch trajectory_planner planner_launch.py map_topic:=/amr/world_map
    ;;

*)
    echo "Unknown pane role: $BRINGUP_PANE"
    exit 1
    ;;

esac