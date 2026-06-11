#!/bin/bash

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
RASP_USER="root"
RASP_IP="10.42.0.50"
RASP_PASS="root"
ROS_DOMAIN_ID=113
X0=0.0
Y0=0.0
THETA0=0.0

RASP_WS="~/collab_nav_ground-rasp"
JETSON_WS="~/collab_nav_ground-jetson"

# ─────────────────────────────────────────────
# Parse optional arguments
# ─────────────────────────────────────────────
for arg in "$@"; do
    case $arg in
        x0:=*)      X0="${arg#*:=}" ;;
        y0:=*)      Y0="${arg#*:=}" ;;
        theta0:=*)  THETA0="${arg#*:=}" ;;
        rasp_ip:=*) RASP_IP="${arg#*:=}" ;;
    esac
done

echo "==========================================="
echo "  AMR Full Bringup"
echo "  Rasp: $RASP_USER@$RASP_IP"
echo "  Initial pose: x=$X0 y=$Y0 θ=$THETA0"
echo "==========================================="

# ─────────────────────────────────────────────
# Cleanup on Ctrl+C — kill everything
# ─────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[bringup] Shutting down..."
    kill $PID_OPTITRACK_CLIENT $PID_OPTITRACK_POSE 2>/dev/null
    sshpass -p "$RASP_PASS" ssh "$RASP_USER@$RASP_IP" "pkill -f 'ros2'" 2>/dev/null
    echo "[bringup] Done."
    exit 0
}
trap cleanup SIGINT SIGTERM

# ─────────────────────────────────────────────
# Source local ROS environment
# ─────────────────────────────────────────────
source /opt/ros/humble/setup.bash
source "$JETSON_WS/install/setup.bash"
export ROS_DOMAIN_ID=$ROS_DOMAIN_ID
export ROS_LOCALHOST_ONLY=0

# ─────────────────────────────────────────────
# Launch local nodes on Jetson
# ─────────────────────────────────────────────
echo "[bringup] Starting optitrack_client..."
ros2 run optitrack_client optitrack_client &
PID_OPTITRACK_CLIENT=$!

echo "[bringup] Starting optitrack_pose_node..."
ros2 run amr_optitrack optitrack_pose_node &
PID_OPTITRACK_POSE=$!

sleep 3

# ─────────────────────────────────────────────
# Launch nodes on Raspberry Pi (background SSH)
# ─────────────────────────────────────────────
echo "[bringup] Starting launch file on Raspberry Pi..."
sshpass -p "$RASP_PASS" ssh -tt "$RASP_USER@$RASP_IP" \
    "source /opt/ros/foxy/setup.bash && \
     source $RASP_WS/install/setup.bash && \
     export ROS_DOMAIN_ID=$ROS_DOMAIN_ID && \
     export ROS_LOCALHOST_ONLY=0 && \
     ros2 launch amr_bringup rasp_launch.py \
         x0:=$X0 y0:=$Y0 theta0:=$THETA0" &
PID_RASP=$!

echo "[bringup] Starting ds3_teleop_node on Raspberry Pi..."
sshpass -p "$RASP_PASS" ssh -tt "$RASP_USER@$RASP_IP" \
    "source /opt/ros/foxy/setup.bash && \
     source $RASP_WS/install/setup.bash && \
     export ROS_DOMAIN_ID=$ROS_DOMAIN_ID && \
     export ROS_LOCALHOST_ONLY=0 && \
     ros2 run amr_bringup ds3_teleop_node" &
PID_TELEOP=$!

sleep 1

# ─────────────────────────────────────────────
# Read one message from /optitrack/rigid_body
# and publish it as initial pose to /aruco/amr/pose
# ─────────────────────────────────────────────
echo "[bringup] Waiting for /optitrack/rigid_body to get initial pose..."

POSE_JSON=$(ros2 topic echo --once /optitrack/rigid_body geometry_msgs/msg/PoseStamped 2>/dev/null)

if [ -z "$POSE_JSON" ]; then
    echo "[bringup] WARNING: No message received from /optitrack/rigid_body — skipping initial pose publish."
else
    # Parse fields with Python (already available in any ROS install)
    read PX PY PZ OX OY OZ OW <<< $(python3 - <<EOF
import sys, re

data = """$POSE_JSON"""

def get(field):
    m = re.search(rf'{field}:\s*([-\d.e+]+)', data)
    return m.group(1) if m else '0.0'

print(
    get('x'), get('y'), get('z'),
    get('x'), get('y'), get('z'), get('w')
)
EOF
    )

    # Re-parse properly separating position and orientation
    read PX PY PZ OX OY OZ OW <<< $(python3 - <<'EOF'
import sys, re

data = """$(ros2 topic echo --once /optitrack/rigid_body geometry_msgs/msg/PoseStamped 2>/dev/null)"""

lines = data.split('\n')
section = ''
pos = {}
ori = {}
for line in lines:
    if 'position:' in line:
        section = 'pos'
    elif 'orientation:' in line:
        section = 'ori'
    m = re.match(r'\s+([xyzw]):\s*([-\d.e+]+)', line)
    if m:
        if section == 'pos':
            pos[m.group(1)] = m.group(2)
        elif section == 'ori':
            ori[m.group(1)] = m.group(2)

print(
    pos.get('x','0'), pos.get('y','0'), pos.get('z','0'),
    ori.get('x','0'), ori.get('y','0'), ori.get('z','0'), ori.get('w','1')
)
EOF
    )

    echo "[bringup] Initial pose from OptiTrack → x=$PX y=$PY z=$PZ | ox=$OX oy=$OY oz=$OZ ow=$OW"

    ros2 topic pub --once /aruco/amr/pose geometry_msgs/msg/PoseWithCovarianceStamped \
        "{header: {}, pose: {pose: {position: {x: $PX, y: $PY, z: $PZ}, orientation: {x: $OX, y: $OY, z: $OZ, w: $OW}}}}"

    echo "[bringup] Initial pose published to /aruco/amr/pose"
fi

echo "[bringup] All nodes launched. Press Ctrl+C to stop."
echo "  PIDs → rasp_ssh=$PID_RASP  teleop=$PID_TELEOP  optitrack_client=$PID_OPTITRACK_CLIENT  optitrack_pose=$PID_OPTITRACK_POSE"

wait