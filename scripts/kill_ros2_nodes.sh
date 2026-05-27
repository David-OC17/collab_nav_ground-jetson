#!/bin/bash
# kill_ros2_nodes.sh
# Gracefully stops then force-kills listed ROS2 nodes on the local machine
# AND on the Raspberry Pi (via SSH), retrying SIGKILL until each process is
# confirmed dead or MAX_RETRIES is exhausted.
#
# Usage:
#   ./kill_ros2_nodes.sh                               # kills all default TARGETS
#   ./kill_ros2_nodes.sh --node orchestrator_node      # kills only that node (local + raspi)
#   ./kill_ros2_nodes.sh --node foo --node bar         # kills foo and bar
#   ./kill_ros2_nodes.sh --no-rasp                     # skip RPi, local only
#   MAX_RETRIES=10 TERM_GRACE_SEC=3 ./kill_ros2_nodes.sh --node foo
#
# Each entry in TARGETS (or --node value) is a pattern passed to `pgrep -f`,
# which matches against the full command line on each machine.
#
# Dependencies: sshpass  (sudo apt install sshpass)

# ── Configuration ──────────────────────────────────────────────────────────────
MAX_RETRIES=${MAX_RETRIES:-5}
TERM_GRACE_SEC=${TERM_GRACE_SEC:-2}
KILL_RETRY_SEC=${KILL_RETRY_SEC:-1}

RASP_HOST=${RASP_HOST:-"10.42.0.50"}
RASP_USER=${RASP_USER:-"root"}
RASP_PASS=${RASP_PASS:-"root"}
RASP_SSH_TIMEOUT=${RASP_SSH_TIMEOUT:-5}

# ── Argument parsing ───────────────────────────────────────────────────────────
CUSTOM_TARGETS=()
SKIP_RASP=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node)
            [[ -z "${2:-}" ]] && { echo "Error: --node requires an argument"; exit 1; }
            CUSTOM_TARGETS+=("$2")
            shift 2
            ;;
        --node=*)
            CUSTOM_TARGETS+=("${1#--node=}")
            shift
            ;;
        --no-rasp)
            SKIP_RASP=true
            shift
            ;;
        *)
            echo "Error: unknown argument '$1'"
            echo "Usage: $0 [--node <pattern>] ... [--no-rasp]"
            exit 1
            ;;
    esac
done

# ── Default target list ────────────────────────────────────────────────────────
# Used only when no --node flags are given.
# One pgrep -f pattern per entry. Comment out targets you don't want to kill.
TARGETS=(
    # ── Mission orchestrator ───────────────────────────────────────────────────
    "orchestrator_node"

    # ── Mapping / SLAM ─────────────────────────────────────────────────────────
    "async_slam_toolbox_node"
    "cartographer_node"
    "cartographer_occupancy_grid_node"

    # ── Perception / Localization ──────────────────────────────────────────────
    "lidar_odometry_node"
    "optitrack_pose_node"
    "ms200_lidar"

    # ── Planning ───────────────────────────────────────────────────────────────
    "astar_planner"
    "trajectory_follower_sim"

    # ── Map / Costmap ──────────────────────────────────────────────────────────
    "map_fusion_node"
    "local_costmap_node"
    "fake_map_publisher"

    # ── Arena services ─────────────────────────────────────────────────────────
    "build_arena_map_server"
    "arena_marker_localizer"

    # ── Tello UAV ──────────────────────────────────────────────────────────────
    "tello_driver_node"
    "tello_controller_node"
    "pos_controller_node"
    "reference_node"
)

if [[ ${#CUSTOM_TARGETS[@]} -gt 0 ]]; then
    TARGETS=("${CUSTOM_TARGETS[@]}")
fi

# ── SSH helpers ────────────────────────────────────────────────────────────────
SELF_PID=$$

remote_exec() {
    sshpass -p "$RASP_PASS" ssh \
        -o ConnectTimeout="$RASP_SSH_TIMEOUT" \
        -o StrictHostKeyChecking=no \
        -o LogLevel=ERROR \
        "${RASP_USER}@${RASP_HOST}" "$@" 2>/dev/null || true
}

check_rasp() {
    if $SKIP_RASP; then
        echo "[INFO] --no-rasp set — skipping RPi"
        return 1
    fi
    if ! command -v sshpass &>/dev/null; then
        echo "[WARN] sshpass not found — RPi kills disabled  (sudo apt install sshpass)"
        return 1
    fi
    if ! sshpass -p "$RASP_PASS" ssh \
            -o ConnectTimeout="$RASP_SSH_TIMEOUT" \
            -o StrictHostKeyChecking=no \
            -o LogLevel=ERROR \
            -o BatchMode=no \
            "${RASP_USER}@${RASP_HOST}" "echo ok" &>/dev/null; then
        echo "[WARN] Cannot reach RPi at ${RASP_HOST} — RPi kills disabled"
        return 1
    fi
    return 0
}

# ── PID / signal helpers (location-aware) ──────────────────────────────────────
# get_pids <loc> <pattern>  →  space-separated PIDs or empty string
get_pids() {
    local loc="$1" pattern="$2" out
    if [[ "$loc" == "local" ]]; then
        out=$(pgrep -f "$pattern" 2>/dev/null | grep -v "^${SELF_PID}$" || true)
    else
        out=$(remote_exec "pgrep -f '$pattern' 2>/dev/null || true")
    fi
    # normalise newlines → spaces so PIDs embed safely in remote shell strings
    echo "$out" | tr '\n' ' ' | xargs
}

# send_signal <loc> <TERM|KILL> <pids...>
send_signal() {
    local loc="$1" sig="$2" pids="${*:3}"
    if [[ "$loc" == "local" ]]; then
        # shellcheck disable=SC2086
        kill "-${sig}" $pids 2>/dev/null || true
    else
        remote_exec "kill -${sig} ${pids} 2>/dev/null || true"
    fi
}

# ── Probe RPi and build location list ─────────────────────────────────────────
LOCATIONS=("local")
if check_rasp; then
    LOCATIONS+=("raspi")
    echo "[INFO] RPi ${RASP_HOST} reachable — kills will run on local + raspi"
else
    echo "[INFO] Kills will run on local only"
fi

# ── Phase 1: SIGTERM ───────────────────────────────────────────────────────────
echo ""
echo "=== Phase 1: SIGTERM (grace period: ${TERM_GRACE_SEC}s) ==="
found_any=false

for target in "${TARGETS[@]}"; do
    for loc in "${LOCATIONS[@]}"; do
        pids=$(get_pids "$loc" "$target")
        if [[ -n "$pids" ]]; then
            echo "  [TERM][$loc]  $target  (pids: $pids)"
            send_signal "$loc" TERM "$pids"
            found_any=true
        else
            echo "  [SKIP][$loc]  $target  (not running)"
        fi
    done
done

if ! $found_any; then
    echo "  No targets were running anywhere."
    exit 0
fi

echo "  Waiting ${TERM_GRACE_SEC}s..."
sleep "$TERM_GRACE_SEC"

# ── Phase 2: SIGKILL loop ──────────────────────────────────────────────────────
echo ""
echo "=== Phase 2: SIGKILL (max retries per target: ${MAX_RETRIES}) ==="
exit_code=0

for target in "${TARGETS[@]}"; do
    for loc in "${LOCATIONS[@]}"; do
        pids=$(get_pids "$loc" "$target")
        [[ -z "$pids" ]] && continue

        for attempt in $(seq 1 "$MAX_RETRIES"); do
            pids=$(get_pids "$loc" "$target")
            [[ -z "$pids" ]] && break
            echo "  [KILL][$loc]  $target  attempt ${attempt}/${MAX_RETRIES}  (pids: $pids)"
            send_signal "$loc" KILL "$pids"
            sleep "$KILL_RETRY_SEC"
        done

        pids=$(get_pids "$loc" "$target")
        if [[ -z "$pids" ]]; then
            echo "  [DEAD][$loc]  $target"
        else
            echo "  [FAIL][$loc]  $target  still alive after ${MAX_RETRIES} retries  (pids: $pids)"
            exit_code=1
        fi
    done
done

echo ""
if [[ $exit_code -eq 0 ]]; then
    echo "=== All targets stopped successfully ==="
else
    echo "=== WARNING: one or more targets could not be killed ==="
fi

exit $exit_code
