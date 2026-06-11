#!/usr/bin/env bash
# generate_certs.sh — mint certificates for all secured nodes in this project.
#
# Run once per deployment from the project root:
#   ./scripts/generate_certs.sh
#
# Output: ./certs/  (gitignored — never commit private keys)
# Requires: openssl on PATH

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GENERATE="$PROJECT_ROOT/security_middleware/scripts/generate_certs.sh"

if [ ! -f "$GENERATE" ]; then
  echo "ERROR: security_middleware not found at $PROJECT_ROOT/security_middleware" >&2
  echo "       Ensure the submodule is initialised: git submodule update --init" >&2
  exit 1
fi

export CERTS_DIR="${CERTS_DIR:-$PROJECT_ROOT/certs}"

echo "Generating certificates in $CERTS_DIR ..."

# Legacy relay processes (bridge uncontrolled C++ drivers)
RELAY_NODES=(
  optitrack_relay
  scan_relay
)

# Controlled Python nodes (all must be signed)
PYTHON_NODES=(
  optitrack_pose_node
  aruco_localizer
  emergency_stop
  lidar_odometry_node
  occupancy_mapper
  map_fusion_node
  astar_planner2
  spline_follower
  frontier_explorer
  explorer_controller
  mission_orchestrator
)

bash "$GENERATE" "${RELAY_NODES[@]}" "${PYTHON_NODES[@]}"

echo ""
echo "Done. Certificates written to: $CERTS_DIR"
echo "  CA:  $CERTS_DIR/ca.crt (distribute to all nodes)"
echo "  Keys: $CERTS_DIR/<node>.key (keep on the machine that runs that node)"
