#!/usr/bin/env python3
"""
Hardware smoke-test: stages 01 (OptiTrack) + 03 (Drone) + 06 (Rasp/AMR/IMU).

Runs OFFLINE (no map builder): OptiTrack bringup, the full drone routine, and
the Raspberry-Pi/AMR bringup (ping → SSH → amr_bringup → IMU). Everything else —
map builder (02), aruco localizer (04), VSLAM (05), e-stop (07), mapping (08),
planner (09) — is a no-op.

On success the script spins as a ROS 2 observer, leaving tello_driver, tello_map
and the AMR service running. Press Ctrl+C to stop the AMR service and exit.

Usage (from workspace root, after sourcing install/setup.bash):
    python3 src/mission_orchestrator/scripts/run_hw_test_drone_amr.py

With a custom config:
    python3 src/mission_orchestrator/scripts/run_hw_test_drone_amr.py \\
        --config /abs/path/to/orchestrator_params.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor

# Allow running directly from the source tree (fallback for non-installed envs)
sys.path.insert(
    0,
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')),
)

from mission_orchestrator.orchestrator_node import MissionOrchestratorNode  # noqa: E402

_DEFAULT_CONFIG = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '..', 'config', 'orchestrator_params.yaml'))


# ─────────────────────────────────────────────────────────────────────────────
# Restricted orchestrator: stages 01 (optitrack) + 03 (drone) + 06 (rasp) run
# ─────────────────────────────────────────────────────────────────────────────

class _HwTestOrchestrator(MissionOrchestratorNode):
    """Optitrack (01), drone (03) and Rasp/AMR (06) run; everything else no-op."""

    # ── No-op: map builder (02) — offline, no map ────────────────────────────
    def _stage_02a_configure_background(self) -> None: pass
    def _stage_02b_configure_mode(self) -> None: pass
    def _send_map_goal_async(self) -> None: pass

    # ── No-op: aruco localizer (04) ──────────────────────────────────────────
    def _stage_04_launch_marker_localizer(self) -> None: pass
    def _stage_04a_call_localize_markers(self):
        return []
    def _stage_04b_publish_aruco_poses(self, markers) -> None: pass

    # ── No-op: VSLAM (05), e-stop (07), mapping (08), planner (09) ────────────
    def _stage_05a_verify_realsense(self) -> None: pass
    def _stage_05b_start_vslam(self) -> None: pass
    def _stage_05c_check_vslam_odometry(self) -> None: pass
    def _stage_07a_emergency_stop(self) -> None: pass
    def _stage_08a_launch_oradar(self) -> None: pass
    def _stage_08b_publish_static_tf(self) -> None: pass
    def _stage_08c_amr_mapper(self) -> None: pass

    def _stage_09_trajectory_planner(self) -> None:
        self._log.info("══════════════════════════════════════════════════")
        self._log.info("  Stages 01 + 03 + 06 PASSED")
        self._log.info("  tello_driver, tello_map and the AMR service are running.")
        self._log.info("  Press Ctrl+C to stop the AMR service and exit.")
        self._log.info("══════════════════════════════════════════════════")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Hardware smoke-test: stages 01 + 03 + 06 (drone + Rasp/AMR).')
    parser.add_argument(
        '--config', default=_DEFAULT_CONFIG,
        help='Path to orchestrator_params.yaml (default: config/ inside this package)')
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        sys.exit(f"ERROR: config not found: {args.config}")

    rclpy.init()

    node = _HwTestOrchestrator(args.config)
    # No map builder in this smoke test → ignore the online stream.
    node._online_enabled = False
    node._cfg.setdefault('map_builder', {})['online'] = False

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True, name='ros-spin')
    spin_thread.start()

    try:
        node.run()
        # Stages done; stay alive as observer so spawned processes keep running
        while rclpy.ok():
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        if not node._mission_complete:
            node._abort()
        node._teardown_ssh()
        executor.shutdown(timeout_sec=5.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
