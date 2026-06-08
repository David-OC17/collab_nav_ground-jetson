#!/usr/bin/env python3
"""
Hardware smoke-test: drone-only pipeline — stages 01 (OptiTrack) + 03 (Drone).

Runs OFFLINE (no map builder): brings up OptiTrack and the full drone routine
(WiFi → driver → preflight → tello_map → state machine → video files → ffmpeg),
then stops. Everything else — map builder (02), aruco localizer (04), VSLAM
(05), Rasp/AMR (06), e-stop (07), mapping (08), planner (09) — is a no-op.

On success the script stays alive as a ROS 2 observer, leaving tello_driver
and tello_map running. Press Ctrl+C to abort the drone (land + kill) and exit.

Usage (from workspace root, after sourcing install/setup.bash):
    python3 src/mission_orchestrator/scripts/run_hw_test_drone.py

With a custom config:
    python3 src/mission_orchestrator/scripts/run_hw_test_drone.py \\
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
# Restricted orchestrator: only stages 01 (optitrack) + 03 (drone) run
# ─────────────────────────────────────────────────────────────────────────────

class _HwTestOrchestrator(MissionOrchestratorNode):
    """Optitrack (01) + drone routine (03) run; everything else is a no-op."""

    # ── No-op: map builder (02) — offline drone-only test, no map ────────────
    def _stage_02a_configure_background(self) -> None: pass
    def _stage_02b_configure_mode(self) -> None: pass
    def _send_map_goal_async(self) -> None: pass

    # ── No-op: aruco localizer (04) ──────────────────────────────────────────
    def _stage_04_launch_marker_localizer(self) -> None: pass
    def _stage_04a_call_localize_markers(self):
        return []
    def _stage_04b_publish_aruco_poses(self, markers) -> None:
        self._log.info("══════════════════════════════════════════════════")
        self._log.info("  Drone stages 01 + 03 PASSED")
        self._log.info("  tello_driver and tello_map are still running.")
        self._log.info("  Press Ctrl+C to abort the drone and exit.")
        self._log.info("══════════════════════════════════════════════════")

    # ── No-op: VSLAM (05), Rasp (06), e-stop (07), mapping (08), planner (09) ─
    def _stage_05a_verify_realsense(self) -> None: pass
    def _stage_05b_start_vslam(self) -> None: pass
    def _stage_05c_check_vslam_odometry(self) -> None: pass
    def _stage_06a_ping(self) -> None: pass
    def _stage_06b_ssh_connect(self) -> None: pass
    def _stage_06c_launch_amr(self) -> None: pass
    def _stage_06d_wait_imu_ready(self) -> None: pass
    def _stage_07a_emergency_stop(self) -> None: pass
    def _stage_08a_launch_oradar(self) -> None: pass
    def _stage_08b_publish_static_tf(self) -> None: pass
    def _stage_08c_amr_mapper(self) -> None: pass
    def _stage_09_trajectory_planner(self) -> None: pass


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Hardware smoke-test: drone-only pipeline (stages 01 + 03).')
    parser.add_argument(
        '--config', default=_DEFAULT_CONFIG,
        help='Path to orchestrator_params.yaml (default: config/ inside this package)')
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        sys.exit(f"ERROR: config not found: {args.config}")

    rclpy.init()

    node = _HwTestOrchestrator(args.config)
    # Drone-only test: no map builder, so the online stream is irrelevant.
    node._online_enabled = False
    node._cfg.setdefault('map_builder', {})['online'] = False

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True, name='ros-spin')
    spin_thread.start()

    try:
        node.run()
        # Drone stages done; stay alive so tello processes keep running
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
