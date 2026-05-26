#!/usr/bin/env python3
"""
Hardware smoke-test: stages 05-11.

Skips stages 01-04 (RPi ping, SSH, AMR bringup, IMU) — the AMR does not need
to be running.  Stages 05-11 run for real:

  05   Check OptiTrack (rigid-body pose topic)
  05b  Connect Tello WiFi
  06   Launch tello_driver
  07   Drone preflight (camera + battery)
  08   Launch tello_map
  09   Observe drone state machine (1 → 2 → 3 → 4)
  10   Wait for video/telemetry file-path topics
  11   Verify video integrity with ffmpeg

Stages 12-20 are no-ops.  On success the script stays alive as a ROS 2
observer, leaving tello_driver and tello_map running for inspection.

Press Ctrl+C to abort the drone (land + kill processes) and exit.

Usage (from workspace root, after sourcing install/setup.bash):
    python3 src/mission_orchestrator/scripts/run_hw_test_s05_s11.py

With a custom config:
    python3 src/mission_orchestrator/scripts/run_hw_test_s05_s11.py \\
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
# Restricted orchestrator: stages 01-04 no-ops; 05-11 real; 12-20 no-ops
# ─────────────────────────────────────────────────────────────────────────────

class _HwTestOrchestrator(MissionOrchestratorNode):
    """Stages 01-04 and 12-20 are no-ops; stages 05-11 run normally."""

    # ── No-op: RPi / AMR pipeline ────────────────────────────────────────────

    def _stage_01_ping(self) -> None:
        pass

    def _stage_02_ssh_connect(self) -> None:
        pass

    def _stage_03_launch_amr(self) -> None:
        pass

    def _stage_04_wait_imu_ready(self) -> None:
        pass

    # ── No-op: post-scan pipeline ─────────────────────────────────────────────

    def _stage_12_launch_trajectory_planner(self) -> None:
        self._log.info("══════════════════════════════════════════════════")
        self._log.info("  Stages 05-11 PASSED")
        self._log.info("  tello_driver and tello_map are still running.")
        self._log.info("  Press Ctrl+C to abort the drone and exit.")
        self._log.info("══════════════════════════════════════════════════")

    def _stage_13_launch_map_fusion(self) -> None:
        pass

    def _stage_14_launch_oradar(self) -> None:
        pass

    def _stage_15_launch_marker_localizer(self) -> None:
        pass

    def _stage_16_call_localize_markers(self):
        return []

    def _stage_17_publish_aruco_poses(self, markers) -> None:
        pass

    def _stage_18_launch_map_builder(self) -> None:
        pass

    def _stage_19_call_map_builder(self):
        return None

    def _stage_20_publish_drone_map(self, grid) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Hardware smoke-test: mission orchestrator stages 05-11.')
    parser.add_argument(
        '--config', default=_DEFAULT_CONFIG,
        help='Path to orchestrator_params.yaml (default: config/ inside this package)')
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        sys.exit(f"ERROR: config not found: {args.config}")

    rclpy.init()

    node = _HwTestOrchestrator(args.config)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True, name='ros-spin')
    spin_thread.start()

    try:
        node.run()
        # Stages 05-11 done; stay alive so tello processes keep running
        while rclpy.ok():
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        if not node._mission_complete:
            node._abort()
        # _teardown_ssh is a no-op here (SSH was never opened)
        node._teardown_ssh()
        executor.shutdown(timeout_sec=5.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
