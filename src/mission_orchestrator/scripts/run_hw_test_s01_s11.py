#!/usr/bin/env python3
"""
Hardware smoke-test: stages 01-11.

Runs the full mission orchestrator pipeline from stage 01 (OptiTrack
check) through stage 11 (ffmpeg video integrity check).  Stages 12-20
are no-ops.  On success the script spins as a ROS 2 observer, leaving
tello_driver and tello_map running so you can inspect the system state.

Stage order: drone pipeline (01-06), RPi/AMR/IMU (07-10), video verify (11).

Press Ctrl+C to stop the AMR service on the RPi and exit.

Usage (from workspace root, after sourcing install/setup.bash):
    python3 src/mission_orchestrator/scripts/run_hw_test_s01_s11.py

With a custom config:
    python3 src/mission_orchestrator/scripts/run_hw_test_s01_s11.py \\
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
# Restricted orchestrator: stages 01-11 for real; 12-20 no-ops
# ─────────────────────────────────────────────────────────────────────────────

class _HwTestOrchestrator(MissionOrchestratorNode):
    """Stages 01-11 run normally (drone first, then RPi/AMR/IMU); stages 12-20 are no-ops."""

    def _stage_12_launch_trajectory_planner(self) -> None:
        self._log.info("══════════════════════════════════════════════════")
        self._log.info("  Stages 01-11 PASSED")
        self._log.info("  tello_driver and tello_map are still running.")
        self._log.info("  Press Ctrl+C to stop the AMR service and exit.")
        self._log.info("══════════════════════════════════════════════════")

    def _stage_13_launch_map_fusion(self) -> None:
        pass

    def _stage_14_launch_oradar(self) -> None:
        pass

    def _stage_14b_launch_emergency_stop(self) -> None:
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
        description='Hardware smoke-test: mission orchestrator stages 01-11 (drone first, then RPi/AMR/IMU).')
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
        # Stages 01-11 done; stay alive as observer so tello processes keep running
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
