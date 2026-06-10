#!/usr/bin/env python3
"""
Hardware test: drone pipeline + ONLINE arena stitching, nothing else.

Combines the drone-only smoke test (run_hw_test_drone.py) with the live
map-builder stitching that save_scan_data.py exercises offline — but here the
map is stitched ONLINE from the drone's processed-image stream during the actual
flight, exactly as the real mission does.

Runs for real:
  01        OptiTrack bringup (drone pose feedback)
  02        Arena map builder bringup (ONLINE mode) + background_path
  03        Drone routine (WiFi → driver → preflight → tello_map takeoff/scan →
            online_start → state machine → video files → ffmpeg)
  (kickoff) online_stop + BuildArenaMap goal (transfer + occupancy)
  04.b      Join the stitched map result and publish it to /drone/map

Skipped (no-ops): 04 marker localizer + 04.a/04.c, 05 VSLAM, 06 Rasp/AMR,
07 e-stop, 08 AMR localizer, 09 mapping, 10 fusion, 11 planner, 12 observer.

On success the stitched OccupancyGrid is published to /drone/map and the script
stays alive (tello_driver/tello_map still running). Press Ctrl+C to land the
drone (abort sequence) and exit.

Usage (from workspace root, after sourcing install/setup.bash):
    python3 src/mission_orchestrator/scripts/run_hw_test_drone_stitch.py

With a custom config:
    python3 src/mission_orchestrator/scripts/run_hw_test_drone_stitch.py \\
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

from mission_orchestrator.orchestrator_node import (  # noqa: E402
    MissionAbortError,
    MissionOrchestratorNode,
)

_DEFAULT_CONFIG = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '..', 'config', 'orchestrator_params.yaml'))


# ─────────────────────────────────────────────────────────────────────────────
# Restricted orchestrator: drone (01, 03) + online stitching (02, kickoff, 04.b)
# ─────────────────────────────────────────────────────────────────────────────

class _DroneStitchOrchestrator(MissionOrchestratorNode):
    """OptiTrack (01), map builder (02), drone routine (03), the BuildArenaMap
    kickoff and the 04.b join run; everything else is a no-op."""

    # ── Stage 04.b override: join the stitched map and publish it ─────────────
    def _stage_04b_publish_aruco_poses(self, markers) -> None:
        self._log.info("╔══ Stage 04.b: Join online-stitched map result")
        # _await_map_result joins the background BuildArenaMap goal kicked off
        # after the flight; it returns None if the build failed.
        map_result = self._await_map_result()
        if map_result is None:
            raise MissionAbortError(
                "Online stitch produced no map — BuildArenaMap failed "
                "(check the map-builder server log / transfer.background_path)")

        grid = map_result.map
        self._log.info(
            f"  Online stitch OK: {grid.info.width}×{grid.info.height} cells, "
            f"{map_result.n_obstacles} obstacles, "
            f"mean consistency={map_result.mean_consistency:.3f}")
        # Publish the stitched grid to /drone/map so it can be inspected in RViz.
        self._publish_drone_map(grid)
        self._log.info("╚══ Stage 04.b OK: stitched map published to /drone/map")

    # ── No-op: marker localizer (04, 04.a, 04.c) — not part of stitching ─────
    def _stage_04_launch_marker_localizer(self) -> None: pass
    def _stage_04a_call_localize_markers(self):
        return []
    def _stage_04c_classify_and_branch(self, markers) -> None: pass

    # ── No-op: everything AMR/nav-side ───────────────────────────────────────
    def _stage_05a_verify_realsense(self) -> None: pass
    def _stage_05b_start_vslam(self) -> None: pass
    def _stage_05c_check_vslam_odometry(self) -> None: pass
    def _stage_06a_ping(self) -> None: pass
    def _stage_06b_ssh_connect(self) -> None: pass
    def _stage_06c_launch_amr(self) -> None: pass
    def _stage_06d_wait_imu_ready(self) -> None: pass
    def _stage_07a_emergency_stop(self) -> None: pass
    def _stage_08_amr_localizer(self) -> None: pass
    def _stage_09a_launch_oradar(self) -> None: pass
    def _stage_09b_publish_static_tf(self) -> None: pass
    def _stage_09c_amr_mapper(self) -> None: pass
    def _stage_10_map_fusion(self) -> None: pass
    def _stage_11_trajectory_planner(self) -> None: pass
    def _stage_12_observer(self) -> None: pass


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Hardware test: drone pipeline + online arena stitching '
                    '(stages 01 + 02 + 03 + map-build kickoff + 04.b join).')
    parser.add_argument(
        '--config', default=_DEFAULT_CONFIG,
        help='Path to orchestrator_params.yaml (default: config/ inside this package)')
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        sys.exit(f"ERROR: config not found: {args.config}")

    rclpy.init()

    node = _DroneStitchOrchestrator(args.config)
    # This test is specifically the ONLINE stitching path: the server is brought
    # up in online mode at 02.a, tello_map publishes processed frames, online_start
    # fires at 03.e, and online_stop finalizes the stitch at the kickoff.
    node._online_enabled = True
    node._cfg.setdefault('map_builder', {})['online'] = True

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True, name='ros-spin')
    spin_thread.start()

    try:
        node.run()
        # Stitch done; stay alive so tello processes keep running and the
        # latched /drone/map stays available for inspection.
        while rclpy.ok():
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        if not node._mission_complete:
            node._abort()
        node._stop_rosbag()
        node._teardown_ssh()
        executor.shutdown(timeout_sec=5.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
