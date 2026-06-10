#!/usr/bin/env python3
"""
Hardware test: bring up ONLY the AMR localization + mapping core.

Runs a minimal slice of the mission — just enough to localize the AMR and build
its odom-based map — with everything else (drone, map builder, marker localizer,
VSLAM, e-stop, oradar, map fusion, planner, observer) skipped.

Runs for real:
  06   Rasp bringup (ping → SSH → amr_bringup → IMU)
  08   AMR aruco_localizer  (/aruco_pose → EKF)
  09.b alignment_node       (/aruco/amr/pose → world->odom tf)
  09.c odom-based world mapper

Skipped (no-ops): 01 optitrack, 02 map builder, 03 drone, 04/04.a/04.b/04.c
marker localizer + map join, 05 VSLAM, 07 e-stop, 09.a oradar lidar, 10 map
fusion, 11 trajectory_planner, 12 observer.

Note: stage 04.b is skipped, so no /aruco/amr/pose is published from a saved map;
alignment_node therefore falls back to the identity world->odom transform until a
real pose arrives on /aruco/amr/pose. The AMR localizer (08) still publishes
/aruco_pose to the EKF as usual.

The script stays alive after bring-up so you can inspect the running nodes.
Press Ctrl+C to stop the AMR service and exit.

Usage (from workspace root, after sourcing install/setup.bash):
    python3 src/mission_orchestrator/scripts/run_hw_test_amr_mapping.py

    # record everything for the run
    python3 src/mission_orchestrator/scripts/run_hw_test_amr_mapping.py --rosbag
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(
    0,
    os.path.normpath(os.path.join(_SCRIPTS_DIR, '..')),
)

from mission_orchestrator.orchestrator_node import MissionOrchestratorNode  # noqa: E402

_DEFAULT_CONFIG = os.path.normpath(
    os.path.join(_SCRIPTS_DIR, '..', 'config', 'orchestrator_params.yaml'))


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator subclass: only Rasp + AMR localizer + alignment + world mapper run
# ─────────────────────────────────────────────────────────────────────────────

class _AmrMappingOrchestrator(MissionOrchestratorNode):
    """Run Rasp (06), AMR aruco_localizer (08), alignment_node (09.b) and the
    odom-based world mapper (09.c); no-op every other stage."""

    # ── No-op: optitrack (01), map builder (02), drone (03), localizer (04) ──
    def _stage_01a_check_optitrack(self) -> None: pass
    def _stage_01b_optitrack_sanity(self) -> None: pass
    def _stage_02a_configure_background(self) -> None: pass
    def _stage_02b_configure_mode(self) -> None: pass
    def _send_map_goal_async(self) -> None: pass
    def _stage_03a_connect_tello_wifi(self) -> None: pass
    def _stage_03b_launch_tello_driver(self) -> None: pass
    def _stage_03c_drone_preflight(self) -> None: pass
    def _stage_03d_launch_tello_map(self) -> None: pass
    def _stage_03f_observe_drone_states(self) -> None: pass
    def _stage_03g_wait_video_files(self) -> None: pass
    def _stage_03h_verify_video_integrity(self) -> None: pass
    def _stage_04_launch_marker_localizer(self) -> None: pass
    def _stage_04a_call_localize_markers(self):
        return []
    def _stage_04b_publish_aruco_poses(self, markers) -> None: pass
    def _stage_04c_classify_and_branch(self, markers) -> None: pass

    # ── No-op: VSLAM (05), e-stop (07) ───────────────────────────────────────
    def _stage_05a_verify_realsense(self) -> None: pass
    def _stage_05b_start_vslam(self) -> None: pass
    def _stage_05c_check_vslam_odometry(self) -> None: pass
    def _stage_07a_emergency_stop(self) -> None: pass

    # ── No-op: oradar (09.a), map fusion (10), planner (11), observer (12) ───
    def _stage_09a_launch_oradar(self) -> None: pass
    def _stage_10_map_fusion(self) -> None: pass
    def _stage_11_trajectory_planner(self) -> None: pass
    def _stage_12a_start_frontier(self) -> None: pass
    def _stage_12_observer(self) -> None: pass

    # Stages 06 (Rasp), 08 (AMR localizer), 09.b (alignment) and 09.c (world
    # mapper) are NOT overridden — they run via the base orchestrator.


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Hardware test: bring up only the AMR localization + mapping '
                    'core (Rasp + aruco_localizer + alignment_node + world '
                    'mapper); all other stages skipped.')
    parser.add_argument(
        '--config', default=_DEFAULT_CONFIG,
        help='Path to orchestrator_params.yaml (default: config/ inside this package)')
    parser.add_argument(
        '--rosbag', action='store_true',
        help='Record a rosbag of all topics for the duration of the run.')
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        sys.exit(f"ERROR: config not found: {args.config}")

    rclpy.init()

    node = _AmrMappingOrchestrator(args.config)

    # No map builder / live stream in this test.
    node._online_enabled = False
    node._cfg.setdefault('map_builder', {})['online'] = False

    if args.rosbag:
        node._cfg.setdefault('rosbag', {})['enabled'] = True

    node._log.info(
        "[run_hw_test_amr_mapping] bringing up Rasp + AMR localizer + "
        "alignment_node + world mapper; all other stages skipped.")

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True, name='ros-spin')
    spin_thread.start()

    try:
        node.run()
        # Stages complete; stay alive so the running nodes can be inspected.
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
