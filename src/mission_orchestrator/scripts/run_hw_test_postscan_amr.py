#!/usr/bin/env python3
"""
Hardware smoke-test: AMR bringup + post-scan pipeline (no drone, no VSLAM).

Skips the drone flight and the Intel RealSense / VSLAM. All Rasp/AMR stages run
for real. Runs OFFLINE (stitches the saved scan.mp4).

Stages that run for real:
  02   Arena map builder bringup (server + background_path)
  03.g/h  Resolve scan.mp4 / telemetry.csv + ffmpeg integrity
  (kickoff) Send BuildArenaMap goal (full stitch + transfer + occupancy)
  04   Launch arena_marker_localizer + 04.a Call /localize_markers
  04.b Join map result → publish /aruco/.../pose + /drone/map
  06   Rasp bringup (ping → SSH → amr_bringup → IMU)
  07   Emergency stop bringup
  08   Mapping bringup (oradar + alignment_node + odom mapper)
  09   Trajectory planner  (skip with --trajectory-planner=false)

Skipped (no-ops): 01 optitrack, 03.a-f drone flight, 05 VSLAM.

Use this to test the full ground-robot pipeline against a pre-recorded drone
scan, without flying the drone. On success the script stays alive as a ROS 2
observer. Press Ctrl+C to stop the AMR service and exit.

Usage (from workspace root, after sourcing install/setup.bash):
    python3 src/mission_orchestrator/scripts/run_hw_test_postscan_amr.py --touch-files
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
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
# Restricted orchestrator: map builder + localizer + AMR bringup + mapping +
# planner; drone flight + VSLAM are no-ops. Stitch OFFLINE from saved scan.mp4.
# ─────────────────────────────────────────────────────────────────────────────

class _HwTestOrchestrator(MissionOrchestratorNode):
    """Stages 02, 03.g/h, 04, 06, 07, 08, 09 run; drone flight + VSLAM no-op."""

    skip_trajectory_planner: bool = False

    # ── No-op: optitrack (01) + drone flight (03.a-f) + VSLAM (05) ───────────
    def _stage_01a_check_optitrack(self) -> None: pass
    def _stage_01b_optitrack_sanity(self) -> None: pass
    def _stage_03a_connect_tello_wifi(self) -> None: pass
    def _stage_03b_launch_tello_driver(self) -> None: pass
    def _stage_03c_drone_preflight(self) -> None: pass
    def _stage_03d_launch_tello_map(self) -> None: pass
    def _stage_03f_observe_drone_states(self) -> None: pass
    def _stage_05a_verify_realsense(self) -> None: pass
    def _stage_05b_start_vslam(self) -> None: pass
    def _stage_05c_check_vslam_odometry(self) -> None: pass
    def _stage_08_amr_localizer(self) -> None: pass
    def _stage_10_map_fusion(self) -> None: pass

    # ── Flag-gated: trajectory_planner (11) ──────────────────────────────────
    def _stage_11_trajectory_planner(self) -> None:
        if self.skip_trajectory_planner:
            self._log.info("  [stage 11] trajectory_planner skipped (--trajectory-planner=false)")
            return
        super()._stage_11_trajectory_planner()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Hardware smoke-test: AMR bringup + post-scan pipeline (no drone, no VSLAM).')
    parser.add_argument(
        '--config', default=_DEFAULT_CONFIG,
        help='Path to orchestrator_params.yaml (default: config/ inside this package)')
    parser.add_argument(
        '--touch-files', action='store_true',
        help=(
            'Touch scan.mp4 and telemetry.csv so the freshness check passes.  '
            'Use this when the files on disk are older than video.max_age_sec.'
        ),
    )
    parser.add_argument(
        '--trajectory-planner', type=lambda v: v.lower() != 'false',
        default=True, metavar='true|false',
        help='Launch trajectory_planner in stage 09 (default: true).',
    )
    parser.add_argument(
        '--file-path', default=None, metavar='DIR',
        help='Directory containing scan.mp4 and telemetry.csv (overrides video.dir).')
    parser.add_argument(
        '--aruco-ids', default=None, metavar='[AMR_ID,GOAL_ID]',
        help="Two ArUco marker IDs as a JSON array, e.g. '[3, 7]' (amr, goal).")
    parser.add_argument(
        '--rosbag', action='store_true',
        help='Record a rosbag of all topics for the duration of the run.')
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        sys.exit(f"ERROR: config not found: {args.config}")

    rclpy.init()

    node = _HwTestOrchestrator(args.config)
    node.skip_trajectory_planner = not args.trajectory_planner

    # No live drone stream here → stitch OFFLINE from the saved scan.mp4.
    node._online_enabled = False
    node._cfg.setdefault('map_builder', {})['online'] = False

    if args.file_path is not None:
        node._cfg['video']['dir'] = args.file_path
        node._log.info(f"[--file-path] video.dir overridden → {args.file_path!r}")

    if args.aruco_ids is not None:
        try:
            ids = json.loads(args.aruco_ids)
            if (not isinstance(ids, list) or len(ids) != 2
                    or not all(isinstance(i, int) for i in ids)):
                raise ValueError
        except (ValueError, TypeError):
            sys.exit("ERROR: --aruco-ids must be a JSON array of exactly two integers, e.g. '[0, 1]'")
        node._cfg['aruco']['amr_marker_id'] = ids[0]
        node._cfg['aruco']['goal_marker_id'] = ids[1]
        node._log.info(f"[--aruco-ids] amr_marker_id={ids[0]}, goal_marker_id={ids[1]}")

    if args.rosbag:
        node._cfg.setdefault('rosbag', {})['enabled'] = True

    if args.touch_files:
        cfg_v = node._cfg['video']
        video_path = os.path.join(cfg_v['dir'], cfg_v['video_filename'])
        telemetry_path = os.path.join(cfg_v['dir'], cfg_v['telemetry_filename'])
        for p in (video_path, telemetry_path):
            if os.path.isfile(p):
                pathlib.Path(p).touch()
                node._log.info(f"[--touch-files] Touched {p!r}")
            else:
                node._log.warning(f"[--touch-files] File not found, cannot touch: {p!r}")

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True, name='ros-spin')
    spin_thread.start()

    try:
        node.run()
        # Stages done; stay alive as observer
        while rclpy.ok():
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        if not node._mission_complete:
            node._abort()
        node._stop_rosbag()
        node._teardown_ssh()  # stops the AMR service and closes the SSH connection
        executor.shutdown(timeout_sec=5.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
