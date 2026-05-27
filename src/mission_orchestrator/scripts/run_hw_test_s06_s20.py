#!/usr/bin/env python3
"""
Hardware smoke-test: post-scan pipeline (stages 06 + 11-20).

Skips the drone pipeline (stages 01-05) and the RPi/AMR/IMU pipeline
(stages 07-10).  Assumes scan.mp4 and telemetry.csv already exist in the
configured video directory from a prior run.

Stages that run for real:

  06   Wait for scan.mp4 and telemetry.csv in the configured video dir
  11   Verify scan.mp4 integrity via ffmpeg
  12   Launch trajectory_planner  (skip with --trajectory-planner=false)
  13   Launch map_fusion
  14   Launch oradar lidar
  15   Launch arena_marker_localizer service node
  16   Call /localize_markers service
  17   Publish /aruco/amr/pose and /aruco/goal/pose
  18   Launch arena_map_builder server
  19   Call BuildArenaMap action
  20   Publish /drone/map

On success the script stays alive as a ROS 2 observer, keeping all spawned
processes running.  Press Ctrl+C to stop and exit.

Usage (from workspace root, after sourcing install/setup.bash):
    python3 src/mission_orchestrator/scripts/run_hw_test_s06_s20.py

When the files on disk are older than max_age_sec (typical for re-tests):
    python3 src/mission_orchestrator/scripts/run_hw_test_s06_s20.py --touch-files

Use a specific directory for scan.mp4 / telemetry.csv (overrides video.dir in YAML):
    python3 src/mission_orchestrator/scripts/run_hw_test_s06_s20.py --file-path /data/run42

Override ArUco marker IDs (AMR then goal):
    python3 src/mission_orchestrator/scripts/run_hw_test_s06_s20.py --aruco-ids '[3, 7]'

Skip trajectory_planner (stage 12):
    python3 src/mission_orchestrator/scripts/run_hw_test_s06_s20.py --trajectory-planner=false

Note on stage numbering: old stage 10 (wait for video files) is now stage 06 in
the new execution order.  The drone pipeline (01-05) and RPi/AMR/IMU pipeline
(07-10) are all skipped by this script.

Record a rosbag of all topics during the run:
    python3 src/mission_orchestrator/scripts/run_hw_test_s06_s20.py --rosbag

With a custom config:
    python3 src/mission_orchestrator/scripts/run_hw_test_s06_s20.py \\
        --config /abs/path/to/orchestrator_params.yaml
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
# Restricted orchestrator: drone (01-05) and RPi/AMR/IMU (07-10) no-ops;
# stage 06 + 11-20 real
# ─────────────────────────────────────────────────────────────────────────────

class _HwTestOrchestrator(MissionOrchestratorNode):
    """Drone stages 01-05 and RPi/AMR/IMU stages 07-10 are no-ops; stage 06 + 11-20 run normally."""

    skip_trajectory_planner: bool = False

    # ── No-op: drone pipeline (01-05) ────────────────────────────────────────

    def _stage_01_check_optitrack(self) -> None:
        pass

    def _stage_01b_connect_tello_wifi(self) -> None:
        pass

    def _stage_02_launch_tello_driver(self) -> None:
        pass

    def _stage_03_drone_preflight(self) -> None:
        pass

    def _stage_04_launch_tello_map(self) -> None:
        pass

    def _stage_05_observe_drone_states(self) -> None:
        pass

    # ── No-op: RPi / AMR / IMU pipeline (07-10) ─────────────────────────────

    def _stage_07_ping(self) -> None:
        pass

    def _stage_08_ssh_connect(self) -> None:
        pass

    def _stage_09_launch_amr(self) -> None:
        pass

    def _stage_10_wait_imu_ready(self) -> None:
        pass

    # ── Flag-gated: trajectory_planner ───────────────────────────────────────

    def _stage_12_launch_trajectory_planner(self) -> None:
        if self.skip_trajectory_planner:
            self._log.info("  [stage 12] trajectory_planner skipped (--trajectory-planner=false)")
            return
        super()._stage_12_launch_trajectory_planner()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Hardware smoke-test: post-scan pipeline (stage 06 + stages 11-20).')
    parser.add_argument(
        '--config', default=_DEFAULT_CONFIG,
        help='Path to orchestrator_params.yaml (default: config/ inside this package)')
    parser.add_argument(
        '--touch-files', action='store_true',
        help=(
            'Touch scan.mp4 and telemetry.csv before stage 06 runs, updating '
            'their mtime to now so the freshness check passes.  Use this when '
            'the files on disk are older than video.max_age_sec.'
        ),
    )
    parser.add_argument(
        '--trajectory-planner', type=lambda v: v.lower() != 'false',
        default=True, metavar='true|false',
        help='Launch trajectory_planner in stage 12 (default: true).',
    )
    parser.add_argument(
        '--file-path', default=None, metavar='DIR',
        help=(
            'Directory containing scan.mp4 and telemetry.csv.  '
            'Overrides video.dir in the YAML; filenames are still taken from '
            'video.video_filename and video.telemetry_filename.'
        ),
    )
    parser.add_argument(
        '--aruco-ids', default=None, metavar='[AMR_ID,GOAL_ID]',
        help=(
            "Two ArUco marker IDs as a JSON array, e.g. '[3, 7]'.  "
            'First is amr_marker_id, second is goal_marker_id.  '
            'Overrides aruco.amr_marker_id and aruco.goal_marker_id in the YAML.'
        ),
    )
    parser.add_argument(
        '--rosbag', action='store_true',
        help=(
            'Record a rosbag of all topics for the duration of the run.  '
            'Output goes to the directory configured under rosbag.output_dir '
            'in the YAML (default: /tmp/mission_orchestrator_logs).'
        ),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        sys.exit(f"ERROR: config not found: {args.config}")

    rclpy.init()

    node = _HwTestOrchestrator(args.config)

    node.skip_trajectory_planner = not args.trajectory_planner

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
            sys.exit(f"ERROR: --aruco-ids must be a JSON array of exactly two integers, e.g. '[0, 1]'")
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
        # Stage 06 + stages 11-20 done; stay alive as observer
        while rclpy.ok():
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        if not node._mission_complete:
            node._abort()
        node._stop_rosbag()
        # _teardown_ssh is a no-op here (SSH was never opened)
        node._teardown_ssh()
        executor.shutdown(timeout_sec=5.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
