#!/usr/bin/env python3
"""
Support script: run aruco localizer + map builder from a pre-recorded drone scan
and save the outputs for use with run_hw_test_amr_nav.py.

No AMR hardware is required — only the drone scan files on disk.

Stages that run:
  06   Wait for scan.mp4 and telemetry.csv
  11   Verify scan.mp4 integrity via ffmpeg
  15   Launch arena_marker_localizer service node
  16   Call /localize_markers service
  17   Publish /aruco/amr/pose and /aruco/goal/pose  → saved to YAML
  18   Launch arena_map_builder server
  19   Call BuildArenaMap action
  20   Publish /drone/map                             → saved to YAML

All other stages (01-05 drone, 07-10 AMR bringup, 12-13 traj/fusion,
14 oradar, 14b emergency stop) are no-ops.

Outputs saved to:
  src/mission_orchestrator/recorded_data/scanX/aruco_amr_pose.yaml
  src/mission_orchestrator/recorded_data/scanX/aruco_goal_pose.yaml
  src/mission_orchestrator/recorded_data/scanX/drone_map.yaml

Usage (from workspace root, after sourcing install/setup.bash):
    python3 src/mission_orchestrator/scripts/save_scan_data.py --scan-id 10

Override ArUco marker IDs (AMR then goal):
    python3 src/mission_orchestrator/scripts/save_scan_data.py --scan-id 10 --aruco-ids '[3, 7]'

With a custom config:
    python3 src/mission_orchestrator/scripts/save_scan_data.py --scan-id 10 \\
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

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.normpath(os.path.join(_SCRIPTS_DIR, '..', '..', '..'))

sys.path.insert(
    0,
    os.path.normpath(os.path.join(_SCRIPTS_DIR, '..')),
)

from mission_orchestrator.orchestrator_node import MissionOrchestratorNode  # noqa: E402
from mission_orchestrator.scan_data_io import (  # noqa: E402
    recorded_data_dir,
    scan_video_dir,
    save_pose,
    save_grid,
)

_DEFAULT_CONFIG = os.path.normpath(
    os.path.join(_SCRIPTS_DIR, '..', 'config', 'orchestrator_params.yaml'))


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator subclass: only stages 06+11+15-20 run; everything else no-ops
# ─────────────────────────────────────────────────────────────────────────────

class _SaveScanOrchestrator(MissionOrchestratorNode):
    """Run localizer + map builder from drone scan files; save aruco poses and map."""

    _output_dir: str = ''

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

    # ── No-op: AMR bringup (07-10) ───────────────────────────────────────────

    def _stage_07_ping(self) -> None:
        pass

    def _stage_08_ssh_connect(self) -> None:
        pass

    def _stage_09_launch_amr(self) -> None:
        pass

    def _stage_10_wait_imu_ready(self) -> None:
        pass

    # ── No-op: trajectory planner + map fusion (not needed for saving) ────────

    def _stage_12_launch_trajectory_planner(self) -> None:
        pass

    def _stage_13_launch_map_fusion(self) -> None:
        pass

    # ── No-op: oradar + emergency stop ────────────────────────────────────────

    def _stage_14_launch_oradar(self) -> None:
        pass

    def _stage_14b_launch_emergency_stop(self) -> None:
        pass

    # ── Stage 17 override: publish + save aruco poses ─────────────────────────

    def _stage_17_publish_aruco_poses(self, markers) -> None:
        super()._stage_17_publish_aruco_poses(markers)

        cfg_a = self._cfg['aruco']
        amr_id = cfg_a['amr_marker_id']
        goal_id = cfg_a['goal_marker_id']
        by_id = {int(m.id): m for m in markers}

        amr_path = os.path.join(self._output_dir, 'aruco_amr_pose.yaml')
        goal_path = os.path.join(self._output_dir, 'aruco_goal_pose.yaml')
        save_pose(amr_path, by_id[amr_id].pose_with_covariance)
        save_pose(goal_path, by_id[goal_id].pose_with_covariance)
        self._log.info(f"  [save] aruco_amr_pose.yaml  → {amr_path!r}")
        self._log.info(f"  [save] aruco_goal_pose.yaml → {goal_path!r}")

    # ── Stage 20 override: publish + save drone map ───────────────────────────

    def _stage_20_publish_drone_map(self, grid) -> None:
        super()._stage_20_publish_drone_map(grid)

        map_path = os.path.join(self._output_dir, 'drone_map.yaml')
        save_grid(map_path, grid)
        self._log.info(
            f"  [save] drone_map.yaml ({grid.info.width}×{grid.info.height} cells)"
            f" → {map_path!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Save aruco poses + drone map from a pre-recorded drone scan '
                    '(no AMR hardware required).')
    parser.add_argument(
        '--scan-id', type=int, required=True, metavar='N',
        help=(
            'Scan number.  Input video is read from '
            'src/arena_map_builder/data/drone_scans/scanN/.  '
            'Outputs are saved to src/mission_orchestrator/recorded_data/scanN/.'
        ),
    )
    parser.add_argument(
        '--config', default=_DEFAULT_CONFIG,
        help='Path to orchestrator_params.yaml (default: config/ inside this package)')
    parser.add_argument(
        '--aruco-ids', default=None, metavar='[AMR_ID,GOAL_ID]',
        help=(
            "Two ArUco marker IDs as a JSON array, e.g. '[3, 7]'.  "
            'Overrides aruco.amr_marker_id and aruco.goal_marker_id in the YAML.'
        ),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        sys.exit(f"ERROR: config not found: {args.config}")

    video_dir = scan_video_dir(_WORKSPACE_ROOT, args.scan_id)
    if not os.path.isdir(video_dir):
        sys.exit(f"ERROR: drone scan directory not found: {video_dir}")

    output_dir = recorded_data_dir(_WORKSPACE_ROOT, args.scan_id)

    rclpy.init()

    node = _SaveScanOrchestrator(args.config)
    node._output_dir = output_dir

    node._cfg['video']['dir'] = video_dir
    node._log.info(f"[--scan-id {args.scan_id}] video.dir → {video_dir!r}")
    node._log.info(f"[--scan-id {args.scan_id}] output dir → {output_dir!r}")

    if args.aruco_ids is not None:
        try:
            ids = json.loads(args.aruco_ids)
            if (not isinstance(ids, list) or len(ids) != 2
                    or not all(isinstance(i, int) for i in ids)):
                raise ValueError
        except (ValueError, TypeError):
            sys.exit("ERROR: --aruco-ids must be a JSON array of exactly two integers, "
                     "e.g. '[0, 1]'")
        node._cfg['aruco']['amr_marker_id'] = ids[0]
        node._cfg['aruco']['goal_marker_id'] = ids[1]
        node._log.info(f"[--aruco-ids] amr_marker_id={ids[0]}, goal_marker_id={ids[1]}")

    cfg_v = node._cfg['video']
    for fname_key in ('video_filename', 'telemetry_filename'):
        p = os.path.join(cfg_v['dir'], cfg_v[fname_key])
        if os.path.isfile(p):
            pathlib.Path(p).touch()
            node._log.info(f"[touch] {p!r}")
        else:
            node._log.warning(f"[touch] File not found, cannot touch: {p!r}")

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True, name='ros-spin')
    spin_thread.start()

    try:
        node.run()
        node._log.info(
            "══════════════════════════════════════════════════\n"
            f"  Saved scan data to: {output_dir}\n"
            "  aruco_amr_pose.yaml\n"
            "  aruco_goal_pose.yaml\n"
            "  drone_map.yaml\n"
            "══════════════════════════════════════════════════")
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
