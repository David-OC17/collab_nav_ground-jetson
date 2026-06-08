#!/usr/bin/env python3
"""
Support script: run aruco localizer + map builder from a pre-recorded drone scan
and save the outputs for use with run_hw_test_amr_nav.py.

No AMR hardware is required — only the drone scan files on disk.

Runs OFFLINE (map_builder.online forced false): stitches from the saved
scan.mp4. The map-builder stages run as in the real mission:
  02.a   Launch arena_map_builder server + set background_path
  03.g/h Resolve scan.mp4 / telemetry.csv + ffmpeg integrity
  (kickoff) Send BuildArenaMap goal (full stitch + transfer + occupancy)
  04     Launch arena_marker_localizer + 04.a Call /localize_markers (ORIENTATION)
  04.b   Join map result (POSITION) → publish + SAVE aruco poses & OccupancyGrid
         (if --ground-truth is given, AMR/goal pose errors vs GT are printed here)

All other stages (01 optitrack, 03.a-f drone, 05 vslam, 06 rasp/AMR,
07 e-stop, 08 mapping, 09 planner, 10 observer) are no-ops.

Outputs saved to:
  src/mission_orchestrator/recorded_data/scanX/aruco_amr_pose.yaml
  src/mission_orchestrator/recorded_data/scanX/aruco_goal_pose.yaml
  src/mission_orchestrator/recorded_data/scanX/drone_map.yaml

Usage (from workspace root, after sourcing install/setup.bash):
    python3 src/mission_orchestrator/scripts/save_scan_data.py --scan-id 10

Override ArUco marker IDs (AMR then goal):
    python3 src/mission_orchestrator/scripts/save_scan_data.py --scan-id 10 --aruco-ids '[3, 7]'

Print AMR/goal pose errors vs a ground-truth file:
    python3 src/mission_orchestrator/scripts/save_scan_data.py --scan-id 20 \\
        --ground-truth src/arena_marker_localizer/config/aruco_pose_gt/scan20.yaml

With a custom config:
    python3 src/mission_orchestrator/scripts/save_scan_data.py --scan-id 10 \\
        --config /abs/path/to/orchestrator_params.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import threading
import time
from typing import Optional

import yaml

import rclpy
from rclpy.executors import MultiThreadedExecutor

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.normpath(os.path.join(_SCRIPTS_DIR, '..', '..', '..'))

sys.path.insert(
    0,
    os.path.normpath(os.path.join(_SCRIPTS_DIR, '..')),
)

from mission_orchestrator.orchestrator_node import (  # noqa: E402
    MissionOrchestratorNode,
    _yaw_from_quat,
)
from mission_orchestrator.scan_data_io import (  # noqa: E402
    recorded_data_dir,
    scan_video_dir,
    save_pose,
    save_grid,
)

_DEFAULT_CONFIG = os.path.normpath(
    os.path.join(_SCRIPTS_DIR, '..', 'config', 'orchestrator_params.yaml'))


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wrap_deg(angle_deg: float) -> float:
    """Wrap an angle in degrees to (-180, 180]."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def _load_ground_truth(path: str) -> dict:
    """Load a marker ground-truth YAML into {int id: {'x', 'y', 'theta'}}.

    Same format the arena_marker_localizer calibration uses, e.g.::

        markers:
          0: {x: 0.76, y: 0.785, theta: 0}
          2: {x: 3.065, y: 0.781, theta: 180}

    x/y are in metres; theta is in DEGREES.
    """
    with open(path, 'r') as fh:
        data = yaml.safe_load(fh) or {}
    markers = data.get('markers', data)
    return {int(k): v for k, v in markers.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator subclass: only stages 06+11+15-20 run; everything else no-ops
# ─────────────────────────────────────────────────────────────────────────────

class _SaveScanOrchestrator(MissionOrchestratorNode):
    """Run localizer + map builder from drone scan files; save aruco poses and map."""

    _output_dir: str = ''
    # When set (via --ground-truth), stage 04.b prints AMR/goal pose errors vs GT.
    _gt_markers: Optional[dict] = None

    # ── No-ops: everything except the map-builder + localizer stages ─────────
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
    def _stage_06a_ping(self) -> None: pass
    def _stage_06b_ssh_connect(self) -> None: pass
    def _stage_06c_launch_amr(self) -> None: pass
    def _stage_06d_wait_imu_ready(self) -> None: pass
    def _stage_07a_emergency_stop(self) -> None: pass
    def _stage_08a_launch_oradar(self) -> None: pass
    def _stage_08b_publish_static_tf(self) -> None: pass
    def _stage_08c_amr_mapper(self) -> None: pass
    def _stage_09_trajectory_planner(self) -> None: pass
    def _stage_10_observer(self) -> None: pass
    def _abort_drone(self) -> None: pass

    # ── Stage 04.b override: join map result → publish + SAVE poses & map ─────

    def _stage_04b_publish_aruco_poses(self, markers) -> None:
        # super() awaits the map result, publishes the poses + OccupancyGrid, and
        # stores self._last_map_result.
        super()._stage_04b_publish_aruco_poses(markers)

        cfg_a = self._cfg['aruco']
        amr_id = cfg_a['amr_marker_id']
        goal_id = cfg_a['goal_marker_id']
        by_id = {int(m.id): m for m in markers}
        mr = self._last_map_result

        # Re-derive the SAME published poses (idempotent) for saving.
        amr_pose = self._build_marker_pose(
            by_id[amr_id], mr.amr_marker_position, "AMR", amr_id)
        goal_pose = self._build_marker_pose(
            by_id[goal_id], mr.goal_marker_position, "goal", goal_id)

        amr_path = os.path.join(self._output_dir, 'aruco_amr_pose.yaml')
        goal_path = os.path.join(self._output_dir, 'aruco_goal_pose.yaml')
        map_path = os.path.join(self._output_dir, 'drone_map.yaml')
        save_pose(amr_path, amr_pose)
        save_pose(goal_path, goal_pose)
        save_grid(map_path, mr.map)
        self._log.info(f"  [save] aruco_amr_pose.yaml  → {amr_path!r}")
        self._log.info(f"  [save] aruco_goal_pose.yaml → {goal_path!r}")
        self._log.info(
            f"  [save] drone_map.yaml ({mr.map.info.width}×{mr.map.info.height} "
            f"cells) → {map_path!r}")

        # If a ground-truth file was provided, report the error of the FINAL
        # (map-builder position + localizer orientation) AMR and goal poses.
        if self._gt_markers is not None:
            self._log.info("  [gt] Marker pose error vs ground truth:")
            self._print_marker_gt_error("AMR", amr_id, amr_pose)
            self._print_marker_gt_error("goal", goal_id, goal_pose)

    def _print_marker_gt_error(self, label: str, marker_id: int, pose) -> None:
        """Print the final pose's error vs ground truth for one marker:
        per-axis (Δx, Δy in metres; Δθ in degrees, wrapped) plus the euclidean
        position error. `pose` is the published PoseWithCovarianceStamped."""
        gt = (self._gt_markers or {}).get(int(marker_id))
        if gt is None:
            self._log.warning(
                f"    {label} (id={marker_id}): no ground-truth entry "
                f"(have ids: {sorted((self._gt_markers or {}).keys())}) — skipped")
            return

        gt_x, gt_y, gt_th = float(gt['x']), float(gt['y']), float(gt['theta'])
        px = pose.pose.pose.position.x
        py = pose.pose.pose.position.y
        yaw_deg = math.degrees(_yaw_from_quat(pose.pose.pose.orientation))

        dx = px - gt_x
        dy = py - gt_y
        dist = math.hypot(dx, dy)
        dth = _wrap_deg(yaw_deg - gt_th)

        self._log.info(
            f"    {label} (id={marker_id}): "
            f"Δx={dx:+.3f} m  Δy={dy:+.3f} m  |Δpos|={dist:.3f} m  Δθ={dth:+.2f}°")
        self._log.info(
            f"        got  x={px:.3f} y={py:.3f} θ={yaw_deg:+.2f}°   "
            f"gt  x={gt_x:.3f} y={gt_y:.3f} θ={gt_th:+.2f}°")


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
    parser.add_argument(
        '--ground-truth', default=None, metavar='PATH',
        help=(
            'Optional marker ground-truth YAML (same format the '
            'arena_marker_localizer calibration uses, e.g. '
            'src/arena_marker_localizer/config/aruco_pose_gt/scanN.yaml).  When '
            'given, the AMR and goal pose errors vs ground truth (Δx, Δy, Δθ and '
            'euclidean) are printed after the poses are saved.'
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

    # This flow has no live drone stream — always stitch from the saved video.
    node._online_enabled = False
    node._cfg.setdefault('map_builder', {})['online'] = False

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

    if args.ground_truth is not None:
        if not os.path.isfile(args.ground_truth):
            sys.exit(f"ERROR: ground-truth file not found: {args.ground_truth}")
        node._gt_markers = _load_ground_truth(args.ground_truth)
        node._log.info(
            f"[--ground-truth] loaded {len(node._gt_markers)} marker(s) from "
            f"{args.ground_truth!r}")

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
