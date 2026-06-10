#!/usr/bin/env python3
"""
Hardware test: AMR localization + mapping core, with map builder + map fusion.

Brings up enough of the mission to localize the AMR, build its odom-based map,
and fuse that with a (saved) drone map — without flying the drone or running the
nav/planner side.

Runs for real:
  02   Arena map builder bringup (build_arena_map_server) — server only
  06   Rasp bringup (ping → SSH → amr_bringup → IMU)
  08   AMR aruco_localizer  (/aruco_pose → EKF)
  09.a oradar lidar
  09.b alignment_node       (/aruco/amr/pose → world->odom tf)
  09.c odom-based world mapper (publishes /amr/world_map)
  10   map fusion (fuses /drone/map + /amr/world_map → /fused_map)

Mocked from saved data (recorded_data/scanN/, via --scan-id):
  04.b Publish the saved /aruco/amr/pose (seeds alignment_node with a real
       world->odom instead of the identity fallback, so the AMR map shares the
       drone map's world frame) AND the saved /drone/map (the immutable base the
       fusion node needs before it will emit /fused_map).

Skipped (no-ops): 01 optitrack, 03 drone, 04/04.a marker localizer, 04.c
classifier, 05 VSLAM, 07 e-stop, 11 trajectory_planner, 12 observer.

Prerequisite: the saved data must exist (run save_scan_data.py --scan-id N first).

The script stays alive after bring-up so you can inspect the running nodes.
Press Ctrl+C to stop the AMR service and exit.

Usage (from workspace root, after sourcing install/setup.bash):
    python3 src/mission_orchestrator/scripts/run_hw_test_amr_mapping.py --scan-id 21

    # record everything for the run
    python3 src/mission_orchestrator/scripts/run_hw_test_amr_mapping.py --scan-id 21 --rosbag
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
_WORKSPACE_ROOT = os.path.normpath(os.path.join(_SCRIPTS_DIR, '..', '..', '..'))

sys.path.insert(
    0,
    os.path.normpath(os.path.join(_SCRIPTS_DIR, '..')),
)

from mission_orchestrator.orchestrator_node import MissionOrchestratorNode  # noqa: E402
from mission_orchestrator.scan_data_io import (  # noqa: E402
    recorded_data_dir,
    load_pose,
    load_grid,
    assert_saved_data_exists,
)

_DEFAULT_CONFIG = os.path.normpath(
    os.path.join(_SCRIPTS_DIR, '..', 'config', 'orchestrator_params.yaml'))


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator subclass: only Rasp + AMR localizer + alignment + world mapper run
# ─────────────────────────────────────────────────────────────────────────────

class _AmrMappingOrchestrator(MissionOrchestratorNode):
    """Run map builder (02), Rasp (06), AMR aruco_localizer (08), oradar lidar
    (09.a), alignment_node (09.b), the odom-based world mapper (09.c) and map
    fusion (10); mock 04.b from saved data; no-op every other stage."""

    _scan_data_dir: str = ''

    # ── No-op: optitrack (01), drone (03), marker localizer (04/04.a/04.c) ───
    # Map builder (02) now runs, but _send_map_goal_async stays a no-op so the
    # launched server idles (no BuildArenaMap goal → no live /drone/map). The
    # /drone/map fusion needs comes from the saved data published at 04.b below.
    def _stage_01a_check_optitrack(self) -> None: pass
    def _stage_01b_optitrack_sanity(self) -> None: pass
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
    def _stage_04c_classify_and_branch(self, markers) -> None: pass

    # ── Mock stage 04.b: publish SAVED /aruco/amr/pose + /drone/map ──────────
    def _stage_04b_publish_aruco_poses(self, markers) -> None:
        self._log.info("╔══ Stage 04.b [saved]: publish saved AMR pose + drone map")
        # AMR pose → seeds alignment_node so world->odom is the real transform
        # (not identity), keeping the AMR map in the same world frame/origin as
        # the drone map for a correct fusion overlay.
        amr_pose = load_pose(os.path.join(self._scan_data_dir, 'aruco_amr_pose.yaml'))
        amr_pose.header.stamp = self.get_clock().now().to_msg()
        self._pub_aruco_amr.publish(amr_pose)
        self._log.info(
            f"  AMR pose → /aruco/amr/pose "
            f"({amr_pose.pose.pose.position.x:.3f}, "
            f"{amr_pose.pose.pose.position.y:.3f})")
        # Drone map → the immutable base the fusion node (10) needs before it
        # will emit /fused_map.
        grid = load_grid(os.path.join(self._scan_data_dir, 'drone_map.yaml'))
        self._publish_drone_map(grid)
        self._log.info(
            f"  drone map → /drone/map "
            f"({grid.info.width}×{grid.info.height} cells)")
        self._log.info("╚══ Stage 04.b [saved] OK")

    # ── No-op: VSLAM (05), e-stop (07) ───────────────────────────────────────
    def _stage_05a_verify_realsense(self) -> None: pass
    def _stage_05b_start_vslam(self) -> None: pass
    def _stage_05c_check_vslam_odometry(self) -> None: pass
    def _stage_07a_emergency_stop(self) -> None: pass

    # ── No-op: planner (11), observer (12) ───────────────────────────────────
    def _stage_11_trajectory_planner(self) -> None: pass
    def _stage_12a_start_frontier(self) -> None: pass
    def _stage_12_observer(self) -> None: pass

    # Stages 02 (map builder), 06 (Rasp), 08 (AMR localizer), 09.a (oradar lidar),
    # 09.b (alignment), 09.c (world mapper) and 10 (map fusion) are NOT
    # overridden — they run via the base orchestrator.


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Hardware test: bring up the AMR localization + mapping core '
                    '(map builder + Rasp + aruco_localizer + oradar lidar + '
                    'alignment_node + world mapper + map fusion); all other '
                    'stages skipped.')
    parser.add_argument(
        '--scan-id', type=int, required=True, metavar='N',
        help=(
            'Scan number whose saved data to publish at stage 04.b '
            '(/aruco/amr/pose + /drone/map).  Data must exist in '
            'src/mission_orchestrator/recorded_data/scanN/ — run '
            'save_scan_data.py --scan-id N first.'
        ),
    )
    parser.add_argument(
        '--config', default=_DEFAULT_CONFIG,
        help='Path to orchestrator_params.yaml (default: config/ inside this package)')
    parser.add_argument(
        '--rosbag', action='store_true',
        help='Record a rosbag of all topics for the duration of the run.')
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        sys.exit(f"ERROR: config not found: {args.config}")

    data_dir = recorded_data_dir(_WORKSPACE_ROOT, args.scan_id)
    try:
        assert_saved_data_exists(data_dir, args.scan_id)
    except FileNotFoundError as exc:
        sys.exit(f"ERROR: {exc}")

    rclpy.init()

    node = _AmrMappingOrchestrator(args.config)
    node._scan_data_dir = data_dir

    # Offline map builder: bring the server up in plain (non-online) mode. No
    # live processed-image stream and no BuildArenaMap goal is sent — the
    # /drone/map fusion needs is the SAVED one published at stage 04.b instead.
    node._online_enabled = False
    node._cfg.setdefault('map_builder', {})['online'] = False

    if args.rosbag:
        node._cfg.setdefault('rosbag', {})['enabled'] = True

    node._log.info(
        f"[--scan-id {args.scan_id}] using saved data from {data_dir!r}")

    node._log.info(
        "[run_hw_test_amr_mapping] bringing up map builder + Rasp + AMR "
        "localizer + oradar lidar + alignment_node + world mapper + map fusion; "
        "all other stages skipped.")

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
