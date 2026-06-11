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
       fusion node needs before it will emit /fused_map). With autonomy enabled
       (--trajectory-planner=true) it ALSO publishes the saved /aruco/goal/pose.

Autonomy (opt-in via --trajectory-planner=true):
  11   Launch trajectory_planner (astar_planner2 + spline_follower). The planner
       plans on /drone/map toward /aruco/goal/pose and the spline_follower emits
       /amr/reference for the AMR controller to follow — driving the car to the
       goal. The script monitors /amr/ekf/odom and logs when the goal is reached.
  Default (--trajectory-planner=false): stage 11 stays a no-op (mapping only,
  e.g. for joystick driving).

Skipped (no-ops): 01 optitrack, 03 drone, 04/04.a marker localizer, 04.c
classifier, 05 VSLAM, 07 e-stop, 12 observer.

Prerequisite: the saved data must exist (run save_scan_data.py --scan-id N first).
For autonomy, the AMR controller must be following /amr/reference (joystick off).

The script stays alive after bring-up so you can inspect the running nodes.
Press Ctrl+C to stop the AMR service and exit.

Usage (from workspace root, after sourcing install/setup.bash):
    # mapping only (default)
    python3 src/mission_orchestrator/scripts/run_hw_test_amr_mapping.py --scan-id 21

    # autonomy: also bring up the planner and drive to the saved goal
    python3 src/mission_orchestrator/scripts/run_hw_test_amr_mapping.py \\
        --scan-id 21 --trajectory-planner=true

    # record everything for the run
    python3 src/mission_orchestrator/scripts/run_hw_test_amr_mapping.py --scan-id 21 --rosbag
"""

from __future__ import annotations

import argparse
import math
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
    fusion (10); mock 04.b from saved data; optionally run the trajectory
    planner (11) for autonomy; no-op every other stage."""

    _scan_data_dir: str = ''
    _run_planner: bool = False          # set from --trajectory-planner
    _goal_tolerance_m: float = 0.15
    _goal_x: float = 0.0
    _goal_y: float = 0.0
    _goal_set: bool = False
    _goal_reached: bool = False

    # ── Extra subscription: AMR EKF pose for goal-reached monitoring ──────────
    def _init_ros_interfaces(self) -> None:
        super()._init_ros_interfaces()
        from nav_msgs.msg import Odometry
        self.create_subscription(Odometry, '/amr/ekf/odom', self._goal_monitor_cb, 10)

    def _goal_monitor_cb(self, msg) -> None:
        if not self._goal_set or self._goal_reached:
            return
        dx = msg.pose.pose.position.x - self._goal_x
        dy = msg.pose.pose.position.y - self._goal_y
        if math.hypot(dx, dy) <= self._goal_tolerance_m:
            self._goal_reached = True
            self._log.info(
                "╔══════════════════════════════════════════════════╗\n"
                f"  GOAL REACHED  (tol={self._goal_tolerance_m} m)\n"
                "  AMR is at goal.  Press Ctrl+C to stop and exit.\n"
                "╚══════════════════════════════════════════════════╝")

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

        # Goal pose — only when the planner will run (autonomy). Gives
        # astar_planner2 a destination on /aruco/goal/pose and arms the
        # goal-reached monitor.
        if self._run_planner:
            goal_pose = load_pose(os.path.join(self._scan_data_dir, 'aruco_goal_pose.yaml'))
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            self._pub_aruco_goal.publish(goal_pose)
            self._goal_x = goal_pose.pose.pose.position.x
            self._goal_y = goal_pose.pose.pose.position.y
            self._goal_set = True
            self._log.info(
                f"  goal pose → /aruco/goal/pose "
                f"({self._goal_x:.3f}, {self._goal_y:.3f})  "
                f"tol={self._goal_tolerance_m} m")
        self._log.info("╚══ Stage 04.b [saved] OK")

    # ── No-op: VSLAM (05), e-stop (07) ───────────────────────────────────────
    def _stage_05a_verify_realsense(self) -> None: pass
    def _stage_05b_start_vslam(self) -> None: pass
    def _stage_05c_check_vslam_odometry(self) -> None: pass
    def _stage_07a_emergency_stop(self) -> None: pass

    # ── No-op: observer (12) ─────────────────────────────────────────────────
    def _stage_12a_start_frontier(self) -> None: pass
    def _stage_12_observer(self) -> None: pass

    # ── Flag-gated: trajectory_planner (11) ──────────────────────────────────
    # Autonomy: plans on /drone/map toward /aruco/goal/pose (published at 04.b).
    def _stage_11_trajectory_planner(self) -> None:
        if not self._run_planner:
            self._log.info(
                "  [stage 11] trajectory_planner skipped (--trajectory-planner=false)")
            return
        super()._stage_11_trajectory_planner()

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
                    'alignment_node + world mapper + map fusion). Optionally add '
                    'the trajectory planner for autonomy (--trajectory-planner=true).')
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
        '--trajectory-planner', type=lambda v: v.lower() != 'false',
        default=False, metavar='true|false',
        help='Run stage 11 trajectory_planner for autonomy (default: false). '
             'When true, the saved /aruco/goal/pose is also published and the '
             'car drives to it.')
    parser.add_argument(
        '--goal-tolerance', type=float, default=0.15, metavar='METRES',
        help='Distance to the goal that counts as reached, for the monitor '
             '(default: 0.15 m). Only used with --trajectory-planner=true.')
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
    node._run_planner = args.trajectory_planner
    node._goal_tolerance_m = args.goal_tolerance

    # Offline map builder: bring the server up in plain (non-online) mode. No
    # live processed-image stream and no BuildArenaMap goal is sent — the
    # /drone/map fusion needs is the SAVED one published at stage 04.b instead.
    node._online_enabled = False
    node._cfg.setdefault('map_builder', {})['online'] = False

    if args.rosbag:
        node._cfg.setdefault('rosbag', {})['enabled'] = True

    node._log.info(
        f"[--scan-id {args.scan_id}] using saved data from {data_dir!r}")

    mode = ("AUTONOMY (planner ON → drive to saved goal)"
            if args.trajectory_planner else "mapping only (planner OFF)")
    node._log.info(
        "[run_hw_test_amr_mapping] bringing up map builder + Rasp + AMR "
        "localizer + oradar lidar + alignment_node + world mapper + map fusion "
        f"— {mode}.")

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
