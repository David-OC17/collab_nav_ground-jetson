#!/usr/bin/env python3
"""
Hardware test: AMR navigation with mocked aruco poses + drone map.

Tests the closed-loop feedback chain without re-running the slow map-builder
and marker-localizer pipeline. Requires saved data produced by save_scan_data.py.

Stages that run for real:
  06   Rasp bringup (ping → SSH → amr_bringup → IMU)
  08.b Launch alignment_node (/aruco/amr/pose → world->odom tf)
  08.c Launch odom-based mapper
  09   Launch trajectory_planner  (skip with --trajectory-planner=false)

Stages mocked from saved data (recorded_data/scanX/):
  04.b Publish /aruco/amr/pose + /aruco/goal/pose (aruco_*.yaml) AND /drone/map
       (drone_map.yaml)  — replaces the real localizer + map-builder join.

Skipped (no-ops): 01 optitrack, 02 map builder, 03 drone, 04/04.a localizer,
05 VSLAM, 07 e-stop, 08.a oradar lidar.

The trajectory planner starts planning as soon as it receives /drone/map and
both aruco poses (all published at stage 04.b with TRANSIENT_LOCAL QoS).

Pass condition: AMR reaches the goal pose within --goal-tolerance metres.
The script stays alive after goal-reached so you can inspect system state.
Press Ctrl+C to stop the AMR service and exit.

Prerequisites:
    python3 src/mission_orchestrator/scripts/save_scan_data.py --scan-id N

Usage (from workspace root, after sourcing install/setup.bash):
    python3 src/mission_orchestrator/scripts/run_hw_test_amr_nav.py --scan-id 10

Skip trajectory_planner:
    python3 src/mission_orchestrator/scripts/run_hw_test_amr_nav.py \\
        --scan-id 10 --trajectory-planner=false

Override goal tolerance (default 0.15 m):
    python3 src/mission_orchestrator/scripts/run_hw_test_amr_nav.py \\
        --scan-id 10 --goal-tolerance 0.20
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

_POSE_TOPIC = '/amr/ekf/odom'   # nav_msgs/Odometry published by the EKF node


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator subclass: Rasp + mapping + planner real; aruco/map mocked
# ─────────────────────────────────────────────────────────────────────────────

class _HwTestOrchestrator(MissionOrchestratorNode):
    """Rasp (06) + mapping (08.b/c) + planner (09) run; aruco/map mocked at 04.b."""

    skip_trajectory_planner: bool = False
    _scan_data_dir: str = ''
    _goal_tolerance_m: float = 0.15
    _goal_reached: bool = False
    _goal_x: float = 0.0
    _goal_y: float = 0.0
    _goal_set: bool = False

    # ── Extra subscription: AMR EKF pose for goal monitoring ──────────────────

    def _init_ros_interfaces(self) -> None:
        super()._init_ros_interfaces()
        from nav_msgs.msg import Odometry
        self.create_subscription(Odometry, _POSE_TOPIC, self._goal_monitor_cb, 10)

    def _goal_monitor_cb(self, msg) -> None:
        if not self._goal_set or self._goal_reached:
            return
        dx = msg.pose.pose.position.x - self._goal_x
        dy = msg.pose.pose.position.y - self._goal_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist <= self._goal_tolerance_m:
            self._goal_reached = True
            self._log.info(
                "╔══════════════════════════════════════════════════╗\n"
                f"  GOAL REACHED  dist={dist:.3f} m  "
                f"(tol={self._goal_tolerance_m} m)\n"
                "  AMR is at goal.  Press Ctrl+C to stop and exit.\n"
                "╚══════════════════════════════════════════════════╝")

    # ── No-op: optitrack (01), map builder (02), drone (03), localizer ───────
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

    # ── No-op: VSLAM (05), e-stop (07), oradar (08.a) ────────────────────────
    def _stage_05a_verify_realsense(self) -> None: pass
    def _stage_05b_start_vslam(self) -> None: pass
    def _stage_05c_check_vslam_odometry(self) -> None: pass
    def _stage_07a_emergency_stop(self) -> None: pass
    def _stage_08a_launch_oradar(self) -> None: pass

    # ── Flag-gated: trajectory_planner (09) ──────────────────────────────────
    def _stage_09_trajectory_planner(self) -> None:
        if self.skip_trajectory_planner:
            self._log.info("  [stage 09] trajectory_planner skipped (--trajectory-planner=false)")
            return
        super()._stage_09_trajectory_planner()

    # ── Mock stage 04.b: publish SAVED aruco poses + SAVED drone map ─────────

    def _stage_04b_publish_aruco_poses(self, markers) -> None:
        self._log.info("╔══ Stage 04.b [MOCK]: publish saved aruco poses + drone map")
        amr_path = os.path.join(self._scan_data_dir, 'aruco_amr_pose.yaml')
        goal_path = os.path.join(self._scan_data_dir, 'aruco_goal_pose.yaml')
        amr_pose = load_pose(amr_path)
        goal_pose = load_pose(goal_path)

        if getattr(self, '_override_start_x', None) is not None:
            amr_pose.pose.pose.position.x = self._override_start_x
            amr_pose.pose.pose.position.y = self._override_start_y
        if getattr(self, '_override_start_yaw', None) is not None:
            yaw = self._override_start_yaw
            amr_pose.pose.pose.orientation.x = 0.0
            amr_pose.pose.pose.orientation.y = 0.0
            amr_pose.pose.pose.orientation.z = math.sin(yaw / 2.0)
            amr_pose.pose.pose.orientation.w = math.cos(yaw / 2.0)

        if getattr(self, '_override_goal_x', None) is not None:
            goal_pose.pose.pose.position.x = self._override_goal_x
            goal_pose.pose.pose.position.y = self._override_goal_y
        if getattr(self, '_override_goal_yaw', None) is not None:
            yaw = self._override_goal_yaw
            goal_pose.pose.pose.orientation.x = 0.0
            goal_pose.pose.pose.orientation.y = 0.0
            goal_pose.pose.pose.orientation.z = math.sin(yaw / 2.0)
            goal_pose.pose.pose.orientation.w = math.cos(yaw / 2.0)

        # Store goal xy for the goal-reached monitor
        self._goal_x = goal_pose.pose.pose.position.x
        self._goal_y = goal_pose.pose.pose.position.y
        self._goal_set = True

        amr_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        self._pub_aruco_amr.publish(amr_pose)
        self._pub_aruco_goal.publish(goal_pose)
        self._log.info(
            f"  AMR  pose: ({amr_pose.pose.pose.position.x:.3f}, "
            f"{amr_pose.pose.pose.position.y:.3f})")
        self._log.info(
            f"  Goal pose: ({self._goal_x:.3f}, {self._goal_y:.3f})  "
            f"tol={self._goal_tolerance_m} m")

        # Publish the saved OccupancyGrid to /drone/map.
        grid = load_grid(os.path.join(self._scan_data_dir, 'drone_map.yaml'))
        self._publish_drone_map(grid)
        self._log.info(
            f"  Loaded {grid.info.width}×{grid.info.height} OccupancyGrid "
            f"(res={grid.info.resolution} m/cell)")
        self._log.info("╚══ Stage 04.b [MOCK] OK: poses + drone map published")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Hardware test: AMR navigation loop with mocked map/poses '
                    '(drone + map-builder pipeline skipped).')
    parser.add_argument(
        '--scan-id', type=int, required=True, metavar='N',
        help=(
            'Scan number whose saved data to use.  '
            'Data must exist in src/mission_orchestrator/recorded_data/scanN/ — '
            'run save_scan_data.py --scan-id N first.'
        ),
    )
    parser.add_argument(
        '--config', default=_DEFAULT_CONFIG,
        help='Path to orchestrator_params.yaml (default: config/ inside this package)')
    parser.add_argument(
        '--trajectory-planner', type=lambda v: v.lower() != 'false',
        default=True, metavar='true|false',
        help='Launch trajectory_planner in stage 09 (default: true).',
    )
    parser.add_argument(
        '--goal-tolerance', type=float, default=0.15, metavar='METRES',
        help='Distance to goal pose that counts as goal-reached (default: 0.15 m).',
    )
    parser.add_argument(
        '--rosbag', action='store_true',
        help='Record a rosbag of all topics for the duration of the run.')

    # Set custom start and goal positions
    parser.add_argument('--start-x', type=float, default=None)
    parser.add_argument('--start-y', type=float, default=None)
    parser.add_argument('--goal-x',  type=float, default=None)
    parser.add_argument('--goal-y',  type=float, default=None)
    parser.add_argument('--start-yaw', type=float, default=None,
                        metavar='RAD', help='Start yaw in radians (overrides saved data).')
    parser.add_argument('--goal-yaw',  type=float, default=None,
                        metavar='RAD', help='Goal yaw in radians (overrides saved data).')

    args = parser.parse_args()

    if not os.path.isfile(args.config):
        sys.exit(f"ERROR: config not found: {args.config}")

    data_dir = recorded_data_dir(_WORKSPACE_ROOT, args.scan_id)
    try:
        assert_saved_data_exists(data_dir, args.scan_id)
    except FileNotFoundError as exc:
        sys.exit(f"ERROR: {exc}")

    rclpy.init()

    node = _HwTestOrchestrator(args.config)
    node.skip_trajectory_planner = not args.trajectory_planner
    node._scan_data_dir = data_dir
    node._goal_tolerance_m = args.goal_tolerance

    # No map builder / live stream in this test.
    node._online_enabled = False
    node._cfg.setdefault('map_builder', {})['online'] = False

    node._override_start_x = args.start_x
    node._override_start_y = args.start_y
    node._override_goal_x  = args.goal_x
    node._override_goal_y  = args.goal_y
    node._override_start_yaw = args.start_yaw
    node._override_goal_yaw  = args.goal_yaw

    if args.rosbag:
        node._cfg.setdefault('rosbag', {})['enabled'] = True

    node._log.info(
        f"[--scan-id {args.scan_id}] using saved data from {data_dir!r}")
    node._log.info(
        f"[--goal-tolerance] {args.goal_tolerance} m  "
        f"(monitoring {_POSE_TOPIC})")

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True, name='ros-spin')
    spin_thread.start()

    try:
        node.run()
        # Stages complete; stay alive as observer — goal monitor fires via callbacks
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
