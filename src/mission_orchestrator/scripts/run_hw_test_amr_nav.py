#!/usr/bin/env python3
"""
Hardware test: AMR navigation with mocked aruco poses + drone map.

Tests the closed-loop feedback chain without re-running the slow map-builder
and marker-localizer pipeline.  Requires saved data produced by save_scan_data.py.

Stages that run for real:
  07   Ping Raspberry Pi
  08   SSH connect to Raspberry Pi
  09   Start amr_bringup systemd service on Raspberry Pi; verify active
  10   Wait for /imu/data_raw (IMU running)
  12   Launch trajectory_planner
  13   Launch map_fusion

Stages mocked from saved data (recorded_data/scanX/):
  15   Publish /aruco/amr/pose  and /aruco/goal/pose  (from aruco_*.yaml)
  19   Return saved OccupancyGrid (from drone_map.yaml)
  20   Publish /drone/map  (uses grid returned by mocked stage 19)

Stages skipped (no-ops):
  01-06  drone pipeline + video-file wait
  11     video integrity check (no video needed)
  14     oradar lidar
  14b    emergency stop
  16-18  marker-localizer service, aruco republish, map-builder launch

The trajectory planner starts planning as soon as it receives /drone/map and
both aruco poses (all published at stage 15/20 with TRANSIENT_LOCAL QoS).

Pass condition: AMR reaches the goal pose within --goal-tolerance metres.
The script stays alive after goal-reached so you can inspect system state.
Press Ctrl+C to stop the AMR service and exit.

Prerequisites:
    python3 src/mission_orchestrator/scripts/save_scan_data.py --scan-id N

Usage (from workspace root, after sourcing install/setup.bash):
    python3 src/mission_orchestrator/scripts/run_hw_test_amr_nav.py --scan-id 10

Skip trajectory_planner (stage 12):
    python3 src/mission_orchestrator/scripts/run_hw_test_amr_nav.py \\
        --scan-id 10 --trajectory-planner=false

Override goal tolerance (default 0.15 m):
    python3 src/mission_orchestrator/scripts/run_hw_test_amr_nav.py \\
        --scan-id 10 --goal-tolerance 0.20

Record a rosbag of all topics during the run:
    python3 src/mission_orchestrator/scripts/run_hw_test_amr_nav.py \\
        --scan-id 10 --rosbag

With a custom config:
    python3 src/mission_orchestrator/scripts/run_hw_test_amr_nav.py \\
        --scan-id 10 --config /abs/path/to/orchestrator_params.yaml
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

_POSE_TOPIC = '/amr/ekf/pose'   # nav_msgs/Odometry published by the EKF node


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator subclass: drone+video skipped; AMR bringup real; stages 14-20 mocked
# ─────────────────────────────────────────────────────────────────────────────

class _HwTestOrchestrator(MissionOrchestratorNode):
    """AMR bringup (07-10) and planning (12-13) run; everything else no-op or mocked."""

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

    # ── No-op: video-file wait + integrity check (no video needed) ────────────

    def _stage_06_wait_video_files(self) -> None:
        pass

    def _stage_11_verify_video_integrity(self) -> None:
        pass

    # ── Flag-gated: trajectory_planner ───────────────────────────────────────

    def _stage_12_launch_trajectory_planner(self) -> None:
        if self.skip_trajectory_planner:
            self._log.info("  [stage 12] trajectory_planner skipped (--trajectory-planner=false)")
            return
        super()._stage_12_launch_trajectory_planner()

    # ── No-op: oradar + emergency stop ────────────────────────────────────────

    def _stage_14_launch_oradar(self) -> None:
        pass

    def _stage_14b_launch_emergency_stop(self) -> None:
        pass

    # ── Mock stage 15: publish saved aruco poses ──────────────────────────────

    def _stage_15_launch_marker_localizer(self) -> None:
        self._log.info("╔══ Stage 15 [MOCK]: Publish saved aruco poses")
        amr_path = os.path.join(self._scan_data_dir, 'aruco_amr_pose.yaml')
        goal_path = os.path.join(self._scan_data_dir, 'aruco_goal_pose.yaml')

        amr_pose = load_pose(amr_path)
        goal_pose = load_pose(goal_path)

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
        self._log.info("╚══ Stage 15 [MOCK] OK: aruco poses published")

    # ── Mock stage 16: bypass real service call ───────────────────────────────

    def _stage_16_call_localize_markers(self):
        return []

    # ── No-op stage 17: already published in stage 15 ────────────────────────

    def _stage_17_publish_aruco_poses(self, markers) -> None:
        pass

    # ── No-op stage 18: no map-builder server needed ─────────────────────────

    def _stage_18_launch_map_builder(self) -> None:
        pass

    # ── Mock stage 19: load saved drone map ───────────────────────────────────

    def _stage_19_call_map_builder(self):
        self._log.info("╔══ Stage 19 [MOCK]: Load saved drone map")
        grid = load_grid(os.path.join(self._scan_data_dir, 'drone_map.yaml'))
        self._log.info(
            f"  Loaded {grid.info.width}×{grid.info.height} OccupancyGrid "
            f"(res={grid.info.resolution} m/cell)")
        self._log.info("╚══ Stage 19 [MOCK] OK")
        return grid

    # Stage 20 runs normally — publishes the grid returned by mock stage 19.


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Hardware test: AMR navigation loop with mocked map/poses '
                    '(drone pipeline skipped).')
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
        help='Launch trajectory_planner in stage 12 (default: true).',
    )
    parser.add_argument(
        '--goal-tolerance', type=float, default=0.15, metavar='METRES',
        help='Distance to goal pose that counts as goal-reached (default: 0.15 m).',
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
