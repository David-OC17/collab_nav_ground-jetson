#!/usr/bin/env python3
"""
Explorer Controller (Mission Controller) — ROS 2 Humble
=========================================================
Fallback exploration state machine. Activated manually by the operator
when the drone map is unavailable. The robot explores the arena using
frontier exploration on its onboard SLAM map until it detects the target
ArUco marker, then navigates directly to it.

States:
  IDLE       — waiting for operator trigger. No goals published.
               Robot stays still.

  EXPLORING  — FrontierExplorer is active. Goals from /frontier/goal are
               forwarded to /aruco/goal/pose (consumed by AStarPlanner2).
               Camera continuously scans for the target ArUco marker.

  HOMING     — Target ArUco detected. FrontierExplorer is silenced.
               Last known ArUco goal forwarded to /aruco/goal/pose.
               If detection is lost → immediately back to EXPLORING.

  DONE       — Robot reached the ArUco goal. Everything stops.
               Terminal state until node is restarted.

Operator triggers:
  Start exploration : ros2 topic pub /mission/start std_msgs/Bool "{data: true}" --once
  Stop / reset      : ros2 topic pub /mission/start std_msgs/Bool "{data: false}" --once

Subscribes:
  - /mission/start    (std_msgs/Bool)                              — operator trigger
  - /frontier/goal    (geometry_msgs/PoseWithCovarianceStamped)   — from FrontierExplorer
  - /aruco/detection  (geometry_msgs/PoseWithCovarianceStamped)   — from ArUcoDetector
  - /follower/pose    (geometry_msgs/PoseWithCovarianceStamped)   — robot pose

Publishes:
  - /aruco/goal/pose          (geometry_msgs/PoseWithCovarianceStamped) — to AStarPlanner2
  - /frontier_explorer/active (std_msgs/Bool)                           — silences FrontierExplorer
  - /mission/status_marker    (visualization_msgs/Marker)               — RViz text overlay
  - /mission/state            (std_msgs/String)                         — current state name

Parameters:
  target_marker_id       0       — ArUco marker ID that triggers HOMING
  goal_reached_dist      0.35    m  — distance to ArUco goal that triggers DONE
  detection_timeout_sec  1.0     s  — seconds without detection → HOMING back to EXPLORING
  world_frame            'map'
  update_rate            10.0    Hz
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                        DurabilityPolicy, HistoryPolicy)

from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

from ros2_security import SecureNodeMixin


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------
class State:
    IDLE      = 'IDLE'
    EXPLORING = 'EXPLORING'
    HOMING    = 'HOMING'
    DONE      = 'DONE'


class ExplorerController(SecureNodeMixin, Node):

    def __init__(self):
        super().__init__('explorer_controller')
        self.declare_parameter('certs_dir', './certs')
        self.security_init(certs_dir=self.get_parameter('certs_dir').value)

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter('target_marker_id',      0)
        self.declare_parameter('goal_reached_dist',     0.35)
        self.declare_parameter('detection_timeout_sec', 1.0)
        self.declare_parameter('world_frame',           'map')
        self.declare_parameter('update_rate',           10.0)

        self.target_marker_id      = int(self.get_parameter('target_marker_id').value)
        self.goal_reached_dist     = float(self.get_parameter('goal_reached_dist').value)
        self.detection_timeout_sec = float(self.get_parameter('detection_timeout_sec').value)
        self.world_frame           = self.get_parameter('world_frame').value
        self.update_rate           = float(self.get_parameter('update_rate').value)

        # ------------------------------------------------------------------
        # State machine
        # ------------------------------------------------------------------
        self.state = State.IDLE

        self.frontier_goal: PoseWithCovarianceStamped | None = None
        self.aruco_goal:    PoseWithCovarianceStamped | None = None
        self.last_detection_time: float = 0.0

        self.robot_x: float = 0.0
        self.robot_y: float = 0.0
        self.pose_received  = False

        self._last_forwarded_goal = None

        # Progress watchdog — re-send goal if robot hasn't moved in N seconds
        self._last_progress_pos:  tuple | None = None
        self._last_progress_time: float        = 0.0
        self._progress_timeout_sec: float      = 4.0   # re-send if stuck this long
        self._progress_min_dist:    float      = 0.05  # m — min movement to count

        # ------------------------------------------------------------------
        # QoS
        # ------------------------------------------------------------------
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        volatile_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,   # ← never replays to late joiners
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ------------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------------
        self.start_sub = self.create_subscription(
            Bool, '/mission/start',
            self._start_callback, volatile_qos)

        self.frontier_sub = self.create_secure_subscription(
            '/frontier/goal', PoseWithCovarianceStamped,
            self._frontier_goal_callback, min_level=None, qos=reliable_qos)

        self.aruco_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/aruco/detection',
            self._aruco_detection_callback, reliable_qos)

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/follower/pose',
            self._pose_callback, reliable_qos)

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        self.goal_pub = self.create_secure_publisher('/aruco/goal/pose', PoseWithCovarianceStamped, latched_qos)

        self.active_pub = self.create_secure_publisher('/frontier_explorer/active', Bool, reliable_qos)

        self.status_marker_pub = self.create_secure_publisher('/mission/status_marker', Marker, reliable_qos)

        self.state_pub = self.create_secure_publisher('/mission/state', String, reliable_qos)

        # ------------------------------------------------------------------
        # Startup — make sure frontier explorer starts silent
        # ------------------------------------------------------------------
        self._set_frontier_explorer_active(False)

        # ------------------------------------------------------------------
        # Timer
        # ------------------------------------------------------------------
        self.create_timer(1.0 / self.update_rate, self._tick)

        self.get_logger().info(
            f'ExplorerController ready\n'
            f'  target_marker_id      = {self.target_marker_id}\n'
            f'  goal_reached_dist     = {self.goal_reached_dist} m\n'
            f'  detection_timeout_sec = {self.detection_timeout_sec} s\n'
            f'  update_rate           = {self.update_rate} Hz\n'
            f'  State                 → {self.state}\n'
            f'  To start: ros2 topic pub /mission/start std_msgs/msg/Bool '
            f'"{{data: true}}" --once\'')

    # ==========================================================================
    # Callbacks
    # ==========================================================================

    def _start_callback(self, msg: Bool):
        if msg.data:
            if self.state == State.IDLE:
                self._transition_to(State.EXPLORING)
            elif self.state == State.DONE:
                self.get_logger().warn(
                    'Mission already DONE. Restart the node to run again.')
            else:
                self.get_logger().info(
                    f'Already running in state {self.state} — ignoring start.')
        else:
            # data=false → operator reset back to IDLE
            if self.state != State.DONE:
                self.get_logger().info('Operator reset → returning to IDLE.')
                self._transition_to(State.IDLE)

    def _frontier_goal_callback(self, msg: PoseWithCovarianceStamped):
        self.frontier_goal        = msg
        self._frontier_goal_dirty = True
        self._last_progress_pos   = None   # reset watchdog for new goal

    def _aruco_detection_callback(self, msg: PoseWithCovarianceStamped):
        # Marker ID is stored as float in covariance[0] by aruco_goal_detector.
        # The pose is already in world_frame — ready to forward to A*.
        detected_id = int(round(msg.pose.covariance[0]))

        if detected_id != self.target_marker_id:
            self.get_logger().debug(
                f'Detected marker {detected_id} ≠ target '
                f'{self.target_marker_id} — ignoring.')
            return

        self.aruco_goal          = msg
        self.last_detection_time = time.time()

        if self.state == State.EXPLORING:
            self.get_logger().info(
                f'Target marker {self.target_marker_id} detected at '
                f'({msg.pose.pose.position.x:.2f}, {msg.pose.pose.position.y:.2f}) '
                f'in world frame → HOMING')
            self._transition_to(State.HOMING)

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        self.robot_x      = msg.pose.pose.position.x
        self.robot_y      = msg.pose.pose.position.y
        self.pose_received = True

    # ==========================================================================
    # State machine tick
    # ==========================================================================

    def _tick(self):
        if self.state == State.IDLE:
            self._tick_idle()
        elif self.state == State.EXPLORING:
            self._tick_exploring()
        elif self.state == State.HOMING:
            self._tick_homing()
        elif self.state == State.DONE:
            self._tick_done()

        self._publish_status_marker()
        self._publish_state()

    def _tick_idle(self):
        self.get_logger().info(
            'IDLE — waiting for operator start signal.\n'
            '  Run: ros2 topic pub /mission/start '
            'std_msgs/msg/Bool "{data: true}" --once',
            throttle_duration_sec=10.0)

    def _tick_exploring(self):
        if self.frontier_goal is None:
            self.get_logger().info(
                'EXPLORING — waiting for first frontier goal…',
                throttle_duration_sec=5.0)
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        # ── Progress watchdog ────────────────────────────────────────────
        # If the robot hasn't moved _progress_min_dist in _progress_timeout_sec,
        # re-send the goal — A* may have missed it or the spline stalled.
        if self.pose_received:
            pos = (self.robot_x, self.robot_y)
            if self._last_progress_pos is None:
                self._last_progress_pos  = pos
                self._last_progress_time = now
            else:
                moved = math.hypot(
                    pos[0] - self._last_progress_pos[0],
                    pos[1] - self._last_progress_pos[1])
                if moved >= self._progress_min_dist:
                    self._last_progress_pos  = pos
                    self._last_progress_time = now
                elif now - self._last_progress_time > self._progress_timeout_sec:
                    self.get_logger().warn(
                        f'No progress for {self._progress_timeout_sec}s — '
                        f're-sending goal to A*')
                    self._frontier_goal_dirty = True
                    self._last_progress_time  = now   # reset to avoid spam

        # ── Forward goal to A* (only when dirty) ────────────────────────
        if not getattr(self, '_frontier_goal_dirty', True):
            return

        self._frontier_goal_dirty = False
        self.secure_publish(self.goal_pub, self.frontier_goal)
        self.get_logger().info(
            f'EXPLORING — forwarding frontier goal '
            f'({self.frontier_goal.pose.pose.position.x:.2f}, '
            f'{self.frontier_goal.pose.pose.position.y:.2f})')

    def _tick_homing(self):
        # Detection lost → resume exploring immediately
        elapsed = time.time() - self.last_detection_time
        if elapsed > self.detection_timeout_sec:
            self.get_logger().warn(
                f'ArUco detection lost ({elapsed:.1f}s) — resuming exploration.')
            self._transition_to(State.EXPLORING)
            return

        # Forward last known ArUco goal
        if self.aruco_goal is not None:
            self.secure_publish(self.goal_pub, self.aruco_goal)

        # Check if robot reached the goal
        if self.pose_received and self.aruco_goal is not None:
            gx   = self.aruco_goal.pose.pose.position.x
            gy   = self.aruco_goal.pose.pose.position.y
            dist = math.hypot(self.robot_x - gx, self.robot_y - gy)

            self.get_logger().info(
                f'HOMING — dist to ArUco goal: {dist:.3f} m',
                throttle_duration_sec=1.0)

            if dist <= self.goal_reached_dist:
                self._transition_to(State.DONE)

    def _tick_done(self):
        self.get_logger().info(
            f'DONE — marker {self.target_marker_id} reached. '
            f'Restart node to run again.',
            throttle_duration_sec=10.0)

    # ==========================================================================
    # Transitions
    # ==========================================================================

    def _transition_to(self, new_state: str):
        old_state  = self.state
        self.state = new_state
        self.get_logger().info(f'State: {old_state} → {new_state}')

        if new_state == State.IDLE:
            self._set_frontier_explorer_active(False)
            self.frontier_goal = None   # discard stale goal on reset

        elif new_state == State.EXPLORING:
            self._set_frontier_explorer_active(True)

        elif new_state == State.HOMING:
            self._set_frontier_explorer_active(False)

        elif new_state == State.DONE:
            self._set_frontier_explorer_active(False)
            self.get_logger().info(
                f'Mission complete — target marker {self.target_marker_id} reached.')

    # ==========================================================================
    # Helpers
    # ==========================================================================

    def _set_frontier_explorer_active(self, active: bool):
        msg      = Bool()
        msg.data = active
        self.secure_publish(self.active_pub, msg)
        self.get_logger().info(
            f'FrontierExplorer {"ACTIVATED" if active else "DEACTIVATED"}')

    def _publish_state(self):
        msg      = String()
        msg.data = self.state
        self.secure_publish(self.state_pub, msg)

    def _publish_status_marker(self):
        colors = {
            State.IDLE:      ColorRGBA(r=0.6, g=0.6, b=0.6, a=1.0),  # grey
            State.EXPLORING: ColorRGBA(r=0.2, g=0.8, b=1.0, a=1.0),  # cyan
            State.HOMING:    ColorRGBA(r=1.0, g=0.6, b=0.0, a=1.0),  # orange
            State.DONE:      ColorRGBA(r=0.2, g=1.0, b=0.2, a=1.0),  # green
        }
        labels = {
            State.IDLE:      'IDLE — awaiting operator start',
            State.EXPLORING: f'EXPLORING  (target marker {self.target_marker_id})',
            State.HOMING:    f'HOMING  →  marker {self.target_marker_id}',
            State.DONE:      f'DONE  —  marker {self.target_marker_id} reached',
        }

        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = self.world_frame
        m.ns     = 'mission_state'
        m.id     = 0
        m.type   = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x    = 0.0
        m.pose.position.y    = 5.5
        m.pose.position.z    = 0.5
        m.pose.orientation.w = 1.0
        m.scale.z = 0.5
        m.color   = colors.get(self.state, ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0))
        m.text    = labels.get(self.state, self.state)
        self.secure_publish(self.status_marker_pub, m)


# ==============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = ExplorerController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()