#!/usr/bin/env python3
"""
Spline Follower Node for ROS2
=====================================
Receives a nav_msgs/Path from astar_planner2, fits a scipy CubicSpline
through the waypoints (parameterised by arc-length), and publishes smooth
position + velocity references at a fixed rate using a trapezoidal speed profile.

Subscribes:
  - /trajectory_planner2/path  (nav_msgs/Path)     — A* waypoints

Publishes:
  - /amr/reference             (nav_msgs/Odometry) — position + velocity in robot frame

Speed profile — trapezoidal:
  ┌──────────────────────────────────────────────┐
  │  v                                           │
  │  ^   ___________________                     │
  │  |  /                   \                    │
  │  | /                     \                   │
  │  |/                       \                  │
  │  0──────────────────────────→ s (arc-length) │
  │    accel      cruise      decel               │
  └──────────────────────────────────────────────┘
  If path is too short to reach max_speed, a triangular profile is used.

Velocity frame — robot (base_link):
  Yaw is taken from the spline tangent direction at every tick.
  Since the reference heading always aligns with the path tangent:
    twist.linear.x  = v(s)         ← forward speed
    twist.linear.y  = 0.0          ← no lateral slip (tangent-aligned)
    twist.angular.z = κ(s) × v(s)  ← curvature × speed

Parameters:
  path_topic       '/trajectory_planner2/path'
  map_frame        'map'
  robot_base_frame 'base_link'
  max_speed        0.30  m/s
  max_accel        0.20  m/s²
  goal_tolerance   0.10  m
  update_rate      20.0  Hz
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Path, Odometry

try:
    from scipy.interpolate import CubicSpline
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


class SplineFollower(Node):

    def __init__(self):
        super().__init__('spline_follower')

        if not _SCIPY_OK:
            self.get_logger().error(
                'scipy is not installed. Run: pip install scipy')

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter('path_topic',       '/trajectory_planner2/path')
        self.declare_parameter('map_frame',        'map')
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('max_speed',        0.30)   # m/s
        self.declare_parameter('max_accel',        0.20)   # m/s²
        self.declare_parameter('goal_tolerance',   0.10)   # m
        self.declare_parameter('update_rate',      20.0)   # Hz

        self.path_topic       = self.get_parameter('path_topic').value
        self.map_frame        = self.get_parameter('map_frame').value
        self.robot_base_frame = self.get_parameter('robot_base_frame').value
        self.max_speed        = float(self.get_parameter('max_speed').value)
        self.max_accel        = float(self.get_parameter('max_accel').value)
        self.goal_tolerance   = float(self.get_parameter('goal_tolerance').value)
        self.update_rate      = float(self.get_parameter('update_rate').value)

        self.dt = 1.0 / self.update_rate

        # ------------------------------------------------------------------
        # Spline state
        # ------------------------------------------------------------------
        self.cs_x        = None   # CubicSpline: x as function of arc-length s
        self.cs_y        = None   # CubicSpline: y as function of arc-length s
        self.total_len   = 0.0    # total arc-length of the current path [m]
        self.s_current   = 0.0    # current arc-length position along spline [m]
        self.goal_reached = True  # start idle — no path yet

        # Trapezoidal profile breakpoints (recomputed on each new path)
        self.s_accel_end   = 0.0  # arc-length where acceleration ends / cruise begins
        self.s_decel_start = 0.0  # arc-length where deceleration begins
        self.peak_speed    = 0.0  # actual cruise speed (≤ max_speed)

        # ------------------------------------------------------------------
        # QoS
        # ------------------------------------------------------------------
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ------------------------------------------------------------------
        # Subscriber / Publisher
        # ------------------------------------------------------------------
        self.path_sub = self.create_subscription(
            Path, self.path_topic, self._path_callback, reliable_qos)

        self.ref_pub = self.create_publisher(
            Odometry, '/amr/reference', reliable_qos)

        # ------------------------------------------------------------------
        # Control timer
        # ------------------------------------------------------------------
        self.timer = self.create_timer(self.dt, self._update)

        self.get_logger().info(
            f'SplineFollower ready\n'
            f'  path    → {self.path_topic}\n'
            f'  out     → /amr/reference\n'
            f'  speed   = {self.max_speed} m/s  accel = {self.max_accel} m/s²\n'
            f'  rate    = {self.update_rate} Hz  goal_tol = {self.goal_tolerance} m'
        )

    # ==========================================================================
    # Path callback — builds spline from new A* waypoints
    # ==========================================================================

    def _path_callback(self, msg: Path):
        """
        Receives a new A* path, fits a CubicSpline through the waypoints
        parameterised by cumulative chord-length, and resets execution to t=0.

        Any in-progress trajectory is discarded and replaced immediately.
        """
        if not _SCIPY_OK:
            self.get_logger().error('scipy unavailable — cannot build spline.')
            return

        if len(msg.poses) < 2:
            self.get_logger().warn('Path has < 2 poses — ignoring.')
            return

        xs = np.array([p.pose.position.x for p in msg.poses], dtype=float)
        ys = np.array([p.pose.position.y for p in msg.poses], dtype=float)

        # ── Arc-length parameterisation (chord-length) ─────────────────────
        dists = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        dists = np.where(dists < 1e-9, 1e-9, dists)   # avoid zero-length segments
        s     = np.concatenate([[0.0], np.cumsum(dists)])

        # CubicSpline requires strictly increasing knots — deduplicate
        _, unique_idx = np.unique(s, return_index=True)
        s  = s[unique_idx]
        xs = xs[unique_idx]
        ys = ys[unique_idx]

        if len(s) < 2:
            self.get_logger().warn(
                'Path deduplication left < 2 unique knots — ignoring.')
            return

        # ── Fit splines ────────────────────────────────────────────────────
        # bc_type='natural': second derivative = 0 at endpoints (no artificial curl)
        self.cs_x      = CubicSpline(s, xs, bc_type='natural')
        self.cs_y      = CubicSpline(s, ys, bc_type='natural')
        self.total_len = float(s[-1])

        # ── Trapezoidal profile ────────────────────────────────────────────
        self._build_trapezoid()

        # ── Reset execution ────────────────────────────────────────────────
        self.s_current  = 0.0
        self.goal_reached = False

        self.get_logger().info(
            f'New spline built: {len(s)} knots | '
            f'length = {self.total_len:.2f} m | '
            f'peak_speed = {self.peak_speed:.2f} m/s | '
            f'accel phase: 0 → {self.s_accel_end:.2f} m | '
            f'decel phase: {self.s_decel_start:.2f} → {self.total_len:.2f} m'
        )

    # ==========================================================================
    # Trapezoidal speed profile
    # ==========================================================================

    def _build_trapezoid(self):
        """
        Computes arc-length breakpoints for the trapezoidal profile.

        Ramp distance formula (from kinematics):  d = v² / (2a)

        Two cases:
          Normal  : 2·d_ramp < L  →  full trapezoid (accel + cruise + decel)
          Triangle: 2·d_ramp ≥ L  →  path too short, peak_speed < max_speed
                     peak = sqrt(a·L),  d_ramp = L/2
        """
        L      = self.total_len
        d_ramp = self.max_speed**2 / (2.0 * self.max_accel)

        if 2.0 * d_ramp >= L:
            # Triangular profile — can't reach max_speed
            self.peak_speed    = math.sqrt(self.max_accel * L)
            d_ramp             = L / 2.0
        else:
            self.peak_speed = self.max_speed

        self.s_accel_end   = d_ramp
        self.s_decel_start = L - d_ramp

    def _speed_at(self, s: float) -> float:
        """
        Returns the commanded speed [m/s] at arc-length s.

        Acceleration phase : v = sqrt(2 · a · s)
        Cruise phase       : v = peak_speed
        Deceleration phase : v = sqrt(2 · a · (L - s))

        A small floor (1e-4 m/s) prevents the sqrt from returning 0
        at the very start/end and causing division issues.
        """
        s   = float(np.clip(s, 0.0, self.total_len))
        eps = 1e-4   # minimum speed floor [m/s]

        if s <= self.s_accel_end:
            return max(math.sqrt(2.0 * self.max_accel * s), eps)
        elif s >= self.s_decel_start:
            remaining = self.total_len - s
            return max(math.sqrt(2.0 * self.max_accel * remaining), eps)
        else:
            return self.peak_speed

    # ==========================================================================
    # Control loop — runs at update_rate Hz
    # ==========================================================================

    def _update(self):
        """
        Each tick:
          1. Evaluate position and tangent of the spline at s_current.
          2. Compute speed from the trapezoidal profile.
          3. Publish Odometry reference (position + velocity in robot frame).
          4. Advance s_current by v·dt for the next tick.

        Step 4 is done AFTER publishing so the reference corresponds exactly
        to the current arc-length, not a look-ahead position.
        """
        # ── Idle: no path or goal already reached ─────────────────────────
        if self.cs_x is None or self.goal_reached:
            self._publish_zero()
            return

        # ── Evaluate spline at current arc-length ─────────────────────────
        s    = float(np.clip(self.s_current, 0.0, self.total_len))

        x    = float(self.cs_x(s))
        y    = float(self.cs_y(s))
        dxds = float(self.cs_x(s, 1))    # first  derivative: dx/ds
        dyds = float(self.cs_y(s, 1))    # first  derivative: dy/ds
        ddx  = float(self.cs_x(s, 2))    # second derivative: d²x/ds²
        ddy  = float(self.cs_y(s, 2))    # second derivative: d²y/ds²

        # ── Yaw from tangent direction ─────────────────────────────────────
        yaw  = math.atan2(dyds, dxds)

        # ── Curvature: κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2) ──────────
        denom = (dxds**2 + dyds**2)**1.5
        kappa = (dxds * ddy - dyds * ddx) / denom if denom > 1e-9 else 0.0

        # ── Speed from trapezoidal profile ─────────────────────────────────
        v     = self._speed_at(s)
        omega = kappa * v

        # ── World-frame velocities (along tangent) ─────────────────────────
        vx_world = dxds * v
        vy_world = dyds * v

        # ── Rotate to robot frame using spline-tangent yaw ─────────────────
        # Since yaw = atan2(dyds, dxds), this rotation always yields:
        #   vx_robot ≈ v  (purely forward)
        #   vy_robot ≈ 0  (no lateral, robot aligned with tangent)
        # Kept explicit for generality and debuggability.
        cos_y    =  math.cos(yaw)
        sin_y    =  math.sin(yaw)
        vx_robot =  cos_y * vx_world + sin_y * vy_world
        vy_robot = -sin_y * vx_world + cos_y * vy_world

        # ── Publish reference ──────────────────────────────────────────────
        self._publish_reference(x, y, yaw, vx_robot, vy_robot, omega)

        # ── Advance arc-length for next tick ──────────────────────────────
        self.s_current += v * self.dt

        # ── Goal check ────────────────────────────────────────────────────
        remaining = self.total_len - self.s_current
        if remaining <= self.goal_tolerance:
            self.goal_reached = True
            self.s_current    = self.total_len
            self.get_logger().info(
                f'Goal reached (remaining={remaining:.3f} m < '
                f'tolerance={self.goal_tolerance} m).')

    # ==========================================================================
    # Publishing
    # ==========================================================================

    def _publish_reference(self, x: float, y: float, yaw: float,
                           vx: float, vy: float, omega: float):
        """
        Publishes a full Odometry reference:
          pose  : position (x, y) + heading (yaw as quaternion)
          twist : linear velocity (vx, vy) in robot frame + angular.z
        """
        msg = Odometry()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.child_frame_id  = self.robot_base_frame

        # Position + orientation
        msg.pose.pose.position.x    = x
        msg.pose.pose.position.y    = y
        msg.pose.pose.position.z    = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        # Velocity in robot frame
        msg.twist.twist.linear.x  = vx
        msg.twist.twist.linear.y  = vy
        msg.twist.twist.linear.z  = 0.0
        msg.twist.twist.angular.z = omega

        self.ref_pub.publish(msg)

    def _publish_zero(self):
        """
        Publishes a zero-velocity reference.
        If the spline exists and goal is reached, uses the final path position.
        Otherwise uses (0, 0, 0).
        """
        if self.cs_x is not None and self.goal_reached:
            # Hold at final position with zero velocity
            s    = self.total_len
            x    = float(self.cs_x(s))
            y    = float(self.cs_y(s))
            dxds = float(self.cs_x(s, 1))
            dyds = float(self.cs_y(s, 1))
            yaw  = math.atan2(dyds, dxds)
            self._publish_reference(x, y, yaw, 0.0, 0.0, 0.0)
        else:
            self._publish_reference(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# ==============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = SplineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
