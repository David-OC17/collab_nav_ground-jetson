#!/usr/bin/env python3
"""
ArUco Visual Servo Node — ROS 2 Humble
========================================
Replaces map-projection + A* homing with direct image-space visual servoing.
No EKF, no TF, no map — just the camera.

Strategy
--------
When enabled via /aruco_servo/enable:
  1. Read marker pose from /aruco/id_{target_id}/pose (camera frame,
     published by aruco_goal_detector using solvePnP).

  2. Compute normalised horizontal pixel error:
       err_x = (tx / tz) / tan(hfov/2)   ∈ [-1, +1]
     Positive = marker is to the RIGHT of image centre.
     tx = horizontal offset in camera frame, tz = depth.

  3. Control law (runs at update_rate Hz):
       angular.z = -Kw * err_x
         → turns robot to keep marker centred horizontally

       linear.x  =  Kv * (depth - stop_dist) * centering_factor
         → drives forward proportional to remaining distance,
           scaled to zero when marker is off-centre so the robot
           centres FIRST then advances (avoids driving past it)

       centering_factor = 0  when |err_x| > centering_threshold
       centering_factor = 1  when |err_x| ≤ centering_threshold

  4. Publish twist on /amr/reference (nav_msgs/Odometry) — same topic
     the spline follower uses, so the AMR controller needs no changes.

  5. When detection is lost for > timeout_sec, publish zero velocity
     and signal /aruco_servo/active = False so explorer_controller
     can fall back to frontier exploration.

  6. When depth ≤ stop_dist_m, publish zero and signal active = False
     (goal reached).

Integration with explorer_controller
-------------------------------------
  explorer_controller publishes /aruco_servo/enable = True on HOMING.
  This node takes over /amr/reference immediately.
  When done or lost, it publishes /aruco_servo/active = False and
  explorer_controller transitions to DONE or EXPLORING.

Subscribes:
  /aruco/id_{target_id}/pose  (PoseWithCovarianceStamped) — aruco_goal_detector
  /aruco_servo/enable         (std_msgs/Bool)             — explorer_controller

Publishes:
  /amr/reference              (nav_msgs/Odometry)         — velocity commands
  /aruco_servo/active         (std_msgs/Bool)             — True while servoing

Parameters:
  target_marker_id    0
  stop_dist_m         0.50     m    — stop this far from the marker
  Kw                  0.60          — angular gain [rad/s per normalised pixel]
  Kv                  0.40          — linear gain  [m/s per metre]
  max_linear          0.25     m/s  — forward speed cap
  max_angular         0.60     rad/s
  centering_threshold 0.10          — |err_x| below which forward motion starts
  timeout_sec         2.0      s    — detection-lost timeout before giving up
  world_frame         'world'
  robot_base_frame    'base_footprint'
  update_rate         20.0     Hz
  tan_hfov_half       0.693         — tan(hfov/2) for D435i at 69.4° ≈ tan(34.7°)
                                      Override if using a different camera.
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool


class ArucoVisualServo(Node):

    def __init__(self):
        super().__init__('aruco_visual_servo')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('target_marker_id',    5)
        self.declare_parameter('stop_dist_m',         0.50)
        self.declare_parameter('Kw',                  0.60)
        self.declare_parameter('Kv',                  0.40)
        self.declare_parameter('max_linear',          0.25)
        self.declare_parameter('max_angular',         0.60)
        self.declare_parameter('centering_threshold', 0.10)
        self.declare_parameter('timeout_sec',         2.0)
        self.declare_parameter('world_frame',         'world')
        self.declare_parameter('robot_base_frame',    'base_footprint')
        self.declare_parameter('update_rate',         20.0)
        self.declare_parameter('tan_hfov_half',       0.693)

        self.target_id    = int(self.get_parameter('target_marker_id').value)
        self.stop_dist    = float(self.get_parameter('stop_dist_m').value)
        self.Kw           = float(self.get_parameter('Kw').value)
        self.Kv           = float(self.get_parameter('Kv').value)
        self.max_linear   = float(self.get_parameter('max_linear').value)
        self.max_angular  = float(self.get_parameter('max_angular').value)
        self.cx_thresh    = float(self.get_parameter('centering_threshold').value)
        self.timeout_sec  = float(self.get_parameter('timeout_sec').value)
        self.world_frame  = self.get_parameter('world_frame').value
        self.base_frame   = self.get_parameter('robot_base_frame').value
        self.tan_hfov     = float(self.get_parameter('tan_hfov_half').value)
        dt                = 1.0 / float(self.get_parameter('update_rate').value)

        # ── State ─────────────────────────────────────────────────────────
        self._enabled          = False
        self._last_detection   = None    # (err_x, depth)
        self._last_detect_time = 0.0
        self._done             = False

        # ── QoS ───────────────────────────────────────────────────────────
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

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            PoseWithCovarianceStamped,
            f'/aruco/id_{self.target_id}/pose',
            self._detection_cb,
            reliable_qos
        )
        self.create_subscription(
            Bool,
            '/aruco_servo/enable',
            self._enable_cb,
            reliable_qos
        )

        # ── Publishers ────────────────────────────────────────────────────
        self._ref_pub = self.create_publisher(
            Odometry, '/amr/reference', reliable_qos)

        self._active_pub = self.create_publisher(
            Bool, '/aruco_servo/active', latched_qos)

        # ── Control timer ─────────────────────────────────────────────────
        self.create_timer(dt, self._control_loop)

        self.get_logger().info(
            f'ArucoVisualServo ready\n'
            f'  target_id    = {self.target_id}\n'
            f'  stop_dist    = {self.stop_dist} m\n'
            f'  Kw={self.Kw}  Kv={self.Kv}\n'
            f'  max_linear={self.max_linear} m/s  '
            f'max_angular={self.max_angular} rad/s\n'
            f'  centering_threshold = {self.cx_thresh}\n'
            f'  timeout      = {self.timeout_sec} s\n'
            f'  listening on /aruco/id_{self.target_id}/pose'
        )

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _enable_cb(self, msg: Bool):
        if msg.data and not self._enabled:
            self.get_logger().info(
                'Visual servo ENABLED — taking over /amr/reference')
            self._done = False
        elif not msg.data and self._enabled:
            self.get_logger().info('Visual servo DISABLED')
            self._publish_zero()
        self._enabled = msg.data

    def _detection_cb(self, msg: PoseWithCovarianceStamped):
        """
        aruco_goal_detector publishes in camera_color_optical_frame:
          x = horizontal offset (+ = marker right of camera centre)
          y = vertical offset   (unused for ground robot)
          z = depth             (distance along optical axis)

        Normalised horizontal angle error (independent of focal length):
          err_x = (tx / tz) / tan(hfov/2)   ∈ [-1, +1]
        """
        tx = msg.pose.pose.position.x
        tz = msg.pose.pose.position.z

        if tz < 0.05:
            return   # degenerate / too close

        err_x = (tx / tz) / self.tan_hfov
        err_x = max(-1.0, min(1.0, err_x))

        self._last_detection   = (err_x, tz)
        self._last_detect_time = time.time()

    # ── Control loop ───────────────────────────────────────────────────────

    def _control_loop(self):
        if not self._enabled or self._done:
            return

        elapsed = time.time() - self._last_detect_time

        # Detection lost
        if self._last_detection is None or elapsed > self.timeout_sec:
            self.get_logger().warn(
                f'ArUco id_{self.target_id} not seen for {elapsed:.1f}s — stopping.',
                throttle_duration_sec=1.0)
            self._publish_zero()
            self._publish_active(False)
            return

        err_x, depth = self._last_detection
        self._publish_active(True)

        # ── Goal reached ──────────────────────────────────────────────────
        if depth <= self.stop_dist:
            self.get_logger().info(
                f'ArUco id_{self.target_id} reached! '
                f'depth={depth:.3f} m ≤ stop_dist={self.stop_dist} m')
            self._publish_zero()
            self._publish_active(False)
            self._done = True
            return

        # ── Control law ───────────────────────────────────────────────────
        # Turn to centre the marker
        angular_z = -self.Kw * err_x
        angular_z = max(-self.max_angular, min(self.max_angular, angular_z))

        # Drive forward only when marker is roughly centred
        if abs(err_x) > self.cx_thresh:
            centering_factor = 0.0   # rotate first
        else:
            centering_factor = 1.0   # drive forward

        linear_x = self.Kv * (depth - self.stop_dist) * centering_factor
        linear_x = max(0.0, min(self.max_linear, linear_x))

        self.get_logger().info(
            f'Servoing → err_x={err_x:+.3f}  depth={depth:.3f}m  '
            f'lin={linear_x:.3f}  ang={angular_z:+.3f}',
            throttle_duration_sec=0.5)

        self._publish_twist(linear_x, angular_z)

    # ── Publishers ─────────────────────────────────────────────────────────

    def _publish_twist(self, linear_x: float, angular_z: float):
        msg = Odometry()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame
        msg.child_frame_id  = self.base_frame
        msg.twist.twist.linear.x  = linear_x
        msg.twist.twist.linear.y  = 0.0
        msg.twist.twist.angular.z = angular_z
        self._ref_pub.publish(msg)

    def _publish_zero(self):
        self._publish_twist(0.0, 0.0)

    def _publish_active(self, active: bool):
        msg = Bool()
        msg.data = active
        self._active_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoVisualServo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()