#!/usr/bin/env python3
"""
2D LiDAR Odometry via point-to-point ICP — publishes BOTH pose and velocity.

Subscribes:
    /scan                       (sensor_msgs/LaserScan)

Publishes:
    /amr/lidar_odom             (nav_msgs/Odometry)         - accumulated pose
    /amr/lidar_vel              (geometry_msgs/TwistStamped) - instantaneous velocity

Why publish both:
    Pose is the integral of velocity. Scan-to-scan ICP velocity is bounded-error
    (each measurement uses only two consecutive scans, no accumulation), while
    accumulated pose drifts unboundedly without loop closure. By publishing both,
    you can plot them and *demonstrate* the drift behavior empirically:

      - lidar_odom.pose vs slam_toolbox pose  →  shows accumulated drift
      - lidar_vel.twist vs gyro / wheel-vel   →  shows instantaneous accuracy

    Use velocity to feed your EKF. Use pose for visualization and comparison.

Algorithm (point-to-point ICP, Besl & McKay 1992; SVD step from Arun et al. 1987):
    For each new scan:
      1. LaserScan → 2D points (filter invalid, downsample).
      2. ICP against previous scan to estimate relative motion (Δx, Δy, Δθ).
      3. Compute velocity = relative motion / dt   (BOUNDED ERROR, fresh per scan).
      4. Accumulate pose by composing relative motion (DRIFTS over time).
      5. Publish both.

ICP inner loop:
      a. Apply current transform guess (R, t) to source points.
      b. KD-tree nearest neighbour: each source point → closest target point.
      c. Reject pairs with distance > max_correspondence_dist (outliers).
      d. SVD on cross-covariance → optimal incremental rotation (proper, det=+1).
      e. Optimal incremental translation from centroids.
      f. Compose, check convergence, repeat or break.

Limitations (acknowledge in thesis):
    - No motion prior (initialised from identity each step). Fast motion may
      exceed ICP's basin of convergence.
    - No degeneracy detection: in a long featureless corridor, ICP slides freely
      along the unobservable axis.
    - Covariance is a crude function of residual error, not a principled
      derivation (e.g. Censi 2007 closed-form ICP covariance is the proper fix).
    - Scan-to-scan (not scan-to-keyframe): errors compound faster than necessary.
"""

import math
import numpy as np
from scipy.spatial import cKDTree

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, TransformStamped
from tf2_ros import TransformBroadcaster

from ros2_security import SecureNodeMixin


class LidarOdometryNode(SecureNodeMixin, Node):

    def __init__(self) -> None:
        super().__init__('lidar_odometry_node')
        self.declare_parameter('certs_dir', './certs')
        self.security_init(certs_dir=self.get_parameter('certs_dir').value)

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('scan_topic',     '/scan')
        self.declare_parameter('odom_topic',     '/amr/lidar_odom')
        self.declare_parameter('vel_topic',      '/amr/lidar_vel')
        self.declare_parameter('odom_frame',     'lidar_odom')
        self.declare_parameter('base_frame',     'base_footprint')
        self.declare_parameter('laser_frame',    'laser')
        self.declare_parameter('publish_tf',     False)

        # ICP tuning
        self.declare_parameter('max_iters',                30)
        self.declare_parameter('tolerance',                1e-4)
        self.declare_parameter('max_correspondence_dist',  0.5)   # metres
        self.declare_parameter('subsample_step',           2)     # keep every Nth
        self.declare_parameter('min_range',                0.05)
        self.declare_parameter('max_range',                12.0)
        self.declare_parameter('min_valid_pairs',          20)

        self._scan_topic = self.get_parameter('scan_topic').value
        self._odom_topic = self.get_parameter('odom_topic').value
        self._vel_topic  = self.get_parameter('vel_topic').value
        self._odom_frame = self.get_parameter('odom_frame').value
        self._base_frame = self.get_parameter('base_frame').value
        self._laser_frame = self.get_parameter('laser_frame').value
        self._publish_tf  = self.get_parameter('publish_tf').value

        self._max_iters  = self.get_parameter('max_iters').value
        self._tol        = self.get_parameter('tolerance').value
        self._max_corr   = self.get_parameter('max_correspondence_dist').value
        self._sub_step   = self.get_parameter('subsample_step').value
        self._min_range  = self.get_parameter('min_range').value
        self._max_range  = self.get_parameter('max_range').value
        self._min_pairs  = self.get_parameter('min_valid_pairs').value

        # ── Accumulated pose (drifts; published for visualization/comparison) ──
        self._x  = 0.0
        self._y  = 0.0
        self._th = 0.0

        # ── State for scan-to-scan matching ───────────────────────────
        self._prev_points: np.ndarray | None = None
        self._prev_stamp:  Time | None       = None

        # ── ROS interfaces ────────────────────────────────────────────
        self._odom_pub = self.create_secure_publisher(self._odom_topic, Odometry, 50)
        self._vel_pub  = self.create_secure_publisher(self._vel_topic, TwistStamped, 50)
        self._tf_br    = TransformBroadcaster(self) if self._publish_tf else None

        self.create_secure_subscription(self._scan_topic, LaserScan, self._scan_cb, min_level=None, qos=10)

        self.get_logger().info(
            f'LiDAR odometry ready | scan={self._scan_topic} | '
            f'odom={self._odom_topic} | vel={self._vel_topic} | '
            f'max_iters={self._max_iters} | max_corr={self._max_corr} m'
        )

    # ── LaserScan → Nx2 point cloud (laser frame) ─────────────────────

    def _scan_to_points(self, scan: LaserScan) -> np.ndarray:
        """Convert a LaserScan to an (N, 2) array of (x, y) in the laser frame."""
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        angles = scan.angle_min + np.arange(ranges.size) * scan.angle_increment

        valid = (
            np.isfinite(ranges)
            & (ranges >= max(scan.range_min, self._min_range))
            & (ranges <= min(scan.range_max, self._max_range))
        )
        ranges = ranges[valid]
        angles = angles[valid]

        pts = np.stack([ranges * np.cos(angles),
                        ranges * np.sin(angles)], axis=1)

        if self._sub_step > 1:
            pts = pts[::self._sub_step]

        return pts

    # ── ICP core (point-to-point, SVD-based) ──────────────────────────

    def _icp(self, src: np.ndarray, tgt: np.ndarray,
             R: np.ndarray, t: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """
        Align src → tgt. Returns final (R, t, mean_error, iters_used).

        src and tgt are (N, 2) point clouds in their respective laser frames.
        R is (2, 2) rotation, t is (2,) translation.
        """
        tree = cKDTree(tgt)
        prev_err = float('inf')
        mean_err = float('inf')
        it = 0

        for it in range(self._max_iters):
            # Apply current guess
            src_T = src @ R.T + t

            # Nearest neighbours
            dists, idx = tree.query(src_T, k=1)

            # Outlier rejection
            mask = dists < self._max_corr
            if mask.sum() < self._min_pairs:
                self.get_logger().warn(
                    f'ICP: only {mask.sum()} valid pairs (need {self._min_pairs}); '
                    f'aborting at iter {it}.'
                )
                break

            P = src_T[mask]
            Q = tgt[idx[mask]]

            # Centroids and centered clouds
            P_mean = P.mean(axis=0)
            Q_mean = Q.mean(axis=0)
            P_c = P - P_mean
            Q_c = Q - Q_mean

            # Cross-covariance → SVD → proper rotation
            H = P_c.T @ Q_c
            U, _, Vt = np.linalg.svd(H)
            d = np.sign(np.linalg.det(Vt.T @ U.T))   # +1 or -1
            D = np.diag([1.0, d])
            R_step = Vt.T @ D @ U.T
            t_step = Q_mean - R_step @ P_mean

            # Compose
            R = R_step @ R
            t = R_step @ t + t_step

            # Convergence on residual
            mean_err = float(dists[mask].mean())
            if abs(prev_err - mean_err) < self._tol:
                break
            prev_err = mean_err

        return R, t, mean_err, it + 1

    # ── Scan callback (main loop) ─────────────────────────────────────

    def _scan_cb(self, scan: LaserScan) -> None:
        points = self._scan_to_points(scan)

        if points.shape[0] < self._min_pairs * 2:
            self.get_logger().warn(
                f'Only {points.shape[0]} valid points in scan; skipping.'
            )
            return

        stamp = Time.from_msg(scan.header.stamp)

        # First scan: stash and exit
        if self._prev_points is None:
            self._prev_points = points
            self._prev_stamp  = stamp
            return

        # Compute dt from actual timestamps (NOT a hardcoded scan rate)
        dt_ns = (stamp.nanoseconds - self._prev_stamp.nanoseconds)
        dt = dt_ns * 1e-9
        if dt <= 0.0 or dt > 1.0:
            self.get_logger().warn(
                f'Suspicious dt={dt:.4f}s between scans; skipping.'
            )
            self._prev_points = points
            self._prev_stamp  = stamp
            return

        # ICP from identity (no motion prior — see Limitations in module docstring)
        R, t, err, iters = self._icp(points, self._prev_points,
                                     np.eye(2), np.zeros(2))

        # (R, t) maps current scan into previous scan's frame:
        #   it expresses the robot's motion in the *previous* laser frame.
        dx  = float(t[0])
        dy  = float(t[1])
        dth = math.atan2(R[1, 0], R[0, 0])

        # ── Velocity (bounded error — safe to feed into EKF) ──────────
        v_lin_x = dx  / dt
        v_lin_y = dy  / dt
        v_ang_z = dth / dt

        # ── Accumulate pose (drifts — for visualization / comparison) ──
        c, s = math.cos(self._th), math.sin(self._th)
        self._x  += c * dx - s * dy
        self._y  += s * dx + c * dy
        self._th  = math.atan2(math.sin(self._th + dth),
                               math.cos(self._th + dth))

        # Bookkeeping for next iteration
        self._prev_points = points
        self._prev_stamp  = stamp

        # ── Publish ────────────────────────────────────────────────────
        self._publish_velocity(scan.header.stamp, v_lin_x, v_lin_y, v_ang_z)
        self._publish_pose(scan.header.stamp, err, dt)

        # Diagnostic log every ~5 seconds (assuming ~10 Hz scans)
        if iters >= self._max_iters:
            self.get_logger().debug(
                f'ICP hit max_iters={self._max_iters}, residual={err:.4f}'
            )

    # ── Publishers ────────────────────────────────────────────────────

    def _publish_velocity(self, stamp, vx, vy, wz) -> None:
        """Publish instantaneous velocity in the base_frame."""
        msg = TwistStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = self._base_frame
        msg.twist.linear.x  = vx
        msg.twist.linear.y  = vy
        msg.twist.angular.z = wz
        self.secure_publish(self._vel_pub, msg)

    def _publish_pose(self, stamp, residual, dt) -> None:
        """Publish accumulated pose as nav_msgs/Odometry."""
        qz = math.sin(self._th * 0.5)
        qw = math.cos(self._th * 0.5)

        msg = Odometry()
        msg.header.stamp     = stamp
        msg.header.frame_id  = self._odom_frame
        msg.child_frame_id   = self._base_frame

        msg.pose.pose.position.x    = self._x
        msg.pose.pose.position.y    = self._y
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        # Crude covariance heuristic, scaled by residual. For thesis-grade work,
        # replace with Censi 2007 closed-form ICP covariance.
        var_xy = max(1e-4, 0.5 * residual ** 2)
        var_th = max(1e-4, residual ** 2)
        msg.pose.covariance[0]  = var_xy   # x
        msg.pose.covariance[7]  = var_xy   # y
        msg.pose.covariance[35] = var_th   # yaw

        # Also fill twist for convenience (same as TwistStamped above)
        # Use the last computed velocity (we don't have access here cleanly;
        # easiest to just leave zeros and rely on /amr/lidar_vel for fusion).
        # If you want twist filled, refactor to pass vx/vy/wz into this method.

        self.secure_publish(self._odom_pub, msg)

        if self._tf_br is not None:
            tf = TransformStamped()
            tf.header.stamp    = stamp
            tf.header.frame_id = self._odom_frame
            tf.child_frame_id  = self._base_frame
            tf.transform.translation.x = self._x
            tf.transform.translation.y = self._y
            tf.transform.rotation.z    = qz
            tf.transform.rotation.w    = qw
            self._tf_br.sendTransform(tf)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
