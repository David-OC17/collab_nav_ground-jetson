#!/usr/bin/env python3
"""
world_odom_tf_node — Publishes a one-shot static TF world → odom.

At the moment an ArUco detection arrives on /aruco/amr/pose, the robot's
pose in the world frame is known.  The EKF is already running so odom →
base_footprint is live.  We recover world → odom as:

    T_world→odom = T_world→base_footprint  ×  T_odom→base_footprint⁻¹

and latch it as a static transform.  After the first successful publish
the node keeps running but ignores subsequent ArUco messages.
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from tf2_ros import StaticTransformBroadcaster, Buffer, TransformListener, TransformException


class WorldOdomTFNode(Node):

    def __init__(self):
        super().__init__('world_odom_tf_node')

        self.declare_parameter('aruco_topic',  '/aruco/amr/pose')
        self.declare_parameter('world_frame',  'world')
        self.declare_parameter('odom_frame',   'odom')
        self.declare_parameter('base_frame',   'base_footprint')

        self._world_frame = self.get_parameter('world_frame').value
        self._odom_frame  = self.get_parameter('odom_frame').value
        self._base_frame  = self.get_parameter('base_frame').value
        aruco_topic       = self.get_parameter('aruco_topic').value

        self._published    = False
        self._pending_msg  = None   # stores pose until TF is ready
        self._ready_time   = self.get_clock().now().nanoseconds + 2_000_000_000  # 2 s grace

        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._static_br   = StaticTransformBroadcaster(self)

        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self._sub = self.create_subscription(
            PoseWithCovarianceStamped, aruco_topic, self._cb, latched_qos)

        self.create_timer(0.5, self._retry)

        self.get_logger().info(f'Waiting for {aruco_topic}…')

    # ──────────────────────────────────────────────────────────────────────────

    def _cb(self, msg: PoseWithCovarianceStamped):
        if self._published:
            return
        self._pending_msg = msg
        self._try_publish()

    def _retry(self):
        if self._published or self._pending_msg is None:
            return
        if self.get_clock().now().nanoseconds < self._ready_time:
            return   # wait for EKF to publish fresh TF
        self._try_publish()

    def _try_publish(self):
        msg = self._pending_msg

        # ── T_world→base_footprint from ArUco ─────────────────────────────
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw_wb = self._yaw(q.x, q.y, q.z, q.w)

        # ── T_odom→base_footprint from EKF TF ─────────────────────────────
        try:
            tf_ob = self._tf_buffer.lookup_transform(
                self._odom_frame, self._base_frame, rclpy.time.Time())
        except TransformException as e:
            self.get_logger().warn(
                f'TF {self._odom_frame}→{self._base_frame} not ready: {e}',
                throttle_duration_sec=1.0)
            return

        ox     = tf_ob.transform.translation.x
        oy     = tf_ob.transform.translation.y
        yaw_ob = self._yaw(
            tf_ob.transform.rotation.x, tf_ob.transform.rotation.y,
            tf_ob.transform.rotation.z, tf_ob.transform.rotation.w)

        # ── T_world→odom = T_WB × T_OB⁻¹ (2-D SE(2)) ─────────────────────
        # Rotation:    yaw_wo = yaw_wb - yaw_ob
        # Translation: t_wo   = t_wb  - R(yaw_wo) × t_ob
        yaw_wo = math.atan2(
            math.sin(yaw_wb - yaw_ob),
            math.cos(yaw_wb - yaw_ob))

        c = math.cos(yaw_wo)
        s = math.sin(yaw_wo)
        tx = p.x - (c * ox - s * oy)
        ty = p.y - (s * ox + c * oy)

        # ── Publish static TF ──────────────────────────────────────────────
        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = self._world_frame
        t.child_frame_id  = self._odom_frame
        t.transform.translation.x = tx
        t.transform.translation.y = ty
        t.transform.translation.z = 0.0
        t.transform.rotation.x    = 0.0
        t.transform.rotation.y    = 0.0
        t.transform.rotation.z    = math.sin(yaw_wo / 2.0)
        t.transform.rotation.w    = math.cos(yaw_wo / 2.0)

        self._static_br.sendTransform(t)
        self._published = True

        self.get_logger().info(
            f'Published static TF {self._world_frame}→{self._odom_frame}: '
            f'x={tx:.3f} m  y={ty:.3f} m  yaw={math.degrees(yaw_wo):.2f}°')

    @staticmethod
    def _yaw(qx: float, qy: float, qz: float, qw: float) -> float:
        return math.atan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy * qy + qz * qz))


def main(args=None):
    rclpy.init(args=args)
    node = WorldOdomTFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
