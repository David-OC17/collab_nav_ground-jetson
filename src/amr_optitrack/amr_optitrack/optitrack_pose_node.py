#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class OptiTrackPoseNode(Node):
    """
    Subscribes to /optitrack/rigid_body (PoseStamped at ~120 Hz).
    Computes body-frame velocity via finite difference + EMA smoothing.
    Publishes pose + velocity to /amr/pose (Odometry).
    """

    def __init__(self):
        super().__init__('optitrack_pose_node')

        self.declare_parameter('vel_alpha', 0.3)
        self._alpha = self.get_parameter('vel_alpha').value

        self._last_msg  = None
        self._last_time = None

        # Filtered velocities
        self._vx = 0.0
        self._vy = 0.0
        self._wz = 0.0

        # Define QoS compatible con BEST_EFFORT
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(
            PoseStamped, '/optitrack/rigid_body', self._cb, qos)

        self._pub = self.create_publisher(Odometry, '/amr/pose', 10)

        self.get_logger().info(
            f'optitrack_pose_node started | vel_alpha={self._alpha}')

    def _cb(self, msg: PoseStamped):
        if msg.header.frame_id != 'AMR':
            return

        stamp = msg.header.stamp
        now   = stamp.sec + stamp.nanosec * 1e-9

        if self._last_msg is None:
            self._last_msg  = msg
            self._last_time = now
            return

        dt = now - self._last_time
        if dt <= 0.0 or dt > 0.5:
            self._last_msg  = msg
            self._last_time = now
            return

        # Current pose
        x   = msg.pose.position.x
        y   = msg.pose.position.y
        yaw = self._quat_to_yaw(msg.pose.orientation)

        # Last pose
        x0  = self._last_msg.pose.position.x
        y0  = self._last_msg.pose.position.y

        # Raw velocity in world frame
        dx_w = (x - x0) / dt
        dy_w = (y - y0) / dt
        dth  = self._wrap(yaw - self._quat_to_yaw(self._last_msg.pose.orientation)) / dt

        # Rotate world-frame velocity → body frame
        c =  math.cos(yaw)
        s =  math.sin(yaw)
        vx_raw =  c * dx_w + s * dy_w
        vy_raw = -s * dx_w + c * dy_w

        # EMA low-pass filter
        a        = self._alpha
        self._vx = a * vx_raw + (1.0 - a) * self._vx
        self._vy = a * vy_raw + (1.0 - a) * self._vy
        self._wz = a * dth   + (1.0 - a) * self._wz

        self._publish(msg, x, y)

        self._last_msg  = msg
        self._last_time = now

    def _publish(self, msg: PoseStamped, x: float, y: float):
        odom = Odometry()
        odom.header.stamp    = msg.header.stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'

        odom.pose.pose.position.x  = x
        odom.pose.pose.position.y  = y
        odom.pose.pose.orientation = msg.pose.orientation

        odom.twist.twist.linear.x  = self._vx
        odom.twist.twist.linear.y  = self._vy
        odom.twist.twist.angular.z = self._wz

        self._pub.publish(odom)

    @staticmethod
    def _quat_to_yaw(q) -> float:
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    @staticmethod
    def _wrap(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))


def main(args=None):
    rclpy.init(args=args)
    node = OptiTrackPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
