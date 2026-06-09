#!/usr/bin/env python3
"""
Relays /amr/reference (nav_msgs/Odometry) → /follower/pose
(geometry_msgs/PoseWithCovarianceStamped) so the SLAM map and
TF broadcaster track the robot's actual planned trajectory.

Publishes the start position immediately at startup to seed the SLAM map
reveal, then holds the last known pose at 20 Hz so TF never goes stale.

The /mission/start gate was removed. It caused a DDS discovery race:
`ros2 topic pub --once` exits before all subscribers are discovered, so
odom_to_pose silently missed the message and stayed locked forever, making
the robot appear frozen even after the mission started.

The (x==0, y==0) guard is sufficient: the spline follower's _publish_zero()
idles at the origin while no path is loaded, and any real path waypoint from
A* starts at the robot's actual cell centre (never exactly 0,0).
"""
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool


class OdomToPose(Node):
    def __init__(self):
        super().__init__('odom_to_pose')

        self.declare_parameter('robot_start_x', 0.0)
        self.declare_parameter('robot_start_y', 0.0)

        self.pub = self.create_publisher(
            PoseWithCovarianceStamped, '/follower/pose', 10)

        self.x   = float(self.get_parameter('robot_start_x').value)
        self.y   = float(self.get_parameter('robot_start_y').value)
        self.yaw = 0.0

        volatile_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(
            Odometry, '/amr/reference', self._cb, volatile_qos)

        # Keep /mission/start subscription for reset-to-start-position on data=false.
        # data=true is intentionally ignored — no gate needed (see module docstring).
        self.create_subscription(
            Bool, '/mission/start', self._mission_cb, volatile_qos)

        self._publish(self.get_clock().now().to_msg())
        self.create_timer(0.05, self._publish_hold)

        self.get_logger().info(
            f'OdomToPose: /amr/reference → /follower/pose '
            f'(start={self.x:.2f},{self.y:.2f})')

    def _mission_cb(self, msg: Bool):
        if not msg.data:
            # Operator reset — snap back to declared start position
            self.x   = float(self.get_parameter('robot_start_x').value)
            self.y   = float(self.get_parameter('robot_start_y').value)
            self.yaw = 0.0
            self.get_logger().info('Mission reset — returning to start position.')

    def _cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Ignore spline idle output at origin (published by _publish_zero()
        # when no path is loaded or goal is already reached).
        if x == 0.0 and y == 0.0:
            return

        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self.x = x
        self.y = y

    def _publish_hold(self):
        self._publish(self.get_clock().now().to_msg())

    def _publish(self, stamp):
        out = PoseWithCovarianceStamped()
        out.header.stamp            = stamp
        out.header.frame_id         = 'map'
        out.pose.pose.position.x    = self.x
        out.pose.pose.position.y    = self.y
        out.pose.pose.position.z    = 0.0
        out.pose.pose.orientation.z = math.sin(self.yaw / 2.0)
        out.pose.pose.orientation.w = math.cos(self.yaw / 2.0)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(OdomToPose())


if __name__ == '__main__':
    main()