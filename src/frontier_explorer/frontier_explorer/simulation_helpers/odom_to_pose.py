#!/usr/bin/env python3
"""
Relays /amr/reference (nav_msgs/Odometry) → /follower/pose
(geometry_msgs/PoseWithCovarianceStamped) so the SLAM map and
TF broadcaster track the robot's actual planned trajectory.

Publishes (0,0) immediately at startup to seed the SLAM map reveal,
then holds the last known pose at 20 Hz so TF never goes stale.
"""
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped


class OdomToPose(Node):
    def __init__(self):
        super().__init__('odom_to_pose')

        self.pub = self.create_publisher(
            PoseWithCovarianceStamped, '/follower/pose', 10)
        self.create_subscription(
            Odometry, '/amr/reference', self._cb, 10)

        self.x    = 0.0
        self.y    = 0.0
        self.yaw  = 0.0
        self.has_pose = False

        # Publish (0,0) immediately so fake_map_publisher seeds the SLAM map
        # at the origin — the robot always starts here
        self._publish(self.get_clock().now().to_msg())

        # Hold last known pose at 20 Hz so TF never goes stale between goals
        self.create_timer(0.05, self._publish_hold)

        self.get_logger().info(
            'OdomToPose: /amr/reference → /follower/pose (with hold)')

    def _cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Ignore the spline's idle (0,0) output only after we've moved away
        # from the origin — avoids snapping back to start mid-run
        if x == 0.0 and y == 0.0 and self.has_pose:
            return

        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self.x    = x
        self.y    = y
        self.has_pose = True

    def _publish_hold(self):
        """Republish at 20 Hz so TF stays alive between spline goals."""
        self._publish(self.get_clock().now().to_msg())

    def _publish(self, stamp):
        out = PoseWithCovarianceStamped()
        out.header.stamp    = stamp
        out.header.frame_id = 'map'
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