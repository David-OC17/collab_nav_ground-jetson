"""
Stub hardware publisher nodes.

These simulate the uncontrolled C++ sensor drivers (oradar LiDAR, Optitrack)
that publish native (unsigned) ROS2 messages.  Used in integration tests and
in the security_test launch file to exercise the relay pattern without real
hardware.
"""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, PointStamped, PoseStamped
from sensor_msgs.msg import LaserScan

import math
import time


class StubScanPublisher(Node):
    """Publishes a minimal LaserScan on /scan at 10 Hz — unsigned, native."""

    def __init__(self):
        super().__init__('stub_scan_pub')
        self._pub = self.create_publisher(LaserScan, '/scan', 10)
        self._timer = self.create_timer(0.1, self._publish)

    def _publish(self):
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser'
        msg.angle_min = -math.pi
        msg.angle_max = math.pi
        msg.angle_increment = math.pi / 180.0
        msg.time_increment = 0.0
        msg.range_min = 0.12
        msg.range_max = 10.0
        n = int((msg.angle_max - msg.angle_min) / msg.angle_increment)
        msg.ranges = [1.0] * n
        msg.intensities = [100.0] * n
        self._pub.publish(msg)


class StubOptitrackPublisher(Node):
    """Publishes fake rigid_body PoseStamped and marker PointStamped — unsigned."""

    def __init__(self):
        super().__init__('stub_optitrack_pub')
        self._pose_pub = self.create_publisher(
            PoseStamped, '/optitrack/rigid_body', 10)
        self._marker_pub = self.create_publisher(
            PointStamped, '/optitrack/marker', 10)
        self._timer = self.create_timer(0.05, self._publish)
        self._t = 0.0

    def _publish(self):
        self._t += 0.05
        now = self.get_clock().now().to_msg()

        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = 'world'
        pose.pose.position.x = math.cos(self._t)
        pose.pose.position.y = math.sin(self._t)
        pose.pose.orientation.w = 1.0
        self._pose_pub.publish(pose)

        marker = PointStamped()
        marker.header.stamp = now
        marker.header.frame_id = 'world'
        marker.point = Point(x=0.0, y=0.0, z=0.5)
        self._marker_pub.publish(marker)


def main_scan(args=None):
    rclpy.init(args=args)
    node = StubScanPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main_optitrack(args=None):
    rclpy.init(args=args)
    node = StubOptitrackPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
