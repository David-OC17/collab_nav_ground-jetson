#!/usr/bin/env python3
"""Publishes a moving fake robot pose for simulation testing."""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped


class FakePosePublisher(Node):
    def __init__(self):
        super().__init__('fake_pose_publisher')
        self.pub = self.create_publisher(
            PoseWithCovarianceStamped, '/follower/pose', 10)
        self.t = 0.0
        self.create_timer(0.1, self._publish)   # 10 Hz

    def _publish(self):
        # Slowly spiral outward from origin to cover the map
        self.t += 0.02
        r = min(self.t * 0.3, 3.5)             # expand radius over time
        x = r * math.cos(self.t)
        y = r * math.sin(self.t)

        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x    = x
        msg.pose.pose.position.y    = y
        msg.pose.pose.orientation.w = 1.0
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FakePosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()