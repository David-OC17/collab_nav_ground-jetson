#!/usr/bin/env python3
"""
Pose to TF Broadcaster — ROS 2 Humble
=======================================
Subscribes to /follower/pose (geometry_msgs/PoseWithCovarianceStamped)
and broadcasts it as a TF transform map → base_footprint.

This allows AStarPlanner2 (which reads robot pose exclusively from TF)
to track the robot's actual position as it follows planned trajectories.

Subscribes:
  - /follower/pose  (geometry_msgs/PoseWithCovarianceStamped)

Broadcasts:
  - TF: map → base_footprint

Parameters:
  pose_topic        '/follower/pose'
  parent_frame      'map'
  child_frame       'base_footprint'
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
import tf2_ros


class PoseToTf(Node):

    def __init__(self):
        super().__init__('pose_to_tf')

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter('pose_topic',   '/follower/pose')
        self.declare_parameter('parent_frame', 'map')
        self.declare_parameter('child_frame',  'base_footprint')

        self.pose_topic   = self.get_parameter('pose_topic').value
        self.parent_frame = self.get_parameter('parent_frame').value
        self.child_frame  = self.get_parameter('child_frame').value

        # ------------------------------------------------------------------
        # TF broadcaster
        # ------------------------------------------------------------------
        self.broadcaster = tf2_ros.TransformBroadcaster(self)

        # ------------------------------------------------------------------
        # Subscriber
        # ------------------------------------------------------------------
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(
            PoseWithCovarianceStamped,
            self.pose_topic,
            self._pose_callback,
            reliable_qos
        )

        self.get_logger().info(
            f'PoseToTf ready\n'
            f'  pose  ← {self.pose_topic}\n'
            f'  TF    → {self.parent_frame} → {self.child_frame}'
        )

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        t = TransformStamped()
        # Use current time if message stamp is zero (no path yet)
        stamp = msg.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            stamp = self.get_clock().now().to_msg()

        t.header.stamp    = stamp
        t.header.frame_id = self.parent_frame
        t.child_frame_id  = self.child_frame

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        t.transform.rotation = msg.pose.pose.orientation

        self.broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = PoseToTf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()