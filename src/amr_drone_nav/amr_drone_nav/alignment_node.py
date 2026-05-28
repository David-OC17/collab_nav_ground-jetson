#!/usr/bin/env python3
"""alignment_node.py — Publishes static TF world → slam_map from ArUco pose.

At t=0, slam_toolbox initializes the slam_map frame at the AMR's starting pose.
So T_world→slam_map = T_world→base_link at the moment of ArUco detection.

This node waits for an /aruco/amr_pose message (PoseStamped in `world` frame),
publishes the static TF once, then idles.
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import StaticTransformBroadcaster


class AlignmentNode(Node):
    def __init__(self):
        super().__init__('alignment_node')
        self.static_br = StaticTransformBroadcaster(self)
        self.published = False

        self.sub = self.create_subscription(
            PoseStamped,
            '/aruco/amr_pose',
            self._on_aruco,
            10,
        )
        self.get_logger().info('Waiting for /aruco/amr_pose in `world` frame…')

    def _on_aruco(self, msg: PoseStamped):
        if self.published:
            return  # one-shot init; see prior discussion on dynamic updates

        if msg.header.frame_id != 'world':
            self.get_logger().error(
                f"Expected frame_id='world', got '{msg.header.frame_id}' — ignoring"
            )
            return

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'slam_map'      # ← must match slam_toolbox's map_frame param
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = 0.0    # 2D arena
        t.transform.rotation = msg.pose.orientation

        self.static_br.sendTransform(t)
        self.published = True

        # Log the yaw for sanity-check
        q = msg.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        self.get_logger().info(
            f'Published world→slam_map: '
            f'x={t.transform.translation.x:.3f} '
            f'y={t.transform.translation.y:.3f} '
            f'yaw={math.degrees(yaw):.1f}°'
        )


def main():
    rclpy.init()
    rclpy.spin(AlignmentNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
