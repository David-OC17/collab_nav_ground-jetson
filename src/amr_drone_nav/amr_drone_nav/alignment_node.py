#!/usr/bin/env python3
"""alignment_node.py — Publishes static TF world → odom from the ArUco AMR pose.

At t=0, slam_toolbox initializes the slam_map frame at the AMR's starting pose.
So T_world→odom = T_world→base_link at the moment of ArUco detection.

This node waits for one /aruco/amr/pose message (PoseWithCovarianceStamped in the
`world` frame), publishes the static TF once, then idles.

Trivial-transform fallback
──────────────────────────
The mission orchestrator publishes an ALL-NaN pose on /aruco/amr/pose when it has
no AMR localization (e.g. the drone map failed quality classification and was
dumped, so there is no reliable map-frame AMR position). A NaN pose means "no
data": this node then publishes the trivial identity transform (world and odom
coincide) instead of a NaN transform, so the TF tree stays valid and downstream
nodes operate in the odom frame.
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from tf2_ros import StaticTransformBroadcaster


class AlignmentNode(Node):
    def __init__(self):
        super().__init__('alignment_node')
        self.static_br = StaticTransformBroadcaster(self)
        self.published = False

        # Matches the orchestrator's publisher type (PoseWithCovarianceStamped);
        # a PoseStamped subscription would silently never connect.
        self.sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/aruco/amr/pose',
            self._on_aruco,
            10,
        )
        self.get_logger().info('Waiting for /aruco/amr/pose in `world` frame…')

    def _on_aruco(self, msg: PoseWithCovarianceStamped):
        if self.published:
            return  # one-shot init; see prior discussion on dynamic updates

        if msg.header.frame_id != 'world':
            self.get_logger().error(
                f"Expected frame_id='world', got '{msg.header.frame_id}' — ignoring"
            )
            return

        pose = msg.pose.pose

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'odom'      # ← must match slam_toolbox's map_frame param

        # NaN position → no AMR data: fall back to the trivial identity transform
        # (world ≡ odom) rather than propagating NaN into the TF tree.
        if math.isnan(pose.position.x) or math.isnan(pose.position.y):
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0
            self.static_br.sendTransform(t)
            self.published = True
            self.get_logger().warn(
                'AMR pose is NaN (no localization data) — published trivial '
                'identity world→odom transform'
            )
            return

        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = 0.0    # 2D arena
        t.transform.rotation = pose.orientation

        self.static_br.sendTransform(t)
        self.published = True

        # Log the yaw for sanity-check
        q = pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        self.get_logger().info(
            f'Published world→odom: '
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
