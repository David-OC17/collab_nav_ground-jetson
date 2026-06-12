#!/usr/bin/env python3
"""
ArUco World Bridge — ROS 2 Humble
====================================
Bridges aruco_goal_detector (camera frame) → explorer_controller (world frame).

aruco_goal_detector publishes each detected marker on /aruco/{id}/pose with
frame_id = camera_color_optical_frame.  explorer_controller expects /aruco/detection
with the pose already in world frame and the marker ID encoded in covariance[0].

This node:
  1. Subscribes to /aruco/id_{id}/pose for the configured target marker ID.
  2. Transforms the pose from camera frame → world frame via TF.
  3. Republishes as /aruco/detection (same format as fake_aruco_detector in sim).

Subscribes:
  - /aruco/id_{target_marker_id}/pose  (geometry_msgs/PoseWithCovarianceStamped)

Publishes:
  - /aruco/detection  (geometry_msgs/PoseWithCovarianceStamped)
      pose in world_frame, covariance[0] = float(marker_id)

Parameters:
  target_marker_id   0
  world_frame        'map'
  camera_frame       'camera_color_optical_frame'
  tf_timeout_sec     0.1
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import rclpy.duration

import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers PoseStamped transform support

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped


class ArucoWorldBridge(Node):

    def __init__(self):
        super().__init__('aruco_world_bridge')

        self.declare_parameter('target_marker_id', 0)
        self.declare_parameter('world_frame',      'map')
        self.declare_parameter('tf_timeout_sec',   0.1)

        self.target_id   = int(self.get_parameter('target_marker_id').value)
        self.world_frame = self.get_parameter('world_frame').value
        self.tf_timeout  = rclpy.duration.Duration(
            seconds=float(self.get_parameter('tf_timeout_sec').value))

        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(
            PoseWithCovarianceStamped,
            f'/aruco/id_{self.target_id}/pose',
            self._detection_callback,
            reliable_qos
        )

        self.pub = self.create_publisher(
            PoseWithCovarianceStamped, '/aruco/detection', reliable_qos)

        self.get_logger().info(
            f'ArucoWorldBridge ready\n'
            f'  in  ← /aruco/id_{self.target_id}/pose (camera frame)\n'
            f'  out → /aruco/detection ({self.world_frame} frame)\n'
            f'  TF timeout = {self.tf_timeout.nanoseconds / 1e9:.2f}s'
        )

    def _detection_callback(self, msg: PoseWithCovarianceStamped):
        # Build a PoseStamped in the camera frame for TF transformation
        pose_cam = PoseStamped()
        pose_cam.header = msg.header   # frame_id = camera_color_optical_frame
        pose_cam.pose   = msg.pose.pose

        try:
            pose_world = self._tf_buffer.transform(
                pose_cam,
                self.world_frame,
                timeout=self.tf_timeout
            )
        except Exception as e:
            self.get_logger().warn(
                f'TF camera→{self.world_frame} failed: {e}',
                throttle_duration_sec=2.0)
            return

        out = PoseWithCovarianceStamped()
        out.header.stamp    = pose_world.header.stamp
        out.header.frame_id = self.world_frame
        out.pose.pose       = pose_world.pose
        # Marker ID in covariance[0] — matches fake_aruco_detector convention
        out.pose.covariance[0] = float(self.target_id)

        self.pub.publish(out)
        self.get_logger().debug(
            f'ArUco {self.target_id} → world: '
            f'({pose_world.pose.position.x:.3f}, {pose_world.pose.position.y:.3f})',
            throttle_duration_sec=0.5)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoWorldBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()