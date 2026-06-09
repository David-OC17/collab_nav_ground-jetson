#!/usr/bin/env python3
"""
Fake ArUco Detector for simulation
====================================
Simulates an ArUco marker placed at a fixed world position (marker_x, marker_y).
Publishes a detection on /aruco/detection whenever the marker falls inside the
robot's camera FOV wedge — using identical geometry to fake_map_publisher so
simulation and real behaviour match exactly.

Detection condition (all must be true):
  1. Euclidean distance robot → marker is within [camera_near_m, camera_range_m]
  2. Bearing to marker is within ±(camera_hfov_deg/2) of robot heading
  3. Line-of-sight is unobstructed (Bresenham on /drone/map)

Publishes:
  - /aruco/detection  (geometry_msgs/PoseWithCovarianceStamped)
      pose.pose.position.{x,y} = marker world position
      pose.covariance[0]       = marker_id (float, same convention as real detector)

  - /aruco/markers    (visualization_msgs/MarkerArray)
      A cyan cube at the marker position for RViz.
      Turns green when currently detected.

Subscribes:
  - /follower/pose  (geometry_msgs/PoseWithCovarianceStamped) — robot pose + yaw
  - /drone/map      (nav_msgs/OccupancyGrid)                  — for LoS checks

Parameters:
  marker_x          0.0     m  — marker world X
  marker_y         -1.5     m  — marker world Y
  marker_id         0           — ArUco ID to report (must match target_marker_id
                                  in explorer_controller)
  camera_hfov_deg  69.4     °  — must match fake_map_publisher
  camera_range_m    4.0     m  — must match fake_map_publisher
  camera_near_m     0.15    m  — must match fake_map_publisher
  publish_rate     10.0     Hz
  world_frame      'map'
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                        DurabilityPolicy, HistoryPolicy)

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import numpy as np


class FakeArucoDetector(Node):

    def __init__(self):
        super().__init__('fake_aruco_detector')

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter('marker_x',        0.0)
        self.declare_parameter('marker_y',       -1.5)
        self.declare_parameter('marker_id',        0)
        self.declare_parameter('camera_hfov_deg', 69.4)
        self.declare_parameter('camera_range_m',   4.0)
        self.declare_parameter('camera_near_m',    0.15)
        self.declare_parameter('publish_rate',    10.0)
        self.declare_parameter('world_frame',     'map')

        self.marker_x        = float(self.get_parameter('marker_x').value)
        self.marker_y        = float(self.get_parameter('marker_y').value)
        self.marker_id       = int(self.get_parameter('marker_id').value)
        self.camera_half_fov = math.radians(
            float(self.get_parameter('camera_hfov_deg').value) / 2.0)
        self.camera_range    = float(self.get_parameter('camera_range_m').value)
        self.camera_near     = float(self.get_parameter('camera_near_m').value)
        self.publish_rate    = float(self.get_parameter('publish_rate').value)
        self.world_frame     = self.get_parameter('world_frame').value

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0
        self.pose_received = False

        # Occupancy grid for LoS
        self.map_data      = None
        self.map_width     = 0
        self.map_height    = 0
        self.map_res       = 0.05
        self.map_origin_x  = 0.0
        self.map_origin_y  = 0.0
        self.map_received  = False

        self.currently_detected = False

        self._tick_count = 0
        self._min_ticks_before_detection = int(self.publish_rate * 2.0)

        # ------------------------------------------------------------------
        # QoS
        # ------------------------------------------------------------------
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ------------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------------
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/follower/pose',
            self._pose_callback,
            reliable_qos
        )
        self.create_subscription(
            OccupancyGrid,
            '/drone/map',
            self._map_callback,
            latched_qos
        )

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        self.detection_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/aruco/detection',
            reliable_qos
        )
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/aruco/markers',
            reliable_qos
        )

        # ------------------------------------------------------------------
        # Timer
        # ------------------------------------------------------------------
        self.create_timer(1.0 / self.publish_rate, self._tick)

        self.get_logger().info(
            f'FakeArucoDetector ready\n'
            f'  marker position = ({self.marker_x:.2f}, {self.marker_y:.2f})\n'
            f'  marker_id       = {self.marker_id}\n'
            f'  camera H-FOV    = {math.degrees(self.camera_half_fov*2):.1f}°\n'
            f'  camera range    = {self.camera_near:.2f} – {self.camera_range:.1f} m'
        )

    # ==========================================================================
    # Callbacks
    # ==========================================================================

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self.pose_received = True

    def _map_callback(self, msg: OccupancyGrid):
        self.map_res      = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y
        self.map_width    = msg.info.width
        self.map_height   = msg.info.height
        self.map_data     = np.array(msg.data, dtype=np.int8).reshape(
            (self.map_height, self.map_width))
        self.map_received = True

    # ==========================================================================
    # Main tick
    # ==========================================================================

    def _tick(self):
        self._tick_count += 1

        # Wait for map and pose regardless
        if not self.pose_received or not self.map_received:
            return

        # Hard tick gate — ensures ROS 2 has applied launch params before we check
        # get_clock() is unreliable in __init__ so we count ticks instead
        if self._tick_count < self._min_ticks_before_detection:
            self._publish_marker()   # show cube in RViz but no detection yet
            return

        self.currently_detected = self._check_detection()
        if self.currently_detected:
            self._publish_detection()
        self._publish_marker()

    # ==========================================================================
    # Detection logic — identical wedge geometry to fake_map_publisher
    # ==========================================================================

    def _check_detection(self) -> bool:
        dx = self.marker_x - self.robot_x
        dy = self.marker_y - self.robot_y
        dist = math.hypot(dx, dy)

        if dist < self.camera_near or dist > self.camera_range:
            return False

        bearing    = math.atan2(dy, dx)
        angle_diff = math.atan2(
            math.sin(bearing - self.robot_yaw),
            math.cos(bearing - self.robot_yaw)
        )
        if abs(angle_diff) > self.camera_half_fov:
            return False

        # LoS always enforced — map is guaranteed by _tick guard above
        return self._has_line_of_sight()

    def _has_line_of_sight(self) -> bool:
        """Bresenham from robot cell to marker cell on the drone map."""
        x0 = int((self.robot_x  - self.map_origin_x) / self.map_res)
        y0 = int((self.robot_y  - self.map_origin_y) / self.map_res)
        x1 = int((self.marker_x - self.map_origin_x) / self.map_res)
        y1 = int((self.marker_y - self.map_origin_y) / self.map_res)

        dx = abs(x1 - x0); sx = 1 if x0 < x1 else -1
        dy = abs(y1 - y0); sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0

        while True:
            if (x, y) == (x1, y1):
                return True
            # Check intermediate cells only (not start or end)
            if (x, y) != (x0, y0):
                if (0 <= x < self.map_width and 0 <= y < self.map_height):
                    if self.map_data[y, x] >= 100:
                        return False
                else:
                    return False  # out of bounds = treat as wall
            e2 = 2 * err
            if e2 > -dy:
                err -= dy; x += sx
            if e2 <  dx:
                err += dx; y += sy

    # ==========================================================================
    # Publishers
    # ==========================================================================

    def _publish_detection(self):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame
        msg.pose.pose.position.x    = self.marker_x
        msg.pose.pose.position.y    = self.marker_y
        msg.pose.pose.position.z    = 0.0
        msg.pose.pose.orientation.w = 1.0
        # Marker ID in covariance[0] — matches real aruco_goal_detector convention
        msg.pose.covariance[0] = float(self.marker_id)
        self.detection_pub.publish(msg)

        self.get_logger().info(
            f'ArUco {self.marker_id} detected at '
            f'({self.marker_x:.2f}, {self.marker_y:.2f}) | '
            f'robot=({self.robot_x:.2f}, {self.robot_y:.2f}) | '
            f'yaw={math.degrees(self.robot_yaw):.1f}°',
            throttle_duration_sec=1.0
        )

    def _publish_marker(self):
        """RViz cube at marker position — cyan when idle, green when detected."""
        array = MarkerArray()

        cube = Marker()
        cube.header.stamp    = self.get_clock().now().to_msg()
        cube.header.frame_id = self.world_frame
        cube.ns     = 'aruco_sim'
        cube.id     = self.marker_id
        cube.type   = Marker.CUBE
        cube.action = Marker.ADD
        cube.pose.position.x    = self.marker_x
        cube.pose.position.y    = self.marker_y
        cube.pose.position.z    = 0.065   # half-height of a 13 cm cube
        cube.pose.orientation.w = 1.0
        cube.scale.x = cube.scale.y = cube.scale.z = 0.13

        if self.currently_detected:
            cube.color = ColorRGBA(r=0.0, g=1.0, b=0.2, a=1.0)   # green
        else:
            cube.color = ColorRGBA(r=0.0, g=0.8, b=0.8, a=0.8)   # cyan

        # Text label above the cube
        label = Marker()
        label.header    = cube.header
        label.ns        = 'aruco_sim_label'
        label.id        = self.marker_id + 1000
        label.type      = Marker.TEXT_VIEW_FACING
        label.action    = Marker.ADD
        label.pose.position.x    = self.marker_x
        label.pose.position.y    = self.marker_y
        label.pose.position.z    = 0.25
        label.pose.orientation.w = 1.0
        label.scale.z = 0.12
        label.color   = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        label.text    = f'ArUco {self.marker_id}'

        array.markers = [cube, label]
        self.marker_pub.publish(array)


# ==============================================================================

def main(args=None):
    rclpy.init(args=None)
    node = FakeArucoDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()