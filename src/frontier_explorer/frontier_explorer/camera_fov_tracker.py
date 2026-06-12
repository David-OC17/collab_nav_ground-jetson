#!/usr/bin/env python3
"""
Camera FOV Tracker Node — ROS 2 Humble
========================================
Tracks which cells of the occupancy map have ever been seen by the RGB
camera (e.g. D435i RealSense).  Publishes the result as an OccupancyGrid
so the FrontierExplorer can avoid re-visiting areas where the LIDAR already
mapped free space but the camera never looked — the ArUco cube could be
sitting in any such cell.

How it works
------------
Every tick:
  1. Look up TF: camera_color_optical_frame → world_frame
  2. Compute the four ground-plane rays that correspond to the image corners
     using the camera intrinsic matrix (fx, fy, cx, cy from CameraInfo).
  3. Find where each ray hits z=0 in world space (the floor plane).
  4. Build a convex frustum polygon from those four hit-points plus the
     camera position projected onto the floor.
  5. Rasterise that polygon into the map grid with OpenCV fillPoly.
  6. OR the result into a persistent boolean mask (`camera_seen`).
  7. Publish the mask as an OccupancyGrid (0 = unseen, 100 = seen) and as
     a semi-transparent RViz polygon marker so you can visualise the FOV
     footprint in real time.

Subscribes:
  - /camera/camera/color/camera_info  (sensor_msgs/CameraInfo)  — intrinsics once
  - /slam/map                          (nav_msgs/OccupancyGrid)  — map size/origin/res
  - /amr/ekf/odom  or  odom_topic     (nav_msgs/Odometry)       — triggers tick

Publishes:
  - /camera/fov_map      (nav_msgs/OccupancyGrid)   — persistent seen mask
  - /camera/fov_marker   (visualization_msgs/Marker) — current FOV footprint polygon

Parameters:
  camera_frame        'camera_color_optical_frame'
  world_frame         'world'
  camera_info_topic   '/camera/camera/color/camera_info'
  map_topic           '/slam/map'
  odom_topic          '/amr/ekf/odom'
  fov_map_topic       '/camera/fov_map'
  fov_marker_topic    '/camera/fov_marker'
  max_ray_length_m    8.0   — clip floor-intersection rays to this length
  tf_timeout_sec      0.1
  update_rate_hz      5.0   — how often to reproject the FOV (Hz)
"""

import math
import numpy as np

import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                       DurabilityPolicy, HistoryPolicy)

import cv2
import tf2_ros
import tf2_geometry_msgs  # noqa: F401

from sensor_msgs.msg import CameraInfo
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA


class CameraFovTracker(Node):

    def __init__(self):
        super().__init__('camera_fov_tracker')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('camera_frame',      'camera_color_optical_frame')
        self.declare_parameter('world_frame',       'world')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('map_topic',         '/slam/map')
        self.declare_parameter('odom_topic',        '/amr/ekf/odom')
        self.declare_parameter('fov_map_topic',     '/camera/fov_map')
        self.declare_parameter('fov_marker_topic',  '/camera/fov_marker')
        self.declare_parameter('max_ray_length_m',  8.0)
        self.declare_parameter('tf_timeout_sec',    0.1)
        self.declare_parameter('update_rate_hz',    5.0)

        self.camera_frame      = self.get_parameter('camera_frame').value
        self.world_frame       = self.get_parameter('world_frame').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.map_topic         = self.get_parameter('map_topic').value
        self.odom_topic        = self.get_parameter('odom_topic').value
        self.fov_map_topic     = self.get_parameter('fov_map_topic').value
        self.fov_marker_topic  = self.get_parameter('fov_marker_topic').value
        self.max_ray_length_m  = float(self.get_parameter('max_ray_length_m').value)
        self.tf_timeout        = rclpy.duration.Duration(
            seconds=float(self.get_parameter('tf_timeout_sec').value))
        update_rate_hz         = float(self.get_parameter('update_rate_hz').value)

        # ── State ─────────────────────────────────────────────────────────
        # Camera intrinsics — filled once from CameraInfo
        self.img_width:  int   = 0
        self.img_height: int   = 0
        self.fx: float         = 0.0
        self.fy: float         = 0.0
        self.cx: float         = 0.0
        self.cy: float         = 0.0
        self.camera_info_ready = False

        # Map geometry — updated on every OccupancyGrid message
        self.map_resolution: float | None = None
        self.map_origin_x:   float | None = None
        self.map_origin_y:   float | None = None
        self.map_width_cells:  int | None = None
        self.map_height_cells: int | None = None
        self.map_header             = None

        # Persistent seen mask — bool array (height × width), grows with map
        self.camera_seen: np.ndarray | None = None

        # ── TF ────────────────────────────────────────────────────────────
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── QoS ───────────────────────────────────────────────────────────
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            CameraInfo, self.camera_info_topic,
            self._camera_info_callback, reliable_qos)

        self.create_subscription(
            OccupancyGrid, self.map_topic,
            self._map_callback, latched_qos)

        self.create_subscription(
            Odometry, self.odom_topic,
            self._odom_callback, sensor_qos)

        # ── Publishers ────────────────────────────────────────────────────
        self.fov_map_pub = self.create_publisher(
            OccupancyGrid, self.fov_map_topic, latched_qos)

        self.fov_marker_pub = self.create_publisher(
            Marker, self.fov_marker_topic, reliable_qos)

        # ── Timer ─────────────────────────────────────────────────────────
        self.create_timer(1.0 / update_rate_hz, self._tick)

        self.get_logger().info(
            f'CameraFovTracker ready\n'
            f'  camera_frame ← {self.camera_frame}\n'
            f'  world_frame  = {self.world_frame}\n'
            f'  map          ← {self.map_topic}\n'
            f'  odom         ← {self.odom_topic}\n'
            f'  fov_map      → {self.fov_map_topic}\n'
            f'  update_rate  = {update_rate_hz} Hz'
        )

    # =========================================================================
    # Callbacks
    # =========================================================================

    def _camera_info_callback(self, msg: CameraInfo):
        """Store intrinsics once — they don't change during a session."""
        if self.camera_info_ready:
            return
        self.img_width  = msg.width
        self.img_height = msg.height
        K = np.array(msg.k, dtype=np.float64).reshape((3, 3))
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        self.camera_info_ready = True
        h_fov = 2.0 * math.degrees(math.atan2(self.img_width  / 2.0, self.fx))
        v_fov = 2.0 * math.degrees(math.atan2(self.img_height / 2.0, self.fy))
        self.get_logger().info(
            f'Camera intrinsics received: '
            f'{self.img_width}×{self.img_height}px  '
            f'hFOV={h_fov:.1f}°  vFOV={v_fov:.1f}°'
        )

    def _map_callback(self, msg: OccupancyGrid):
        """Update map geometry; extend the seen mask if the map grew."""
        new_w = msg.info.width
        new_h = msg.info.height
        new_res = msg.info.resolution
        new_ox  = msg.info.origin.position.x
        new_oy  = msg.info.origin.position.y

        geometry_changed = (
            new_res != self.map_resolution
            or new_ox != self.map_origin_x
            or new_oy != self.map_origin_y
            or new_w  != self.map_width_cells
            or new_h  != self.map_height_cells
        )

        self.map_resolution    = new_res
        self.map_origin_x      = new_ox
        self.map_origin_y      = new_oy
        self.map_width_cells   = new_w
        self.map_height_cells  = new_h
        self.map_header        = msg.header

        if geometry_changed or self.camera_seen is None:
            # Allocate or resize the mask, preserving existing data
            new_mask = np.zeros((new_h, new_w), dtype=bool)
            if self.camera_seen is not None:
                old_h, old_w = self.camera_seen.shape
                copy_h = min(old_h, new_h)
                copy_w = min(old_w, new_w)
                new_mask[:copy_h, :copy_w] = self.camera_seen[:copy_h, :copy_w]
            self.camera_seen = new_mask
            self.get_logger().info(
                f'Map geometry updated: {new_w}×{new_h} cells, '
                f'res={new_res} m/cell, '
                f'origin=({new_ox:.2f}, {new_oy:.2f})'
            )

    def _odom_callback(self, msg: Odometry):
        """Keep the latest stamp for TF lookups — actual tick is timer-driven."""
        self._latest_stamp = msg.header.stamp

    # =========================================================================
    # Main tick
    # =========================================================================

    def _tick(self):
        if not self.camera_info_ready:
            self.get_logger().warn(
                'Waiting for camera intrinsics…', throttle_duration_sec=5.0)
            return
        if self.camera_seen is None:
            self.get_logger().warn(
                'Waiting for map…', throttle_duration_sec=5.0)
            return

        # ── 1. Get camera pose in world frame ────────────────────────────
        try:
            tf_stamped = self._tf_buffer.lookup_transform(
                self.world_frame,
                self.camera_frame,
                rclpy.time.Time(),          # latest available
                timeout=self.tf_timeout,
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f'TF {self.camera_frame}→{self.world_frame}: {e}',
                throttle_duration_sec=2.0)
            return

        t = tf_stamped.transform.translation
        q = tf_stamped.transform.rotation

        # Camera origin in world XY (project onto floor plane z=0)
        cam_wx = t.x
        cam_wy = t.y
        cam_wz = t.z   # height above floor

        # Rotation matrix from quaternion (camera → world)
        R = self._quat_to_rotation_matrix(q.x, q.y, q.z, q.w)

        # ── 2. Project image corners → world floor plane ──────────────────
        # Image corners in pixel coordinates (u, v):
        #   top-left, top-right, bottom-right, bottom-left
        corners_px = [
            (0,                  0),
            (self.img_width - 1, 0),
            (self.img_width - 1, self.img_height - 1),
            (0,                  self.img_height - 1),
        ]

        floor_pts = []
        for (u, v) in corners_px:
            pt = self._pixel_to_floor(u, v, R, cam_wx, cam_wy, cam_wz)
            if pt is not None:
                floor_pts.append(pt)

        if len(floor_pts) < 3:
            # Camera is looking straight up or rays don't hit the floor
            self.get_logger().warn(
                'FOV rays do not intersect floor plane — skipping tick.',
                throttle_duration_sec=5.0)
            return

        # ── 3. Rasterise polygon into the map grid ────────────────────────
        res = self.map_resolution
        ox  = self.map_origin_x
        oy  = self.map_origin_y
        W   = self.map_width_cells
        H   = self.map_height_cells

        # Convert world XY → grid pixel coordinates (col, row)
        grid_pts = []
        for (wx, wy) in floor_pts:
            ci = int((wx - ox) / res)
            cj = int((wy - oy) / res)
            # Clamp to map bounds — rays may extend beyond the map edge
            ci = max(0, min(W - 1, ci))
            cj = max(0, min(H - 1, cj))
            grid_pts.append([ci, cj])

        polygon = np.array(grid_pts, dtype=np.int32).reshape((-1, 1, 2))

        # Draw filled polygon into a temporary mask, then OR into persistent mask
        frame = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(frame, [polygon], color=1)
        self.camera_seen |= frame.astype(bool)

        # ── 4. Publish ────────────────────────────────────────────────────
        self._publish_fov_map()
        self._publish_fov_marker(floor_pts, cam_wx, cam_wy)

    # =========================================================================
    # Geometry helpers
    # =========================================================================

    def _pixel_to_floor(
        self,
        u: int, v: int,
        R: np.ndarray,
        cam_wx: float, cam_wy: float, cam_wz: float,
    ) -> tuple[float, float] | None:
        """
        Back-project pixel (u,v) through the camera into world space and
        find where the resulting ray intersects z = 0 (the floor plane).

        Returns (world_x, world_y) or None if the ray points upward or
        is clipped to max_ray_length_m.
        """
        # Normalised ray in camera frame (OpenCV convention: Z forward)
        ray_cam = np.array([
            (u - self.cx) / self.fx,
            (v - self.cy) / self.fy,
            1.0,
        ])
        ray_cam /= np.linalg.norm(ray_cam)

        # Rotate into world frame
        ray_world = R @ ray_cam   # shape (3,)

        # Intersect with z = 0: cam_wz + t * ray_world[2] = 0
        if abs(ray_world[2]) < 1e-6:
            return None   # ray nearly horizontal — no floor intersection
        t = -cam_wz / ray_world[2]
        if t < 0:
            return None   # floor is behind the camera

        wx = cam_wx + t * ray_world[0]
        wy = cam_wy + t * ray_world[1]

        # Clip excessively long rays (camera high up, nearly horizontal)
        dist = math.hypot(wx - cam_wx, wy - cam_wy)
        if dist > self.max_ray_length_m:
            scale = self.max_ray_length_m / dist
            wx = cam_wx + (wx - cam_wx) * scale
            wy = cam_wy + (wy - cam_wy) * scale

        return (wx, wy)

    @staticmethod
    def _quat_to_rotation_matrix(qx, qy, qz, qw) -> np.ndarray:
        """Quaternion → 3×3 rotation matrix (world = R @ camera)."""
        n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if n < 1e-9:
            return np.eye(3)
        qx /= n; qy /= n; qz /= n; qw /= n
        return np.array([
            [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ])

    # =========================================================================
    # Publishing
    # =========================================================================

    def _publish_fov_map(self):
        msg = OccupancyGrid()
        if self.map_header is not None:
            msg.header = self.map_header
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame

        msg.info.resolution = self.map_resolution
        msg.info.width      = self.map_width_cells
        msg.info.height     = self.map_height_cells
        msg.info.origin.position.x = self.map_origin_x
        msg.info.origin.position.y = self.map_origin_y
        msg.info.origin.orientation.w = 1.0

        # 100 = seen by camera, 0 = never seen
        flat = self.camera_seen.flatten().astype(np.int8) * 100
        msg.data = flat.tolist()
        self.fov_map_pub.publish(msg)

    def _publish_fov_marker(self, floor_pts: list, cam_wx: float, cam_wy: float):
        """Publish the current-frame FOV footprint as a LINE_STRIP marker."""
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = self.world_frame
        m.ns     = 'camera_fov'
        m.id     = 0
        m.type   = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.05   # line width in metres
        m.color   = ColorRGBA(r=0.2, g=0.8, b=1.0, a=0.7)
        m.pose.orientation.w = 1.0

        # Camera position → corners → back to camera, drawing the frustum
        origin = Point(x=cam_wx, y=cam_wy, z=0.01)
        m.points.append(origin)
        for (wx, wy) in floor_pts:
            m.points.append(Point(x=wx, y=wy, z=0.01))
        m.points.append(origin)   # close the loop

        m.lifetime.sec     = 0
        m.lifetime.nanosec = int(0.5e9)   # vanishes if not refreshed within 0.5 s
        self.fov_marker_pub.publish(m)

    # =========================================================================
    # Public API (used by FrontierExplorer)
    # =========================================================================

    def cell_seen_by_camera(self, world_x: float, world_y: float) -> bool:
        """
        Returns True if the map cell at (world_x, world_y) has been seen
        by the camera at least once.  Safe to call even before the map arrives.
        """
        if self.camera_seen is None or self.map_resolution is None:
            return False
        ci = int((world_x - self.map_origin_x) / self.map_resolution)
        cj = int((world_y - self.map_origin_y) / self.map_resolution)
        if not (0 <= ci < self.map_width_cells and 0 <= cj < self.map_height_cells):
            return False
        return bool(self.camera_seen[cj, ci])


# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = CameraFovTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()