#!/usr/bin/env python3
"""
ArUco Detector Node for ROS 2 Humble — Intel RealSense D435i + Isaac ROS VSLAM
================================================================================
Detects ArUco markers in the RGB stream, back-projects the marker centre using
the aligned depth frame to get a 3D point in camera coordinates, then transforms
it to the world frame using TF (published by Isaac ROS Visual SLAM).

The goal pose is published as a fixed offset IN FRONT of the marker so the
robot stands facing it rather than standing on top of it.

Subscribes:
  - /camera/color/image_raw                    (sensor_msgs/Image)      — RGB frame
  - /camera/aligned_depth_to_color/image_raw   (sensor_msgs/Image)      — aligned depth
  - /camera/color/camera_info                  (sensor_msgs/CameraInfo) — intrinsics

Publishes:
  - /aruco/detection  (geometry_msgs/PoseWithCovarianceStamped)
      Detected marker pose in world frame.
      header.frame_id carries the marker ID as a string (e.g. '7') so
      MissionController can filter by target_marker_id.

  - /aruco/markers    (visualization_msgs/MarkerArray)
      RViz visualisation: sphere at marker position + text label.

TF used:
  - camera_color_optical_frame → map   (provided by Isaac ROS VSLAM)

Parameters:
  rgb_topic          '/camera/color/image_raw'
  depth_topic        '/camera/aligned_depth_to_color/image_raw'
  camera_info_topic  '/camera/color/camera_info'
  world_frame        'map'
  camera_frame       'camera_color_optical_frame'
  aruco_dict         'DICT_4X4_50'      — OpenCV ArUco dictionary name
  marker_size_m      0.10               — physical marker side length in metres
  standoff_m         0.50               — how far in front of marker to place goal
  detection_rate     10.0               — Hz, how often to run detection
  min_depth_m        0.10               — ignore depth readings below this (noise)
  max_depth_m        6.0                — ignore depth readings above this
  depth_sample_r     3                  — pixel radius to median-sample depth
"""

import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import tf2_ros
from tf2_ros import TransformException
import tf2_geometry_msgs   # needed for do_transform_pose

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import (PoseWithCovarianceStamped, PoseStamped,
                                Point, Vector3)
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge


# ---------------------------------------------------------------------------
# ArUco dictionary map
# ---------------------------------------------------------------------------
ARUCO_DICTS = {
    'DICT_4X4_50':       cv2.aruco.DICT_4X4_50,
    'DICT_4X4_100':      cv2.aruco.DICT_4X4_100,
    'DICT_4X4_250':      cv2.aruco.DICT_4X4_250,
    'DICT_5X5_50':       cv2.aruco.DICT_5X5_50,
    'DICT_5X5_100':      cv2.aruco.DICT_5X5_100,
    'DICT_6X6_50':       cv2.aruco.DICT_6X6_50,
    'DICT_ARUCO_ORIGINAL': cv2.aruco.DICT_ARUCO_ORIGINAL,
}


class ArucoDetector(Node):

    def __init__(self):
        super().__init__('aruco_detector')

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter('rgb_topic',
                               '/camera/color/image_raw')
        self.declare_parameter('depth_topic',
                               '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic',
                               '/camera/color/camera_info')
        self.declare_parameter('world_frame',      'map')
        self.declare_parameter('camera_frame',     'camera_color_optical_frame')
        self.declare_parameter('aruco_dict',       'DICT_4X4_50')
        self.declare_parameter('marker_size_m',    0.10)
        self.declare_parameter('standoff_m',       0.50)
        self.declare_parameter('detection_rate',   10.0)
        self.declare_parameter('min_depth_m',      0.10)
        self.declare_parameter('max_depth_m',      6.0)
        self.declare_parameter('depth_sample_r',   3)

        self.rgb_topic         = self.get_parameter('rgb_topic').value
        self.depth_topic       = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.world_frame       = self.get_parameter('world_frame').value
        self.camera_frame      = self.get_parameter('camera_frame').value
        self.marker_size_m     = float(self.get_parameter('marker_size_m').value)
        self.standoff_m        = float(self.get_parameter('standoff_m').value)
        self.detection_rate    = float(self.get_parameter('detection_rate').value)
        self.min_depth_m       = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m       = float(self.get_parameter('max_depth_m').value)
        self.depth_sample_r    = int(self.get_parameter('depth_sample_r').value)

        # ------------------------------------------------------------------
        # ArUco setup
        # ------------------------------------------------------------------
        dict_name = self.get_parameter('aruco_dict').value
        if dict_name not in ARUCO_DICTS:
            self.get_logger().error(
                f'Unknown aruco_dict "{dict_name}". '
                f'Valid options: {list(ARUCO_DICTS.keys())}')
            raise ValueError(f'Unknown aruco_dict: {dict_name}')

        aruco_dict    = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
        aruco_params  = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.bridge          = CvBridge()
        self.camera_matrix   = None   # 3x3 np.ndarray
        self.dist_coeffs     = None   # 1x5 np.ndarray
        self.camera_info_ok  = False

        self.latest_rgb      = None   # sensor_msgs/Image
        self.latest_depth    = None   # sensor_msgs/Image

        # ------------------------------------------------------------------
        # TF
        # ------------------------------------------------------------------
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ------------------------------------------------------------------
        # QoS
        # ------------------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
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
        self.rgb_sub = self.create_subscription(
            Image, self.rgb_topic,
            self._rgb_callback, sensor_qos)

        self.depth_sub = self.create_subscription(
            Image, self.depth_topic,
            self._depth_callback, sensor_qos)

        self.info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic,
            self._camera_info_callback, latched_qos)

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
        # Detection timer
        # ------------------------------------------------------------------
        self.create_timer(1.0 / self.detection_rate, self._detect)

        self.get_logger().info(
            f'ArucoDetector ready\n'
            f'  rgb        ← {self.rgb_topic}\n'
            f'  depth      ← {self.depth_topic}\n'
            f'  dict       = {dict_name}\n'
            f'  marker_size= {self.marker_size_m} m\n'
            f'  standoff   = {self.standoff_m} m\n'
            f'  rate       = {self.detection_rate} Hz\n'
            f'  TF         : {self.camera_frame} → {self.world_frame}'
        )

    # ==========================================================================
    # Callbacks — just buffer the latest frames
    # ==========================================================================

    def _rgb_callback(self, msg: Image):
        self.latest_rgb = msg

    def _depth_callback(self, msg: Image):
        self.latest_depth = msg

    def _camera_info_callback(self, msg: CameraInfo):
        if self.camera_info_ok:
            return   # only need it once
        self.camera_matrix  = np.array(msg.k, dtype=np.float64).reshape((3, 3))
        self.dist_coeffs    = np.array(msg.d, dtype=np.float64)
        self.camera_info_ok = True
        self.get_logger().info(
            f'Camera intrinsics received:\n'
            f'  fx={self.camera_matrix[0,0]:.1f}  fy={self.camera_matrix[1,1]:.1f}\n'
            f'  cx={self.camera_matrix[0,2]:.1f}  cy={self.camera_matrix[1,2]:.1f}')

    # ==========================================================================
    # Detection loop
    # ==========================================================================

    def _detect(self):
        # Guard: need intrinsics and both frames
        if not self.camera_info_ok:
            self.get_logger().warn(
                'Waiting for camera_info…', throttle_duration_sec=5.0)
            return
        if self.latest_rgb is None or self.latest_depth is None:
            self.get_logger().warn(
                'Waiting for RGB + depth frames…', throttle_duration_sec=5.0)
            return

        # Convert to OpenCV
        try:
            bgr   = self.bridge.imgmsg_to_cv2(
                self.latest_rgb, desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(
                self.latest_depth, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'CvBridge error: {e}')
            return

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return   # nothing detected this frame

        # Process each detected marker
        for idx, marker_id in enumerate(ids.flatten()):
            corner_pts = corners[idx][0]   # shape (4, 2), float32

            # ── 1. Marker centre in pixel space ───────────────────────────
            cx_px = float(np.mean(corner_pts[:, 0]))
            cy_px = float(np.mean(corner_pts[:, 1]))

            # ── 2. Depth at marker centre (median over a small patch) ─────
            depth_m = self._sample_depth(depth, cx_px, cy_px)
            if depth_m is None:
                self.get_logger().warn(
                    f'Marker {marker_id}: invalid depth at '
                    f'({cx_px:.0f}, {cy_px:.0f}) — skipping.')
                continue

            # ── 3. Back-project to 3D in camera frame ────────────────────
            #   X_cam = (cx_px - cx) * Z / fx
            #   Y_cam = (cy_px - cy) * Z / fy
            #   Z_cam = depth_m
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]

            x_cam = (cx_px - cx) * depth_m / fx
            y_cam = (cy_px - cy) * depth_m / fy
            z_cam = depth_m

            # ── 4. Estimate marker normal using solvePnP ─────────────────
            #   This gives us the rotation of the marker so we can compute
            #   the standoff direction correctly.
            half = self.marker_size_m / 2.0
            obj_pts = np.array([
                [-half,  half, 0.0],
                [ half,  half, 0.0],
                [ half, -half, 0.0],
                [-half, -half, 0.0],
            ], dtype=np.float64)

            img_pts = corner_pts.astype(np.float64)

            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)

            if not ok:
                self.get_logger().warn(
                    f'Marker {marker_id}: solvePnP failed — using '
                    f'depth centre without standoff rotation.')
                normal_cam = np.array([0.0, 0.0, -1.0])  # fallback: face camera
            else:
                # Marker normal in camera frame = rotation of Z-axis of marker
                R, _ = cv2.Rodrigues(rvec)
                normal_cam = -R[:, 2]   # marker faces along -Z of marker frame

            # ── 5. Compute goal = marker centre + standoff * normal ───────
            marker_pos_cam = np.array([x_cam, y_cam, z_cam])
            goal_pos_cam   = marker_pos_cam + self.standoff_m * normal_cam

            # ── 6. Transform goal from camera frame to world frame ────────
            goal_world = self._transform_to_world(
                goal_pos_cam, normal_cam,
                self.latest_rgb.header.stamp)

            if goal_world is None:
                continue   # TF not available yet

            # ── 7. Publish detection + RViz marker ───────────────────────
            self._publish_detection(marker_id, goal_world)
            self._publish_rviz_marker(marker_id, goal_world, depth_m)

            self.get_logger().info(
                f'Marker {marker_id} detected | '
                f'depth={depth_m:.2f} m | '
                f'goal=({goal_world.pose.pose.position.x:.2f}, '
                f'{goal_world.pose.pose.position.y:.2f})',
                throttle_duration_sec=1.0)

    # ==========================================================================
    # Depth sampling — median over a small patch for robustness
    # ==========================================================================

    def _sample_depth(self, depth_img: np.ndarray,
                      cx_px: float, cy_px: float) -> float | None:
        """
        Samples depth_img within a square patch of radius depth_sample_r
        around (cx_px, cy_px).  Returns the median in metres, or None if
        no valid readings exist.

        The D435i depth image is uint16 in millimetres.
        """
        r   = self.depth_sample_r
        h, w = depth_img.shape[:2]
        x0  = max(0,   int(cx_px) - r)
        x1  = min(w-1, int(cx_px) + r)
        y0  = max(0,   int(cy_px) - r)
        y1  = min(h-1, int(cy_px) + r)

        patch = depth_img[y0:y1+1, x0:x1+1].astype(np.float32)
        valid = patch[(patch > 0)]   # 0 = no measurement

        if valid.size == 0:
            return None

        depth_m = float(np.median(valid)) / 1000.0   # mm → m

        if not (self.min_depth_m <= depth_m <= self.max_depth_m):
            return None

        return depth_m

    # ==========================================================================
    # TF transform: camera frame → world frame
    # ==========================================================================

    def _transform_to_world(
            self,
            pos_cam: np.ndarray,
            normal_cam: np.ndarray,
            stamp) -> PoseWithCovarianceStamped | None:
        """
        Builds a PoseStamped in camera frame, transforms it to world frame
        using Isaac ROS VSLAM's TF tree, and returns a
        PoseWithCovarianceStamped ready to publish.

        The orientation is set so the robot faces the marker (yaw aligned
        with the inverse of the marker normal projected onto the XY plane).
        """
        try:
            tf_stamped = self._tf_buffer.lookup_transform(
                self.world_frame,
                self.camera_frame,
                rclpy.time.Time(),          # latest available
                timeout=rclpy.duration.Duration(seconds=0.1))
        except TransformException as e:
            self.get_logger().warn(
                f'TF {self.camera_frame} → {self.world_frame} '
                f'unavailable: {e}',
                throttle_duration_sec=3.0)
            return None

        # Build PoseStamped in camera frame
        pose_cam = PoseStamped()
        pose_cam.header.stamp    = stamp
        pose_cam.header.frame_id = self.camera_frame
        pose_cam.pose.position.x = float(pos_cam[0])
        pose_cam.pose.position.y = float(pos_cam[1])
        pose_cam.pose.position.z = float(pos_cam[2])
        pose_cam.pose.orientation.w = 1.0   # orientation handled after transform

        # Transform position to world frame
        pose_world = tf2_geometry_msgs.do_transform_pose(pose_cam, tf_stamped)

        # Compute goal yaw: robot should face toward the marker from the goal.
        # The marker normal in world frame points FROM marker TOWARD goal.
        # Robot faces opposite direction (back toward marker).
        # We project the normal onto XY plane and compute yaw.
        n = normal_cam / (np.linalg.norm(normal_cam) + 1e-9)

        # Rotate normal to world frame (rotation only, no translation)
        import tf_transformations
        q = tf_stamped.transform.rotation
        q_arr = [q.x, q.y, q.z, q.w]
        R_world = tf_transformations.quaternion_matrix(q_arr)[:3, :3]
        n_world = R_world @ n

        # Robot faces from goal BACK toward marker → opposite of normal
        yaw = math.atan2(-n_world[1], -n_world[0])

        # Pack into PoseWithCovarianceStamped
        result = PoseWithCovarianceStamped()
        result.header.stamp    = stamp
        result.header.frame_id = str(0)   # placeholder; overwritten below
        result.pose.pose.position.x    = pose_world.pose.position.x
        result.pose.pose.position.y    = pose_world.pose.position.y
        result.pose.pose.position.z    = 0.0   # ground plane
        result.pose.pose.orientation.z = math.sin(yaw / 2.0)
        result.pose.pose.orientation.w = math.cos(yaw / 2.0)

        return result

    # ==========================================================================
    # Publishing
    # ==========================================================================

    def _publish_detection(self, marker_id: int,
                           pose: PoseWithCovarianceStamped):
        """
        Publishes the detection with the marker ID encoded in header.frame_id
        so MissionController can filter by target_marker_id without a custom msg.
        """
        pose.header.frame_id = str(marker_id)
        self.detection_pub.publish(pose)

    def _publish_rviz_marker(self, marker_id: int,
                              pose: PoseWithCovarianceStamped,
                              depth_m: float):
        """Publishes a sphere + text label for RViz."""
        now   = self.get_clock().now().to_msg()
        array = MarkerArray()

        # Sphere at detected goal position
        sphere = Marker()
        sphere.header.stamp    = now
        sphere.header.frame_id = self.world_frame
        sphere.ns     = 'aruco_detections'
        sphere.id     = marker_id * 2
        sphere.type   = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose   = pose.pose.pose
        sphere.scale  = Vector3(x=0.15, y=0.15, z=0.15)
        sphere.color  = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
        array.markers.append(sphere)

        # Text label above the sphere
        label = Marker()
        label.header.stamp    = now
        label.header.frame_id = self.world_frame
        label.ns     = 'aruco_labels'
        label.id     = marker_id * 2 + 1
        label.type   = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose   = pose.pose.pose
        label.pose.position.z += 0.25
        label.scale.z = 0.25
        label.color   = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        label.text    = f'ID:{marker_id}  {depth_m:.2f}m'
        array.markers.append(label)

        self.marker_pub.publish(array)


# ==============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()