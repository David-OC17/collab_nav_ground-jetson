#!/usr/bin/env python3
"""
ArUco Detector Node — ROS 2 Humble
=====================================
Detects ArUco markers from an RGB camera stream and publishes each
detected marker's ID and pose relative to the camera frame.

For each detected marker, publishes:
  - Marker ID encoded in header.frame_id
  - Pose: translation + rotation of the marker relative to camera_color_optical_frame

If the same marker ID appears multiple times in one frame (e.g. two faces
of a cube both visible), the poses are averaged before publishing.

Subscribes:
  - /camera/color/image_raw   (sensor_msgs/Image)      — RGB image
  - /camera/color/camera_info (sensor_msgs/CameraInfo) — intrinsics for solvePnP

Publishes:
  - /aruco/detections  (geometry_msgs/PoseArray)              — all markers this frame
  - /aruco/markers     (visualization_msgs/MarkerArray)       — RViz cubes
  - /aruco/{id}/pose   (geometry_msgs/PoseWithCovarianceStamped) — per-ID topic

Parameters:
  marker_size_m       0.13     m  — physical side length of the ArUco marker
  camera_frame        'camera_color_optical_frame'
  image_topic         '/camera/color/image_raw'
  camera_info_topic   '/camera/color/camera_info'
  aruco_dict          'DICT_4X4_50'  — ArUco dictionary to use
  min_detection_area  100      px²  — discard tiny/noisy detections
  publish_debug_image True          — publish annotated image on /aruco/debug_image
"""

import math
import numpy as np

import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


# ---------------------------------------------------------------------------
# ArUco dictionary map
# ---------------------------------------------------------------------------
ARUCO_DICTS = {
    'DICT_4X4_50':        cv2.aruco.DICT_4X4_50,
    'DICT_4X4_100':       cv2.aruco.DICT_4X4_100,
    'DICT_4X4_250':       cv2.aruco.DICT_4X4_250,
    'DICT_5X5_50':        cv2.aruco.DICT_5X5_50,
    'DICT_5X5_100':       cv2.aruco.DICT_5X5_100,
    'DICT_6X6_50':        cv2.aruco.DICT_6X6_50,
    'DICT_ARUCO_ORIGINAL': cv2.aruco.DICT_ARUCO_ORIGINAL,
}


class ArucoDetector(Node):

    def __init__(self):
        super().__init__('aruco_detector')

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter('marker_size_m',      0.13)
        self.declare_parameter('camera_frame',       'camera_color_optical_frame')
        self.declare_parameter('image_topic',        '/camera/camera/color/image_raw')
        self.declare_parameter('camera_info_topic',  '/camera/camera/color/camera_info')
        self.declare_parameter('aruco_dict',         'DICT_4X4_50')
        self.declare_parameter('min_detection_area', 100)
        self.declare_parameter('publish_debug_image', True)

        self.marker_size_m      = float(self.get_parameter('marker_size_m').value)
        self.camera_frame       = self.get_parameter('camera_frame').value
        self.image_topic        = self.get_parameter('image_topic').value
        self.camera_info_topic  = self.get_parameter('camera_info_topic').value
        aruco_dict_name         = self.get_parameter('aruco_dict').value
        self.min_detection_area = int(self.get_parameter('min_detection_area').value)
        self.publish_debug      = bool(self.get_parameter('publish_debug_image').value)

        # ------------------------------------------------------------------
        # ArUco setup
        # ------------------------------------------------------------------
        if aruco_dict_name not in ARUCO_DICTS:
            self.get_logger().error(
                f'Unknown aruco_dict "{aruco_dict_name}". '
                f'Valid options: {list(ARUCO_DICTS.keys())}')
            raise ValueError(f'Unknown ArUco dictionary: {aruco_dict_name}')

        self.aruco_dict   = cv2.aruco.Dictionary_get(ARUCO_DICTS[aruco_dict_name])
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # ------------------------------------------------------------------
        # Camera intrinsics (populated on first CameraInfo message)
        # ------------------------------------------------------------------
        self.camera_matrix = None   # 3×3 np.ndarray
        self.dist_coeffs   = None   # 1×5 np.ndarray
        self.camera_info_received = False

        # ------------------------------------------------------------------
        # 3D marker corner template (centred on marker, Z forward into scene)
        # half the marker size from centre to edge
        half = self.marker_size_m / 2.0
        self.obj_points = np.array([
            [-half,  half, 0.0],   # top-left
            [ half,  half, 0.0],   # top-right
            [ half, -half, 0.0],   # bottom-right
            [-half, -half, 0.0],   # bottom-left
        ], dtype=np.float32)

        # ------------------------------------------------------------------
        # Bridge
        # ------------------------------------------------------------------
        self.bridge = CvBridge()

        # ------------------------------------------------------------------
        # QoS
        # ------------------------------------------------------------------
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # ------------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------------
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self._image_callback, sensor_qos)

        self.info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic,
            self._camera_info_callback, reliable_qos)

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        # All detections this frame as a PoseArray
        self.pose_array_pub = self.create_publisher(
            PoseArray, '/aruco/detections', reliable_qos)

        # Per-marker-ID pose topic: /aruco/{id}/pose
        # Built dynamically as new IDs are seen
        self._id_publishers: dict = {}

        # RViz marker array
        self.marker_pub = self.create_publisher(
            MarkerArray, '/aruco/markers', reliable_qos)

        # Debug annotated image
        if self.publish_debug:
            self.debug_pub = self.create_publisher(
                Image, '/aruco/debug_image', sensor_qos)

        self.get_logger().info(
            f'ArucoDetector ready\n'
            f'  image        ← {self.image_topic}\n'
            f'  camera_info  ← {self.camera_info_topic}\n'
            f'  camera_frame = {self.camera_frame}\n'
            f'  marker_size  = {self.marker_size_m} m\n'
            f'  aruco_dict   = {aruco_dict_name}\n'
            f'  detections   → /aruco/detections\n'
            f'  per-ID       → /aruco/{{id}}/pose'
        )

    # ==========================================================================
    # Camera info callback — capture intrinsics once
    # ==========================================================================

    def _camera_info_callback(self, msg: CameraInfo):
        if self.camera_info_received:
            return   # only need it once

        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape((3, 3))
        self.dist_coeffs   = np.array(msg.d, dtype=np.float64)
        self.camera_info_received = True

        self.get_logger().info(
            f'Camera intrinsics received\n'
            f'  fx={self.camera_matrix[0,0]:.1f}  fy={self.camera_matrix[1,1]:.1f}\n'
            f'  cx={self.camera_matrix[0,2]:.1f}  cy={self.camera_matrix[1,2]:.1f}\n'
            f'  distortion={self.dist_coeffs.tolist()}'
        )

    # ==========================================================================
    # Image callback — main detection pipeline
    # ==========================================================================

    def _image_callback(self, msg: Image):
        if not self.camera_info_received:
            self.get_logger().warn(
                'Waiting for camera intrinsics…',
                throttle_duration_sec=5.0)
            return

        # Convert ROS image → OpenCV BGR
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Detect markers ────────────────────────────────────────────────
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if self.publish_debug:
            debug_frame = frame.copy()

        if ids is None or len(ids) == 0:
            if self.publish_debug:
                self._publish_debug(debug_frame, msg.header)
            return

        # ── Filter tiny detections ────────────────────────────────────────
        valid_corners = []
        valid_ids     = []
        for i, corner in enumerate(corners):
            area = cv2.contourArea(corner[0])
            if area >= self.min_detection_area:
                valid_corners.append(corner)
                valid_ids.append(ids[i][0])

        if not valid_corners:
            if self.publish_debug:
                self._publish_debug(debug_frame, msg.header)
            return

        # ── Solve PnP for each marker ─────────────────────────────────────
        # Group by marker ID to average duplicate detections (e.g. cube faces)
        id_poses: dict = {}   # marker_id → list of (tvec, rvec)

        for i, corner in enumerate(valid_corners):
            marker_id = valid_ids[i]

            success, rvec, tvec = cv2.solvePnP(
                self.obj_points,
                corner[0],
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE   # best for square markers
            )

            if not success:
                continue

            if marker_id not in id_poses:
                id_poses[marker_id] = []
            id_poses[marker_id].append((tvec.flatten(), rvec.flatten()))

            if self.publish_debug:
                cv2.aruco.drawDetectedMarkers(debug_frame, [corner], np.array([[marker_id]]))
                cv2.drawFrameAxes(
                    debug_frame, self.camera_matrix, self.dist_coeffs,
                    rvec, tvec, self.marker_size_m * 0.5)

        # ── Average duplicates and publish ───────────────────────────────
        now        = msg.header.stamp
        pose_array = PoseArray()
        pose_array.header.stamp    = now
        pose_array.header.frame_id = self.camera_frame
        rviz_array = MarkerArray()
        rviz_id    = 0

        for marker_id, pose_list in id_poses.items():

            # Average translation and rotation across all detections of this ID
            avg_tvec = np.mean([p[0] for p in pose_list], axis=0)
            avg_rvec = np.mean([p[1] for p in pose_list], axis=0)

            # Convert rvec → quaternion
            qx, qy, qz, qw = self._rvec_to_quaternion(avg_rvec)

            # ── Build Pose (camera frame) ─────────────────────────────────
            pose = Pose()
            pose.position.x    = float(avg_tvec[0])
            pose.position.y    = float(avg_tvec[1])
            pose.position.z    = float(avg_tvec[2])
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw
            pose_array.poses.append(pose)

            # ── PoseWithCovarianceStamped on /aruco/{id}/pose ─────────────
            pwcs = PoseWithCovarianceStamped()
            pwcs.header.stamp    = now
            pwcs.header.frame_id = self.camera_frame   # ← camera frame
            # Encode marker ID in child frame_id for downstream consumers
            pwcs.pose.pose = pose

            pub = self._get_id_publisher(marker_id)
            pub.publish(pwcs)

            # ── RViz marker ───────────────────────────────────────────────
            dist = float(np.linalg.norm(avg_tvec))
            rviz_array.markers.append(
                self._make_rviz_marker(
                    rviz_id, now, pose, marker_id, dist))
            rviz_id += 1

            self.get_logger().info(
                f'ArUco ID={marker_id} | '
                f'pos=({avg_tvec[0]:.3f}, {avg_tvec[1]:.3f}, {avg_tvec[2]:.3f}) m '
                f'in {self.camera_frame} | '
                f'dist={dist:.3f} m'
                + (f' [averaged {len(pose_list)} faces]'
                   if len(pose_list) > 1 else ''),
                throttle_duration_sec=0.5
            )

        self.pose_array_pub.publish(pose_array)
        self.marker_pub.publish(rviz_array)

        if self.publish_debug:
            self._publish_debug(debug_frame, msg.header)

    # ==========================================================================
    # Helpers
    # ==========================================================================

    def _get_id_publisher(self, marker_id: int):
        """Lazily create a publisher for /aruco/{id}/pose."""
        if marker_id not in self._id_publishers:
            topic = f'/aruco/{marker_id}/pose'
            reliable_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10
            )
            self._id_publishers[marker_id] = self.create_publisher(
                PoseWithCovarianceStamped, topic, reliable_qos)
            self.get_logger().info(
                f'New publisher created: {topic}')
        return self._id_publishers[marker_id]

    @staticmethod
    def _rvec_to_quaternion(rvec: np.ndarray):
        """
        Converts an OpenCV rotation vector (Rodrigues) to a quaternion (x,y,z,w).
        """
        angle = np.linalg.norm(rvec)
        if angle < 1e-9:
            return 0.0, 0.0, 0.0, 1.0

        axis  = rvec / angle
        s     = math.sin(angle / 2.0)
        return (float(axis[0] * s),
                float(axis[1] * s),
                float(axis[2] * s),
                float(math.cos(angle / 2.0)))

    def _make_rviz_marker(self, marker_id_rviz: int, stamp,
                          pose: Pose, marker_id: int, dist: float) -> Marker:
        m = Marker()
        m.header.stamp    = stamp
        m.header.frame_id = self.camera_frame
        m.ns     = 'aruco_detections'
        m.id     = marker_id_rviz
        m.type   = Marker.CUBE
        m.action = Marker.ADD
        m.pose   = pose
        m.scale.x = self.marker_size_m
        m.scale.y = self.marker_size_m
        m.scale.z = 0.01   # flat marker representation
        m.color  = ColorRGBA(r=1.0, g=0.8, b=0.0, a=0.85)

        # Lifetime: marker disappears if not re-detected within 0.5 s
        m.lifetime.sec     = 0
        m.lifetime.nanosec = 500_000_000
        return m

    def _publish_debug(self, frame: np.ndarray, header):
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            debug_msg.header = header
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f'Debug image publish failed: {e}')


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