#!/usr/bin/env python3
"""
aruco_localizer_node.py
────────────────────────────────────────────────────────────────────────────
Estima la pose del robot en el frame 'world' detectando marcadores ArUco
cuyas poses en el mundo son conocidas (publicadas como TFs estáticas).

Pipeline por frame:
  1. Detectar marcadores ArUco en la imagen de color (RealSense D435i).
  2. Resolver PnP (IPPE_SQUARE) → T_camera_aruco por cada marcador visible.
  3. Componer:
       T_w_bf = T_w_ar  *  inv(T_cam_ar)  *  inv(T_bf_cam)
  4. Fusionar estimaciones de múltiples marcadores (peso = 1/dist²).
  5. Publicar PoseWithCovarianceStamped en /aruco_pose para el EKF.

TF tree esperado:
  world ──(static)──► aruco_0
  world ──(static)──► aruco_1    (un TF por cada ID conocido)
  base_footprint ──(static)──► camera_color_optical_frame

Integración con robot_localization EKF:
  ekf_node:
    ros__parameters:
      pose0: /aruco_pose
      pose0_config: [true, true, false,   # x, y, z
                     false, false, true,  # roll, pitch, yaw
                     false, false, false, false, false, false,
                     false, false, false]
      pose0_differential: false
      pose0_relative: false
"""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import (
    PoseWithCovarianceStamped, TransformStamped, Point, Quaternion
)
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformListener, TransformBroadcaster

from cv_bridge import CvBridge

from ros2_security import SecureNodeMixin


# ─────────────────────────────────────────────────────────────────────────────
class ArucoLocalizerNode(SecureNodeMixin, Node):

    def __init__(self):
        super().__init__('aruco_localizer')
        self.declare_parameter('certs_dir', './certs')
        self.security_init(certs_dir=self.get_parameter('certs_dir').value)

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('marker_size',    0.135)        # metros
        self.declare_parameter('aruco_dict_id',  cv2.aruco.DICT_4X4_50)
        self.declare_parameter('marker_ids',     [15, 16, 17, 21])
        self.declare_parameter('world_frame',    'world')
        self.declare_parameter('base_frame',     'base_footprint')
        self.declare_parameter('camera_frame',   'camera_color_optical_frame')
        self.declare_parameter('image_topic',
            '/camera/realsense2_camera/color/image_raw')
        self.declare_parameter('camera_info_topic',
            '/camera/realsense2_camera/color/camera_info')
        self.declare_parameter('pose_topic',     '/aruco_pose')
        self.declare_parameter('debug_image',    True)
        self.declare_parameter('publish_tf',     False)   # TF directo (debug)
        # Modelo de covarianza: sigma = base + dist_factor * distancia
        self.declare_parameter('cov_base_xy',    0.01)    # m
        self.declare_parameter('cov_dist_xy',    0.02)    # m por metro de distancia
        self.declare_parameter('cov_base_yaw',   0.02)    # rad
        self.declare_parameter('cov_dist_yaw',   0.03)    # rad por metro
        # Umbral: descartar detecciones más lejos de este valor (m)
        self.declare_parameter('max_marker_dist', 1.5)
        # Mínimo de píxeles de perímetro del marcador (filtra detecciones ruidosas)
        self.declare_parameter('min_perimeter_px', 50.0)

        p = self.get_parameter

        self.marker_size   = p('marker_size').value
        self.world_frame   = p('world_frame').value
        self.base_frame    = p('base_frame').value
        self.camera_frame  = p('camera_frame').value
        self.marker_ids    = set(int(x) for x in p('marker_ids').value)
        self.debug_image   = p('debug_image').value
        self.publish_tf    = p('publish_tf').value
        self.max_dist      = p('max_marker_dist').value
        self.min_perim     = p('min_perimeter_px').value

        # ── Detector ArUco ────────────────────────────────────────────────────
        dict_id      = int(p('aruco_dict_id').value)
        self.aruco_dict   = cv2.aruco.Dictionary_get(dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        # Ajustes para arena con iluminación artificial
        self.aruco_params.adaptiveThreshWinSizeMin  = 3
        self.aruco_params.adaptiveThreshWinSizeMax  = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        self.aruco_params.minMarkerPerimeterRate    = 0.03
        self.aruco_params.maxMarkerPerimeterRate    = 4.0
        self.aruco_params.polygonalApproxAccuracyRate = 0.05
        self.aruco_params.cornerRefinementMethod    = cv2.aruco.CORNER_REFINE_SUBPIX
        

        # Esquinas 3D del marcador en su frame propio (plano z=0)
        s = self.marker_size / 2.0
        self._marker_pts_3d = np.array([
            [-s,  s, 0.0],
            [ s,  s, 0.0],
            [ s, -s, 0.0],
            [-s, -s, 0.0],
        ], dtype=np.float32)

        # ── Intrínsecos de cámara (llenados desde /camera_info) ───────────────
        self.K: np.ndarray | None = None
        self.D: np.ndarray | None = None

        # ── TF ────────────────────────────────────────────────────────────────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        if self.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self)

        # Cache de T_base_footprint_camera (estático, no cambia)
        self._T_bf_cam: np.ndarray | None = None

        # ── CV Bridge ─────────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── Suscripciones ─────────────────────────────────────────────────────
        self.create_subscription(
            CameraInfo, p('camera_info_topic').value, self._cb_info, 1)
        self.create_subscription(
            Image, p('image_topic').value, self._cb_image, 10)

        # ── Publicadores ──────────────────────────────────────────────────────
        self.pose_pub = self.create_secure_publisher(p('pose_topic').value, PoseWithCovarianceStamped, 10)

        if self.debug_image:
            self.dbg_pub = self.create_secure_publisher('/aruco_localizer/debug_image', Image, 10)

        # Marcadores RViz para visualizar dónde se "ven" los ArUcos en el mundo
        self.viz_pub = self.create_secure_publisher('/aruco_localizer/detections_viz', MarkerArray, 10)

        self.get_logger().info(
            f'ArUco localizer listo. IDs conocidos: {sorted(self.marker_ids)}, '
            f'tamaño marcador: {self.marker_size} m')

    # ─── Callback: camera_info ────────────────────────────────────────────────
    def _cb_info(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.D = np.array(msg.d, dtype=np.float64)
            self.get_logger().info(
                f'Intrínsecos recibidos. fx={self.K[0,0]:.1f} fy={self.K[1,1]:.1f}')

    # ─── Callback principal: imagen ───────────────────────────────────────────
    def _cb_image(self, msg: Image):
        if self.K is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None or len(ids) == 0:
            if self.debug_image:
                self._publish_debug(frame, msg.header.stamp)
            return

        # ── Caché del TF estático base_footprint → camera ─────────────────────
        if self._T_bf_cam is None:
            self._T_bf_cam = self._lookup_matrix(
                self.base_frame, self.camera_frame)
            if self._T_bf_cam is None:
                return   # aún no disponible, reintentar en el próximo frame

        # ── Procesar cada marcador detectado ──────────────────────────────────
        estimates = []        # [(T_w_bf, dist_m, sigma_xy, sigma_yaw, marker_id)]
        viz_markers = MarkerArray()

        for i, mid in enumerate(ids.flatten()):
            mid = int(mid)
            if mid not in self.marker_ids:
                continue

            # Filtro por tamaño mínimo (descarta detecciones borrosas/lejanas)
            perim = cv2.arcLength(corners[i][0], closed=True)
            if perim < self.min_perim:
                continue

            # ── PnP: obtiene T_camera_aruco ───────────────────────────────────
            ok, rvec, tvec = cv2.solvePnP(
                self._marker_pts_3d,
                corners[i][0].astype(np.float32),
                self.K, self.D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                continue

            dist_m = float(np.linalg.norm(tvec))
            if dist_m > self.max_dist:
                self.get_logger().debug(
                    f'Marcador {mid} demasiado lejos ({dist_m:.2f} m), descartado.')
                continue

            # ── Lookup world → aruco_<id> (estático) ──────────────────────────
            T_w_ar = self._lookup_matrix(self.world_frame, f'aruco_{mid}')
            if T_w_ar is None:
                self.get_logger().warn(
                    f'TF world→aruco_{mid} no encontrado.  '
                    '¿Publicaste los TFs estáticos de los marcadores?',
                    throttle_duration_sec=3.0)
                continue

            # ── Composición de la pose ─────────────────────────────────────────
            # T_w_bf = T_w_ar * inv(T_cam_ar) * inv(T_bf_cam)
            R_cam_ar, _ = cv2.Rodrigues(rvec)
            T_cam_ar = self._build_mat(R_cam_ar, tvec.flatten())

            T_ar_cam = np.linalg.inv(T_cam_ar)          # aruco → camera
            T_cam_bf = np.linalg.inv(self._T_bf_cam)    # camera → base_footprint

            T_w_bf = T_w_ar @ T_ar_cam @ T_cam_bf

            # ── Covarianza basada en distancia ────────────────────────────────
            sigma_xy  = (self.get_parameter('cov_base_xy').value
                         + self.get_parameter('cov_dist_xy').value * dist_m)
            sigma_yaw = (self.get_parameter('cov_base_yaw').value
                         + self.get_parameter('cov_dist_yaw').value * dist_m)

            estimates.append((T_w_bf, dist_m, sigma_xy, sigma_yaw, mid))

            # Marcador RViz en la posición estimada del robot
            viz_markers.markers.append(
                self._make_viz_marker(T_w_bf, mid, msg.header.stamp))

            # Debug: dibujar ejes sobre imagen
            if self.debug_image:
                cv2.aruco.drawDetectedMarkers(frame, [corners[i]], np.array([[mid]]))
                cv2.drawFrameAxes(
                    frame, self.K, self.D, rvec, tvec, self.marker_size * 0.5)
                cx = int(corners[i][0][:, 0].mean())
                cy = int(corners[i][0][:, 1].mean())
                cv2.putText(frame, f'ID:{mid} d={dist_m:.2f}m',
                            (cx - 40, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if self.debug_image:
            self._publish_debug(frame, msg.header.stamp)

        if viz_markers.markers:
            self.secure_publish(self.viz_pub, viz_markers)

        if not estimates:
            return

        # ── Fusionar estimaciones (peso = 1/dist²) ────────────────────────────
        T_final, sigma_xy_f, sigma_yaw_f = self._fuse_poses(estimates)

        # ── Publicar PoseWithCovarianceStamped ────────────────────────────────
        self._publish_pose(T_final, sigma_xy_f, sigma_yaw_f, msg.header.stamp)

        # ── Opción: broadcast TF directo (solo para debug, no usar con EKF) ───
        if self.publish_tf:
            self._broadcast_tf(T_final, msg.header.stamp)

    # ─── Fusión de múltiples poses ────────────────────────────────────────────
    def _fuse_poses(self, estimates):
        """
        Promedio ponderado por 1/dist² de todas las estimaciones de marcadores
        simultáneamente visibles.

        Args:
            estimates: lista de (T_w_bf, dist_m, sigma_xy, sigma_yaw, mid)

        Returns:
            (T_fused 4x4, sigma_xy, sigma_yaw)
        """
        if len(estimates) == 1:
            T, _, sx, sy, _ = estimates[0]
            return T, sx, sy

        # Pesos inversamente proporcionales al cuadrado de la distancia
        dists   = np.array([e[1] for e in estimates])
        weights = 1.0 / (dists ** 2 + 1e-6)
        weights /= weights.sum()

        # Promedio ponderado de la traslación
        t_avg = sum(w * e[0][:3, 3] for w, e in zip(weights, estimates))

        # Promedio ponderado de quaterniones (lineal + renormalización)
        # Funciona bien cuando las estimaciones son cercanas entre sí
        quats = np.array([
            Rotation.from_matrix(e[0][:3, :3]).as_quat() for e in estimates
        ])
        # Resolver ambigüedad de signo del quaternión
        ref = quats[0]
        for j in range(1, len(quats)):
            if np.dot(quats[j], ref) < 0.0:
                quats[j] = -quats[j]
        q_avg = (weights[:, None] * quats).sum(axis=0)
        q_avg /= np.linalg.norm(q_avg)

        T_fused = self._build_mat(
            Rotation.from_quat(q_avg).as_matrix(), t_avg)

        sx_avg = float(sum(w * e[2] for w, e in zip(weights, estimates)))
        sy_avg = float(sum(w * e[3] for w, e in zip(weights, estimates)))

        n = len(estimates)
        ids_str = [str(e[4]) for e in estimates]
        self.get_logger().debug(
            f'Fusión de {n} marcadores {ids_str}, '
            f'sigma_xy={sx_avg:.3f} m, sigma_yaw={sy_avg:.3f} rad')

        return T_fused, sx_avg, sy_avg

    # ─── Publicar pose ────────────────────────────────────────────────────────
    def _publish_pose(self, T: np.ndarray, sigma_xy: float,
                      sigma_yaw: float, stamp):
        out = PoseWithCovarianceStamped()
        out.header.stamp    = stamp
        out.header.frame_id = self.world_frame

        pos  = T[:3, 3]
        quat = Rotation.from_matrix(T[:3, :3]).as_quat()   # [x, y, z, w]

        out.pose.pose.position    = Point(
            x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
        out.pose.pose.orientation = Quaternion(
            x=float(quat[0]), y=float(quat[1]),
            z=float(quat[2]), w=float(quat[3]))

        # Matriz 6×6 (x, y, z, rx, ry, rz) — row-major
        # Para robot terrestre solo son relevantes x, y, yaw
        # z, roll, pitch con varianza muy alta → el EKF los ignora
        HIGH_VAR = 9999.0
        cov = np.diag([
            sigma_xy  ** 2,   # x
            sigma_xy  ** 2,   # y
            HIGH_VAR,         # z
            HIGH_VAR,         # roll
            HIGH_VAR,         # pitch
            sigma_yaw ** 2,   # yaw
        ])
        out.pose.covariance = cov.flatten().tolist()

        self.secure_publish(self.pose_pub, out)

    # ─── Broadcast TF directo (solo debug) ───────────────────────────────────
    def _broadcast_tf(self, T: np.ndarray, stamp):
        ts = TransformStamped()
        ts.header.stamp    = stamp
        ts.header.frame_id = self.world_frame
        ts.child_frame_id  = 'aruco_base_footprint_est'   # frame separado

        pos  = T[:3, 3]
        quat = Rotation.from_matrix(T[:3, :3]).as_quat()

        ts.transform.translation.x = float(pos[0])
        ts.transform.translation.y = float(pos[1])
        ts.transform.translation.z = float(pos[2])
        ts.transform.rotation.x    = float(quat[0])
        ts.transform.rotation.y    = float(quat[1])
        ts.transform.rotation.z    = float(quat[2])
        ts.transform.rotation.w    = float(quat[3])

        self.tf_broadcaster.sendTransform(ts)

    # ─── Helpers TF ───────────────────────────────────────────────────────────
    def _lookup_matrix(self, target: str, source: str) -> np.ndarray | None:
        """Busca TF target←source y lo devuelve como matriz 4×4."""
        try:
            tf = self.tf_buffer.lookup_transform(target, source, Time())
            return self._ros_tf_to_mat(tf)
        except Exception as exc:
            self.get_logger().debug(
                f'lookup_transform({target}←{source}): {exc}')
            return None

    @staticmethod
    def _ros_tf_to_mat(tf) -> np.ndarray:
        tr = tf.transform.translation
        ro = tf.transform.rotation
        M  = np.eye(4)
        M[:3, :3] = Rotation.from_quat(
            [ro.x, ro.y, ro.z, ro.w]).as_matrix()
        M[:3, 3] = [tr.x, tr.y, tr.z]
        return M

    @staticmethod
    def _build_mat(R_3x3: np.ndarray, t: np.ndarray) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = R_3x3
        M[:3, 3]  = t
        return M

    # ─── Debug image ──────────────────────────────────────────────────────────
    def _publish_debug(self, frame: np.ndarray, stamp):
        img_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        img_msg.header.stamp = stamp
        self.secure_publish(self.dbg_pub, img_msg)

    # ─── RViz marker para la pose estimada ───────────────────────────────────
    def _make_viz_marker(self, T: np.ndarray, mid: int, stamp) -> Marker:
        pos  = T[:3, 3]
        quat = Rotation.from_matrix(T[:3, :3]).as_quat()

        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp    = stamp
        m.ns              = 'aruco_robot_pose'
        m.id              = mid
        m.type            = Marker.ARROW
        m.action          = Marker.ADD

        m.pose.position.x    = float(pos[0])
        m.pose.position.y    = float(pos[1])
        m.pose.position.z    = 0.0
        m.pose.orientation.x = float(quat[0])
        m.pose.orientation.y = float(quat[1])
        m.pose.orientation.z = float(quat[2])
        m.pose.orientation.w = float(quat[3])

        m.scale.x = 0.3
        m.scale.y = 0.05
        m.scale.z = 0.05

        # Color por ID de marcador
        colors = [
            (1.0, 0.2, 0.2, 1.0),   # rojo
            (0.2, 1.0, 0.2, 1.0),   # verde
            (0.2, 0.2, 1.0, 1.0),   # azul
            (1.0, 1.0, 0.2, 1.0),   # amarillo
        ]
        r, g, b, a = colors[mid % len(colors)]
        m.color = ColorRGBA(r=r, g=g, b=b, a=a)
        m.lifetime.sec = 1
        return m


# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = ArucoLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
