#!/usr/bin/env python3
"""
alignment_node.py
─────────────────────────────────────────────────────────────────────────────
Mantiene el TF dinámico world → odom en dos fases:

  Fase 1 — Inicialización con el dron (one-shot):
    Suscribe /aruco/amr/pose (PoseWithCovarianceStamped, world frame)
    Publica world→odom asumiendo T_odom_bf = identity
    (el EKF acaba de arrancar, robot no se ha movido)

  Fase 2 — Correcciones continuas con la cámara del AMR:
    Suscribe /aruco_pose (PoseWithCovarianceStamped, world frame)
    Computa T_world_odom = T_world_bf x inv(T_odom_bf)
    lookupando el TF odom→base_footprint del EKF en tiempo real

  Timer a 20 Hz republica el TF dinámico para que no expire en el buffer.

Transición de fases:
  - El nodo arranca en estado WAITING.
  - Al recibir el primer /aruco/amr/pose pasa a INITIALIZED.
  - Al recibir el primer /aruco_pose (con nodo inicializado) pasa a CORRECTING.
  - Una vez en CORRECTING, los mensajes de /aruco/amr/pose se ignoran.
"""

import math
import threading

import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from tf2_ros import Buffer, TransformListener, TransformBroadcaster


# ─────────────────────────────────────────────────────────────────────────────
class AlignmentNode(Node):

    def __init__(self):
        super().__init__('alignment_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('world_frame',      'world')
        self.declare_parameter('odom_frame',       'odom')
        self.declare_parameter('base_frame',       'base_footprint')
        self.declare_parameter('publish_rate',     20.0)    # Hz
        self.declare_parameter('init_topic',       '/aruco/amr/pose')
        self.declare_parameter('correction_topic', '/aruco_pose')

        self._world_frame = self.get_parameter('world_frame').value
        self._odom_frame  = self.get_parameter('odom_frame').value
        self._base_frame  = self.get_parameter('base_frame').value

        # ── Estado ────────────────────────────────────────────────────────────
        # 'waiting'     → sin inicializar
        # 'initialized' → pose inicial del dron recibida
        # 'correcting'  → correcciones del AMR activas
        self._phase: str                   = 'waiting'
        self._T_world_odom: np.ndarray | None = None   # 4×4 homogénea
        self._lock = threading.Lock()

        # ── TF ────────────────────────────────────────────────────────────────
        self.tf_buffer      = Buffer()
        self.tf_listener    = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ── Suscripciones ─────────────────────────────────────────────────────
        init_topic       = self.get_parameter('init_topic').value
        correction_topic = self.get_parameter('correction_topic').value

        self.create_subscription(
            PoseWithCovarianceStamped, init_topic, self._cb_init, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, correction_topic, self._cb_correction, 10)

        # ── Timer de republishing ─────────────────────────────────────────────
        rate = self.get_parameter('publish_rate').value
        self.create_timer(1.0 / rate, self._publish_tf)

        self.get_logger().info(
            f'alignment_node listo | init={init_topic} | '
            f'correction={correction_topic} | {rate:.0f} Hz')

    # ─── Fase 1: inicialización con el dron ───────────────────────────────────
    def _cb_init(self, msg: PoseWithCovarianceStamped):
        with self._lock:
            if self._phase != 'waiting':
                return   # ya inicializado, ignorar

        if msg.header.frame_id != self._world_frame:
            self.get_logger().error(
                f'[Fase 1] frame_id esperado={self._world_frame}, '
                f'recibido={msg.header.frame_id} — ignorando')
            return

        # NaN position → sin datos de localización: publicar identity como fallback
        if (math.isnan(msg.pose.pose.position.x) or
                math.isnan(msg.pose.pose.position.y)):
            T_world_odom = np.eye(4)
            with self._lock:
                self._T_world_odom = T_world_odom
                self._phase = 'initialized'
            self.get_logger().warn(
                '[Fase 1] Pose con NaN — publicando identity world→odom como fallback.')
            return

        # Intentar lookup de T_odom_bf. Si no está disponible, asumir identity.
        try:
            tf_ob = self.tf_buffer.lookup_transform(
                self._odom_frame, self._base_frame, Time())
            T_odom_bf = self._ros_tf_to_mat(tf_ob)
        except Exception:
            self.get_logger().warn(
                '[Fase 1] odom→base_footprint no disponible, asumiendo identity.')
            T_odom_bf = np.eye(4)

        T_world_bf   = self._pose_to_mat(msg.pose.pose)
        T_world_odom = T_world_bf @ np.linalg.inv(T_odom_bf)

        with self._lock:
            self._T_world_odom = T_world_odom
            self._phase = 'initialized'

        yaw = math.degrees(self._yaw_from_mat(T_world_odom))
        self.get_logger().info(
            f'[Fase 1] world→odom inicializado: '
            f'x={T_world_odom[0,3]:.3f} '
            f'y={T_world_odom[1,3]:.3f} '
            f'yaw={yaw:.1f}°')

    # ─── Fase 2: correcciones con la cámara del AMR ───────────────────────────
    def _cb_correction(self, msg: PoseWithCovarianceStamped):
        with self._lock:
            phase = self._phase

        if phase == 'waiting':
            self.get_logger().warn(
                '[Fase 2] Corrección recibida pero no inicializado aún. '
                'Espera a que el dron dé la pose inicial.',
                throttle_duration_sec=2.0)
            return

        if msg.header.frame_id != self._world_frame:
            return

        # Lookup T_odom_bf del EKF (pose actual del robot en odom)
        try:
            tf_ob = self.tf_buffer.lookup_transform(
                self._odom_frame, self._base_frame, Time())
            T_odom_bf = self._ros_tf_to_mat(tf_ob)
        except Exception as e:
            self.get_logger().warn(
                f'[Fase 2] lookup odom→base_footprint falló: {e}',
                throttle_duration_sec=1.0)
            return

        # T_world_bf del ArUco de la cámara del AMR
        T_world_bf   = self._pose_to_mat(msg.pose.pose)

        # T_world_odom = T_world_bf × inv(T_odom_bf)
        T_world_odom = T_world_bf @ np.linalg.inv(T_odom_bf)

        with self._lock:
            self._T_world_odom = T_world_odom
            if phase == 'initialized':
                self._phase = 'correcting'
                self.get_logger().info(
                    '[Fase 2] Primera corrección AMR recibida — '
                    'cambiando a modo corrección continua.')

    # ─── Timer: republica world→odom a 20 Hz ─────────────────────────────────
    def _publish_tf(self):
        with self._lock:
            T = self._T_world_odom

        if T is None:
            return

        ts = TransformStamped()
        ts.header.stamp    = self.get_clock().now().to_msg()
        ts.header.frame_id = self._world_frame
        ts.child_frame_id  = self._odom_frame

        ts.transform.translation.x = float(T[0, 3])
        ts.transform.translation.y = float(T[1, 3])
        ts.transform.translation.z = 0.0

        q = Rotation.from_matrix(T[:3, :3]).as_quat()   # [x, y, z, w]
        ts.transform.rotation.x = float(q[0])
        ts.transform.rotation.y = float(q[1])
        ts.transform.rotation.z = float(q[2])
        ts.transform.rotation.w = float(q[3])

        self.tf_broadcaster.sendTransform(ts)

    # ─── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _pose_to_mat(pose) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = Rotation.from_quat([
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w,
        ]).as_matrix()
        M[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        return M

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
    def _yaw_from_mat(T: np.ndarray) -> float:
        return float(Rotation.from_matrix(T[:3, :3]).as_euler('xyz')[2])


# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = AlignmentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
