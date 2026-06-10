#!/usr/bin/env python3
"""
measure_extrinsics.py
─────────────────────────────────────────────────────────────────────────────
Script de ayuda para calcular el TF estático
  base_footprint → camera_color_optical_frame

Coloca UN marcador ArUco en una posición conocida del mundo y mueve el robot
a una posición conocida.  El script resuelve los extrínsecos comparando la
pose medida con la esperada.

Uso rápido (si conoces los offsets físicos con cinta métrica):
  Solo ajusta las variables PHYSICAL_OFFSET en el launch file.

Uso con este script (calibración precisa):
  1. Coloca aruco_0 en (0, 0, 0) orientado hacia la cámara.
  2. Posiciona el robot con base_footprint en (−1.0, 0, 0), yaw=0.
  3. Ejecuta:
       python3 measure_extrinsics.py
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from scipy.spatial.transform import Rotation
import cv2


def rvec_tvec_to_mat(rvec, tvec):
    R_mat, _ = cv2.Rodrigues(rvec)
    M = np.eye(4)
    M[:3, :3] = R_mat
    M[:3, 3]  = tvec.flatten()
    return M


def mat_to_tf_args(M):
    """Convierte 4×4 a los args de static_transform_publisher."""
    t    = M[:3, 3]
    q    = Rotation.from_matrix(M[:3, :3]).as_quat()   # [x,y,z,w]
    rpy  = Rotation.from_matrix(M[:3, :3]).as_euler('xyz', degrees=False)
    print('\n── static_transform_publisher args ──────────────────────────')
    print(f'  x={t[0]:.4f}  y={t[1]:.4f}  z={t[2]:.4f}')
    print(f'  roll={rpy[0]:.4f}  pitch={rpy[1]:.4f}  yaw={rpy[2]:.4f}')
    print(f'  (qx={q[0]:.4f} qy={q[1]:.4f} qz={q[2]:.4f} qw={q[3]:.4f})')


def compute_extrinsics(T_world_basefoot, T_world_aruco, T_cam_aruco):
    """
    Dado:
        T_world_basefoot : pose conocida del robot en el mundo
        T_world_aruco    : pose conocida del aruco en el mundo
        T_cam_aruco      : pose medida del aruco desde la cámara (PnP)

    Devuelve:
        T_basefoot_cam   : transform que debes usar como TF estático
    """
    # T_world_cam = T_world_aruco * inv(T_cam_aruco)
    T_world_cam = T_world_aruco @ np.linalg.inv(T_cam_aruco)

    # T_basefoot_cam = inv(T_world_basefoot) * T_world_cam
    T_basefoot_cam = np.linalg.inv(T_world_basefoot) @ T_world_cam

    return T_basefoot_cam


if __name__ == '__main__':
    # ── Ejemplo con valores sintéticos ─────────────────────────────────────
    print('Ejemplo de calibración de extrínsecos (valores sintéticos)')

    # Robot en (−1.0, 0, 0), orientado a +x (yaw=0)
    T_w_bf = np.eye(4)
    T_w_bf[:3, 3] = [-1.0, 0.0, 0.0]

    # Marcador en origen del mundo, en el suelo
    T_w_ar = np.eye(4)

    # PnP devuelve aruco visto desde la cámara:
    # cámara a 1 m frente al marcador, a 15 cm de altura
    # (esto es lo que obtendría estimatePoseSingleMarkers)
    T_cam_ar_synthetic = np.eye(4)
    T_cam_ar_synthetic[:3, 3] = [0.0, 0.15, 1.0]   # x, y, z en frame óptico
    # Pequeña rotación para simular inclinación
    T_cam_ar_synthetic[:3, :3] = Rotation.from_euler(
        'xyz', [90, 0, 0], degrees=True).as_matrix()

    T_bf_cam = compute_extrinsics(T_w_bf, T_w_ar, T_cam_ar_synthetic)

    print('\nT_basefoot_camera calculado:')
    print(np.round(T_bf_cam, 4))
    mat_to_tf_args(T_bf_cam)
    print()
    print('Copia estos valores al launch file en CAMERA_OFFSET')
