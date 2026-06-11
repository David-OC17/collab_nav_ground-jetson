#!/usr/bin/env python3
"""
aruco_pose_viewer.py
────────────────────────────────────────────────────────────────
Ve la pose de marcadores ArUco respecto a la cámara D435i.
Sin ROS. Solo pyrealsense2 + OpenCV.

Uso:
    python3 aruco_pose_viewer.py [--size 0.15] [--dict 4x4_50] [--id 0]

Teclas:
    q / Esc  — salir
    s        — guardar frame actual como PNG
    r        — resetear promedio de pose
"""

import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

try:
    import pyrealsense2 as rs
    USE_REALSENSE = True
except ImportError:
    print('[WARN] pyrealsense2 no encontrado — usando webcam (OpenCV VideoCapture).')
    USE_REALSENSE = False


# ─────────────────────────────────────────────────────────────────────────────
DICT_MAP = {
    '4x4_50':  cv2.aruco.DICT_4X4_50,
    '4x4_100': cv2.aruco.DICT_4X4_100,
    '5x5_50':  cv2.aruco.DICT_5X5_50,
    '5x5_100': cv2.aruco.DICT_5X5_100,
    '6x6_250': cv2.aruco.DICT_6X6_250,
}


# ─────────────────────────────────────────────────────────────────────────────
def get_realsense_stream(width=640, height=480, fps=30):
    """Inicializa la D435i y devuelve (pipeline, profile)."""
    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(cfg)
    # Esperar a que el auto-exposure se estabilice
    for _ in range(30):
        pipeline.wait_for_frames()
    return pipeline, profile


def get_realsense_intrinsics(profile):
    """Extrae K y D directamente del perfil de la cámara."""
    stream   = profile.get_stream(rs.stream.color)
    intr     = stream.as_video_stream_profile().get_intrinsics()
    K = np.array([
        [intr.fx,    0,    intr.ppx],
        [   0,    intr.fy, intr.ppy],
        [   0,       0,       1   ],
    ], dtype=np.float64)
    D = np.array(intr.coeffs, dtype=np.float64)
    print(f'[INFO] Intrínsecos RealSense:')
    print(f'       fx={intr.fx:.2f}  fy={intr.fy:.2f}')
    print(f'       cx={intr.ppx:.2f}  cy={intr.ppy:.2f}')
    print(f'       dist={np.round(D,5)}')
    return K, D


def get_webcam_stream(device=0, width=640, height=480):
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


# ─────────────────────────────────────────────────────────────────────────────
def marker_3d_corners(size):
    s = size / 2.0
    return np.array([
        [-s,  s, 0],
        [ s,  s, 0],
        [ s, -s, 0],
        [-s, -s, 0],
    ], dtype=np.float32)


def rvec_tvec_to_pose(rvec, tvec):
    """Convierte rvec+tvec a (x,y,z en m) y (roll,pitch,yaw en grados)."""
    t   = tvec.flatten()
    rpy = Rotation.from_rotvec(rvec.flatten()).as_euler('xyz', degrees=True)
    return t, rpy


def draw_overlay(frame, K, D, corners, ids, rvecs, tvecs, marker_size,
                 pose_history, target_id):
    """Dibuja ejes, bounding-box, y texto de pose sobre el frame."""
    h, w = frame.shape[:2]

    for i, mid in enumerate(ids.flatten()):
        rvec = rvecs[i]
        tvec = tvecs[i]
        t, rpy = rvec_tvec_to_pose(rvec, tvec)
        dist = np.linalg.norm(t)

        # Ejes sobre el marcador
        cv2.drawFrameAxes(frame, K, D, rvec, tvec, marker_size * 0.6, thickness=3)
        cv2.aruco.drawDetectedMarkers(frame, [corners[i]], np.array([[mid]]))

        # Coordenadas del centro del marcador en imagen
        cx = int(corners[i][0][:, 0].mean())
        cy = int(corners[i][0][:, 1].mean())

        # Cuadro de texto
        tag = f'ID {mid}'
        cv2.rectangle(frame, (cx - 45, cy - 60), (cx + 140, cy - 5),
                      (20, 20, 20), -1)
        cv2.putText(frame, tag,
                    (cx - 40, cy - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 80), 2)
        cv2.putText(frame,
                    f'x={t[0]:+.3f}m y={t[1]:+.3f}m z={t[2]:.3f}m',
                    (cx - 40, cy - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 255, 100), 1)
        cv2.putText(frame,
                    f'R={rpy[0]:+.1f} P={rpy[1]:+.1f} Y={rpy[2]:+.1f} deg',
                    (cx - 40, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 200, 255), 1)

        # Si es el marcador objetivo, guardar en historial para promedio
        if mid == target_id:
            pose_history.append((t.copy(), rpy.copy()))

    # Panel de estadísticas (esquina superior izquierda)
    if pose_history and target_id is not None:
        poses = list(pose_history)
        t_arr   = np.array([p[0] for p in poses])
        rpy_arr = np.array([p[1] for p in poses])
        t_mean  = t_arr.mean(axis=0)
        t_std   = t_arr.std(axis=0)
        rpy_mean = rpy_arr.mean(axis=0)

        panel_lines = [
            f'ID {target_id}  (avg de {len(poses)} frames)',
            f'x = {t_mean[0]:+.4f} m   std={t_std[0]:.4f}',
            f'y = {t_mean[1]:+.4f} m   std={t_std[1]:.4f}',
            f'z = {t_mean[2]:+.4f} m   std={t_std[2]:.4f}',
            f'roll  = {rpy_mean[0]:+.2f} deg',
            f'pitch = {rpy_mean[1]:+.2f} deg',
            f'yaw   = {rpy_mean[2]:+.2f} deg',
        ]
        box_h = len(panel_lines) * 22 + 14
        cv2.rectangle(frame, (8, 8), (310, 8 + box_h), (20, 20, 20), -1)
        cv2.rectangle(frame, (8, 8), (310, 8 + box_h), (80, 80, 80), 1)
        for j, line in enumerate(panel_lines):
            color = (255, 255, 80) if j == 0 else (220, 220, 220)
            cv2.putText(frame, line, (14, 28 + j * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 1)

    # FPS contador (esquina inferior derecha)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='ArUco pose viewer — RealSense D435i')
    ap.add_argument('--size',   type=float, default=0.15,
                    help='Tamaño del marcador en metros (default: 0.15)')
    ap.add_argument('--dict',   type=str,   default='4x4_50',
                    choices=list(DICT_MAP.keys()),
                    help='Diccionario ArUco (default: 4x4_50)')
    ap.add_argument('--id',     type=int,   default=None,
                    help='ID del marcador a monitorear en el panel (default: cualquiera)')
    ap.add_argument('--avg',    type=int,   default=30,
                    help='Ventana de promedio de frames (default: 30)')
    ap.add_argument('--width',  type=int,   default=640)
    ap.add_argument('--height', type=int,   default=480)
    ap.add_argument('--fps',    type=int,   default=30)
    args = ap.parse_args()

    marker_size = args.size
    target_id   = args.id
    pose_history = deque(maxlen=args.avg)

    # ── Inicializar cámara ────────────────────────────────────────────────────
    if USE_REALSENSE:
        pipeline, profile = get_realsense_stream(args.width, args.height, args.fps)
        K, D = get_realsense_intrinsics(profile)
    else:
        cap = get_webcam_stream(0, args.width, args.height)
        # Intrínsecos genéricos (poco precisos — solo para prueba rápida)
        fx = fy = max(args.width, args.height) * 1.2
        K = np.array([[fx, 0, args.width/2],
                      [0, fy, args.height/2],
                      [0,  0,       1      ]], dtype=np.float64)
        D = np.zeros(5, dtype=np.float64)
        print('[WARN] Usando intrínsecos aproximados. Para precisión usa la D435i.')

    # ── Detector ArUco (API OpenCV 4.5.x) ────────────────────────────────────
    aruco_dict   = cv2.aruco.Dictionary_get(DICT_MAP[args.dict])
    aruco_params = cv2.aruco.DetectorParameters_create()
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    pts3d = marker_3d_corners(marker_size)

    print(f'\n[INFO] Buscando marcadores ArUco {args.dict.upper()}, '
          f'tamaño = {marker_size} m')
    print('[INFO] Teclas:  q/Esc=salir   s=guardar frame   r=resetear promedio\n')

    t_last = time.time()
    fps_disp = 0.0
    frame_count = 0

    # ── Loop principal ────────────────────────────────────────────────────────
    while True:
        # Capturar frame
        if USE_REALSENSE:
            frames     = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
        else:
            ok, frame = cap.read()
            if not ok:
                break

        frame_count += 1
        now = time.time()
        if now - t_last >= 1.0:
            fps_disp = frame_count / (now - t_last)
            frame_count = 0
            t_last = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=aruco_params)

        if ids is not None and len(ids) > 0:
            rvecs, tvecs = [], []
            for i in range(len(ids)):
                ok, rvec, tvec = cv2.solvePnP(
                    pts3d,
                    corners[i][0].astype(np.float32),
                    K, D,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                rvecs.append(rvec)
                tvecs.append(tvec)

                # También imprimir en terminal (throttle: 1 vez / segundo)
                mid = int(ids[i])
                if frame_count == 0:   # solo 1 vez por segundo
                    t_v, rpy = rvec_tvec_to_pose(rvec, tvec)
                    print(f'ID {mid:3d} | '
                          f'x={t_v[0]:+.3f} y={t_v[1]:+.3f} z={t_v[2]:.3f} m | '
                          f'roll={rpy[0]:+.1f}° pitch={rpy[1]:+.1f}° yaw={rpy[2]:+.1f}°')

            frame = draw_overlay(frame, K, D, corners, ids,
                                 rvecs, tvecs, marker_size,
                                 pose_history, target_id)
        else:
            # Sin marcadores
            cv2.putText(frame, 'Sin marcadores detectados',
                        (20, args.height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)

        # FPS en esquina inferior derecha
        cv2.putText(frame, f'{fps_disp:.1f} fps',
                    (args.width - 100, args.height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow('ArUco Pose Viewer — D435i', frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # q o Esc
            break
        elif key == ord('s'):
            fname = f'aruco_frame_{int(time.time())}.png'
            cv2.imwrite(fname, frame)
            print(f'[INFO] Frame guardado: {fname}')
        elif key == ord('r'):
            pose_history.clear()
            print('[INFO] Promedio reseteado.')

    # Cleanup
    cv2.destroyAllWindows()
    if USE_REALSENSE:
        pipeline.stop()
    else:
        cap.release()
    print('[INFO] Cerrado.')


if __name__ == '__main__':
    main()