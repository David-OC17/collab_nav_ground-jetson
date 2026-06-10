"""
aruco_localizer.launch.py
──────────────────────────────────────────────────────────────────────────────
Lanza:
  1. RealSense D435i (realsense2_camera) — solo stream de color + camera_info
  2. TFs estáticos  world → aruco_<id>  para cada marcador de la arena
  3. TF estático    base_footprint → camera_color_optical_frame  (extrínsecos)
  4. Nodo aruco_localizer

Argumentos de línea de comandos:
  use_realsense   (default: true)  — desactivar si ya está corriendo en otro nodo
  use_rviz        (default: false) — lanzar RViz con config de debug

Ejemplo:
  ros2 launch aruco_localizer aruco_localizer.launch.py use_rviz:=true
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


# ─── Poses conocidas de los marcadores en el frame 'world' ───────────────────
# Formato: (id, x, y, z, roll_deg, pitch_deg, yaw_deg)
#
#
MARKER_POSES = [
    # id    x      y     z    roll  pitch  yaw
    (15,  0.1344, 0.1344, 0.19,  90.0,  0.0,   135.0),   # sur-oeste
    (16,  0.1344, 3.735, 0.19,  90.0,  0.0,   45.0),   # pared norte-oeste
    (17,  3.724, 3.735, 0.19,  90.0,  0.0, -45.0),   # pared norte-este
    (18,  3.715, 0.1344, 0.19,  90.0,  0.0, -135.0),   # pared sur-este
]

# ─── Posición de la cámara en el robot ───────────────────────────────────────
# TF: base_footprint → camera_color_optical_frame
# Mide estas distancias con cinta métrica en tu robot.
# Recuerda que camera_color_optical_frame tiene ejes: +x=derecha, +y=abajo, +z=adelante
CAMERA_OFFSET = dict(
    x=0.07,    # 
    y=0.035,    #
    z=0.20,    # 
    roll=0.0,
    pitch=0.0,  # ángulo de inclinación de la cámara (positivo = apuntando hacia abajo)
    yaw=0.0,
)


def generate_launch_description():
    pkg_share = FindPackageShare('aruco_localizer')
    config    = PathJoinSubstitution([pkg_share, 'config', 'params.yaml'])

    use_realsense = LaunchConfiguration('use_realsense')
    use_rviz      = LaunchConfiguration('use_rviz')

    args = [
        DeclareLaunchArgument('use_realsense', default_value='true',
            description='Lanzar nodo realsense2_camera'),
        DeclareLaunchArgument('use_rviz', default_value='false',
            description='Lanzar RViz con config de debug'),
    ]

    nodes = []

    # ── 1. RealSense D435i ─────────────────────────────────────────────────────
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='realsense2_camera',
        parameters=[{
            'enable_color':              True,
            'enable_depth':              False,
            'enable_infra1':             False,
            'enable_infra2':             False,
            'rgb_camera.color_profile':  '640x480x30',   # ← nuevo formato
            'align_depth.enable':        False,
        }],
        output='screen',
        condition=IfCondition(use_realsense),
    )
    nodes.append(realsense_node)

    # ── 2. TFs estáticos: world → aruco_<id> ──────────────────────────────────
    import math
    for mid, x, y, z, roll_d, pitch_d, yaw_d in MARKER_POSES:
        r = math.radians(roll_d)
        p = math.radians(pitch_d)
        w = math.radians(yaw_d)
        nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=f'static_tf_aruco_{mid}',
            arguments=[
                '--x',     str(x),
                '--y',     str(y),
                '--z',     str(z),
                '--roll',  str(r),
                '--pitch', str(p),
                '--yaw',   str(w),
                '--frame-id',       'world',
                '--child-frame-id', f'aruco_{mid}',
            ],
        ))

    # ── 3. TF estático: base_footprint → camera_camera_link ───────────
    cam = CAMERA_OFFSET
    # Conversión de extrínsecos físicos a la convención óptica de ROS:
    # El frame óptico tiene +z hacia adelante, +x derecha, +y abajo
    # Desde base_footprint (+x adelante, +z arriba) hay un roll de -90° y yaw de -90°
    nodes.append(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_camera_extrinsics',
        arguments=[
            '--x',     str(cam['x']),
            '--y',     str(cam['y']),
            '--z',     str(cam['z']),
            '--roll',  str(math.radians(cam['roll'])),
            '--pitch', str(math.radians(cam['pitch'])),
            '--yaw',   str(math.radians(cam['yaw'])),
            '--frame-id',       'base_footprint',
            '--child-frame-id', 'camera_link',
        ],
    ))

    # ── 4. Nodo aruco_localizer ────────────────────────────────────────────────
    localizer_node = Node(
        package='aruco_localizer',
        executable='aruco_localizer_node',
        name='aruco_localizer',
        parameters=[config],
        output='screen',
        emulate_tty=True,
    )
    nodes.append(localizer_node)

    # ── 5. RViz (opcional) ────────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        condition=IfCondition(use_rviz),
    )
    # nodes.append(rviz_node)

    return LaunchDescription(args + nodes)
