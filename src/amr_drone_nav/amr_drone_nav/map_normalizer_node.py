#!/usr/bin/env python3
"""map_normalizer_node.py — fuerza el mapa de SLAM a dimensiones fijas.

Suscribe a /map (slam_toolbox) y republica en /map_normalized siempre
en TARGET_SIZE×TARGET_SIZE celdas, usando coordenadas métricas para el
crop/pad — no índices de celda.

Lógica por celda del output:
  - Celda conocida del SLAM (0 o 100) dentro del bounds  → copiar valor
  - Celda desconocida (-1) dentro del bounds del SLAM    → mantener -1
  - Celda fuera del bounds del SLAM                      → 0 (libre)
    └→ el drone_layer rellenará esas zonas via
       combination_method: 2 (MaxWithoutUnknownOverwrite)

El output mantiene:
  - Mismo frame_id que el SLAM map (slam_map)
  - Origin fijo en (master_origin_x, master_origin_y)
  - Resolución fija (target_resolution)
  - Tamaño fijo (target_size × target_size)

Topics
------
  Sub: /map             nav_msgs/OccupancyGrid  (slam_toolbox)
  Pub: /map_normalized  nav_msgs/OccupancyGrid  (tamaño fijo para Nav2)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from nav_msgs.msg import OccupancyGrid

# ── QoS — transient local en ambos extremos ───────────────────────────────
_QOS = QoSProfile(
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
)

# ── Parámetros por defecto ────────────────────────────────────────────────
PARAM_SPEC = [
    ("input_topic",       "/map"),
    ("output_topic",      "/map_normalized"),
    ("target_size",       80),      # celdas — 390 cm / 5 cm
    ("target_resolution", 0.05),    # m/cell — debe coincidir con el master
    ("master_origin_x",   0.0),     # esquina inferior-izquierda del master en world
    ("master_origin_y",   0.0),
]


class MapNormalizerNode(Node):
    """Republica el mapa de SLAM con tamaño fijo usando crop/pad métrico."""

    def __init__(self):
        super().__init__("map_normalizer_node")

        # ── Parámetros ───────────────────────────────────────────────
        self.p = {}
        for name, default in PARAM_SPEC:
            self.declare_parameter(name, default)
            self.p[name] = self.get_parameter(name).value

        size = self.p["target_size"]
        res  = self.p["target_resolution"]
        ox   = self.p["master_origin_x"]
        oy   = self.p["master_origin_y"]

        # Pre-calcula coordenadas métricas de los centros de celda del output
        # en el frame del SLAM map (mismo frame que el input).
        cols = np.arange(size, dtype=np.float64)
        rows = np.arange(size, dtype=np.float64)
        self._x_metric = ox + (cols + 0.5) * res   # (size,) coords x de cada col
        self._y_metric = oy + (rows + 0.5) * res   # (size,) coords y de cada row

        # ── Publishers / Subscriptions ───────────────────────────────
        self._pub = self.create_publisher(
            OccupancyGrid, self.p["output_topic"], _QOS
        )
        self.create_subscription(
            OccupancyGrid, self.p["input_topic"], self._cb, _QOS
        )

        self.get_logger().info(
            f"map_normalizer_node iniciado: "
            f"{self.p['input_topic']} → {self.p['output_topic']} "
            f"@ {size}×{size} celdas, {res} m/cell, "
            f"origin=({ox}, {oy})"
        )

    # ─────────────────────────────────────────────────────────────────
    def _cb(self, msg: OccupancyGrid):
        """Recibe el mapa de SLAM, normaliza y publica."""

        size = self.p["target_size"]
        res  = self.p["target_resolution"]

        # ── Convierte el input a array NumPy ─────────────────────────
        slam_arr = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width
        )
        slam_res = msg.info.resolution
        slam_ox  = msg.info.origin.position.x
        slam_oy  = msg.info.origin.position.y
        slam_h   = msg.info.height
        slam_w   = msg.info.width

        # ── Canvas de salida ──────────────────────────────────────────
        # Fuera del bounds del SLAM → 0 (libre)
        # El drone_layer rellenará estas zonas via MaxWithoutUnknownOverwrite
        out = np.zeros((size, size), dtype=np.int8)

        # ── Índices en el SLAM map para cada celda del output ─────────
        slam_cols = np.round(
            (self._x_metric - slam_ox) / slam_res - 0.5
        ).astype(int)   # (size,)

        slam_rows = np.round(
            (self._y_metric - slam_oy) / slam_res - 0.5
        ).astype(int)   # (size,)

        # Máscaras de validez (dentro de los bounds del SLAM map)
        valid_c = (slam_cols >= 0) & (slam_cols < slam_w)   # (size,) bool
        valid_r = (slam_rows >= 0) & (slam_rows < slam_h)   # (size,) bool

        # ── Copia preservando -1 donde el SLAM los tiene ──────────────
        # Dentro del bounds del SLAM:
        #   - valor conocido (0 o 100) → copiar
        #   - valor desconocido (-1)   → mantener -1 (no 0)
        # Fuera del bounds del SLAM:
        #   - dejar 0 (canvas inicial)
        for out_r in range(size):
            if not valid_r[out_r]:
                continue   # fuera del SLAM map → deja 0
            mask = valid_c
            slam_values = slam_arr[slam_rows[out_r], slam_cols[mask]]
            out[out_r, mask] = slam_values   # copia tal cual, incluyendo -1

        # ── Construye el mensaje de salida ────────────────────────────
        out_msg = OccupancyGrid()
        out_msg.header.stamp    = msg.header.stamp
        out_msg.header.frame_id = msg.header.frame_id   # conserva slam_map

        out_msg.info.map_load_time = msg.info.map_load_time
        out_msg.info.resolution    = res
        out_msg.info.width         = size
        out_msg.info.height        = size
        out_msg.info.origin.position.x    = self.p["master_origin_x"]
        out_msg.info.origin.position.y    = self.p["master_origin_y"]
        out_msg.info.origin.position.z    = 0.0
        out_msg.info.origin.orientation.w = 1.0

        out_msg.data = out.flatten(order='C').tolist()
        self._pub.publish(out_msg)

        # ── Log de cobertura ─────────────────────────────────────────
        n_known   = int((out != -1).sum())
        n_unknown = int((out == -1).sum())
        n_total   = size * size
        self.get_logger().debug(
            f"[normalizer] input {slam_w}×{slam_h} "
            f"@ origin({slam_ox:.2f},{slam_oy:.2f}) → "
            f"output {size}×{size} | "
            f"known={n_known} unknown={n_unknown} "
            f"({100*n_known/n_total:.0f}% coverage)"
        )


# ── Entry point ───────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(MapNormalizerNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
