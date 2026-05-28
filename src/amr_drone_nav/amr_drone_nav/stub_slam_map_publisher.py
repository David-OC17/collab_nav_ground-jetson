#!/usr/bin/env python3
"""stub_slam_map_publisher.py — simula el mapa que publicaría slam_toolbox.

Publica en /map con frame_id: slam_map (cobertura parcial del arena).
Requiere que exista el TF world → slam_map para que el slam_layer
del costmap pueda reproyectar correctamente.

Casos de prueba que cubre:
  1. Celdas libres en zona conocida     → SLAM overwrite al drone
  2. Obstáculo propio del SLAM          → aparece en costmap aunque drone no lo tenga
  3. Zona desconocida (-1)              → drone conserva sus valores ahí
  4. SLAM dice libre donde drone ocupa  → SLAM gana (verifica combination_method: 0)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import OccupancyGrid


# ── Parámetros del grid ────────────────────────────────────────────────────
RES  = 0.05   # m/cell — igual que drone map y master costmap
SIZE = 80     # 80×80 celdas = 4×4 m

# Cobertura parcial: SLAM solo conoce la mitad inferior del arena
# (filas 0–49 en coords mundo, y ∈ [0, 2.5 m])
# Las filas 50–79 quedan como -1 (desconocido)
SLAM_KNOWN_ROWS = 50   # de fila 0 hasta esta (exclusive de arriba)


class StubSlamMapPublisher(Node):
    def __init__(self):
        super().__init__('stub_slam_map_publisher')

        qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

        self.pub = self.create_publisher(OccupancyGrid, '/map', qos)

        grid = self._build_grid()
        msg  = self._make_msg(grid)
        self.pub.publish(msg)

        # Resumen para debugging
        n_occ  = int((grid == 100).sum())
        n_free = int((grid == 0).sum())
        n_unk  = int((grid == -1).sum())
        self.get_logger().info(
            f'Published stub SLAM map: {SIZE}×{SIZE} @ {RES} m/cell | '
            f'occ={n_occ} free={n_free} unk={n_unk} '
            f'(coverage: {100*(n_occ+n_free)/(SIZE*SIZE):.0f}%)'
        )
        self.get_logger().info(
            'frame_id: slam_map — asegúrate de tener el TF world→slam_map publicado'
        )

    def _build_grid(self) -> np.ndarray:
        """Construye el grid de prueba con los 4 casos de fusión."""
        grid = np.full((SIZE, SIZE), -1, dtype=np.int8)  # todo desconocido por default

        # ── Zona conocida (mitad inferior: y ∈ [0, 2.5 m]) ──────────────
        # Marca como libre toda la zona conocida
        grid[:SLAM_KNOWN_ROWS, :] = 0

        # Paredes de la zona conocida
        grid[0, :]                    = 100   # pared sur
        grid[:SLAM_KNOWN_ROWS, 0]     = 100   # pared oeste
        grid[:SLAM_KNOWN_ROWS, -1]    = 100   # pared este
        # No hay pared norte del SLAM (la zona desconocida empieza ahí)

        # ── Caso 2: obstáculo propio del SLAM (no está en drone map) ─────
        # Caja pequeña en (1.75, 0.75) → (2.0, 1.0) — zona que el drone
        # no tiene marcada como ocupada
        grid[15:20, 35:40] = 100   # y ∈ [0.75,1.0], x ∈ [1.75,2.0]

        # ── Caso 4: SLAM dice libre donde drone dice ocupado ──────────────
        # El drone stub tiene una caja en grid[30:40, 20:25]
        # (y ∈ [1.5,2.0], x ∈ [1.0,1.25])
        # El SLAM "desacuerda" y marca esa zona como libre:
        grid[30:40, 20:25] = 0     # SLAM-wins debería ganar → libre en costmap

        return grid

    def _make_msg(self, grid: np.ndarray) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'slam_map'   # ← frame del SLAM, NO world

        msg.info.resolution = RES
        msg.info.width      = SIZE
        msg.info.height     = SIZE
        # Mismo origen que el drone map y el costmap master
        # (asumiendo slam_map ≡ world en Stage 2 via TF)
        msg.info.origin.position.x    = 0.0
        msg.info.origin.position.y    = 0.0
        msg.info.origin.position.z    = 0.0
        msg.info.origin.orientation.w = 1.0

        msg.data = grid.flatten(order='C').tolist()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = StubSlamMapPublisher()
    rclpy.spin(node)   # keep alive — transient_local requiere que el publisher siga vivo
    rclpy.shutdown()


if __name__ == '__main__':
    main()
