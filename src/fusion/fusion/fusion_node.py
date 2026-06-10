#!/usr/bin/env python3
"""
fusion_node.py
──────────────────────────────────────────────────────────────────────────────
Fuses the drone OccupancyGrid (arena_map_builder) with the AMR live map
(world_mapper). Both grids share the same resolution, origin, dimensions and
frame_id (world), so fusion is a direct element-wise operation.

Fusion semantics — Drone immutable + AMR additive
──────────────────────────────────────────────────
  ┌──────────────────────┬──────────────────────────────────────────────────┐
  │ drone cell           │ fused result                                     │
  ├──────────────────────┼──────────────────────────────────────────────────┤
  │ >= DRONE_OCC_THRESH  │ 100  — immutable; AMR scans cannot clear this    │
  │ -1  (unknown)        │ AMR value as-is  (or -1 if AMR also unknown)     │
  │ 25  (free floor)     │ max(25, AMR)  — AMR can add obstacles, not erase │
  └──────────────────────┴──────────────────────────────────────────────────┘

Topics
──────
  Subscriptions:
    /drone/map   (nav_msgs/OccupancyGrid) — TRANSIENT_LOCAL, published once
    /world_map   (nav_msgs/OccupancyGrid) — continuous AMR map
  Publications:
    /fused_map   (nav_msgs/OccupancyGrid) — published on every AMR update
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
)
from nav_msgs.msg import OccupancyGrid


class MapFusionNode(Node):

    DRONE_OCC_THRESH = 65   # cells >= this are treated as occupied (wall/obstacle)
    DRONE_FREE_VAL   = 25   # value used by arena_map_builder for free floor

    def __init__(self) -> None:
        super().__init__('map_fusion_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('drone_map_topic', '/drone/map')
        self.declare_parameter('amr_map_topic',   '/world_map')
        self.declare_parameter('fused_map_topic', '/fused_map')

        drone_topic = self.get_parameter('drone_map_topic').value
        amr_topic   = self.get_parameter('amr_map_topic').value
        fused_topic = self.get_parameter('fused_map_topic').value

        # ── State ─────────────────────────────────────────────────────────────
        self._drone_arr: np.ndarray | None = None  # (H, W) int8
        self._drone_info = None                     # MapMetaData, kept for output

        # ── Subscriptions ─────────────────────────────────────────────────────
        # TRANSIENT_LOCAL so we receive the drone map even if it was published
        # before this node started.
        drone_qos = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            OccupancyGrid, drone_topic, self._drone_cb, drone_qos
        )
        self.create_subscription(
            OccupancyGrid, amr_topic, self._amr_cb, 10
        )

        # ── Publisher ─────────────────────────────────────────────────────────
        self._pub = self.create_publisher(OccupancyGrid, fused_topic, 10)

        self.get_logger().info(
            f'map_fusion_node ready\n'
            f'  drone  → {drone_topic}\n'
            f'  amr    → {amr_topic}\n'
            f'  output → {fused_topic}'
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def _drone_cb(self, msg: OccupancyGrid) -> None:
        self._drone_info = msg.info
        self._drone_arr  = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width
        )
        self.get_logger().info(
            f'Drone map received: {msg.info.width}x{msg.info.height} cells '
            f'@ {msg.info.resolution:.3f} m/cell'
        )

    def _amr_cb(self, msg: OccupancyGrid) -> None:
        if self._drone_arr is None:
            self.get_logger().warn(
                'AMR map received but drone map not yet available — skipping.',
                throttle_duration_sec=5.0,
            )
            return

        self._pub.publish(self._fuse(msg))

    # ──────────────────────────────────────────────────────────────────────────
    # Fusion
    # ──────────────────────────────────────────────────────────────────────────

    def _fuse(self, amr_msg: OccupancyGrid) -> OccupancyGrid:
        d = self._drone_arr
        a = np.array(amr_msg.data, dtype=np.int8).reshape(
            amr_msg.info.height, amr_msg.info.width
        )

        drone_occ  = d >= self.DRONE_OCC_THRESH
        drone_unk  = d == -1
        drone_free = (~drone_occ) & (~drone_unk)

        fused = np.full(d.shape, -1, dtype=np.int8)

        # Rule 1 — drone occupied → immutable
        fused[drone_occ] = 100

        # Rule 2 — drone unknown → defer entirely to AMR
        fused[drone_unk] = a[drone_unk]

        # Rule 3 — drone free → AMR can only add obstacles, not erase
        fused[drone_free] = np.where(
            a[drone_free] == -1,
            np.int8(self.DRONE_FREE_VAL),                          # AMR has no data: keep drone free
            np.maximum(                                             # AMR has data: take the higher value
                np.int16(self.DRONE_FREE_VAL),
                a[drone_free].astype(np.int16),
            ).astype(np.int8),
        )

        out = OccupancyGrid()
        out.header.stamp    = self.get_clock().now().to_msg()
        out.header.frame_id = 'world'
        out.info            = self._drone_info
        out.data            = fused.flatten().tolist()
        return out


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MapFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()