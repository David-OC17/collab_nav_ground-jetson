#!/usr/bin/env python3
"""
Local Costmap Node for ROS2 Humble — map-only version
=======================================================
Assumes GMapping continuously updates /map with new obstacles.

Subscribes to:
  - /map   (nav_msgs/OccupancyGrid)  — full updated map from GMapping

Uses:
  - TF (map → base_link)             — to find robot position in map

Publishes:
  - /local_costmap/costmap          (nav_msgs/OccupancyGrid)  — local window
  - /local_costmap/costmap_inflated (nav_msgs/OccupancyGrid)  — inflated window
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.duration import Duration

from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

import tf2_ros

# ---------------------------------------------------------------------------
# Cost values (Nav2 convention)
# ---------------------------------------------------------------------------
FREE_SPACE   = 0
LETHAL       = 100
INSCRIBED    = 99
MAX_INFLATED = 1

GMAPPING_OCC_THRESHOLD = 65   # GMapping cells above this → LETHAL


class LocalCostmapNode(Node):

    def __init__(self):
        super().__init__('local_costmap_node')

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter('resolution',          0.05)
        self.declare_parameter('width',               4.0)
        self.declare_parameter('height',              4.0)
        self.declare_parameter('inflation_radius',    0.55)
        self.declare_parameter('cost_scaling_factor', 3.5)
        self.declare_parameter('robot_radius',        0.2)
        self.declare_parameter('publish_frequency',   5.0)
        self.declare_parameter('map_frame',           'map')
        self.declare_parameter('robot_base_frame',    'base_link')
        self.declare_parameter('map_topic',           '/map')

        self.resolution      = self.get_parameter('resolution').value
        self.width_m         = self.get_parameter('width').value
        self.height_m        = self.get_parameter('height').value
        self.inflation_radius= self.get_parameter('inflation_radius').value
        self.cost_scaling    = self.get_parameter('cost_scaling_factor').value
        self.robot_radius    = self.get_parameter('robot_radius').value
        self.publish_freq    = self.get_parameter('publish_frequency').value
        self.map_frame       = self.get_parameter('map_frame').value
        self.robot_base_frame= self.get_parameter('robot_base_frame').value
        self.map_topic       = self.get_parameter('map_topic').value

        # Local window dimensions in cells
        self.cells_x = int(self.width_m  / self.resolution)
        self.cells_y = int(self.height_m / self.resolution)

        # Robot position in map frame (from TF)
        self.robot_x = 0.0
        self.robot_y = 0.0

        # Rolling window origin (updated every publish cycle)
        self.origin_x = 0.0
        self.origin_y = 0.0

        # Output grids
        self.local_grid    = np.full((self.cells_y, self.cells_x),
                                      FREE_SPACE, dtype=np.float32)
        self.inflated_grid = np.full((self.cells_y, self.cells_x),
                                      FREE_SPACE, dtype=np.float32)

        # Full GMapping map storage
        self.gmap_data       = None
        self.gmap_resolution = None
        self.gmap_origin_x   = None
        self.gmap_origin_y   = None
        self.gmap_width      = None
        self.gmap_height     = None
        self.map_received    = False

        # Precompute inflation LUT
        self._inflation_cells = int(math.ceil(self.inflation_radius / self.resolution))
        self._build_inflation_lut()

        # ------------------------------------------------------------------
        # TF — only used to get robot position in map frame
        # ------------------------------------------------------------------
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ------------------------------------------------------------------
        # QoS
        # ------------------------------------------------------------------
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ------------------------------------------------------------------
        # Subscriber — only /map
        # ------------------------------------------------------------------
        self.map_sub = self.create_subscription(
            OccupancyGrid, self.map_topic, self._map_callback, map_qos)

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        self.costmap_pub = self.create_publisher(
            OccupancyGrid, '/local_costmap/costmap', latched_qos)

        self.inflated_pub = self.create_publisher(
            OccupancyGrid, '/local_costmap/costmap_inflated', latched_qos)

        # ------------------------------------------------------------------
        # Timer — extract window + publish at fixed rate
        # ------------------------------------------------------------------
        self.publish_timer = self.create_timer(
            1.0 / self.publish_freq, self._update_and_publish)

        self.get_logger().info(
            f'LocalCostmap started (map-only mode) | '
            f'window={self.width_m}x{self.height_m}m | '
            f'res={self.resolution}m | '
            f'inflation={self.inflation_radius}m'
        )

    # ==========================================================================
    # Map callback — store full GMapping map
    # ==========================================================================

    def _map_callback(self, msg: OccupancyGrid):
        """
        Called every time GMapping publishes an updated map.
        Just store it — the timer will extract the local window.
        """
        self.gmap_resolution = msg.info.resolution
        self.gmap_origin_x   = msg.info.origin.position.x
        self.gmap_origin_y   = msg.info.origin.position.y
        self.gmap_width      = msg.info.width
        self.gmap_height     = msg.info.height

        self.gmap_data = np.array(msg.data, dtype=np.int8).reshape(
            (self.gmap_height, self.gmap_width))

        self.map_received = True

        self.get_logger().info(
            f'Map updated: {self.gmap_width}x{self.gmap_height} cells',
            throttle_duration_sec=5.0
        )

    # ==========================================================================
    # Timer — get robot pose from TF, extract window, inflate, publish
    # ==========================================================================

    def _update_and_publish(self):
        if not self.map_received:
            self.get_logger().warn(
                f'Waiting for map on {self.map_topic}...',
                throttle_duration_sec=3.0)
            return

        # --- Step 1: get robot position in map frame from TF ---
        if not self._update_robot_pose():
            return

        # --- Step 2: set rolling window origin centered on robot ---
        half_w = (self.cells_x // 2) * self.resolution
        half_h = (self.cells_y // 2) * self.resolution
        self.origin_x = self.robot_x - half_w
        self.origin_y = self.robot_y - half_h

        # --- Step 3: extract local window from full GMapping map ---
        self._extract_local_window()

        # --- Step 4: inflate ---
        self._inflate()

        # --- Step 5: publish ---
        self.costmap_pub.publish(
            self._build_occupancy_grid(self.local_grid))
        self.inflated_pub.publish(
            self._build_occupancy_grid(self.inflated_grid))

        self.get_logger().debug(
            f'Published | robot=({self.robot_x:.2f},{self.robot_y:.2f}) | '
            f'origin=({self.origin_x:.2f},{self.origin_y:.2f})',
            throttle_duration_sec=2.0
        )

    # ==========================================================================
    # Get robot position in map frame via TF
    # ==========================================================================

    def _update_robot_pose(self) -> bool:
        """
        Look up map → base_link TF to get robot position.
        Returns False if TF is not available yet.
        """
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_base_frame,
                rclpy.time.Time(),          # latest available
                timeout=Duration(seconds=0.1)
            )
            self.robot_x = tf_stamped.transform.translation.x
            self.robot_y = tf_stamped.transform.translation.y
            return True

        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f'TF {self.map_frame}→{self.robot_base_frame} not available: {e}',
                throttle_duration_sec=3.0)
            return False

    # ==========================================================================
    # Extract local window from full GMapping map
    # ==========================================================================

    def _extract_local_window(self):
        """
        For each cell in the local rolling window, sample the
        corresponding cell from the full GMapping map and convert
        its occupancy probability to a cost value.
        """
        self.local_grid[:] = FREE_SPACE

        for cj in range(self.cells_y):
            for ci in range(self.cells_x):
                # World position of this local cell
                wx, wy = self._cell_to_world(ci, cj)

                # Corresponding cell in full GMapping map
                gci = int((wx - self.gmap_origin_x) / self.gmap_resolution)
                gcj = int((wy - self.gmap_origin_y) / self.gmap_resolution)

                if 0 <= gci < self.gmap_width and 0 <= gcj < self.gmap_height:
                    val = int(self.gmap_data[gcj, gci])
                    if val > GMAPPING_OCC_THRESHOLD:
                        self.local_grid[cj, ci] = LETHAL
                    else:
                        self.local_grid[cj, ci] = FREE_SPACE
                # Cells outside the GMapping map boundary stay FREE

    # ==========================================================================
    # Inflation
    # ==========================================================================

    def _inflate(self):
        self.inflated_grid[:] = self.local_grid
        lethal_cells = np.argwhere(self.local_grid >= LETHAL)

        for (cj, ci) in lethal_cells:
            for (di, dj), cost in self._lut.items():
                ni = ci + di
                nj = cj + dj
                if self._in_bounds(ni, nj):
                    if self.inflated_grid[nj, ni] < cost:
                        self.inflated_grid[nj, ni] = cost

    def _build_inflation_lut(self):
        r = self._inflation_cells
        self._lut = {}
        inscribed_cells = int(math.ceil(self.robot_radius / self.resolution))

        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                dist_cells = math.sqrt(di * di + dj * dj)
                dist_m     = dist_cells * self.resolution
                if dist_m > self.inflation_radius:
                    continue
                if dist_cells <= inscribed_cells:
                    cost = INSCRIBED
                else:
                    factor = math.exp(
                        -self.cost_scaling * (dist_m - self.robot_radius))
                    cost = max(int(INSCRIBED * factor), MAX_INFLATED)
                self._lut[(di, dj)] = cost

        self.get_logger().info(
            f'Inflation LUT built: {len(self._lut)} cells')

    # ==========================================================================
    # Coordinate helpers
    # ==========================================================================

    def _world_to_cell(self, wx: float, wy: float):
        ci = int((wx - self.origin_x) / self.resolution)
        cj = int((wy - self.origin_y) / self.resolution)
        return ci, cj

    def _cell_to_world(self, ci: int, cj: int):
        wx = self.origin_x + (ci + 0.5) * self.resolution
        wy = self.origin_y + (cj + 0.5) * self.resolution
        return wx, wy

    def _in_bounds(self, ci: int, cj: int) -> bool:
        return 0 <= ci < self.cells_x and 0 <= cj < self.cells_y

    # ==========================================================================
    # Publishing
    # ==========================================================================

    def _build_occupancy_grid(self, grid: np.ndarray) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame

        msg.info.resolution = self.resolution
        msg.info.width      = self.cells_x
        msg.info.height     = self.cells_y
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        flat = grid.flatten().astype(np.int8)
        msg.data = flat.tolist()
        return msg


# ==============================================================================
# Entry point
# ==============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = LocalCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()