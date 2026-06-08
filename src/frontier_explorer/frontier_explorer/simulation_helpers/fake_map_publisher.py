#!/usr/bin/env python3
"""
Fake Map Publisher for ROS 2 Humble
=====================================
Publishes two OccupancyGrid maps for simulation:

  /drone/map  — complete, fully known map. Published once at startup and
                latched. Used by AStarPlanner2 for safe path planning.

  /slam/map   — partial map that starts fully unknown (-1) and progressively
                reveals cells as the robot moves, simulating onboard SLAM.
                Used by FrontierExplorer to detect frontier boundaries.
                Reveal uses Bresenham line-of-sight raycast from each visited
                robot position within sensor_range_m.

Subscribes:
  - /follower/pose  (geometry_msgs/PoseWithCovarianceStamped) — robot pose
    Used to grow the SLAM map reveal as the robot explores.

Publishes:
  - /drone/map  (nav_msgs/OccupancyGrid, TRANSIENT_LOCAL) — full fused map
  - /slam/map   (nav_msgs/OccupancyGrid, TRANSIENT_LOCAL) — partial SLAM map

Parameters:
  scenario              'room'   — environment: 'room' | 'room2' | 'corridor'
  map_resolution        0.05     m/cell
  map_width_m           12.0     m
  map_height_m          12.0     m
  publish_rate          2.0      Hz  — how often to republish /slam/map
  sensor_range_m        4.0      m   — robot sensor reveal radius
  position_log_spacing  0.20     m   — min distance moved before logging new pose
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                        DurabilityPolicy, HistoryPolicy)

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped


class FakeMapPublisher(Node):

    def __init__(self):
        super().__init__('fake_map_publisher')

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter('scenario',             'room')
        self.declare_parameter('map_resolution',       0.05)
        self.declare_parameter('map_width_m',          12.0)
        self.declare_parameter('map_height_m',         12.0)
        self.declare_parameter('publish_rate',         2.0)
        self.declare_parameter('sensor_range_m',       4.0)
        self.declare_parameter('position_log_spacing', 0.20)

        self.scenario      = self.get_parameter('scenario').value
        self.map_res       = float(self.get_parameter('map_resolution').value)
        self.map_width_m   = float(self.get_parameter('map_width_m').value)
        self.map_height_m  = float(self.get_parameter('map_height_m').value)
        self.publish_rate  = float(self.get_parameter('publish_rate').value)
        self.sensor_range  = float(self.get_parameter('sensor_range_m').value)
        self.log_spacing   = float(self.get_parameter('position_log_spacing').value)

        # Derived
        self.map_cells_x  = int(self.map_width_m  / self.map_res)
        self.map_cells_y  = int(self.map_height_m / self.map_res)
        self.map_origin_x = -self.map_width_m  / 2.0
        self.map_origin_y = -self.map_height_m / 2.0

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.visited_positions = []       # list of (x, y) robot has visited
        self._last_logged_pos  = None     # last position we logged

        # Pre-build the full grid once — reused for drone map and LoS checks
        self._full_grid = self._build_full_grid()

        # ------------------------------------------------------------------
        # QoS — latched so late subscribers always receive the map
        # ------------------------------------------------------------------
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        self.drone_map_pub = self.create_publisher(
            OccupancyGrid, '/drone/map', latched_qos)

        self.slam_map_pub = self.create_publisher(
            OccupancyGrid, '/slam/map', latched_qos)

        # ------------------------------------------------------------------
        # Subscriber
        # ------------------------------------------------------------------
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/follower/pose',
            self._pose_callback,
            reliable_qos
        )

        # ------------------------------------------------------------------
        # Publish drone map once (full, latched)
        # ------------------------------------------------------------------
        self.drone_map_pub.publish(self._make_msg(self._full_grid))
        self.get_logger().info(
            f'FakeMapPublisher ready\n'
            f'  scenario     = {self.scenario}\n'
            f'  size         = {self.map_cells_x} x {self.map_cells_y} cells\n'
            f'  resolution   = {self.map_res} m/cell\n'
            f'  sensor_range = {self.sensor_range} m\n'
            f'  /drone/map published (full, latched)\n'
            f'  /slam/map   will grow as robot moves'
        )

        # Publish an initial fully-unknown SLAM map immediately
        unknown_grid = [-1] * (self.map_cells_x * self.map_cells_y)
        self.slam_map_pub.publish(self._make_msg(unknown_grid))

        # ------------------------------------------------------------------
        # Timer — republish growing SLAM map
        # ------------------------------------------------------------------
        self.create_timer(1.0 / self.publish_rate, self._publish_slam_map)

    # ==========================================================================
    # Pose callback — log position if robot moved enough
    # ==========================================================================

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self._last_logged_pos is None or \
                math.hypot(x - self._last_logged_pos[0],
                           y - self._last_logged_pos[1]) >= self.log_spacing:
            self.visited_positions.append((x, y))
            self._last_logged_pos = (x, y)

    # ==========================================================================
    # SLAM map publisher — grows as robot visits new positions
    # ==========================================================================

    def _publish_slam_map(self):
        slam_grid = self._build_partial_slam_grid()
        self.slam_map_pub.publish(self._make_msg(slam_grid))

        revealed = sum(1 for c in slam_grid if c != -1)
        total    = len(slam_grid)
        self.get_logger().debug(
            f'SLAM map: {revealed}/{total} cells revealed '
            f'({100*revealed/total:.1f}%) | '
            f'{len(self.visited_positions)} poses logged',
            throttle_duration_sec=2.0)

    # ==========================================================================
    # Partial SLAM grid — Bresenham line-of-sight reveal
    # ==========================================================================

    def _build_partial_slam_grid(self) -> list:
        """
        Starts fully unknown (-1). For each logged robot position, reveals
        all cells within sensor_range_m that have clear line-of-sight.
        """
        revealed = [-1] * (self.map_cells_x * self.map_cells_y)
        sensor_cells = int(self.sensor_range / self.map_res)

        for (rx, ry) in self.visited_positions:
            rci = int((rx - self.map_origin_x) / self.map_res)
            rcj = int((ry - self.map_origin_y) / self.map_res)

            for dj in range(-sensor_cells, sensor_cells + 1):
                for di in range(-sensor_cells, sensor_cells + 1):
                    ci = rci + di
                    cj = rcj + dj

                    if not (0 <= ci < self.map_cells_x and
                            0 <= cj < self.map_cells_y):
                        continue

                    if math.sqrt(di*di + dj*dj) > sensor_cells:
                        continue

                    if self._has_line_of_sight(rci, rcj, ci, cj):
                        idx = cj * self.map_cells_x + ci
                        revealed[idx] = self._full_grid[idx]

        return revealed

    def _has_line_of_sight(self, x0: int, y0: int,
                            x1: int, y1: int) -> bool:
        """
        Bresenham raycast from (x0,y0) to (x1,y1).
        Returns False if any intermediate cell is lethal (value >= 100).
        """
        dx = abs(x1 - x0); sx = 1 if x0 < x1 else -1
        dy = abs(y1 - y0); sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0

        while True:
            if (x, y) == (x1, y1):
                return True
            if (x, y) != (x0, y0):
                idx = y * self.map_cells_x + x
                if 0 <= idx < len(self._full_grid) and \
                        self._full_grid[idx] >= 100:
                    return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy; x += sx
            if e2 <  dx:
                err += dx; y += sy

    # ==========================================================================
    # Message builder
    # ==========================================================================

    def _make_msg(self, grid: list) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.stamp              = self.get_clock().now().to_msg()
        msg.header.frame_id           = 'map'
        msg.info.resolution           = self.map_res
        msg.info.width                = self.map_cells_x
        msg.info.height               = self.map_cells_y
        msg.info.origin.position.x    = self.map_origin_x
        msg.info.origin.position.y    = self.map_origin_y
        msg.info.origin.position.z    = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data                      = grid
        return msg

    # ==========================================================================
    # Full grid builder
    # ==========================================================================

    def _build_full_grid(self) -> list:
        size = self.map_cells_x * self.map_cells_y
        grid = [0] * size

        if self.scenario == 'room':
            self._build_room(grid)
        elif self.scenario == 'room2':
            self._build_room2(grid)
        elif self.scenario == 'corridor':
            self._build_corridor(grid)
        else:
            self.get_logger().warn(
                f'Unknown scenario "{self.scenario}" — using empty room.')
            self._rect(grid, -4.0, -4.0, 8.0, 8.0, fill=False)

        return grid

    # ==========================================================================
    # Scenario builders
    # ==========================================================================

    def _build_room(self, grid: list):
        self._rect(grid, -4.0, -4.0, 8.0, 8.0, fill=False)
        self._rect(grid, -2.5, -2.5, 0.5, 0.5, fill=True)
        self._rect(grid,  2.0, -2.5, 0.5, 0.5, fill=True)
        self._rect(grid, -2.5,  2.0, 0.5, 0.5, fill=True)
        self._rect(grid,  2.0,  2.0, 0.5, 0.5, fill=True)

    def _build_corridor(self, grid: list):
        self._rect(grid, -5.0, -0.75, 10.0, 1.5, fill=False)
        for ox, oy in [(-3.5, -0.3), (-1.5, 0.3), (0.5, -0.3),
                       ( 2.0,  0.3), ( 3.5, -0.3)]:
            self._rect(grid, ox - 0.15, oy - 0.15, 0.3, 0.3, fill=True)

    def _build_room2(self, grid: list):
        self._rect(grid, -4.0, -4.0, 8.0, 8.0, fill=False)
        self._rect(grid, -3.8,  1.3,  3.2, 0.2, fill=True)
        self._rect(grid,  0.6,  1.3,  3.2, 0.2, fill=True)
        self._rect(grid, -3.2,  2.2,  0.4, 0.4, fill=True)
        self._rect(grid,  1.5,  0.0,  0.8, 0.8, fill=True)
        self._rect(grid, -1.5, -0.8,  0.25, 1.6, fill=True)
        self._rect(grid,  2.5, -3.2,  0.5, 0.5, fill=True)
        self._rect(grid, -3.5, -3.5,  0.2, 2.5, fill=True)
        self._rect(grid, -3.5, -3.5,  2.0, 0.2, fill=True)
        self._rect(grid,  1.5, -3.5,  0.2, 2.0, fill=True)
        self._rect(grid,  3.2, -3.5,  0.2, 2.0, fill=True)
        self._rect(grid,  1.5, -3.5,  1.9, 0.2, fill=True)
        for ox, oy in [(-2.5,-1.5),(-1.8,-0.5),(-1.0,0.5),(-0.2,1.0)]:
            self._rect(grid, ox, oy, 0.35, 0.35, fill=True)
        self._rect(grid,  2.2, -1.5,  1.5, 0.2, fill=True)
        self._rect(grid,  2.2, -1.5,  0.2, 1.0, fill=True)
        self._rect(grid,  1.0, -2.5,  1.4, 0.2, fill=True)
        self._rect(grid,  2.2, -2.5,  0.2, 0.8, fill=True)
        self._rect(grid,  2.2, -3.3,  1.5, 0.2, fill=True)
        for ox, oy in [(2.0,2.5),(2.6,2.0),(3.0,2.8),(2.3,3.2)]:
            self._rect(grid, ox, oy, 0.3, 0.3, fill=True)
        self._rect(grid, -3.0, -0.1,  2.0, 0.15, fill=True)
        self._rect(grid,  0.5, -0.1,  1.0, 0.15, fill=True)

    # ==========================================================================
    # Primitive
    # ==========================================================================

    def _rect(self, grid: list, x: float, y: float,
              w: float, h: float, thickness: float = 0.15,
              fill: bool = False):
        ci0 = max(0, min(int((x     - self.map_origin_x) / self.map_res),
                         self.map_cells_x - 1))
        ci1 = max(0, min(int((x + w - self.map_origin_x) / self.map_res),
                         self.map_cells_x - 1))
        cj0 = max(0, min(int((y     - self.map_origin_y) / self.map_res),
                         self.map_cells_y - 1))
        cj1 = max(0, min(int((y + h - self.map_origin_y) / self.map_res),
                         self.map_cells_y - 1))
        t   = max(1, int(thickness / self.map_res))

        for cj in range(cj0, cj1 + 1):
            for ci in range(ci0, ci1 + 1):
                is_border = (cj < cj0 + t or cj > cj1 - t or
                             ci < ci0 + t or ci > ci1 - t)
                if fill or is_border:
                    grid[cj * self.map_cells_x + ci] = 100


# ==============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = FakeMapPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()