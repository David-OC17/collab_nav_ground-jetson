#!/usr/bin/env python3
"""
Fake Map Publisher for ROS 2 Humble
=====================================
Publishes three OccupancyGrid maps for simulation:

  /drone/map      — complete, fully known map. Published once at startup
                    and latched. Used by AStarPlanner2 for safe path planning.

  /slam/map       — partial map that starts fully unknown (-1) and progressively
                    reveals cells as the robot moves, simulating onboard LIDAR
                    SLAM with a 360-degree sensor. Used by FrontierExplorer.

  /camera/fov_map — cells that have been seen by the simulated RGB camera.
                    The camera is modelled as a forward-facing wedge:
                      • H-FOV  ±34.7°  (D435i colour sensor at 69.4° total)
                      • Range  0.15 m … 4.0 m  (scaled for 4 m map)
                    Line-of-sight blocked by occupied cells, same as SLAM.
                    Published as 0 (unseen) / 100 (seen) and latched.

Subscribes:
  - /follower/pose  (geometry_msgs/PoseWithCovarianceStamped) — robot pose + yaw

Publishes:
  - /drone/map      (nav_msgs/OccupancyGrid, TRANSIENT_LOCAL) — full map
  - /slam/map       (nav_msgs/OccupancyGrid, TRANSIENT_LOCAL) — growing LIDAR map
  - /camera/fov_map (nav_msgs/OccupancyGrid, TRANSIENT_LOCAL) — camera seen mask

Scenarios (4 × 4 m room, origin at centre → extents −2 … +2 m):
  room1    — scattered pillars and a partial wall creating blind corners
  room2    — L-shaped wall dividing the space with small box obstacles
  room3    — two tall vertical bars near opposite walls, corridor between them

Parameters:
  scenario              'room3'  — environment: 'room1' | 'room2' | 'room3'
  map_resolution        0.05     m/cell
  map_width_m           4.0      m
  map_height_m          4.0      m
  publish_rate          2.0      Hz
  sensor_range_m        2.0      m   — LIDAR reveal radius (360°)
  max_camera_range_m    4.0      m   — camera far-clip distance
  camera_hfov_deg       69.4     °   — D435i colour sensor H-FOV
  camera_near_m         0.15     m   — near-clip
  position_log_spacing  0.10     m
  robot_start_x        -1.5      m   — initial pose logged on first tick
  robot_start_y        -1.5      m
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
        self.declare_parameter('scenario',             'room3')
        self.declare_parameter('map_resolution',       0.05)
        self.declare_parameter('map_width_m',          4.0)
        self.declare_parameter('map_height_m',         4.0)
        self.declare_parameter('publish_rate',         2.0)
        self.declare_parameter('sensor_range_m',       2.0)
        self.declare_parameter('max_camera_range_m',   4.0)
        self.declare_parameter('camera_hfov_deg',      69.4)
        self.declare_parameter('camera_near_m',        0.15)
        self.declare_parameter('position_log_spacing', 0.10)
        self.declare_parameter('robot_start_x',       1.7)
        self.declare_parameter('robot_start_y',       1.7)

        self.scenario          = self.get_parameter('scenario').value
        self.map_res           = float(self.get_parameter('map_resolution').value)
        self.map_width_m       = float(self.get_parameter('map_width_m').value)
        self.map_height_m      = float(self.get_parameter('map_height_m').value)
        self.publish_rate      = float(self.get_parameter('publish_rate').value)
        self.sensor_range      = float(self.get_parameter('sensor_range_m').value)
        self.camera_range      = float(self.get_parameter('max_camera_range_m').value)
        self.camera_half_angle = math.radians(
            float(self.get_parameter('camera_hfov_deg').value) / 2.0)
        self.camera_near       = float(self.get_parameter('camera_near_m').value)
        self.log_spacing       = float(self.get_parameter('position_log_spacing').value)
        self.robot_start_x     = float(self.get_parameter('robot_start_x').value)
        self.robot_start_y     = float(self.get_parameter('robot_start_y').value)

        # Derived
        self.map_cells_x  = int(self.map_width_m  / self.map_res)
        self.map_cells_y  = int(self.map_height_m / self.map_res)
        self.map_origin_x = -self.map_width_m  / 2.0
        self.map_origin_y = -self.map_height_m / 2.0

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.visited_poses: list[tuple[float, float, float]] = []
        self._last_logged_pos = None

        # Persistent camera-seen boolean mask
        self._camera_seen = [False] * (self.map_cells_x * self.map_cells_y)

        self._full_grid = self._build_full_grid()

        # Seed the LIDAR reveal at start position (yaw unknown — no camera wedge yet)
        self._log_pose(self.robot_start_x, self.robot_start_y, None)

        # ------------------------------------------------------------------
        # QoS
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
            OccupancyGrid, '/drone/map',      latched_qos)
        self.slam_map_pub  = self.create_publisher(
            OccupancyGrid, '/slam/map',       latched_qos)
        self.cam_fov_pub   = self.create_publisher(
            OccupancyGrid, '/camera/fov_map', latched_qos)

        # ------------------------------------------------------------------
        # Subscriber
        # ------------------------------------------------------------------
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/follower/pose',
            self._pose_callback,
            reliable_qos
        )

        # ------------------------------------------------------------------
        # Startup publications
        # ------------------------------------------------------------------
        self.drone_map_pub.publish(self._make_msg(self._full_grid))
        self.slam_map_pub.publish(self._make_msg(
            [-1] * (self.map_cells_x * self.map_cells_y)))
        self.cam_fov_pub.publish(self._make_msg(
            [0]  * (self.map_cells_x * self.map_cells_y)))

        self.get_logger().info(
            f'FakeMapPublisher ready\n'
            f'  scenario     = {self.scenario}\n'
            f'  map size     = {self.map_width_m} × {self.map_height_m} m  '
            f'({self.map_cells_x} × {self.map_cells_y} cells)\n'
            f'  resolution   = {self.map_res} m/cell\n'
            f'  LIDAR range  = {self.sensor_range} m (360°)\n'
            f'  Camera H-FOV = {math.degrees(self.camera_half_angle*2):.1f}°  '
            f'range {self.camera_near:.2f}–{self.camera_range:.1f} m\n'
            f'  robot start  = ({self.robot_start_x}, {self.robot_start_y})'
        )

        self.create_timer(1.0 / self.publish_rate, self._publish_maps)

    # ==========================================================================
    # Pose callback
    # ==========================================================================

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        if self._last_logged_pos is None or \
                math.hypot(x - self._last_logged_pos[0],
                           y - self._last_logged_pos[1]) >= self.log_spacing:
            self._log_pose(x, y, yaw)   # real yaw from odometry

    def _log_pose(self, x: float, y: float, yaw: float | None):
        self.visited_poses.append((x, y, yaw or 0.0))
        self._last_logged_pos = (x, y)
        # Only paint the camera wedge when we have a real yaw from odometry
        if yaw is not None:
            self._update_camera_seen(x, y, yaw)

    # ==========================================================================
    # Timer — publish all three maps
    # ==========================================================================

    def _publish_maps(self):
        slam_grid = self._build_partial_slam_grid()
        self.slam_map_pub.publish(self._make_msg(slam_grid))

        fov_grid = [100 if s else 0 for s in self._camera_seen]
        self.cam_fov_pub.publish(self._make_msg(fov_grid))

        revealed = sum(1 for c in slam_grid if c != -1)
        cam_seen = sum(self._camera_seen)
        total    = len(slam_grid)
        self.get_logger().debug(
            f'SLAM {100*revealed/total:.1f}% | '
            f'Camera {100*cam_seen/total:.1f}% | '
            f'{len(self.visited_poses)} poses',
            throttle_duration_sec=2.0)

    # ==========================================================================
    # LIDAR — 360° circle reveal with LoS
    # ==========================================================================

    def _build_partial_slam_grid(self) -> list:
        revealed = [-1] * (self.map_cells_x * self.map_cells_y)
        sensor_cells = int(self.sensor_range / self.map_res)

        for (rx, ry, _yaw) in self.visited_poses:
            rci = int((rx - self.map_origin_x) / self.map_res)
            rcj = int((ry - self.map_origin_y) / self.map_res)

            for dj in range(-sensor_cells, sensor_cells + 1):
                for di in range(-sensor_cells, sensor_cells + 1):
                    if math.sqrt(di*di + dj*dj) > sensor_cells:
                        continue
                    ci = rci + di
                    cj = rcj + dj
                    if not (0 <= ci < self.map_cells_x and
                            0 <= cj < self.map_cells_y):
                        continue
                    if self._has_line_of_sight(rci, rcj, ci, cj):
                        idx = cj * self.map_cells_x + ci
                        revealed[idx] = self._full_grid[idx]

        return revealed

    # ==========================================================================
    # Camera — forward wedge, accumulated
    # ==========================================================================

    def _update_camera_seen(self, rx: float, ry: float, yaw: float):
        rci = int((rx - self.map_origin_x) / self.map_res)
        rcj = int((ry - self.map_origin_y) / self.map_res)

        far_cells  = int(self.camera_range / self.map_res)
        near_cells = max(1, int(self.camera_near / self.map_res))

        for dj in range(-far_cells, far_cells + 1):
            for di in range(-far_cells, far_cells + 1):
                dist_cells = math.sqrt(di * di + dj * dj)
                if dist_cells < near_cells or dist_cells > far_cells:
                    continue
                ci = rci + di
                cj = rcj + dj
                if not (0 <= ci < self.map_cells_x and
                        0 <= cj < self.map_cells_y):
                    continue
                # atan2(dj, di): dj=world ΔY, di=world ΔX — correct in ROS grid
                cell_angle = math.atan2(dj, di)
                # Robust wrap into (−π, +π] avoiding modulo edge cases
                delta = math.atan2(
                    math.sin(cell_angle - yaw),
                    math.cos(cell_angle - yaw)
                )
                if abs(delta) > self.camera_half_angle:
                    continue
                if self._has_line_of_sight(rci, rcj, ci, cj):
                    self._camera_seen[cj * self.map_cells_x + ci] = True

    # ==========================================================================
    # Bresenham LoS
    # ==========================================================================

    def _has_line_of_sight(self, x0: int, y0: int,
                            x1: int, y1: int) -> bool:
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
        grid = [0] * (self.map_cells_x * self.map_cells_y)
        builders = {
            'room1': self._build_room1,
            'room2': self._build_room2,
            'room3': self._build_room3,
        }
        if self.scenario in builders:
            builders[self.scenario](grid)
        else:
            self.get_logger().warn(
                f'Unknown scenario "{self.scenario}" — valid: room1 | room2 | room3')
            self._build_room1(grid)
        return grid

    # ==========================================================================
    # Scenario builders  (4 × 4 m, origin at centre, extents −2 … +2 m)
    # ==========================================================================

    def _build_room1(self, grid: list):
        """
        Random-ish scatter: four small square pillars near the corners and a
        short diagonal-ish wall in the centre-left, creating blind pockets that
        the LIDAR maps but the camera may not see.

        Walls at ±2 m, obstacles well inside so the robot has room to navigate.
        """
        # Outer walls
        self._rect(grid, -2.0, -2.0, 4.0, 4.0, fill=False)

        # Four corner pillars (0.3 × 0.3 m), inset 0.5 m from each corner
        self._rect(grid, -1.7, -1.7, 0.3, 0.3, fill=True)   # SW
        self._rect(grid,  1.4, -1.7, 0.3, 0.3, fill=True)   # SE
        self._rect(grid, -1.7,  1.4, 0.3, 0.3, fill=True)   # NW
        self._rect(grid,  1.4,  1.4, 0.3, 0.3, fill=True)   # NE

        # Short wall stub from the west side towards the centre
        self._rect(grid, -2.0, -0.15, 1.0, 0.15, fill=True)

        # Small isolated box near centre-right
        self._rect(grid,  0.5, -0.4, 0.3, 0.3, fill=True)

    def _build_room2(self, grid: list):
        """
        L-shaped wall dividing the room into two connected zones, plus two
        small box obstacles to create additional occlusion.

        The gap in the L-wall is wide enough for the robot to pass through.
        """
        # Outer walls
        self._rect(grid, -2.0, -2.0, 4.0, 4.0, fill=False)

        # L-shaped divider:
        #   horizontal arm — runs from the west wall almost to centre
        self._rect(grid, -2.0,  0.2, 1.6, 0.15, fill=True)
        #   vertical arm — drops from the end of the horizontal arm downward
        self._rect(grid, -0.55, -0.8, 0.15, 1.0, fill=True)

        # Two small square boxes on the eastern side
        self._rect(grid,  0.8, -1.4, 0.3, 0.3, fill=True)
        self._rect(grid,  0.8,  0.8, 0.3, 0.3, fill=True)

    def _build_room3(self, grid: list):
        """
        Two tall narrow vertical bars in the middle area of the room,
        staggered so there is a corridor between them.

          Left bar:  x ∈ [−1.3, −0.9],  y ∈ [−1.8,  0.0]  (lower half)
          Right bar: x ∈ [ 0.9,  1.3],  y ∈ [ 0.0,  1.8]  (upper half)
        """
        # Outer walls
        self._rect(grid, -2.0, -2.0, 4.0, 4.0, fill=False)

        # Left bar — lower half, shifted toward centre
        self._rect(grid, -1.3, -1.8, 0.4, 1.8, fill=True)

        # Right bar — upper half, shifted toward centre
        self._rect(grid,  0.9,  0.0, 0.4, 1.8, fill=True)

    # ==========================================================================
    # Rect primitive  _rect(grid, x, y, w, h)  — bottom-left corner + size
    # ==========================================================================

    def _rect(self, grid: list, x: float, y: float,
              w: float, h: float, thickness: float = 0.10,
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