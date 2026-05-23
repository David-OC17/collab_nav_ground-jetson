#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header

from visualization_msgs.msg import Marker
from std_msgs.msg import Float32, ColorRGBA
from geometry_msgs.msg import PoseWithCovarianceStamped



class FakeMapPublisher(Node):

    def __init__(self):
        super().__init__('fake_map_publisher')

        self.declare_parameter('scenario',            'room')
        self.declare_parameter('map_resolution',      0.05)
        self.declare_parameter('map_width_m',         12.0)
        self.declare_parameter('map_height_m',        12.0)
        self.declare_parameter('publish_rate',        1.0)
        self.declare_parameter('move_robot',          False)
        self.declare_parameter('confidence_ramp_sec', 50.0)
        self.declare_parameter('slam_reveal_sec',     120.0)

        self.scenario             = self.get_parameter('scenario').value
        self.map_res              = self.get_parameter('map_resolution').value
        self.map_width_m          = self.get_parameter('map_width_m').value
        self.map_height_m         = self.get_parameter('map_height_m').value
        self.publish_rate         = self.get_parameter('publish_rate').value
        self.move_robot           = self.get_parameter('move_robot').value
        self.confidence_ramp_sec  = self.get_parameter('confidence_ramp_sec').value
        self.slam_reveal_sec      = self.get_parameter('slam_reveal_sec').value

        self.map_cells_x  = int(self.map_width_m  / self.map_res)
        self.map_cells_y  = int(self.map_height_m / self.map_res)
        self.map_origin_x = -self.map_width_m  / 2.0
        self.map_origin_y = -self.map_height_m / 2.0

        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0

        self.start_time        = self.get_clock().now()
        self._drone_published  = False   # guard so drone map only goes out once

        # SLAM map
        self.visited_positions = []          # history of (x, y) the robot has been at
        self.declare_parameter('sensor_range_m',       4.0)
        self.declare_parameter('position_log_spacing', 0.3)   # only log if moved this far
        self._last_logged_pos = None

        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.drone_map_pub   = self.create_publisher(OccupancyGrid, '/drone/map',                  latched_qos)
        self.slam_map_pub    = self.create_publisher(OccupancyGrid, '/fusion/slam_reprojected',    latched_qos)
        self.confidence_pub  = self.create_publisher(Float32,       '/fusion/confidence',          latched_qos)
        self.status_marker_pub = self.create_publisher(Marker,      '/fusion/status_marker',       latched_qos)

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/follower/pose',
            self._pose_callback,
            10
        )

        # Single timer drives everything
        self.create_timer(1.0 / self.publish_rate, self._publish)

        self.get_logger().info(
            f'FakeMapPublisher ready | scenario={self.scenario} | '
            f'{self.map_cells_x}x{self.map_cells_y} cells | '
            f'confidence_ramp={self.confidence_ramp_sec}s | '
            f'slam_reveal={self.slam_reveal_sec}s'
        )

    # TO reveal addjacent cells to robot in SLAM
    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        spacing = self.get_parameter('position_log_spacing').value
        if self._last_logged_pos is None or \
        math.hypot(x - self._last_logged_pos[0], y - self._last_logged_pos[1]) >= spacing:
            self.visited_positions.append((x, y))
            self._last_logged_pos = (x, y)


    def _publish_drone_once(self):
        """Drone map — full complete map, published once and latched."""
        grid = self._build_full_grid()
        self.drone_map_pub.publish(self._make_msg(grid))
        self.get_logger().info('Drone map published (static).')
        # Cancel this timer — drone map never changes
        raise rclpy.executors.ExternalShutdownException  # or just store timer ref and cancel

    
    def _build_full_grid(self):
        size = self.map_cells_x * self.map_cells_y
        grid = [0] * size
        if self.scenario == 'room':
            self._build_room(grid)
        elif self.scenario == 'room2':
            self._build_room2(grid)
        elif self.scenario == 'corridor':
            self._build_corridor(grid)
        else:
            self._rect(grid, -4.0, -4.0, 8.0, 8.0, fill=False)
        return grid

    
    def _publish_slam_and_confidence(self):
        """SLAM map grows from nothing as time increases. Confidence follows."""
        now     = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds * 1e-9

        # Confidence ramps from 0 to 1 over confidence_ramp_sec
        confidence = min(1.0, elapsed / self.confidence_ramp_sec)

        # SLAM map reveals itself gradually — mask out cells beyond the reveal frontier
        reveal_fraction = min(1.0, elapsed / self.slam_reveal_sec)
        grid = self._build_partial_slam_grid(reveal_fraction)

        self.slam_map_pub.publish(self._make_msg(grid))
        self.confidence_pub.publish(Float32(data=float(confidence)))

        self.get_logger().info(
            f'SLAM reveal={reveal_fraction:.2f} confidence={confidence:.2f}',
            throttle_duration_sec=2.0)

    
    def _build_partial_slam_grid(self, visited_positions: list, sensor_range_m: float = 4.0):
        """
        Realistic SLAM reveal: only expose cells within sensor_range of a
        visited robot position AND with clear line-of-sight (no lethal obstacle
        between the robot and the cell).  Everything else stays -1 (unknown).
        """
        full_grid = self._build_full_grid()
        revealed  = [-1] * len(full_grid)   # start fully unknown

        sensor_range_cells = int(sensor_range_m / self.map_res)

        for (rx, ry) in visited_positions:
            # Robot cell
            rci = int((rx - self.map_origin_x) / self.map_res)
            rcj = int((ry - self.map_origin_y) / self.map_res)

            # Scan a bounding box around the robot position
            for dj in range(-sensor_range_cells, sensor_range_cells + 1):
                for di in range(-sensor_range_cells, sensor_range_cells + 1):
                    ci = rci + di
                    cj = rcj + dj
                    if not (0 <= ci < self.map_cells_x and 0 <= cj < self.map_cells_y):
                        continue
                    # Euclidean range check
                    dist = math.sqrt(di*di + dj*dj)
                    if dist > sensor_range_cells:
                        continue
                    # Line-of-sight check via Bresenham raycast
                    if self._has_line_of_sight(full_grid, rci, rcj, ci, cj):
                        revealed[cj * self.map_cells_x + ci] = full_grid[cj * self.map_cells_x + ci]

        return revealed


    def _has_line_of_sight(self, grid: list, x0: int, y0: int, x1: int, y1: int) -> bool:
        """
        Bresenham line from (x0,y0) to (x1,y1).
        Returns False if any cell along the way (excluding the endpoint) is lethal.
        This simulates lidar being blocked by walls.
        """
        dx = abs(x1 - x0);  sx = 1 if x0 < x1 else -1
        dy = abs(y1 - y0);  sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            if (x, y) == (x1, y1):
                return True   # reached target — clear LoS
            # Check current cell for lethal obstacle (but not the start cell)
            if (x, y) != (x0, y0):
                idx = y * self.map_cells_x + x
                if 0 <= idx < len(grid) and grid[idx] >= 100:
                    return False  # ray blocked by wall
            e2 = 2 * err
            if e2 > -dy:
                err -= dy;  x += sx
            if e2 <  dx:
                err += dx;  y += sy


    def _make_msg(self, grid):
        """Build an OccupancyGrid message from a flat grid list."""
        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.resolution = self.map_res
        msg.info.width      = self.map_cells_x
        msg.info.height     = self.map_cells_y
        msg.info.origin.position.x   = self.map_origin_x
        msg.info.origin.position.y   = self.map_origin_y
        msg.info.origin.position.z   = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid
        return msg


    # PUBLISH MAP CONSTANTLY (NOT USING IT RN, TRIGGERS REMAPPING)
    def _publish(self):
        """Single timer callback — publishes drone map once, SLAM + confidence every tick."""
        elapsed    = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        confidence = min(1.0, elapsed / max(self.confidence_ramp_sec, 1e-9))
        reveal     = min(1.0, elapsed / max(self.slam_reveal_sec,     1e-9))

        # Drone map — latched, only needs to go out once
        if not self._drone_published:
            grid = self._build_full_grid()
            self.drone_map_pub.publish(self._make_msg(grid))
            self._drone_published = True
            self.get_logger().info('Drone map published (static, latched).')

        # SLAM map — grows over time
        sensor_range_m = self.get_parameter('sensor_range_m').value
        slam_grid = self._build_partial_slam_grid(self.visited_positions, sensor_range_m)
        self.slam_map_pub.publish(self._make_msg(slam_grid))

        # Confidence score
        self.confidence_pub.publish(Float32(data=float(confidence)))

        # Status marker floating above the map in RViz
        self._publish_status_marker(confidence)

        self.get_logger().info(
            f'reveal={reveal:.2f} confidence={confidence:.2f}',
            throttle_duration_sec=2.0)

    
    def _publish_status_marker(self, confidence: float):
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.ns     = 'fusion_status'
        m.id     = 0
        m.type   = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x    = self.map_origin_x + 0.5
        m.pose.position.y    = self.map_origin_y + self.map_height_m + 0.3
        m.pose.position.z    = 0.2
        m.pose.orientation.w = 1.0
        m.scale.z = 0.4
        m.color   = ColorRGBA(
            r=float(1.0 - confidence),
            g=float(confidence),
            b=0.0,
            a=1.0)
        m.text = f'Fusion conf: {confidence:.2f}'
        self.status_marker_pub.publish(m)



    # ==========================================================================
    # Scenario builders
    # ==========================================================================

    def _build_room(self, grid):
        """Original room — 4 symmetric small boxes."""
        self._rect(grid, -4.0, -4.0, 8.0, 8.0, fill=False)
        self._rect(grid, -2.5, -2.5, 0.5, 0.5, fill=True)
        self._rect(grid,  2.0, -2.5, 0.5, 0.5, fill=True)
        self._rect(grid, -2.5,  2.0, 0.5, 0.5, fill=True)
        self._rect(grid,  2.0,  2.0, 0.5, 0.5, fill=True)

    def _build_corridor(self, grid):
        """Narrow corridor with staggered obstacles."""
        self._rect(grid, -5.0, -0.75, 10.0, 1.5, fill=False)
        for ox, oy in [(-3.5,-0.3),(-1.5,0.3),(0.5,-0.3),(2.0,0.3),(3.5,-0.3)]:
            self._rect(grid, ox-0.15, oy-0.15, 0.3, 0.3, fill=True)

    def _build_room2(self, grid):
        """
        room2 — asymmetric 8x8m test environment designed to stress A*.
        """
        # --- Outer boundary ---
        self._rect(grid, -4.0, -4.0, 8.0, 8.0, fill=False)

        # -------------------------------------------------------------------
        # CHOKE POINT — horizontal wall across upper half with a narrow gap
        # -------------------------------------------------------------------
        self._rect(grid, -3.8,  1.3,  3.2, 0.2, fill=True)   # left segment
        self._rect(grid,  0.6,  1.3,  3.2, 0.2, fill=True)   # right segment

        # -------------------------------------------------------------------
        # SCATTERED BOXES — different sizes, asymmetrically placed
        # -------------------------------------------------------------------
        self._rect(grid, -3.2,  2.2,  0.4, 0.4, fill=True)   # [A] top-left
        self._rect(grid,  1.5,  0.0,  0.8, 0.8, fill=True)   # [B] centre-right
        self._rect(grid, -1.5, -0.8,  0.25, 1.6, fill=True)  # [C] tall pillar
        self._rect(grid,  2.5, -3.2,  0.5, 0.5, fill=True)   # [D] bottom-right

        # -------------------------------------------------------------------
        # L-SHAPED WALL — bottom-left
        # -------------------------------------------------------------------
        self._rect(grid, -3.5, -3.5,  0.2, 2.5, fill=True)
        self._rect(grid, -3.5, -3.5,  2.0, 0.2, fill=True)

        # -------------------------------------------------------------------
        # U-SHAPED POCKET — bottom-right
        # -------------------------------------------------------------------
        self._rect(grid,  1.5, -3.5,  0.2, 2.0, fill=True)
        self._rect(grid,  3.2, -3.5,  0.2, 2.0, fill=True)
        self._rect(grid,  1.5, -3.5,  1.9, 0.2, fill=True)

        # -------------------------------------------------------------------
        # NEW: DIAGONAL STEPPING STONES — forces slalom across centre
        # -------------------------------------------------------------------
        for i, (ox, oy) in enumerate([
            (-2.5, -1.5),
            (-1.8, -0.5),
            (-1.0,  0.5),
            (-0.2,  1.0),
        ]):
            self._rect(grid, ox, oy, 0.35, 0.35, fill=True)

        # -------------------------------------------------------------------
        # NEW: ZIGZAG BARRIER — right side, creates forced detour
        # -------------------------------------------------------------------
        self._rect(grid,  2.2, -1.5,  1.5, 0.2, fill=True)   # top arm →
        self._rect(grid,  2.2, -1.5,  0.2, 1.0, fill=True)   # drop down
        self._rect(grid,  1.0, -2.5,  1.4, 0.2, fill=True)   # middle arm ←
        self._rect(grid,  2.2, -2.5,  0.2, 0.8, fill=True)   # drop down again
        self._rect(grid,  2.2, -3.3,  1.5, 0.2, fill=True)   # bottom arm →

        # -------------------------------------------------------------------
        # NEW: PILLAR CLUSTER — top-right quadrant, forces route choice
        # -------------------------------------------------------------------
        for ox, oy in [(2.0, 2.5), (2.6, 2.0), (3.0, 2.8), (2.3, 3.2)]:
            self._rect(grid, ox, oy, 0.3, 0.3, fill=True)

        # -------------------------------------------------------------------
        # NEW: THIN HORIZONTAL DIVIDER — centre area, partial wall
        # -------------------------------------------------------------------
        self._rect(grid, -3.0, -0.1,  2.0, 0.15, fill=True)  # left of centre
        self._rect(grid,  0.5, -0.1,  1.0, 0.15, fill=True)  # right fragment

    # ==========================================================================
    # Primitive: filled or hollow rectangle
    # ==========================================================================

    def _rect(self, grid, x, y, w, h, thickness=0.15, fill=False):
        ci0 = max(0, min(int((x     - self.map_origin_x) / self.map_res), self.map_cells_x-1))
        ci1 = max(0, min(int((x + w - self.map_origin_x) / self.map_res), self.map_cells_x-1))
        cj0 = max(0, min(int((y     - self.map_origin_y) / self.map_res), self.map_cells_y-1))
        cj1 = max(0, min(int((y + h - self.map_origin_y) / self.map_res), self.map_cells_y-1))
        t   = max(1, int(thickness / self.map_res))

        for cj in range(cj0, cj1 + 1):
            for ci in range(ci0, ci1 + 1):
                is_border = (cj < cj0+t or cj > cj1-t or
                             ci < ci0+t or ci > ci1-t)
                if fill or is_border:
                    idx = cj * self.map_cells_x + ci
                    grid[idx] = 100


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