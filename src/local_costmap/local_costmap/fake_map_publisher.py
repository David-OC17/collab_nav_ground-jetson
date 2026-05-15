#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header

import tf2_ros


class FakeMapPublisher(Node):

    def __init__(self):
        super().__init__('fake_map_publisher')

        self.declare_parameter('scenario',       'room')
        self.declare_parameter('map_resolution', 0.05)
        self.declare_parameter('map_width_m',    12.0)
        self.declare_parameter('map_height_m',   12.0)
        self.declare_parameter('publish_rate',   1.0)
        self.declare_parameter('robot_speed',    0.2)
        self.declare_parameter('move_robot',     False)

        self.scenario     = self.get_parameter('scenario').value
        self.map_res      = self.get_parameter('map_resolution').value
        self.map_width_m  = self.get_parameter('map_width_m').value
        self.map_height_m = self.get_parameter('map_height_m').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.robot_speed  = self.get_parameter('robot_speed').value
        self.move_robot   = self.get_parameter('move_robot').value

        self.map_cells_x  = int(self.map_width_m  / self.map_res)
        self.map_cells_y  = int(self.map_height_m / self.map_res)
        self.map_origin_x = -self.map_width_m  / 2.0
        self.map_origin_y = -self.map_height_m / 2.0

        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0
        self.t         = 0.0

        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', latched_qos)
        # Publish constantly 
        # self.timer   = self.create_timer(1.0 / self.publish_rate, self._publish)
        # Publish once at startup 
        self.timer   = self.create_timer(1.0, self._publish_once)

        self.get_logger().info(
            f'FakeMapPublisher ready | scenario={self.scenario} | '
            f'{self.map_cells_x}x{self.map_cells_y} cells'
        )


    def _publish_once(self):
        self._publish()
        self.timer.cancel()   # stop the timer after first publish
        self.get_logger().info('Map published once — timer stopped.')


    # PUBLISH MAP CONSTANTLY (NOT USING IT RN, TRIGGERS REMAPPING)
    def _publish(self):
        self.t += 1.0 / self.publish_rate

        if self.move_robot:
            self.robot_x   = 2.0 * math.cos(self.t * self.robot_speed)
            self.robot_y   = 2.0 * math.sin(self.t * self.robot_speed)
            self.robot_yaw = self.t * self.robot_speed + math.pi / 2.0

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

        nonzero = sum(1 for v in grid if v > 0)
        self.get_logger().info(
            f't={self.t:.1f} | robot=({self.robot_x:.2f},{self.robot_y:.2f}) | '
            f'nonzero cells={nonzero}'
        )

        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.resolution = self.map_res
        msg.info.width      = self.map_cells_x
        msg.info.height     = self.map_cells_y
        msg.info.origin.position.x  = self.map_origin_x
        msg.info.origin.position.y  = self.map_origin_y
        msg.info.origin.position.z  = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid
        self.map_pub.publish(msg)

        tf_msg = TransformStamped()
        tf_msg.header.stamp    = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = 'map'
        tf_msg.child_frame_id  = 'base_link'
        tf_msg.transform.translation.x = self.robot_x
        tf_msg.transform.translation.y = self.robot_y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.z = math.sin(self.robot_yaw / 2.0)
        tf_msg.transform.rotation.w = math.cos(self.robot_yaw / 2.0)
        self.tf_broadcaster.sendTransform(tf_msg)

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
        # Wall runs from x=-3.5 to x=3.5 at y=1.5, gap at x=[-0.4, 0.4]
        # -------------------------------------------------------------------
        self._rect(grid, -3.8,  1.3,  3.2, 0.2, fill=True)   # left segment
        self._rect(grid,  0.6,  1.3,  3.2, 0.2, fill=True)   # right segment
        # gap is between x=[-0.6, 0.6] — only ~1.2m wide

        # -------------------------------------------------------------------
        # SCATTERED BOXES — different sizes, asymmetrically placed
        # -------------------------------------------------------------------
        # [A] Small box top-left quadrant
        self._rect(grid, -3.2,  2.2,  0.4, 0.4, fill=True)

        # [B] Medium box centre-right, below choke
        self._rect(grid,  1.5,  0.0,  0.8, 0.8, fill=True)

        # [C] Tall thin pillar centre-left
        self._rect(grid, -1.5, -0.8,  0.25, 1.6, fill=True)

        # [D] Small box bottom-right
        self._rect(grid,  2.5, -3.2,  0.5, 0.5, fill=True)

        # -------------------------------------------------------------------
        # L-SHAPED WALL — bottom-left, forces detour around corner
        # Vertical arm + horizontal arm
        # -------------------------------------------------------------------
        self._rect(grid, -3.5, -3.5,  0.2, 2.5, fill=True)   # vertical arm
        self._rect(grid, -3.5, -3.5,  2.0, 0.2, fill=True)   # horizontal arm

        # -------------------------------------------------------------------
        # U-SHAPED PARTIAL WALL — bottom-right, creates a pocket
        # Open side faces left — robot must go around, not through
        # -------------------------------------------------------------------
        self._rect(grid,  1.5, -3.5,  0.2, 2.0, fill=True)   # left wall of U
        self._rect(grid,  3.2, -3.5,  0.2, 2.0, fill=True)   # right wall of U
        self._rect(grid,  1.5, -3.5,  1.9, 0.2, fill=True)   # bottom of U

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