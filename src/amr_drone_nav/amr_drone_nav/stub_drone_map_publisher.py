#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import OccupancyGrid

SIZE = 80
RES  = 0.05

_QOS = QoSProfile(
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
)


class StubDroneMap(Node):
    def __init__(self):
        super().__init__('stub_drone_map_publisher')
        self.pub = self.create_publisher(OccupancyGrid, '/drone/map', _QOS)
        self._msg = self._build_msg()

        # Publica una vez inmediatamente
        self._publish()

        # Y luego cada 1 segundo para mantener has_updated_data_ activo
        self.create_timer(1.0, self._publish)

    def _build_msg(self) -> OccupancyGrid:
        grid = np.zeros((SIZE, SIZE), dtype=np.int8)
        grid[0, :]   = 100
        grid[-1, :]  = 100
        grid[:, 0]   = 100
        grid[:, -1]  = 100
        grid[30:40, 20:25] = 100
        grid[55:60, 50:60] = 100

        msg = OccupancyGrid()
        msg.header.frame_id = 'world'
        msg.info.resolution = RES
        msg.info.width      = SIZE
        msg.info.height     = SIZE
        msg.info.origin.position.x    = 0.0
        msg.info.origin.position.y    = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.flatten(order='C').tolist()
        return msg

    def _publish(self):
        self._msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self._msg)
        self.get_logger().debug('Drone map republished')


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(StubDroneMap())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
