"""
ImuMockNode — publishes /imu/data_raw to simulate the AMR's IMU.
"""

from __future__ import annotations

from rclpy.node import Node
from sensor_msgs.msg import Imu


class ImuMockNode(Node):
    """
    Publishes Imu on /imu/data_raw at 100 Hz.

    active: when False, no messages are published (simulates IMU not running).
    """

    def __init__(self, active: bool = True) -> None:
        super().__init__('imu_mock')
        self._active = active
        self._pub = self.create_publisher(Imu, '/imu/data_raw', 10)
        self.create_timer(0.01, self._publish)  # 100 Hz

    def _publish(self) -> None:
        if not self._active:
            return
        self._pub.publish(Imu())
