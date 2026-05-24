"""
RaspMockNode — publishes /amr/ekf/odom to simulate the AMR's EKF odometry.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class RaspMockNode(Node):
    """
    Publishes Odometry on /amr/ekf/odom at 10 Hz.

    velocity_mps: linear.x velocity to embed in each message.
                  Set to 0.0 for a stable EKF; set above the threshold
                  (default 0.05 m/s) to simulate an unstable EKF.
    """

    def __init__(self, velocity_mps: float = 0.0) -> None:
        super().__init__('rasp_mock')
        self._velocity = velocity_mps
        self._pub = self.create_publisher(Odometry, '/amr/ekf/odom', 10)
        self.create_timer(0.1, self._publish)

    def _publish(self) -> None:
        msg = Odometry()
        msg.twist.twist.linear.x = self._velocity
        self._pub.publish(msg)
