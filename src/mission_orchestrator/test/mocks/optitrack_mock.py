"""
OptiTrackMockNode — publishes /optitrack/rigid_body to simulate the OptiTrack system.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class OptiTrackMockNode(Node):
    """
    Publishes PoseStamped on /optitrack/rigid_body at 10 Hz.

    frame_id:        header.frame_id of each message (default 'drone').
    active:          if False, no messages are published at all.
    stamp_offset_sec: added to the current clock time when building the stamp.
                     Use a large negative value (e.g. -5.0) to simulate stale messages.
    """

    def __init__(
        self,
        frame_id: str = 'drone',
        active: bool = True,
        stamp_offset_sec: float = 0.0,
    ) -> None:
        super().__init__('optitrack_mock')
        self._frame_id = frame_id
        self._stamp_offset_sec = stamp_offset_sec
        self._pub = self.create_publisher(PoseStamped, '/optitrack/rigid_body', 10)
        if active:
            self.create_timer(0.1, self._publish)

    def _publish(self) -> None:
        msg = PoseStamped()
        msg.header.frame_id = self._frame_id

        now_ns = self.get_clock().now().nanoseconds
        offset_ns = int(self._stamp_offset_sec * 1e9)
        adjusted_ns = max(0, now_ns + offset_ns)

        from rclpy.time import Time
        msg.header.stamp = Time(nanoseconds=adjusted_ns).to_msg()
        self._pub.publish(msg)
