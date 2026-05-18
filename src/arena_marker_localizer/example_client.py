"""Minimal example client for the LocalizeMarkers service.

Usage:
    # Terminal 1
    ros2 launch arena_marker_localizer marker_localizer.launch.py
    ros2 param set /marker_localizer_service intrinsics_path /abs/path/calib.yaml

    # Terminal 2
    python3 example_client.py /abs/path/video.mp4 /abs/path/optitrack.csv
"""

import sys
import rclpy
from rclpy.node import Node

from arena_marker_localizer_interfaces.srv import LocalizeMarkers


class ExampleClient(Node):
    def __init__(self):
        super().__init__("marker_localizer_example_client")
        self._client = self.create_client(LocalizeMarkers, "localize_markers")

    def call(self, video_path: str, csv_path: str):
        while not self._client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /localize_markers...")
        req = LocalizeMarkers.Request()
        req.video_path = video_path
        req.optitrack_csv = csv_path

        future = self._client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()

        if not res.success:
            self.get_logger().error(f"Service failed: {res.message}")
            return

        self.get_logger().info(res.message)
        for m in res.markers:
            self.get_logger().info(
                f"  id={m.id:3d}  "
                f"xyz=({m.pose_3d.position.x:+.3f}, "
                f"{m.pose_3d.position.y:+.3f}, "
                f"{m.pose_3d.position.z:+.3f})  "
                f"yaw={m.pose_2d.theta:+.3f} rad  "
                f"cell=({m.cell_x}, {m.cell_y})  "
                f"obs={m.n_observations}"
            )


def main():
    if len(sys.argv) < 3:
        print("Usage: example_client.py <video_path> <optitrack_csv>")
        sys.exit(1)
    rclpy.init()
    node = ExampleClient()
    node.call(sys.argv[1], sys.argv[2])
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
