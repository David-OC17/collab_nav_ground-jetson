"""
Minimal example action client. Usage:

    ros2 run arena_map_builder build_arena_map_server     # in one terminal
    # then in another, set the background param + send a goal:
    ros2 param set /build_arena_map_server transfer.background_path /abs/path/background.png
    python3 example_client.py /abs/path/video.mp4
"""

import sys
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from arena_map_builder_msgs.action import BuildArenaMap


class ExampleClient(Node):
    def __init__(self):
        super().__init__("build_arena_map_example_client")
        self._client = ActionClient(self, BuildArenaMap, "build_arena_map")

    def send(self, video_path: str):
        self._client.wait_for_server()
        goal = BuildArenaMap.Goal()
        goal.video_path = video_path

        future = self._client.send_goal_async(goal, feedback_callback=self._fb)
        future.add_done_callback(self._goal_accepted)

    def _fb(self, fb_msg):
        f = fb_msg.feedback
        self.get_logger().info(f"[{f.stage}] {f.progress*100:5.1f}%  {f.message}")

    def _goal_accepted(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error("Goal rejected.")
            rclpy.shutdown()
            return
        self.get_logger().info("Goal accepted. Awaiting result...")
        handle.get_result_async().add_done_callback(self._done)

    def _done(self, future):
        res = future.result().result
        if res.success:
            self.get_logger().info(
                f"Done. {res.n_obstacles} obstacle(s), "
                f"mean confidence {res.mean_consistency:.2f}.\n"
                f"OccupancyGrid: {res.map.info.width}x{res.map.info.height} "
                f"cells @ {res.map.info.resolution:.3f} m/cell.\n"
                f"Debug images at: {res.debug_dir}"
            )
        else:
            self.get_logger().error(f"Failed: {res.message}")
        rclpy.shutdown()


def main():
    if len(sys.argv) < 2:
        print("Usage: example_client.py <video_path>")
        sys.exit(1)
    rclpy.init()
    node = ExampleClient()
    node.send(sys.argv[1])
    rclpy.spin(node)


if __name__ == "__main__":
    main()
