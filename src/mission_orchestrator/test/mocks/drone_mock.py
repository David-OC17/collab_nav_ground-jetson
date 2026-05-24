"""
DroneMockNode — simulates the Tello drone's ROS 2 interfaces.

Publishes:
  /drone/state           std_msgs/Int32       — internal state machine
  /camera/image_raw      sensor_msgs/Image    — dummy camera frames
  /battery_state         sensor_msgs/BatteryState
  /drone/video_filename  std_msgs/String      — path after landing (state 4)
  /drone/telemetry_filename std_msgs/String   — path after landing (state 4)

State machine progression (driven in a background thread after start_mission()):
  0 → 1 → 2 → 3 → 4
  After reaching state 4, scan.mp4 + telemetry.csv are copied from the
  scan10 dataset to *video_dir* and their paths are published.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, Image
from std_msgs.msg import Int32, String

# Path to the real scan10 dataset bundled in arena_map_builder.
# From  src/mission_orchestrator/test/mocks/ we go up three levels to src/
_SRC_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SCAN10_DIR = os.path.join(
    _SRC_DIR, 'arena_map_builder', 'data', 'drone_scans', 'scan10')


class DroneMockNode(Node):
    """
    Simulates drone telemetry and the scanning state machine.

    video_dir:       directory where scan.mp4 / telemetry.csv will be written.
    battery_pct:     reported battery percentage (use < 20 to trigger low-battery abort).
    camera_active:   if False, never publishes camera frames (triggers camera timeout).
    stuck_at_state:  if set, the state machine publishes this state and then freezes,
                     causing the orchestrator to time out waiting for the next state.
    state_delays:    per-state sleep (seconds) before publishing that state.
    """

    def __init__(
        self,
        video_dir: str,
        battery_pct: float = 80.0,
        camera_active: bool = True,
        stuck_at_state: Optional[int] = None,
        state_delays: Optional[Dict[int, float]] = None,
    ) -> None:
        super().__init__('drone_mock')
        self._video_dir = video_dir
        self._battery_pct = battery_pct
        self._camera_active = camera_active
        self._stuck_at_state = stuck_at_state
        self._state_delays = state_delays or {0: 0.05, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05}

        self._pub_state = self.create_publisher(Int32, '/drone/state', 10)
        self._pub_camera = self.create_publisher(Image, '/camera/image_raw', 10)
        self._pub_battery = self.create_publisher(BatteryState, '/battery_state', 10)
        self._pub_video = self.create_publisher(String, '/drone/video_filename', 10)
        self._pub_telemetry = self.create_publisher(String, '/drone/telemetry_filename', 10)

        self._mission_started = threading.Event()
        self._sm_thread = threading.Thread(
            target=self._run_state_machine, daemon=True, name='drone-mock-sm')
        self._sm_thread.start()

        # Publish camera and battery at 5 Hz from the start so preflight checks pass.
        self.create_timer(0.2, self._publish_sensors)

    # ── public API ───────────────────────────────────────────────────────────

    def start_mission(self) -> None:
        """Unblock the state machine thread (call when tello_map would be launched)."""
        self._mission_started.set()

    # ── internal ─────────────────────────────────────────────────────────────

    def _publish_sensors(self) -> None:
        if self._camera_active:
            self._pub_camera.publish(Image())
        bat = BatteryState()
        bat.percentage = self._battery_pct
        self._pub_battery.publish(bat)

    def _run_state_machine(self) -> None:
        self._mission_started.wait()

        for state in (0, 1, 2, 3, 4):
            delay = self._state_delays.get(state, 0.05)
            time.sleep(delay)

            msg = Int32()
            msg.data = state
            self._pub_state.publish(msg)

            if self._stuck_at_state is not None and state == self._stuck_at_state:
                # Stay in this state forever — orchestrator will time out
                time.sleep(3600.0)
                return

        # State 4 reached: copy files and publish paths
        self._publish_video_paths()

    def _publish_video_paths(self) -> None:
        os.makedirs(self._video_dir, exist_ok=True)
        video_dst = os.path.join(self._video_dir, 'scan.mp4')
        telem_dst = os.path.join(self._video_dir, 'telemetry.csv')

        shutil.copy2(os.path.join(SCAN10_DIR, 'scan.mp4'), video_dst)
        shutil.copy2(os.path.join(SCAN10_DIR, 'telemetry.csv'), telem_dst)

        time.sleep(0.1)  # small gap so subscribers are ready

        vid = String()
        vid.data = video_dst
        self._pub_video.publish(vid)

        tel = String()
        tel.data = telem_dst
        self._pub_telemetry.publish(tel)
