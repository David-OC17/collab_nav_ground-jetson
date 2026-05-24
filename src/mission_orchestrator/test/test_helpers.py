"""
Shared non-fixture helpers for mission_orchestrator tests.

Imported as  `from test.test_helpers import ...`  because test/__init__.py
makes the test/ directory a Python package; package-qualified imports are
the only reliable way to share code between modules inside that package.
"""

from __future__ import annotations

import os
from typing import Optional
from unittest.mock import MagicMock

import yaml

from mission_orchestrator.orchestrator_node import MissionOrchestratorNode


# ─────────────────────────────────────────────────────────────────────────────
# Fake process stand-in
# ─────────────────────────────────────────────────────────────────────────────

class _FakeProc:
    """Minimal subprocess.Popen stand-in — all operations are no-ops."""
    pid = 0

    def poll(self):
        return None  # appear still running so _kill_proc proceeds cleanly

    def send_signal(self, sig):
        pass

    def wait(self, timeout=None):
        pass

    def kill(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Testable orchestrator subclass
# ─────────────────────────────────────────────────────────────────────────────

class TestableOrchestratorNode(MissionOrchestratorNode):
    """
    Subclass used in tests:
    - records whether _abort() was called
    - suppresses real process/SSH teardown in _abort()
    """

    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        self.abort_called: bool = False

    @property
    def mission_complete(self) -> bool:
        return self._mission_complete

    def _abort(self) -> None:
        self.abort_called = True
        if self._ssh is not None:
            try:
                self._ssh.close()
            except Exception:
                pass
            self._ssh = None


# ─────────────────────────────────────────────────────────────────────────────
# Config builder
# ─────────────────────────────────────────────────────────────────────────────

def make_test_config(tmp_path, overrides: Optional[dict] = None) -> str:
    """Write a minimal orchestrator_params.yaml to *tmp_path* and return its path."""
    cfg: dict = {
        'orchestrator': {
            'rasp': {
                'ip': '127.0.0.1',
                'user': 'test',
                'password': 'test',
                'amr_launch_cmd': 'echo PID:1234',
                'ping_count': 1,
                'ping_timeout_sec': 1.0,
                'ssh_connect_timeout_sec': 5.0,
            },
            'ekf': {
                'topic': '/amr/ekf/odom',
                'window_size': 5,
                'velocity_threshold_mps': 0.05,
                'timeout_sec': 10.0,
            },
            'optitrack': {
                'topic': '/optitrack/rigid_body',
                'expected_frame_id': 'drone',
                'max_stamp_age_sec': 1.0,
                'check_timeout_sec': 5.0,
                'retry_delay_sec': 0.3,
            },
            'drone': {
                'state_topic': '/drone/state',
                'camera_topic': '/camera/image_raw',
                'battery_topic': '/battery_state',
                'cmd_vel_topic': '/cmd_vel',
                'battery_min_pct': 20.0,
                'camera_timeout_sec': 5.0,
                'battery_timeout_sec': 5.0,
                'state1_timeout_sec': 10.0,
                'state2_timeout_sec': 10.0,
                'state3_timeout_sec': 10.0,
                'state4_timeout_sec': 10.0,
                'video_filename_topic': '/drone/video_filename',
                'telemetry_filename_topic': '/drone/telemetry_filename',
            },
            'video': {
                'file_appear_timeout_sec': 15.0,
            },
            'aruco': {
                'amr_marker_id': 0,
                'goal_marker_id': 1,
                'amr_pose_topic': '/aruco/amr/pose',
                'goal_pose_topic': '/aruco/goal/pose',
            },
            'marker_localizer': {
                'workspace_path': str(tmp_path),
                'service_name': '/localize_markers',
                'service_timeout_sec': 30.0,
                'server_ready_timeout_sec': 15.0,
            },
            'map_builder': {
                'background_image_path': '/tmp/bg.png',
                'action_name': 'build_arena_map',
                'action_timeout_sec': 60.0,
                'server_ready_timeout_sec': 15.0,
                'drone_map_topic': '/drone/map',
            },
            'logging': {
                'log_dir': str(tmp_path / 'logs'),
                'log_level': 'DEBUG',
            },
        }
    }

    if overrides:
        _deep_merge(cfg['orchestrator'], overrides)

    p = tmp_path / 'orchestrator_params.yaml'
    p.write_text(yaml.dump(cfg))
    return str(p)


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# SSH mock factory
# ─────────────────────────────────────────────────────────────────────────────

def make_ssh_mock(pid: int = 9999) -> MagicMock:
    """Return a paramiko.SSHClient mock whose exec_command echoes a PID line."""
    client = MagicMock()
    client.connect.return_value = None
    stdout = MagicMock()
    stdout.read.return_value = f'PID:{pid}\n'.encode()
    client.exec_command.return_value = (MagicMock(), stdout, MagicMock())
    return client
