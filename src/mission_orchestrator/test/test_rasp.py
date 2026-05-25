"""
Integration tests for mission_orchestrator stages 01-04 (Raspberry Pi + EKF).

The _OrchestratorUnderTest subclass no-ops stages 05-20 so only the rasp/EKF
pipeline runs.  External calls (subprocess ping, paramiko SSH) are patched.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import paramiko

from test.test_helpers import TestableOrchestratorNode, make_ssh_mock, make_test_config
from test.mocks.rasp_mock import RaspMockNode


# ─────────────────────────────────────────────────────────────────────────────
# Restricted orchestrator: only runs stages 01-04
# ─────────────────────────────────────────────────────────────────────────────

class _OrchestratorUnderTest(TestableOrchestratorNode):
    """Stages 05-20 are no-ops; stages 01-04 run normally."""

    def _stage_05_check_optitrack(self):
        pass

    def _stage_05b_connect_tello_wifi(self):
        pass

    def _stage_06_launch_tello_driver(self):
        pass

    def _stage_07_drone_preflight(self):
        pass

    def _stage_08_launch_tello_map(self):
        pass

    def _stage_09_observe_drone_states(self):
        pass

    def _stage_10_wait_video_topics(self):
        pass

    def _stage_11_verify_video_integrity(self):
        pass

    def _stage_12_launch_trajectory_planner(self):
        pass

    def _stage_13_launch_map_fusion(self):
        pass

    def _stage_14_launch_oradar(self):
        pass

    def _stage_15_launch_marker_localizer(self):
        pass

    def _stage_16_call_localize_markers(self):
        return []

    def _stage_17_publish_aruco_poses(self, markers):
        pass

    def _stage_18_launch_map_builder(self):
        pass

    def _stage_19_call_map_builder(self):
        return None

    def _stage_20_publish_drone_map(self, grid):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# subprocess.run side-effects
# ─────────────────────────────────────────────────────────────────────────────

def _ping_ok(_cmd, **_kw):
    return MagicMock(returncode=0)


def _ping_fail(_cmd, **_kw):
    return MagicMock(returncode=1)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_rasp_happy_path(executor, tmp_path):
    """Stages 01-04 all succeed: ping OK, SSH OK, AMR launched, EKF stable."""
    cfg = make_test_config(tmp_path)
    ssh_client = make_ssh_mock()

    with patch('subprocess.run', side_effect=_ping_ok), \
         patch('paramiko.SSHClient', return_value=ssh_client):

        node = _OrchestratorUnderTest(cfg)
        rasp = RaspMockNode(velocity_mps=0.0)
        executor.add_node(node)
        executor.add_node(rasp)
        try:
            node.run()
            assert node.mission_complete, "Mission should complete when all stages succeed"
            assert not node.abort_called
        finally:
            executor.remove_node(node)
            executor.remove_node(rasp)
            node.destroy_node()
            rasp.destroy_node()


def test_rasp_abort_ping_fails(executor, tmp_path):
    """Stage 01 fails: ping returns non-zero → MissionAbortError."""
    cfg = make_test_config(tmp_path)

    with patch('subprocess.run', side_effect=_ping_fail):
        node = _OrchestratorUnderTest(cfg)
        executor.add_node(node)
        try:
            node.run()
            assert node.abort_called, "Abort should be triggered when ping fails"
            assert not node.mission_complete
        finally:
            executor.remove_node(node)
            node.destroy_node()


def test_rasp_abort_ssh_auth_fails(executor, tmp_path):
    """Stage 02 fails: paramiko raises AuthenticationException → MissionAbortError."""
    cfg = make_test_config(tmp_path)

    bad_client = MagicMock()
    bad_client.connect.side_effect = paramiko.AuthenticationException("bad creds")

    with patch('subprocess.run', side_effect=_ping_ok), \
         patch('paramiko.SSHClient', return_value=bad_client):

        node = _OrchestratorUnderTest(cfg)
        executor.add_node(node)
        try:
            node.run()
            assert node.abort_called, "Abort should be triggered when SSH auth fails"
            assert not node.mission_complete
        finally:
            executor.remove_node(node)
            node.destroy_node()


def test_rasp_abort_ekf_timeout(executor, tmp_path):
    """Stage 04 fails: no EKF messages published → timeout → MissionAbortError."""
    cfg = make_test_config(tmp_path, overrides={'ekf': {'timeout_sec': 2.0}})
    ssh_client = make_ssh_mock()

    with patch('subprocess.run', side_effect=_ping_ok), \
         patch('paramiko.SSHClient', return_value=ssh_client):

        node = _OrchestratorUnderTest(cfg)
        executor.add_node(node)
        try:
            node.run()
            assert node.abort_called, "Abort should be triggered when EKF times out"
            assert not node.mission_complete
        finally:
            executor.remove_node(node)
            node.destroy_node()


def test_rasp_abort_ekf_unstable(executor, tmp_path):
    """Stage 04 fails: EKF publishes high velocity, window never stabilises → timeout."""
    cfg = make_test_config(tmp_path, overrides={'ekf': {'timeout_sec': 2.0}})
    ssh_client = make_ssh_mock()

    with patch('subprocess.run', side_effect=_ping_ok), \
         patch('paramiko.SSHClient', return_value=ssh_client):

        node = _OrchestratorUnderTest(cfg)
        rasp = RaspMockNode(velocity_mps=1.0)  # 1.0 >> threshold 0.05
        executor.add_node(node)
        executor.add_node(rasp)
        try:
            node.run()
            assert node.abort_called, "Abort should be triggered when EKF stays unstable"
            assert not node.mission_complete
        finally:
            executor.remove_node(node)
            executor.remove_node(rasp)
            node.destroy_node()
            rasp.destroy_node()
