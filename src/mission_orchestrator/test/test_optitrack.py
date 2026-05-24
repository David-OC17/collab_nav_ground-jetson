"""
Integration tests for mission_orchestrator stage 05 (OptiTrack verification).

The _OrchestratorUnderTest subclass no-ops stages 01-04 (already tested in
test_rasp.py) and stages 06-20, so only stage 05 runs.
"""

from __future__ import annotations

from test.test_helpers import TestableOrchestratorNode, make_test_config
from test.mocks.optitrack_mock import OptiTrackMockNode


# ─────────────────────────────────────────────────────────────────────────────
# Restricted orchestrator: only runs stage 05
# ─────────────────────────────────────────────────────────────────────────────

class _OrchestratorUnderTest(TestableOrchestratorNode):
    """Stages 01-04 and 06-20 are no-ops; only stage 05 runs normally."""

    def _stage_01_ping(self):
        pass

    def _stage_02_ssh_connect(self):
        pass

    def _stage_03_launch_amr(self):
        pass

    def _stage_04_wait_ekf_stable(self):
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
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_optitrack_happy_path(executor, tmp_path):
    """Stage 05 succeeds: fresh messages with the expected frame_id arrive."""
    cfg = make_test_config(tmp_path)

    node = _OrchestratorUnderTest(cfg)
    optitrack = OptiTrackMockNode(frame_id='drone', active=True, stamp_offset_sec=0.0)
    executor.add_node(node)
    executor.add_node(optitrack)
    try:
        node.run()
        assert node.mission_complete, "Mission should complete when OptiTrack is healthy"
        assert not node.abort_called
    finally:
        executor.remove_node(node)
        executor.remove_node(optitrack)
        node.destroy_node()
        optitrack.destroy_node()


def test_optitrack_abort_no_message(executor, tmp_path):
    """Stage 05 fails: no OptiTrack messages published → timeout → abort."""
    cfg = make_test_config(tmp_path, overrides={
        'optitrack': {'check_timeout_sec': 2.0, 'retry_delay_sec': 0.3}
    })

    node = _OrchestratorUnderTest(cfg)
    optitrack = OptiTrackMockNode(frame_id='drone', active=False)
    executor.add_node(node)
    executor.add_node(optitrack)
    try:
        node.run()
        assert node.abort_called, "Abort should be triggered when no OptiTrack message arrives"
        assert not node.mission_complete
    finally:
        executor.remove_node(node)
        executor.remove_node(optitrack)
        node.destroy_node()
        optitrack.destroy_node()


def test_optitrack_abort_wrong_frame_id(executor, tmp_path):
    """Stage 05 fails: messages arrive but frame_id doesn't match → abort."""
    cfg = make_test_config(tmp_path)

    node = _OrchestratorUnderTest(cfg)
    optitrack = OptiTrackMockNode(frame_id='world', active=True, stamp_offset_sec=0.0)
    executor.add_node(node)
    executor.add_node(optitrack)
    try:
        node.run()
        assert node.abort_called, "Abort should be triggered on frame_id mismatch"
        assert not node.mission_complete
    finally:
        executor.remove_node(node)
        executor.remove_node(optitrack)
        node.destroy_node()
        optitrack.destroy_node()


def test_optitrack_abort_stale_stamp(executor, tmp_path):
    """Stage 05 fails: messages arrive but stamp is older than max_stamp_age_sec → abort."""
    cfg = make_test_config(tmp_path, overrides={
        'optitrack': {'max_stamp_age_sec': 1.0}
    })

    node = _OrchestratorUnderTest(cfg)
    # -5.0 s puts the stamp well beyond max_stamp_age_sec=1.0
    optitrack = OptiTrackMockNode(frame_id='drone', active=True, stamp_offset_sec=-5.0)
    executor.add_node(node)
    executor.add_node(optitrack)
    try:
        node.run()
        assert node.abort_called, "Abort should be triggered when OptiTrack stamp is stale"
        assert not node.mission_complete
    finally:
        executor.remove_node(node)
        executor.remove_node(optitrack)
        node.destroy_node()
        optitrack.destroy_node()
