"""
Pytest fixtures for mission_orchestrator integration tests.

Shared helpers (TestableOrchestratorNode, _FakeProc, make_test_config, …)
live in test_helpers.py and are imported with the package-qualified form
`from test.test_helpers import ...` because test/__init__.py makes this
directory a Python package.
"""

from __future__ import annotations

import threading

import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor


@pytest.fixture(scope='session')
def ros_init():
    rclpy.init()
    yield
    rclpy.try_shutdown()


@pytest.fixture
def executor(ros_init):
    exec_ = MultiThreadedExecutor(num_threads=4)
    spin_thread = threading.Thread(target=exec_.spin, daemon=True, name='ros-spin-test')
    spin_thread.start()
    yield exec_
    exec_.shutdown(timeout_sec=3.0)
