"""Unit tests for the map_fusion algorithm modules.

These exercise pure-numpy/scipy code only -- no ROS runtime is required, so
they run under plain ``pytest`` as well as ``colcon test``.
"""

import math

import numpy as np

from map_fusion.coarse_search import coarse_search
from map_fusion.geometry import (angle_diff, apply_se2, compose, invert,
                                 rigid_fit_2d)
from map_fusion.grid_utils import GridInfo
from map_fusion.icp import icp_align, residual_to_confidence


def test_compose_invert_roundtrip():
    t = (1.3, -0.8, math.radians(35.0))
    identity = compose(t, invert(t))
    assert abs(identity[0]) < 1e-9
    assert abs(identity[1]) < 1e-9
    assert abs(angle_diff(identity[2], 0.0)) < 1e-9


def test_rigid_fit_recovers_transform():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-2.0, 2.0, size=(50, 2))
    t_gt = (0.7, -0.4, math.radians(20.0))
    moved = apply_se2(t_gt, pts)
    t_est = rigid_fit_2d(pts, moved)
    assert math.hypot(t_est[0] - t_gt[0], t_est[1] - t_gt[1]) < 1e-9
    assert abs(angle_diff(t_est[2], t_gt[2])) < 1e-9


def test_icp_refines_perturbed_initial_guess():
    rng = np.random.default_rng(1)
    # Target cloud in the world frame.
    target = rng.uniform(0.0, 4.0, size=(120, 2))
    t_gt = (1.30, -0.80, math.radians(35.0))
    # Source = target expressed in the slam frame, so apply_se2(t_gt) recovers it.
    source = apply_se2(invert(t_gt), target)

    init = (t_gt[0] + 0.05, t_gt[1] - 0.04, t_gt[2] + math.radians(3.0))
    params = {'icp_max_correspondence_m': 0.30,
              'icp_max_iterations': 50,
              'icp_convergence_epsilon': 1e-5}
    result = icp_align(source, target, init, params)

    assert result['converged']
    assert math.hypot(result['t'][0] - t_gt[0],
                      result['t'][1] - t_gt[1]) < 1e-3
    assert abs(angle_diff(result['t'][2], t_gt[2])) < math.radians(0.5)
    assert result['residual'] < 1e-3


def test_residual_to_confidence_bounds():
    assert residual_to_confidence(0.0, 0.05) == 1.0
    assert 0.0 < residual_to_confidence(0.05, 0.05) < 1.0
    assert residual_to_confidence(float('inf'), 0.05) == 0.0


def test_coarse_search_finds_correct_basin():
    # Drone edge image: an L-shaped contour on a 0.05 m/cell grid.
    res = 0.05
    drone_info = GridInfo(res, 120, 120, 0.0, 0.0, 0.0)
    edge_img = np.zeros((120, 120), dtype=np.uint8)
    edge_img[30:90, 30] = 1          # vertical segment
    edge_img[30, 30:90] = 1          # horizontal segment
    edge_img[60, 30:70] = 1          # a distinguishing stub (breaks symmetry)

    rows, cols = np.where(edge_img == 1)
    world_pts = np.stack([(cols + 0.5) * res, (rows + 0.5) * res], axis=-1)

    t_gt = (0.40, -0.30, math.radians(15.0))
    slam_pts = apply_se2(invert(t_gt), world_pts)

    params = {'coarse_rotation_step_deg': 5.0,
              'coarse_translation_step_m': 0.10,
              'coarse_translation_radius_m': 1.0,
              'coarse_top_k': 5,
              'symmetry_score_tolerance': 0.05}
    candidates, _ = coarse_search(slam_pts, edge_img, drone_info,
                                  params, seed=(t_gt[0], t_gt[1]))
    assert candidates
    best = candidates[0][1]
    # Coarse search is discrete: expect within one rotation/translation step.
    assert math.hypot(best[0] - t_gt[0], best[1] - t_gt[1]) < 0.15
    assert abs(angle_diff(best[2], t_gt[2])) < math.radians(7.5)
