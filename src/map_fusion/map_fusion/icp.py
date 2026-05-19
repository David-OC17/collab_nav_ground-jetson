"""Stage 5 of the pipeline: point-to-point ICP (Besl & McKay) refinement.

The expensive nearest-neighbour search is delegated to ``scipy.spatial.cKDTree``;
the iteration loop itself is hand-written so that the spec's per-iteration
outlier rejection (drop correspondences beyond 2x the median distance) can be
applied exactly -- something the off-the-shelf library ICPs do not expose.
"""

import math

import numpy as np
from scipy.spatial import cKDTree

from .geometry import apply_se2, compose, rigid_fit_2d


def icp_align(source, target, init_t, params):
    """Refine ``init_t`` so that ``source`` (slam frame) aligns onto ``target``
    (world frame).

    Parameters
    ----------
    source : (N, 2) array
        SLAM edge point cloud, in the ``slam_map`` frame.
    target : (M, 2) array
        Drone edge point cloud, in the ``world`` frame.
    init_t : (tx, ty, theta)
        Initial transform (a coarse-search candidate, or the warm-start prior).
    params : dict
        Tuning parameters.

    Returns
    -------
    dict with keys ``t`` (refined transform), ``residual`` (mean inlier
    correspondence distance), ``converged`` (bool), ``inliers`` (int).
    """
    max_corr = params['icp_max_correspondence_m']
    max_iter = int(params['icp_max_iterations'])
    eps = params['icp_convergence_epsilon']

    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(source) < 3 or len(target) < 3:
        return {'t': tuple(init_t), 'residual': float('inf'),
                'converged': False, 'inliers': 0}

    tree = cKDTree(target)
    t = tuple(init_t)
    converged = False

    for _ in range(max_iter):
        src_world = apply_se2(t, source)
        dist, idx = tree.query(src_world)

        # Reject correspondences beyond the gate distance.
        gated = dist <= max_corr
        if int(gated.sum()) < 3:
            break

        # Per-iteration outlier rejection: drop beyond 2x the median distance.
        gated_dist = dist[gated]
        keep = gated_dist <= 2.0 * np.median(gated_dist)
        sel = np.where(gated)[0][keep]

        delta = rigid_fit_2d(src_world[sel], target[idx[sel]])
        t = compose(delta, t)

        if math.hypot(delta[0], delta[1]) < eps and abs(delta[2]) < eps:
            converged = True
            break

    # Final residual / inlier count on the converged transform.
    src_world = apply_se2(t, source)
    dist, _ = tree.query(src_world)
    gated = dist <= max_corr
    if int(gated.sum()) >= 3:
        residual = float(np.mean(dist[gated]))
        inliers = int(gated.sum())
    else:
        residual = float('inf')
        inliers = int(gated.sum())

    return {'t': t, 'residual': residual,
            'converged': converged, 'inliers': inliers}


def residual_to_confidence(residual, residual_scale):
    """Map a mean ICP residual to a confidence in [0, 1]: ``exp(-r / scale)``."""
    if not math.isfinite(residual):
        return 0.0
    return float(math.exp(-residual / max(residual_scale, 1e-9)))
