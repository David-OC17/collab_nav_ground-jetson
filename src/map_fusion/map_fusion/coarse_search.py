"""Stage 4 of the pipeline: coarse global SE(2) search.

For each discretised rotation theta the SLAM edge cloud is rotated, rasterised,
and cross-correlated against the drone edge image with an FFT. The correlation
peak over all (tx, ty) for that theta is found in one transform, collapsing the
inner translation search from O(N_t^2) to O(N log N) per rotation.

Scoring. The spec defines the per-candidate score as

    score(T) = |S' n D_edges n M| / |S' n M|

where ``M`` is the SLAM known-mask. Every SLAM edge cell is derived from an
*occupied* cell, and occupied cells are by definition known, so ``S' ⊆ M``
always holds. The denominator therefore reduces to ``|S'| = |S|`` (constant per
rotation), and the score is simply the FFT correlation peak normalised by the
SLAM edge-cell count. ``M`` still matters downstream for Stage 7 reprojection.
"""

import math

import numpy as np
from scipy.signal import fftconvolve

from .geometry import angle_diff, apply_se2


def _rasterize(points, res):
    """Rasterise (N, 2) metric points to a binary image at resolution ``res``.

    Returns ``(image, corner_xy)`` where ``corner_xy`` is the metric coordinate
    of the lower-left corner of cell (0, 0).
    """
    corner = points.min(axis=0)
    cols = np.floor((points[:, 0] - corner[0]) / res).astype(int)
    rows = np.floor((points[:, 1] - corner[1]) / res).astype(int)
    img = np.zeros((int(rows.max()) + 1, int(cols.max()) + 1), dtype=np.float32)
    img[rows, cols] = 1.0
    return img, corner


def _nms_peaks(corr, min_sep_px, k):
    """Greedy non-maximum suppression on a correlation map -> [(score, r, c)]."""
    peaks = []
    work = corr.copy()
    for _ in range(k):
        flat = int(np.argmax(work))
        r, c = np.unravel_index(flat, work.shape)
        score = work[r, c]
        if not np.isfinite(score) or score <= 0:
            break
        peaks.append((float(score), int(r), int(c)))
        r0, r1 = max(0, r - min_sep_px), min(work.shape[0], r + min_sep_px + 1)
        c0, c1 = max(0, c - min_sep_px), min(work.shape[1], c + min_sep_px + 1)
        work[r0:r1, c0:c1] = -np.inf
    return peaks


def coarse_search(slam_pts, drone_edge_img, drone_info, params, seed=None):
    """Find coarse SE(2) candidates aligning ``slam_pts`` to the drone edges.

    Parameters
    ----------
    slam_pts : (N, 2) array
        SLAM edge points in the ``slam_map`` frame.
    drone_edge_img : (H, W) array
        Binary drone edge image (Stage 1 output).
    drone_info : GridInfo
        Geometry of the drone map (defines the world-frame raster).
    params : dict
        Tuning parameters (see the node's parameter table).
    seed : (tx, ty) or None
        Optional translation seed. When given, the search is restricted to a
        window of radius ``coarse_translation_radius_m`` around it.

    Returns
    -------
    (candidates, ambiguous)
        ``candidates`` is a list of ``(score, (tx, ty, theta))`` sorted by
        score descending; ``ambiguous`` flags a possible symmetry (top two
        candidates within ``symmetry_score_tolerance``).
    """
    res = drone_info.resolution
    n_src = len(slam_pts)
    if n_src == 0:
        return [], False

    rot_step = math.radians(params['coarse_rotation_step_deg'])
    n_rot = max(1, int(round(2.0 * math.pi / rot_step)))
    drone_img = np.asarray(drone_edge_img, dtype=np.float32)
    min_sep_px = max(1, int(round(params['coarse_translation_step_m'] / res)))
    radius = params['coarse_translation_radius_m']

    candidates = []
    for i in range(n_rot):
        theta = -math.pi + i * rot_step
        rotated = apply_se2((0.0, 0.0, theta), slam_pts)
        src_img, corner = _rasterize(rotated, res)
        h_b, w_b = src_img.shape

        # Cross-correlation == convolution with the flipped kernel.
        corr = fftconvolve(drone_img, src_img[::-1, ::-1], mode='full')

        # Correlation cell (u, v) corresponds to placing the source image with
        # its (0,0) cell at drone-grid cell (u - h_b + 1, v - w_b + 1), which
        # maps to translation tx/ty below.
        if seed is not None:
            us = np.arange(corr.shape[0])
            vs = np.arange(corr.shape[1])
            tx_row = drone_info.origin_x + (vs - w_b + 1) * res - corner[0]
            ty_col = drone_info.origin_y + (us - h_b + 1) * res - corner[1]
            ok = np.outer(np.abs(ty_col - seed[1]) <= radius,
                          np.abs(tx_row - seed[0]) <= radius)
            corr = np.where(ok, corr, -np.inf)

        for score, u, v in _nms_peaks(corr, min_sep_px, k=3):
            row0, col0 = u - h_b + 1, v - w_b + 1
            tx = drone_info.origin_x + col0 * res - corner[0]
            ty = drone_info.origin_y + row0 * res - corner[1]
            candidates.append((score / n_src, (tx, ty, theta)))

    candidates.sort(key=lambda c: c[0], reverse=True)

    # Non-maximum suppression across rotations: keep a candidate only if it is
    # well separated from every accepted one in translation OR rotation.
    t_sep = params['coarse_translation_step_m']
    r_sep = rot_step
    kept = []
    for score, t in candidates:
        if all(math.hypot(t[0] - k[1][0], t[1] - k[1][1]) > t_sep or
               abs(angle_diff(t[2], k[1][2])) > r_sep for k in kept):
            kept.append((score, t))
        if len(kept) >= params['coarse_top_k']:
            break

    ambiguous = (len(kept) >= 2 and
                 kept[0][0] - kept[1][0] < params['symmetry_score_tolerance'])
    return kept, ambiguous
