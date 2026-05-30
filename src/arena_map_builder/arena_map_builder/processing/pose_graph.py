"""
pose_graph.py
─────────────────────────────────────────────────────────────────────────────
Lightweight, dependency-free 2D pose-graph optimiser for drone-map stitching.

Why this exists
───────────────
The incremental stitcher places each frame by composing a relative similarity
onto a reference frame's absolute pose: H_i = H_ref @ H_pair.  Pairwise
estimates are individually good but their errors compound along the chain, and
nothing ever reconciles the chain globally.  The visible symptoms are:

  * patches that align beautifully to their neighbour but disagree where two
    chains meet (drift, no loop closure), and
  * a slowly bowing / non-rigid grid and a global scale "spiral".

This module fixes both by treating the whole flight as ONE optimisation:
frame poses and ArUco-marker landmarks are solved jointly so that every
overlap constraint and every repeat sighting of a marker is satisfied at once.
Accumulated error is redistributed around the loops instead of dumped at a seam.

The key simplification
──────────────────────
Parametrise every pose as a similarity in (a, b, tx, ty) form, i.e. the 2x2
block is [[a, -b], [b, a]] (a = s·cosθ, b = s·sinθ).  Then mapping a point
p = (x, y) is

    T(p) = ( a·x - b·y + tx ,  b·x + a·y + ty )
         = B(p) @ [a, b, tx, ty]^T          with   B(p) = [[x, -y, 1, 0],
                                                           [y,  x, 0, 1]]

which is **linear in the parameters**.  Every constraint we use is "two poses
must map known points to the same place", so every residual is linear and the
global solve is a single linear least-squares system (re-solved a few times
with Huber weights to stay robust to bad loop closures).

No SLAM library, no manifold bookkeeping — just numpy.  Node counts here are
small (keyframes + a handful of markers → a few hundred unknowns), so dense
normal equations are more than fast enough, even on a Jetson.

Coordinate convention
──────────────────────
Each frame node's transform maps that frame's pixel coordinates → MAP (canvas)
coordinates, matching the 3x3 H stored in MapReconstructor.  Marker landmark
poses map a marker's canonical corner template → MAP coordinates.

Public surface
──────────────
    g = PoseGraph()
    fid = g.add_frame(node_id, H_init, frame_w, frame_h, fixed=False)
    g.add_relative(ref_id, cur_id, H_pair, weight)      # H_pair: cur px -> ref px
    g.add_marker_obs(frame_id, marker_id, corners_px, weight)
    report = g.optimize(iterations=8, huber_delta=4.0)
    H_opt = g.get_H(node_id)                             # optimised 3x3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ───────────────────────────────────────────────────────────────────────────
# Similarity (a, b, tx, ty)  ⇄  3x3 homography helpers
# ───────────────────────────────────────────────────────────────────────────

def sim_from_H(H: np.ndarray) -> np.ndarray:
    """Extract the (a, b, tx, ty) similarity parameters from a 3x3 matrix.

    Assumes H is (close to) a similarity — the case produced by
    estimateAffinePartial2D promoted to 3x3.  The (a, b) are read directly
    from the top-left 2x2; any small shear is ignored by construction.
    """
    a  = 0.5 * (H[0, 0] + H[1, 1])
    b  = 0.5 * (H[1, 0] - H[0, 1])
    tx = H[0, 2]
    ty = H[1, 2]
    return np.array([a, b, tx, ty], dtype=np.float64)


def H_from_sim(p: np.ndarray) -> np.ndarray:
    """Build a 3x3 homography from (a, b, tx, ty) similarity parameters."""
    a, b, tx, ty = p
    return np.array([[a, -b, tx],
                     [b,  a, ty],
                     [0,  0, 1.0]], dtype=np.float64)


def _B(pt: np.ndarray) -> np.ndarray:
    """The 2x4 design block such that T(pt) = B(pt) @ (a, b, tx, ty).

    pt may be (2,) for a single point or (N, 2) for a stack, in which case the
    result is (2N, 4).
    """
    pt = np.asarray(pt, dtype=np.float64)
    if pt.ndim == 1:
        x, y = pt
        return np.array([[x, -y, 1.0, 0.0],
                         [y,  x, 0.0, 1.0]], dtype=np.float64)
    x = pt[:, 0]
    y = pt[:, 1]
    n = pt.shape[0]
    out = np.zeros((2 * n, 4), dtype=np.float64)
    out[0::2, 0] = x
    out[0::2, 1] = -y
    out[0::2, 2] = 1.0
    out[1::2, 0] = y
    out[1::2, 1] = x
    out[1::2, 3] = 1.0
    return out


def _apply_sim(p: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply similarity params p to an (N, 2) stack of points → (N, 2)."""
    a, b, tx, ty = p
    x = pts[:, 0]
    y = pts[:, 1]
    return np.stack([a * x - b * y + tx, b * x + a * y + ty], axis=1)


# ───────────────────────────────────────────────────────────────────────────
# Graph data structures
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class _FrameNode:
    node_id: int
    p: np.ndarray            # (4,) similarity params, mutated in-place by solve
    frame_w: int
    frame_h: int
    fixed: bool = False       # gauge anchor: held at its initial value
    col: int = -1             # column offset into the global state vector

    def corners(self) -> np.ndarray:
        """Four frame corners used as evaluation points for relative edges."""
        w, h = float(self.frame_w), float(self.frame_h)
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)


@dataclass
class _MarkerNode:
    marker_id: int
    p: np.ndarray            # (4,) similarity params for the marker placement
    col: int = -1


@dataclass
class _Edge:
    """A generic 'two poses agree at known points' constraint.

    Residual (stacked over the K evaluation points) is

        r = B(pts_i) @ p[node_i]  -  B(pts_j) @ p[node_j]

    where pts_i are points in node_i's local frame and pts_j the
    corresponding points in node_j's local frame.  weight scales the whole
    block; a robust (Huber) reweighting is applied on top during optimise().
    """
    node_i: int              # column-resolved later via a (kind, key) handle
    node_j: int
    pts_i: np.ndarray        # (K, 2)
    pts_j: np.ndarray        # (K, 2)
    weight: float
    tag: str = "rel"         # "rel" | "marker" — for telemetry only


@dataclass
class _Prior:
    """Soft anchor pulling a node's params toward a target (gauge / conditioning)."""
    col: int
    target: np.ndarray       # (4,)
    weight: float


# ───────────────────────────────────────────────────────────────────────────
# Pose graph
# ───────────────────────────────────────────────────────────────────────────

class PoseGraph:
    """Linear-similarity pose graph with frame poses + marker landmarks.

    Build it incrementally during the online pass, then call optimize() once
    the stream ends.  All transforms map LOCAL coords → MAP coords, identical
    in convention to MapReconstructor's stored H.
    """

    # Canonical marker corner template (unit square, marker-local frame).
    # cv2.aruco returns corners in a consistent order, so corner j is the
    # same physical corner across every frame that sees the marker.  The
    # absolute size is irrelevant because the marker's placement node carries
    # its own scale; only the *shape* (a square) and the *ordering* matter.
    _MARKER_TEMPLATE = np.array(
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=np.float64
    )

    def __init__(self, soft_prior_weight: float = 1e-3):
        self._frames: Dict[int, _FrameNode] = {}
        self._markers: Dict[int, _MarkerNode] = {}
        self._edges: List[_Edge] = []
        # node handle = ("f", frame_id) or ("m", marker_id) resolved to a column
        self._edge_handles: List[Tuple[Tuple[str, int], Tuple[str, int]]] = []
        self._priors: List[Tuple[Tuple[str, int], np.ndarray, float]] = []
        # Weak prior pulling every node toward its initial estimate.  Keeps the
        # system full-rank even if a node is only weakly connected, and gently
        # regularises toward the (already decent) online solution.
        self._soft_prior_weight = float(soft_prior_weight)

    # ── construction ─────────────────────────────────────────────────────────

    def add_frame(self, node_id: int, H_init: np.ndarray,
                  frame_w: int, frame_h: int, fixed: bool = False) -> None:
        """Register a frame (keyframe) node with its online pose as the seed."""
        self._frames[node_id] = _FrameNode(
            node_id=node_id,
            p=sim_from_H(np.asarray(H_init, dtype=np.float64)),
            frame_w=int(frame_w),
            frame_h=int(frame_h),
            fixed=bool(fixed),
        )

    def add_relative(self, ref_id: int, cur_id: int,
                     H_pair: np.ndarray, weight: float = 1.0) -> None:
        """Add a relative-overlap constraint between two frame nodes.

        H_pair maps CUR pixel coords → REF pixel coords (the convention of
        MapReconstructor's _pairwise_H).  The constraint says: a point in cur
        and its H_pair-image in ref must land at the same MAP location.
        """
        if ref_id not in self._frames or cur_id not in self._frames:
            raise KeyError("both frame nodes must be added before the edge")
        H_pair = np.asarray(H_pair, dtype=np.float64)
        # Evaluation points: the cur frame's corners and their image in ref.
        pts_cur = self._frames[cur_id].corners()
        ones = np.ones((4, 1))
        ph = (H_pair @ np.hstack([pts_cur, ones]).T).T
        pts_ref = ph[:, :2] / ph[:, 2:3]
        self._edges.append(_Edge(
            node_i=0, node_j=0, pts_i=pts_cur, pts_j=pts_ref,
            weight=float(weight), tag="rel",
        ))
        self._edge_handles.append((("f", cur_id), ("f", ref_id)))

    def add_marker_obs(self, frame_id: int, marker_id: int,
                       corners_px: np.ndarray, weight: float = 4.0) -> None:
        """Add an observation of a marker's 4 corners in a frame.

        corners_px : (4, 2) float, in the SAME order cv2.aruco returns them.
        A marker landmark node is created lazily on first sighting and seeded
        from this observation so the initial guess is sensible.
        """
        if frame_id not in self._frames:
            raise KeyError("frame node must be added before its marker obs")
        corners_px = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)

        if marker_id not in self._markers:
            # Seed the marker placement from this first observation:
            # map the observed corners into MAP space using the frame's
            # current pose, then fit a similarity template→map.
            f = self._frames[frame_id]
            map_corners = _apply_sim(f.p, corners_px)
            p_seed = _fit_similarity(self._MARKER_TEMPLATE, map_corners)
            self._markers[marker_id] = _MarkerNode(marker_id=marker_id, p=p_seed)

        self._edges.append(_Edge(
            node_i=0, node_j=0,
            pts_i=corners_px,                 # in frame-i local coords
            pts_j=self._MARKER_TEMPLATE,      # in marker-local coords
            weight=float(weight), tag="marker",
        ))
        self._edge_handles.append((("f", frame_id), ("m", marker_id)))

    def add_prior(self, node_id: int, target_H: np.ndarray,
                  weight: float, is_marker: bool = False) -> None:
        """Add an explicit strong prior on a node (e.g. to fix the gauge)."""
        key = ("m" if is_marker else "f", node_id)
        self._priors.append((key, sim_from_H(np.asarray(target_H, np.float64)),
                             float(weight)))

    # ── optimisation ─────────────────────────────────────────────────────────

    def optimize(self, iterations: int = 8, huber_delta: float = 4.0,
                 verbose: bool = False) -> dict:
        """Solve the joint least-squares system with Huber IRLS.

        Returns a small report dict with before/after RMS residuals.
        """
        cols = self._assign_columns()
        n = cols
        if n == 0:
            return {"nodes": 0, "edges": 0, "rms_before": 0.0, "rms_after": 0.0}

        # Stable indexing for residual evaluation.
        handles = self._edge_handles
        edges = self._edges

        def node_col(handle: Tuple[str, int]) -> int:
            kind, key = handle
            return (self._frames[key].col if kind == "f"
                    else self._markers[key].col)

        def node_params(handle: Tuple[str, int]) -> np.ndarray:
            kind, key = handle
            return (self._frames[key].p if kind == "f"
                    else self._markers[key].p)

        rms_before = self._rms(handles, edges, node_params)

        for it in range(max(1, iterations)):
            # Gauss-Newton normal equations  H dx = g  (linear model → the
            # step is exact for fixed weights; Huber weights change between
            # iterations, hence the small loop).
            H, g = self._build_system(n, handles, edges, node_col, node_params,
                                      huber_delta)

            # ── priors (gauge anchor + soft conditioning) ──────────────────
            self._add_priors(H, g, node_col, node_params)

            # ── solve and update ───────────────────────────────────────────
            try:
                dx = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(H, g, rcond=None)[0]

            self._apply_update(dx)

            if verbose:
                rms = self._rms(handles, edges, node_params)
                print(f"  [pose_graph] iter {it}: rms={rms:.3f}px")

        rms_after = self._rms(handles, edges, node_params)
        return {
            "nodes": len(self._frames) + len(self._markers),
            "frames": len(self._frames),
            "markers": len(self._markers),
            "edges": len(self._edges),
            "rms_before": rms_before,
            "rms_after": rms_after,
        }

    # ── retrieval ─────────────────────────────────────────────────────────────

    def get_H(self, node_id: int) -> np.ndarray:
        """Optimised 3x3 homography for a frame node."""
        return H_from_sim(self._frames[node_id].p)

    def get_marker_H(self, marker_id: int) -> np.ndarray:
        return H_from_sim(self._markers[marker_id].p)

    def frame_ids(self) -> List[int]:
        return list(self._frames.keys())

    # ── internals ─────────────────────────────────────────────────────────────

    def _assign_columns(self) -> int:
        col = 0
        for f in self._frames.values():
            f.col = col
            col += 4
        for m in self._markers.values():
            m.col = col
            col += 4
        return col

    def _build_system(self, n, handles, edges, node_col, node_params, delta):
        H = np.zeros((n, n), dtype=np.float64)
        g = np.zeros(n, dtype=np.float64)
        for e, (hi, hj) in zip(edges, handles):
            ci, cj = node_col(hi), node_col(hj)
            Bi = _B(e.pts_i)
            Bj = _B(e.pts_j)
            r = Bi @ node_params(hi) - Bj @ node_params(hj)     # (2K,)
            w = e.weight * _huber_w(r.reshape(-1, 2), delta)    # (2K,)
            J = np.hstack([Bi, -Bj])                            # (2K, 8)
            Wd = w[:, None]
            JT_W = (J * Wd).T                                   # (8, 2K)
            Hblk = JT_W @ J                                     # (8, 8)
            gblk = JT_W @ r                                     # Gauss-Newton: H dx = -g, dx reduces r
            idx = np.r_[ci:ci+4, cj:cj+4]
            H[np.ix_(idx, idx)] += Hblk
            g[idx] += gblk
        return H, g

    def _add_priors(self, H, g, node_col, node_params):
        # Explicit strong priors (gauge anchors).
        for key, target, w in self._priors:
            kind, k = key
            c = (self._frames[k].col if kind == "f" else self._markers[k].col)
            p = node_params(key)
            r = p - target
            H[c:c+4, c:c+4] += w * np.eye(4)
            g[c:c+4] += w * r
        # Weak soft prior toward each node's current estimate (conditioning).
        wsp = self._soft_prior_weight
        if wsp > 0:
            for f in self._frames.values():
                if f.fixed:
                    # Hard-ish anchor for gauge fixing.
                    H[f.col:f.col+4, f.col:f.col+4] += 1e6 * np.eye(4)
                    # target == current value → g unchanged (residual 0)
                else:
                    H[f.col:f.col+4, f.col:f.col+4] += wsp * np.eye(4)
            for m in self._markers.values():
                H[m.col:m.col+4, m.col:m.col+4] += wsp * np.eye(4)

    def _apply_update(self, dx: np.ndarray):
        # Gauss-Newton step minimises ||J x_total - 0||; we solved H dx = g with
        # g = J^T W r, so the descent step is p <- p - dx.
        for f in self._frames.values():
            if f.fixed:
                continue
            f.p = f.p - dx[f.col:f.col+4]
        for m in self._markers.values():
            m.p = m.p - dx[m.col:m.col+4]

    def _rms(self, handles, edges, node_params) -> float:
        if not edges:
            return 0.0
        sse = 0.0
        cnt = 0
        for e, (hi, hj) in zip(edges, handles):
            Bi = _B(e.pts_i)
            Bj = _B(e.pts_j)
            r = Bi @ node_params(hi) - Bj @ node_params(hj)
            sse += float(r @ r)
            cnt += r.size // 2
        return float(np.sqrt(sse / max(cnt, 1)))


# ───────────────────────────────────────────────────────────────────────────
# helpers
# ───────────────────────────────────────────────────────────────────────────

def _huber_w(res_xy: np.ndarray, delta: float) -> np.ndarray:
    """Per-residual Huber weights, expanded to per-coordinate (2 per point).

    res_xy : (K, 2) residual vectors.  Returns (2K,) weights so a point whose
    residual norm exceeds delta is down-weighted ∝ delta / |r|.
    """
    norm = np.linalg.norm(res_xy, axis=1)                 # (K,)
    w = np.ones_like(norm)
    big = norm > delta
    w[big] = delta / norm[big]
    return np.repeat(w, 2)


def _fit_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Closed-form least-squares similarity (a, b, tx, ty) mapping src→dst.

    src, dst : (N, 2).  Used to seed a marker landmark from its first sighting.
    """
    A = _B(src)                       # (2N, 4)
    bvec = dst.reshape(-1)            # (2N,)
    p, *_ = np.linalg.lstsq(A, bvec, rcond=None)
    return p
