"""
stitch_graph.py
─────────────────────────────────────────────────────────────────────────────
The glue between MapReconstructor's online pass and the global solve.

Design (the keyframe-only approach you green-lit)
─────────────────────────────────────────────────
  * Graph NODES are keyframes only (a few hundred at most → the dense
    numpy solver in pose_graph.py is plenty fast).
  * Graph EDGES:
      - odometry  : between consecutive keyframes, derived from the online
                    pose estimate (this is the drifted chain),
      - loop      : direct keyframe↔keyframe pairwise matches when the online
                    pass happened to match a keyframe against another keyframe
                    (e.g. the backward pass flying back over the forward pass),
      - marker    : every keyframe's ArUco sightings (the zero-ambiguity loop
                    closures that do most of the work).
  * Every PLACED frame (keyframe or not) is recorded in placement order with
    its online pose.  Non-keyframes are not optimised individually; instead,
    at finalize() each keyframe gets a correction similarity
        C_k = H_opt_k · H_online_k⁻¹      (old-map → new-map)
    and a non-keyframe between keyframes a and b inherits the correction
    interpolated between C_a and C_b by its fractional position.  Adjacent
    keyframes are only ~keyframe_interval frames apart, so the residual local
    drift across a bracket is tiny and the interpolation is smooth (no step at
    keyframe boundaries).

Re-render (how the corrected map is produced)
─────────────────────────────────────────────
Because the online pass is streaming (no frame images are retained), the
corrected map is produced by a second pass that re-reads the source frames in
the same deterministic order and warps each at its corrected pose.  See
corrected_H_for_record() / iter_corrections().  For a live feed that finishes,
record it to a temp file during capture and re-stream that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

try:
    from .pose_graph import PoseGraph, H_from_sim, sim_from_H
except ImportError:
    from pose_graph import PoseGraph, H_from_sim, sim_from_H


@dataclass
class StitchGraphConfig:
    soft_prior_weight: float = 1e-4
    marker_weight: float = 10.0     # subpixel ArUco corners → trust them most
    odom_weight: float = 1.0        # online consecutive-keyframe chain
    loop_weight: float = 2.0        # direct keyframe↔keyframe overlap matches
    iterations: int = 10
    huber_delta: float = 4.0        # px; robust to bad loop closures


@dataclass
class _Record:
    """One placed frame, in placement order."""
    online_H: np.ndarray            # (3,3) the online absolute pose
    is_keyframe: bool
    kf_id: int = -1                 # this record's keyframe id (if a keyframe)


class StitchGraph:
    """Accumulates constraints online; solves + provides corrections on finalize.

    Minimal call surface from MapReconstructor:
        sg = StitchGraph()
        ...first frame...
        sg.on_placed(H0, is_keyframe=True, frame_w=w, frame_h=h, fixed=True)
        sg.on_markers(markers)                      # for the just-placed frame
        ...each later placed frame...
        sg.on_placed(best_H, is_keyframe=<bool>, frame_w=w, frame_h=h,
                     matched_kf_id=<id or None>, matched_H_pair=<H or None>,
                     matched_n_inliers=<int>)
        sg.on_markers(markers)
        ...on stream stop...
        report = sg.finalize()
        for rec_idx, H_corr in sg.iter_corrections():  # aligned to placement order
            ...re-warp that frame at H_corr...
    """

    def __init__(self, cfg: Optional[StitchGraphConfig] = None):
        self.cfg = cfg or StitchGraphConfig()
        self._g = PoseGraph(soft_prior_weight=self.cfg.soft_prior_weight)
        self._records: List[_Record] = []
        # keyframe id -> (record_index, online_H, w, h)
        self._kf: Dict[int, dict] = {}
        self._kf_order: List[int] = []          # keyframe ids in placement order
        self._prev_kf_id: Optional[int] = None
        self._next_kf_id = 0                     # monotonic keyframe id allocator
        self._finalized = False
        self._corrections: Dict[int, np.ndarray] = {}   # kf_id -> C_k sim params

    # ── online accumulation ──────────────────────────────────────────────────

    def on_placed(
        self,
        H: np.ndarray,
        is_keyframe: bool,
        frame_w: int,
        frame_h: int,
        fixed: bool = False,
        matched_kf_id: Optional[int] = None,
        matched_H_pair: Optional[np.ndarray] = None,
        matched_n_inliers: int = 0,
    ) -> Optional[int]:
        """Register a just-placed frame. Returns its keyframe id if it became
        one, else None. Call once per successfully placed frame, in order."""
        H = np.asarray(H, dtype=np.float64)
        rec = _Record(online_H=H.copy(), is_keyframe=is_keyframe)
        rec_idx = len(self._records)

        kf_id = None
        if is_keyframe:
            kf_id = self._next_kf_id
            self._next_kf_id += 1
            rec.kf_id = kf_id
            self._kf[kf_id] = {"rec": rec_idx, "H": H.copy(),
                               "w": int(frame_w), "h": int(frame_h)}
            self._g.add_frame(kf_id, H, frame_w, frame_h, fixed=fixed)

            # Odometry edge from the previous keyframe (the online chain).
            if self._prev_kf_id is not None:
                Hp = self._kf[self._prev_kf_id]["H"]
                H_rel = np.linalg.inv(Hp) @ H          # cur-kf px -> prev-kf px
                self._g.add_relative(self._prev_kf_id, kf_id, H_rel,
                                     weight=self.cfg.odom_weight)

            # Loop-closure edge if this keyframe was matched against another
            # keyframe (not just a recent neighbour) during the online pass.
            if (matched_kf_id is not None and matched_kf_id in self._kf
                    and matched_kf_id != self._prev_kf_id
                    and matched_H_pair is not None):
                w = self.cfg.loop_weight * max(1, matched_n_inliers) / 50.0
                self._g.add_relative(matched_kf_id, kf_id,
                                     np.asarray(matched_H_pair, np.float64),
                                     weight=max(self.cfg.loop_weight, w))

            self._kf_order.append(kf_id)
            self._prev_kf_id = kf_id

        self._records.append(rec)
        self._pending_kf_for_markers = kf_id    # markers attach to this kf
        return kf_id

    def on_markers(self, markers) -> None:
        """Attach the just-placed frame's marker sightings. Only sightings on
        keyframes enter the graph (non-keyframes inherit corrections), which is
        sufficient because each marker is seen by many frames."""
        kf_id = getattr(self, "_pending_kf_for_markers", None)
        if kf_id is None or not markers:
            return
        for m in markers:
            self._g.add_marker_obs(kf_id, int(m.marker_id), m.corners_px,
                                   weight=self.cfg.marker_weight)

    # ── finalize ──────────────────────────────────────────────────────────────

    def finalize(self, verbose: bool = False) -> dict:
        """Run the global solve and precompute per-keyframe corrections."""
        report = self._g.optimize(iterations=self.cfg.iterations,
                                   huber_delta=self.cfg.huber_delta,
                                   verbose=verbose)
        # C_k = H_opt_k · H_online_k^{-1}  (maps OLD map coords -> NEW map coords)
        self._corrections.clear()
        for kf_id, info in self._kf.items():
            H_opt = self._g.get_H(kf_id)
            C = H_opt @ np.linalg.inv(info["H"])
            self._corrections[kf_id] = sim_from_H(C)
        self._finalized = True
        return report

    # ── corrected poses for re-render ──────────────────────────────────────────

    def corrected_H_for_record(self, rec_idx: int) -> np.ndarray:
        """Corrected absolute pose for the rec_idx-th placed frame.

        Keyframes use their own correction; non-keyframes interpolate the
        correction between their bracketing keyframes.
        """
        if not self._finalized:
            raise RuntimeError("call finalize() first")
        rec = self._records[rec_idx]
        C = self._interp_correction(rec_idx)
        return H_from_sim(C) @ rec.online_H

    def iter_corrections(self):
        """Yield (rec_idx, corrected_H) for every placed frame, in order.
        Aligns with the placement order so a re-render second pass can zip it
        against the re-streamed frames."""
        for i in range(len(self._records)):
            yield i, self.corrected_H_for_record(i)

    # ── internals ──────────────────────────────────────────────────────────────

    def _interp_correction(self, rec_idx: int) -> np.ndarray:
        """Correction similarity (a,b,tx,ty) for the rec_idx-th frame."""
        if not self._kf_order:
            return sim_from_H(np.eye(3))          # nothing to correct against

        # keyframe record indices in order
        kf_recs = [(self._kf[k]["rec"], k) for k in self._kf_order]

        # before the first / after the last keyframe → clamp to nearest
        if rec_idx <= kf_recs[0][0]:
            return self._corrections[kf_recs[0][1]]
        if rec_idx >= kf_recs[-1][0]:
            return self._corrections[kf_recs[-1][1]]

        # find bracketing keyframes a (<= rec_idx) and b (> rec_idx)
        a_rec, a_id = kf_recs[0]
        b_rec, b_id = kf_recs[-1]
        for (r, k) in kf_recs:
            if r <= rec_idx:
                a_rec, a_id = r, k
            if r > rec_idx:
                b_rec, b_id = r, k
                break

        if b_rec == a_rec:
            return self._corrections[a_id]
        t = (rec_idx - a_rec) / (b_rec - a_rec)
        Ca = self._corrections[a_id]
        Cb = self._corrections[b_id]
        # Linear interp of similarity params. Corrections between adjacent
        # keyframes are small, so linear blend of (a,b,tx,ty) is smooth and
        # stays close to a valid similarity.
        return (1.0 - t) * Ca + t * Cb
