"""
run_matrix_v2.py  –  Expanded overnight sweep (new-video / mirror-recalibration)
══════════════════════════════════════════════════════════════════════════════════
128 runs  ·  estimated ≈ 7.8 h on laptop with 4070ti Mobile GPU
  sift / no-pg runs  ≈ 3.5 min  ·  sp+lg (GPU) runs ≈ 2.5 min  ·  pg-on ≈ 5 min

v1 winner on the new video
──────────────────────────
  sift + ratio_test  ·  grid=False  ·  pg=False  ·  fid=False  ·  exc=blue_tape
  match_ratio=0.80, mad_factor=6.0, lookback=4  (all CODE DEFAULTS)

New in v2 vs v1
────────────────
  • processing_scale         (group D, K)  — "slightly smaller image" may shift optimum
  • blur_thresh + max_movement             (group J)
  • static_pixel_thresh + artifact_thresh (group N)
  • grid HSV recalibration                (group G2) — mirror shifts apparent hue
  • PG with wider weight ranges           (group H)
  • 3-way combo: match_ratio × mad_factor × lookback  (group L)
  • SP+LG across processing_scale         (group M)  — leverages 4070ti

Groups
──────
  A  (24)  architecture × grid × solver × exclusion     [same scope as v1, re-run on new video]
  B  (15)  match_ratio × mad_factor                     [0.55–0.80 × 2.0–6.0]
  C  ( 9)  lookback × keyframe_interval                 [up to lb=16, ki=20]
  D  ( 6)  processing_scale                             [0.25–1.0, NEW]
  E  ( 6)  feature_exclude_dilate_px                    [3–20 px]
  F  ( 9)  min_keypoint_bins + min_inliers              [wider / new ranges]
  G  ( 8)  grid re-exploration: grid_match_dist × HSV
  H  ( 9)  pose-graph re-exploration: marker_weight × huber_delta
  I  ( 6)  target_fps × min_movement
  J  ( 6)  blur_thresh + max_movement                   [NEW]
  K  ( 9)  processing_scale × match_ratio               [key interaction, NEW]
  L  ( 8)  3-way: match_ratio × mad_factor × lookback  [interaction, NEW]
  M  ( 6)  SP × processing_scale                        [GPU benefit, NEW]
  N  ( 7)  static_pixel_thresh + artifact_thresh        [NEW]
  ─────────────────────────────────────────────────────────────────────
  Total   128 runs                                        ≈ 7.8 h

Baseline (_BASELINE) used by groups B–N
  = v1 winning config; update below if your v1 sweep found a better run.
"""

from __future__ import annotations
from typing import Any, Dict, List


# ── Code defaults (what you get without any override) ─────────────────────────
_D: Dict[str, Any] = {
    # ExtractionConfig
    "target_fps":                5.0,
    "min_movement":              0.015,
    "max_movement":              0.55,
    "blur_thresh":               60.0,
    "artifact_thresh":           2.0,
    "static_pixel_thresh":       3.0,
    # ReconstructConfig — feature backend
    "feature_extractor":         "sift",
    "feature_matcher":           "ratio_test",
    "processing_scale":          0.5,
    # ReconstructConfig — alignment flags
    "use_grid_intersections":    True,
    "use_pose_graph":            True,
    "use_fiducials":             True,
    # ReconstructConfig — feature quality
    "feature_exclude_hsv":       "none",    # "none" | "blue_tape"
    "feature_exclude_dilate_px": 5,
    "match_ratio":               0.80,
    "mad_factor":                6.0,
    "min_inliers":               10,
    "min_keypoint_bins":         10,
    # ReconstructConfig — temporal
    "lookback":                  4,
    "keyframe_interval":         10,
    # ReconstructConfig — grid detection
    "grid_match_dist":           25.0,
    "grid_min_intersections":    4,
    "grid_hsv_h_lo":             90,
    "grid_hsv_h_hi":             130,
    "grid_hsv_s_lo":             50,
    "grid_hsv_v_lo":             50,
    # ReconstructConfig — pose graph
    "pg_marker_weight":          15.0,
    "pg_odom_weight":            1.0,
    "pg_loop_weight":            2.0,
    "pg_huber_delta":            4.0,
    "pg_iterations":             10,
}


# ── v1 winning config on the new (mirror-recalibrated) video ──────────────────
# Update this if your v1 sweep found something better before running v2.
_BASELINE: Dict[str, Any] = {
    **_D,
    "feature_extractor":         "sift",
    "feature_matcher":           "ratio_test",
    "use_grid_intersections":    False,
    "use_pose_graph":            False,
    "use_fiducials":             False,
    "feature_exclude_hsv":       "blue_tape",
    # Continuous params at code defaults — v1 single-param sweeps
    # did NOT improve on these; groups B/C/K/L probe combinations.
}


def _R(run_id: str, group: str, base: Dict[str, Any] = None, **ov) -> Dict[str, Any]:
    """Build run dict: base (default=_D) overridden by ov."""
    b = _D if base is None else base
    return {"run_id": run_id, "group": group, **b, **ov}


def _RB(run_id: str, group: str, **ov) -> Dict[str, Any]:
    """Build run dict from _BASELINE."""
    return {"run_id": run_id, "group": group, **_BASELINE, **ov}


RUNS: List[Dict[str, Any]] = []


# ══════════════════════════════════════════════════════════════════════════════
# GROUP A  –  Architecture × grid × solver × feature-exclusion
#             Uses _D (code defaults) as base for a fair cross-backend comparison.
#             sift+ratio: 2 grid × 3 solver × 2 exclude = 12 runs
#             sp+ratio:   2 grid × 3 solver               =  6 runs
#             sp+lg:      2 grid × 3 solver               =  6 runs
#             Total: 24 runs
# ══════════════════════════════════════════════════════════════════════════════

_a = 0
for _grid in [False, True]:
    for _pg, _fid in [(False, False), (True, False), (True, True)]:
        for _exc in ["none", "blue_tape"]:
            _a += 1
            RUNS.append(_R(f"A{_a:02d}", "A",
                feature_extractor="sift",
                feature_matcher="ratio_test",
                use_grid_intersections=_grid,
                use_pose_graph=_pg,
                use_fiducials=_fid,
                feature_exclude_hsv=_exc))

for _grid in [False, True]:
    for _pg, _fid in [(False, False), (True, False), (True, True)]:
        _a += 1
        RUNS.append(_R(f"A{_a:02d}", "A",
            feature_extractor="superpoint",
            feature_matcher="ratio_test",
            use_grid_intersections=_grid,
            use_pose_graph=_pg,
            use_fiducials=_fid,
            feature_exclude_hsv="none"))   # SP ignores this field

for _grid in [False, True]:
    for _pg, _fid in [(False, False), (True, False), (True, True)]:
        _a += 1
        RUNS.append(_R(f"A{_a:02d}", "A",
            feature_extractor="superpoint",
            feature_matcher="lightglue",
            use_grid_intersections=_grid,
            use_pose_graph=_pg,
            use_fiducials=_fid,
            feature_exclude_hsv="none"))

assert sum(r["group"] == "A" for r in RUNS) == 24, "A count"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP B  –  match_ratio × mad_factor  (wider than v1: 0.55–0.80 × 2–6)
#             v1 tested ratio in [0.65–0.80] and mad in [4.0–6.0];
#             v2 extends both bounds and keeps sift-only for clean comparison.
#             Base: _BASELINE  /  5 × 3 = 15 runs
# ══════════════════════════════════════════════════════════════════════════════

_b = 0
for _mr in [0.55, 0.65, 0.70, 0.75, 0.80]:
    for _mf in [2.0, 4.0, 6.0]:
        _b += 1
        RUNS.append(_RB(f"B{_b:02d}", "B", match_ratio=_mr, mad_factor=_mf))

assert sum(r["group"] == "B" for r in RUNS) == 15, "B count"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP C  –  lookback × keyframe_interval  (wider: lb up to 16, ki up to 20)
#             v1 tested lb in [4–8] and ki in [5–15].
#             Adding lb=16 and ki=[3, 20] to catch scenarios where the drone
#             re-enters an area far back in the queue.
#             Base: _BASELINE  /  3 × 3 = 9 runs
# ══════════════════════════════════════════════════════════════════════════════

_c = 0
for _lb in [4, 8, 16]:
    for _ki in [3, 10, 20]:
        _c += 1
        RUNS.append(_RB(f"C{_c:02d}", "C", lookback=_lb, keyframe_interval=_ki))

assert sum(r["group"] == "C" for r in RUNS) == 9, "C count"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP D  –  processing_scale  (NEW — not in v1)
#             The recalibrated camera produces a "slightly smaller image."
#             0.5 was hardcoded in v1; this tests whether a different scale
#             changes matching quality (larger scale = finer pixel grid for
#             SIFT; smaller scale = smaller buffers and faster, wider FOV gaps).
#             Base: _BASELINE  /  6 runs
# ══════════════════════════════════════════════════════════════════════════════

for _d, _ps in enumerate([0.25, 0.33, 0.50, 0.66, 0.75, 1.0], 1):
    RUNS.append(_RB(f"D{_d:02d}", "D", processing_scale=_ps))

assert sum(r["group"] == "D" for r in RUNS) == 6, "D count"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP E  –  feature_exclude_dilate_px  (wider range: 3–20 px)
#             The mirror angle change means the blue tape appears at a
#             different apparent width. v1 tested [3–12]; v2 adds 16 and 20.
#             Base: _BASELINE  /  6 runs
# ══════════════════════════════════════════════════════════════════════════════

for _e, _dp in enumerate([3, 5, 8, 12, 16, 20], 1):
    RUNS.append(_RB(f"E{_e:02d}", "E", feature_exclude_dilate_px=_dp))

assert sum(r["group"] == "E" for r in RUNS) == 6, "E count"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP F  –  min_keypoint_bins (4 runs) + min_inliers (5 runs) = 9 runs
#             Tested separately to identify which gate is binding.
#             min_keypoint_bins=0 disables the spatial spread check entirely.
#             min_inliers=6 is below the default 10 — tests if the gate is
#             over-rejecting frames that have few but correct matches.
#             Base: _BASELINE
# ══════════════════════════════════════════════════════════════════════════════

_f = 0
for _mkb in [0, 5, 10, 20]:
    _f += 1
    RUNS.append(_RB(f"F{_f:02d}", "F", min_keypoint_bins=_mkb))

for _mi in [6, 8, 12, 15, 20]:
    _f += 1
    RUNS.append(_RB(f"F{_f:02d}", "F", min_inliers=_mi))

assert sum(r["group"] == "F" for r in RUNS) == 9, "F count"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP G  –  Grid re-exploration  (grid was OFF in v1 winner)
#             Re-enables grid with tuned grid_match_dist and adjusted HSV
#             to compensate for the mirror's hue/saturation shift.
#             Base: _BASELINE + grid=True, match_ratio=0.70, mad_factor=4.0
#
#   G1 (4 runs): grid_match_dist  [10, 15, 25, 40]
#   G2 (4 runs): grid_hsv variants  (h_lo, h_hi, s_lo, v_lo)
# ══════════════════════════════════════════════════════════════════════════════

_G_FIXED = dict(
    use_grid_intersections=True,
    match_ratio=0.70,
    mad_factor=4.0,
)

_g = 0
for _gmd in [10, 15, 25, 40]:
    _g += 1
    RUNS.append(_RB(f"G{_g:02d}", "G", **_G_FIXED, grid_match_dist=_gmd))

for _hlo, _hhi, _slo, _vlo in [
    ( 90, 130, 50, 50),   # code default
    ( 85, 135, 40, 40),   # wider / more permissive (low-sat tape edges)
    ( 95, 125, 65, 65),   # narrower / stricter (avoid near-purple/cyan)
    (100, 140, 50, 50),   # blue-shifted (mirror can shift hue toward cyan)
]:
    _g += 1
    RUNS.append(_RB(f"G{_g:02d}", "G", **_G_FIXED,
        grid_hsv_h_lo=_hlo, grid_hsv_h_hi=_hhi,
        grid_hsv_s_lo=_slo, grid_hsv_v_lo=_vlo))

assert sum(r["group"] == "G" for r in RUNS) == 8, "G count"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP H  –  Pose-graph re-exploration  (pg was OFF in v1 winner)
#             Re-enables PG with much wider weight ranges.
#             High pg_marker_weight (50) tests whether overwhelming the odometry
#             edges with marker constraints overrides the bad loop closures.
#             High pg_huber_delta (12) softens the robust loss so weaker
#             constraints aren't discarded by the Huber kernel.
#             Base: _BASELINE + pg=True, fid=True, tighter matching
#             pg_marker_weight × pg_huber_delta: [5, 20, 50] × [2, 4, 12] = 9 runs
# ══════════════════════════════════════════════════════════════════════════════

_H_FIXED = dict(
    use_pose_graph=True,
    use_fiducials=True,
    match_ratio=0.70,
    mad_factor=4.0,
)

_h = 0
for _pw in [5.0, 20.0, 50.0]:
    for _hd in [2.0, 4.0, 12.0]:
        _h += 1
        RUNS.append(_RB(f"H{_h:02d}", "H", **_H_FIXED,
            pg_marker_weight=_pw, pg_huber_delta=_hd))

assert sum(r["group"] == "H" for r in RUNS) == 9, "H count"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP I  –  target_fps × min_movement
#             target_fps: [3.0, 5.0, 8.0]  ×  min_movement: [0.010, 0.020]
#             Lower fps = larger inter-frame motion = harder to match.
#             Higher min_movement = fewer frames = less cumulative drift.
#             Base: _BASELINE  /  6 runs
# ══════════════════════════════════════════════════════════════════════════════

_i = 0
for _fps in [3.0, 5.0, 8.0]:
    for _mm in [0.010, 0.020]:
        _i += 1
        RUNS.append(_RB(f"I{_i:02d}", "I", target_fps=_fps, min_movement=_mm))

assert sum(r["group"] == "I" for r in RUNS) == 6, "I count"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP J  –  blur_thresh  +  max_movement  (NEW — not in v1)
#             blur_thresh:  [30, 80, 150]  — mirror introduces blur from
#               reflective surface imperfections; 30=permissive, 150=strict.
#             max_movement: [0.35, 0.45, 0.55]  — tighter gate drops jerk
#               frames that contribute to staircase drift.
#             Tested separately; base: _BASELINE  /  3 + 3 = 6 runs
# ══════════════════════════════════════════════════════════════════════════════

_j = 0
for _bt in [30, 80, 150]:
    _j += 1
    RUNS.append(_RB(f"J{_j:02d}", "J", blur_thresh=float(_bt)))

for _mxm in [0.35, 0.45, 0.55]:
    _j += 1
    RUNS.append(_RB(f"J{_j:02d}", "J", max_movement=_mxm))

assert sum(r["group"] == "J" for r in RUNS) == 6, "J count"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP K  –  processing_scale × match_ratio  (key interaction, NEW)
#             A smaller image means SIFT descriptors cover a larger field
#             patch → more likely to match across grid cells; tighter ratio
#             may be needed to compensate.  This is the first test of that
#             interaction.
#             Base: _BASELINE  /  3 × 3 = 9 runs
# ══════════════════════════════════════════════════════════════════════════════

_k = 0
for _ps in [0.25, 0.50, 0.75]:
    for _mr in [0.60, 0.70, 0.80]:
        _k += 1
        RUNS.append(_RB(f"K{_k:02d}", "K", processing_scale=_ps, match_ratio=_mr))

assert sum(r["group"] == "K" for r in RUNS) == 9, "K count"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP L  –  3-way: match_ratio × mad_factor × lookback  (NEW)
#             v1 varied these ONE AT A TIME and gained nothing over defaults.
#             This group tests joint effects — stricter ratio AND tighter MAD
#             AND deeper lookback together may break the staircase where
#             any individual change does not.
#             2 × 2 × 2 = 8 runs
# ══════════════════════════════════════════════════════════════════════════════

_l = 0
for _mr in [0.65, 0.70]:
    for _mf in [3.0, 4.0]:
        for _lb in [8, 16]:
            _l += 1
            RUNS.append(_RB(f"L{_l:02d}", "L",
                match_ratio=_mr, mad_factor=_mf, lookback=_lb))

assert sum(r["group"] == "L" for r in RUNS) == 8, "L count"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP M  –  SuperPoint × processing_scale  (leverages 4070ti GPU)
#             SP+LG is now ~2-3 min per run vs 5+ on Jetson.
#             Same alignment flags as v1 winner (no grid, no pg).
#             sp+ratio: exc="none" (SP ignores the mask anyway)
#             sp+lg:    exc="none"
#             3 scales × 2 backends = 6 runs
# ══════════════════════════════════════════════════════════════════════════════

_m = 0
for _ext, _mat in [("superpoint", "ratio_test"), ("superpoint", "lightglue")]:
    for _ps in [0.25, 0.50, 0.75]:
        _m += 1
        RUNS.append(_RB(f"M{_m:02d}", "M",
            feature_extractor=_ext,
            feature_matcher=_mat,
            feature_exclude_hsv="none",
            processing_scale=_ps))

assert sum(r["group"] == "M" for r in RUNS) == 6, "M count"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP N  –  static_pixel_thresh (4 runs) + artifact_thresh (3 runs) = 7 runs
#             Both are ExtractionConfig quality gates not tested in v1.
#             static_pixel_thresh=0 disables the pixel-diff fallback entirely.
#             Higher artifact_thresh (4.0) is more permissive about DCT blocks —
#             useful if the new codec compresses more aggressively.
#             Base: _BASELINE
# ══════════════════════════════════════════════════════════════════════════════

_n = 0
for _spt in [0.0, 1.5, 3.0, 6.0]:
    _n += 1
    RUNS.append(_RB(f"N{_n:02d}", "N", static_pixel_thresh=_spt))

for _at in [1.0, 2.0, 4.0]:
    _n += 1
    RUNS.append(_RB(f"N{_n:02d}", "N", artifact_thresh=_at))

assert sum(r["group"] == "N" for r in RUNS) == 7, "N count"


# ── final count ───────────────────────────────────────────────────────────────
assert len(RUNS) == 128, f"Expected 128 runs, got {len(RUNS)}"


# ─────────────────────────────────────────────────────────────────────────────
# Conversion helper  →  (ExtractionConfig, ReconstructConfig)
# ─────────────────────────────────────────────────────────────────────────────

def to_configs(run: Dict[str, Any]):
    """
    Convert a run dict → (ExtractionConfig, ReconstructConfig).

    sp_ort_provider is hardcoded to "cuda" since this sweep runs on the
    laptop with 4070ti.  Change to "cpu" if testing on the Jetson Orin.

    Example
    ───────
        from drone_map_grid_gen import (
            ExtractionConfig, ReconstructConfig, ColorRangeMask, MapReconstructor
        )
        from run_matrix_v2 import RUNS, to_configs

        for run in RUNS:
            extract_cfg, recon_cfg = to_configs(run)
            rec = MapReconstructor(recon_cfg)
            rec.add_video("new_flight.mp4", extract_cfg=extract_cfg)
            result = rec.finalize() if recon_cfg.use_pose_graph else rec.get_map()
            save_result(run["run_id"], result)
    """
    from drone_map_grid_gen import ExtractionConfig, ReconstructConfig, ColorRangeMask

    exclude = (
        [ColorRangeMask.blue_tape()]
        if run["feature_exclude_hsv"] == "blue_tape"
        else []
    )

    extract_cfg = ExtractionConfig(
        target_fps          = run["target_fps"],
        min_movement        = run["min_movement"],
        max_movement        = run["max_movement"],
        blur_thresh         = run["blur_thresh"],
        artifact_thresh     = run["artifact_thresh"],
        static_pixel_thresh = run["static_pixel_thresh"],
    )

    recon_cfg = ReconstructConfig(
        # backend
        feature_extractor         = run["feature_extractor"],
        feature_matcher           = run["feature_matcher"],
        processing_scale          = run["processing_scale"],
        sp_ort_provider           = "cuda",   # 4070ti — change to "cpu" on Jetson
        # alignment flags
        use_grid_intersections    = run["use_grid_intersections"],
        use_pose_graph            = run["use_pose_graph"],
        use_fiducials             = run["use_fiducials"],
        # feature quality
        feature_exclude_hsv       = exclude,
        feature_exclude_dilate_px = run["feature_exclude_dilate_px"],
        match_ratio               = run["match_ratio"],
        mad_factor                = run["mad_factor"],
        min_inliers               = run["min_inliers"],
        min_keypoint_bins         = run["min_keypoint_bins"],
        # temporal
        lookback                  = run["lookback"],
        keyframe_interval         = run["keyframe_interval"],
        # grid
        grid_match_dist           = run["grid_match_dist"],
        grid_min_intersections    = run["grid_min_intersections"],
        grid_hsv_h_lo             = run["grid_hsv_h_lo"],
        grid_hsv_h_hi             = run["grid_hsv_h_hi"],
        grid_hsv_s_lo             = run["grid_hsv_s_lo"],
        grid_hsv_v_lo             = run["grid_hsv_v_lo"],
        # pose graph
        pg_marker_weight          = run["pg_marker_weight"],
        pg_odom_weight            = run["pg_odom_weight"],
        pg_loop_weight            = run["pg_loop_weight"],
        pg_huber_delta            = run["pg_huber_delta"],
        pg_iterations             = run["pg_iterations"],
    )

    return extract_cfg, recon_cfg


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from collections import Counter

    gc = Counter(r["group"] for r in RUNS)

    descs = {
        "A": "architecture × grid × solver × exclusion",
        "B": "match_ratio × mad_factor          [0.55–0.80 × 2–6]",
        "C": "lookback × keyframe_interval      [lb up to 16, ki up to 20]",
        "D": "processing_scale                  [0.25–1.0, NEW]",
        "E": "feature_exclude_dilate_px         [3–20 px]",
        "F": "min_keypoint_bins + min_inliers",
        "G": "grid re-exploration (match_dist + HSV recalibration)",
        "H": "pose-graph re-exploration (wider weights)",
        "I": "target_fps × min_movement",
        "J": "blur_thresh + max_movement        [NEW]",
        "K": "processing_scale × match_ratio    [interaction, NEW]",
        "L": "3-way: match_ratio × mad × lookback [NEW]",
        "M": "SP × processing_scale             [GPU, NEW]",
        "N": "static_pixel_thresh + artifact_thresh [NEW]",
    }

    print(f"\nTotal runs: {len(RUNS)}\n")
    print(f"{'Group':<6} {'Runs':>5}   Description")
    print("─" * 72)
    for g in sorted(gc):
        print(f"  {g:<4} {gc[g]:>5}   {descs.get(g, '')}")
    print("─" * 72)

    # show the _BASELINE for easy inspection
    print("\n_BASELINE (v1 winner):")
    keys = ["feature_extractor", "feature_matcher", "use_grid_intersections",
            "use_pose_graph", "feature_exclude_hsv", "match_ratio", "mad_factor",
            "lookback", "processing_scale"]
    for k in keys:
        print(f"  {k}: {_BASELINE[k]}")

    print("\nSample — first run of each group:")
    seen: set = set()
    for r in RUNS:
        if r["group"] not in seen:
            seen.add(r["group"])
            print(f"  {r['run_id']}  ext={r['feature_extractor'][:4]}  "
                  f"mat={r['feature_matcher'][:5]}  grid={r['use_grid_intersections']}  "
                  f"pg={r['use_pose_graph']}  exc={r['feature_exclude_hsv'][:4]}  "
                  f"mr={r['match_ratio']}  mf={r['mad_factor']}  "
                  f"lb={r['lookback']}  ps={r['processing_scale']}")