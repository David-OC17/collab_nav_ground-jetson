"""
run_matrix_v3.py  –  Exhaustive overnight sweep (focused on what v2 revealed)
══════════════════════════════════════════════════════════════════════════════

v2 findings
───────────
  • The ONLY configs that found more than 2 obstacles were in Group L (3-way
    factorial: match_ratio × mad_factor × lookback). Single-param sweeps at
    lookback=4 all produced identical degenerate results.

  • L07 (mr=0.70, mf=4.0, lb=8)  → 0 obstacles.
    L08 (mr=0.70, mf=4.0, lb=16) → 8 obstacles.
    One parameter flip, binary outcome: lookback is a stability threshold, not
    a quality knob. Raise it high enough and matching becomes possible at all.

  • Dozens of runs share consistency=0.9963812828063965 (exact float). That
    means the grid-mode pass silently fell back to bbox → perfect agreement
    by construction → fake ~1.0 score. Not real detections.

  • keyframe_interval was always 10 in the top configs because Group L never
    varied it. Completely unexplored as a dimension.

  • LightGlue dropped by user. SP+ratio_test included; SP runs on CUDA GPU.

v3 strategy
───────────
  A (150)  Dense 3-way: match_ratio × mad_factor × lookback
           Densify the sweet spot found by v2 Group L. Extend lookback to 24.
  B ( 60)  keyframe_interval × lookback for two anchored (mr, mf) combos.
           Completely new dimension — not explored in v1 or v2.
  C ( 45)  SP+ratio_test in the productive region.
  D ( 22)  Extraction params (target_fps, min_movement, blur_thresh, …)
           fixed at best estimated stitching config.
  E ( 18)  processing_scale × match_ratio interaction.
  F ( 20)  feature_exclude_dilate_px × lookback.
  G ( 20)  min_keypoint_bins + min_inliers.
  ───────────────────────────────────────────────────────────────────────────
  Total   335 runs
  Estimated wall time:
    SIFT runs  (290) × ~65 s  ≈ 314 min
    SP runs    ( 45) × ~110 s ≈  83 min
    Grand total                ≈ 397 min  ≈ 6.6 h   (safe for 10-h window)

Baseline (_BASELINE) = v2 rank-1 config (L03)
  sift + ratio_test, no grid, no pg, blue_tape exclusion
  match_ratio=0.65, mad_factor=4.0, lookback=8, keyframe_interval=10
  processing_scale=0.5, feature_exclude_dilate_px=5
  + all other params at code defaults
"""

from __future__ import annotations
from typing import Any, Dict, List


# ── Code defaults (what ReconstructConfig / ExtractionConfig give when
#    you don't override anything) ────────────────────────────────────────────
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
    # ReconstructConfig — alignment flags (all OFF — v2 confirmed these hurt)
    "use_grid_intersections":    False,
    "use_pose_graph":            False,
    "use_fiducials":             False,
    # ReconstructConfig — feature quality
    "feature_exclude_hsv":       "blue_tape",  # ON by default now; v2 confirmed essential
    "feature_exclude_dilate_px": 5,
    "match_ratio":               0.80,   # code default (server overrides to 0.65)
    "mad_factor":                6.0,    # code default (server overrides to 4.0)
    "min_inliers":               10,
    "min_keypoint_bins":         10,
    # ReconstructConfig — temporal
    "lookback":                  4,      # code default (server overrides to 8)
    "keyframe_interval":         10,
    # ReconstructConfig — grid (not used, but must round-trip through to_configs)
    "grid_match_dist":           25.0,
    "grid_min_intersections":    4,
    "grid_hsv_h_lo":             90,
    "grid_hsv_h_hi":             130,
    "grid_hsv_s_lo":             50,
    "grid_hsv_v_lo":             50,
    # ReconstructConfig — pose graph (not used)
    "pg_marker_weight":          15.0,
    "pg_odom_weight":            1.0,
    "pg_loop_weight":            2.0,
    "pg_huber_delta":            4.0,
    "pg_iterations":             10,
}

# ── v2 rank-1 baseline (L03) ─────────────────────────────────────────────────
_BASELINE: Dict[str, Any] = {
    **_D,
    # Override the code defaults that v2 showed matter:
    "match_ratio":               0.65,
    "mad_factor":                4.0,
    "lookback":                  8,
    "keyframe_interval":         10,
    "processing_scale":          0.5,
    "feature_exclude_dilate_px": 5,
    "feature_exclude_hsv":       "blue_tape",
    "feature_extractor":         "sift",
    "feature_matcher":           "ratio_test",
}


def _R(run_id: str, group: str, base: Dict[str, Any] = None, **ov) -> Dict[str, Any]:
    b = _D if base is None else base
    return {"run_id": run_id, "group": group, **b, **ov}


def _RB(run_id: str, group: str, **ov) -> Dict[str, Any]:
    return {"run_id": run_id, "group": group, **_BASELINE, **ov}


RUNS: List[Dict[str, Any]] = []


# ══════════════════════════════════════════════════════════════════════════════
# GROUP A  –  Dense 3-way: match_ratio × mad_factor × lookback
#
#  match_ratio: [0.58, 0.62, 0.65, 0.67, 0.70, 0.73]  (6 values)
#  mad_factor:  [2.5, 3.0, 3.5, 4.0, 5.0]              (5 values)
#  lookback:    [8, 10, 12, 16, 20]                     (5 values)
#  = 6 × 5 × 5 = 150 runs
#
#  All other params at _BASELINE (ki=10, ps=0.5, dilate=5, sift, no-grid/pg).
#  Rationale: v2 showed only this 3-way interaction drives real improvement.
#  Extends both below (mr=0.58) and above (lb=20) the v2-tested ranges.
# ══════════════════════════════════════════════════════════════════════════════

_a = 0
for _mr in [0.58, 0.62, 0.65, 0.67, 0.70, 0.73]:
    for _mf in [2.5, 3.0, 3.5, 4.0, 5.0]:
        for _lb in [8, 10, 12, 16, 20]:
            _a += 1
            RUNS.append(_RB(f"A{_a:03d}", "A",
                match_ratio=_mr, mad_factor=_mf, lookback=_lb))

assert sum(r["group"] == "A" for r in RUNS) == 150, "A count"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP B  –  keyframe_interval × lookback  (ENTIRELY NEW DIMENSION)
#
#  Two anchored (match_ratio, mad_factor) combos:
#    B_lo: mr=0.65, mf=3.0  (v2 L01 — found 9 obstacles, highest count)
#    B_hi: mr=0.65, mf=4.0  (v2 L03 — rank 1 winner)
#  lookback:         [8, 10, 12, 16, 20]   (5)
#  keyframe_interval:[3, 5, 7, 10, 12, 15] (6)
#  = 2 × 5 × 6 = 60 runs
#
#  keyframe_interval controls how often a long-range anchor is created.
#  Denser keyframes (ki=3) give more re-localization candidates; sparser
#  (ki=15) reduces memory but may leave gaps. Fully unexplored in v1/v2.
# ══════════════════════════════════════════════════════════════════════════════

_b = 0
for _mr, _mf in [(0.65, 3.0), (0.65, 4.0)]:
    for _lb in [8, 10, 12, 16, 20]:
        for _ki in [3, 5, 7, 10, 12, 15]:
            _b += 1
            RUNS.append(_RB(f"B{_b:03d}", "B",
                match_ratio=_mr, mad_factor=_mf,
                lookback=_lb, keyframe_interval=_ki))

assert sum(r["group"] == "B" for r in RUNS) == 60, "B count"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP C  –  SuperPoint + ratio_test in the productive region
#
#  feature_extractor: superpoint  (CUDA on 4070ti)
#  feature_matcher:   ratio_test  (LightGlue dropped by user)
#  feature_exclude_hsv: "none"    (SP ignores this field — no mask API)
#
#  match_ratio:  [0.60, 0.65, 0.70, 0.75, 0.80]  (5)
#  mad_factor:   [3.0, 4.0, 6.0]                  (3)
#  lookback:     [8, 12, 16]                       (3)
#  = 5 × 3 × 3 = 45 runs   (~110 s each on GPU)
#
#  Note: SP runs do NOT use feature_exclude_hsv (set to "none" explicitly).
#  SP may be worse on this grid arena than SIFT+exclusion — this sweep tests
#  whether the GPU-native descriptor helps in any range.
# ══════════════════════════════════════════════════════════════════════════════

_c = 0
for _mr in [0.60, 0.65, 0.70, 0.75, 0.80]:
    for _mf in [3.0, 4.0, 6.0]:
        for _lb in [8, 12, 16]:
            _c += 1
            RUNS.append(_RB(f"C{_c:03d}", "C",
                feature_extractor="superpoint",
                feature_matcher="ratio_test",
                feature_exclude_hsv="none",    # SP ignores the mask
                match_ratio=_mr, mad_factor=_mf, lookback=_lb))

assert sum(r["group"] == "C" for r in RUNS) == 45, "C count"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP D  –  Extraction parameters
#             Fixed: _BASELINE stitching (mr=0.65, mf=4.0, lb=8, ki=10)
#             Tests whether the extraction pipeline itself is a bottleneck.
#
#  D1 (9): target_fps × min_movement — frame density vs static-motion gate
#  D2 (5): blur_thresh — how strict the Laplacian sharpness test is
#  D3 (4): max_movement — upper bound on frame-to-frame motion allowed
#  D4 (4): static_pixel_thresh — fallback static-frame gate (0 = disable)
#  Total:  22 runs
# ══════════════════════════════════════════════════════════════════════════════

_d = 0
# D1: target_fps × min_movement
for _fps in [3.0, 5.0, 7.0]:
    for _mm in [0.010, 0.015, 0.020]:
        _d += 1
        RUNS.append(_RB(f"D{_d:03d}", "D",
            target_fps=_fps, min_movement=_mm))

# D2: blur_thresh
for _bt in [30.0, 60.0, 80.0, 120.0, 150.0]:
    _d += 1
    RUNS.append(_RB(f"D{_d:03d}", "D", blur_thresh=_bt))

# D3: max_movement
for _mxm in [0.30, 0.40, 0.50, 0.55]:
    _d += 1
    RUNS.append(_RB(f"D{_d:03d}", "D", max_movement=_mxm))

# D4: static_pixel_thresh (0 = disable the gate entirely)
for _spt in [0.0, 1.5, 3.0, 6.0]:
    _d += 1
    RUNS.append(_RB(f"D{_d:03d}", "D", static_pixel_thresh=_spt))

assert sum(r["group"] == "D" for r in RUNS) == 22, f"D count: {sum(r['group']=='D' for r in RUNS)}"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP E  –  processing_scale × match_ratio  (key interaction)
#             Fixed: mad_factor=3.0, lookback=8, ki=10, sift
#
#  processing_scale: [0.40, 0.45, 0.50, 0.55, 0.60, 0.66]  (6)
#  match_ratio:      [0.60, 0.65, 0.70]                     (3)
#  = 18 runs
#
#  Smaller scale → larger patch coverage per descriptor → grid intersections
#  more likely to generate near-duplicate descriptors. This interaction
#  may need tighter match_ratio at smaller scales.
# ══════════════════════════════════════════════════════════════════════════════

_e = 0
for _ps in [0.40, 0.45, 0.50, 0.55, 0.60, 0.66]:
    for _mr in [0.60, 0.65, 0.70]:
        _e += 1
        RUNS.append(_RB(f"E{_e:03d}", "E",
            processing_scale=_ps, match_ratio=_mr,
            mad_factor=3.0, lookback=8))

assert sum(r["group"] == "E" for r in RUNS) == 18, "E count"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP F  –  feature_exclude_dilate_px × lookback
#             Fixed: mr=0.65, mf=3.0, ki=10, sift, ps=0.5
#
#  feature_exclude_dilate_px: [3, 5, 7, 9, 12]  (5)
#  lookback:                  [8, 10, 12, 16]    (4)
#  = 20 runs
#
#  Dilation determines how far keypoints are pushed from blue-tape edges.
#  Under the new mirror angle the tape edges may appear at different widths;
#  testing dilation × lookback reveals whether these interact.
# ══════════════════════════════════════════════════════════════════════════════

_f = 0
for _dp in [3, 5, 7, 9, 12]:
    for _lb in [8, 10, 12, 16]:
        _f += 1
        RUNS.append(_RB(f"F{_f:03d}", "F",
            feature_exclude_dilate_px=_dp,
            match_ratio=0.65, mad_factor=3.0, lookback=_lb))

assert sum(r["group"] == "F" for r in RUNS) == 20, "F count"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP G  –  min_keypoint_bins + min_inliers
#             Fixed: mr=0.65, mf=3.0, lb=8, ki=10, sift
#
#  G1 (5): min_keypoint_bins sweep [0, 5, 8, 10, 15]   (fixed min_inliers=10)
#  G2 (6): min_inliers sweep       [6, 8, 10, 12, 15, 20] (fixed mkb=10)
#  G3 (9): 2-way interaction       [0,5,10] × [8,12,20]
#  = 20 runs
#
#  min_keypoint_bins=0 disables the spatial-spread gate entirely; worth
#  testing because frames with few but excellent keypoints may be wrongly
#  rejected. min_inliers interacts: raising it rejects weaker homographies
#  but also reduces frame count, which may help or hurt depending on video.
# ══════════════════════════════════════════════════════════════════════════════

_g = 0
_base_g = dict(match_ratio=0.65, mad_factor=3.0, lookback=8)

# G1: min_keypoint_bins (vary, min_inliers fixed at 10)
for _mkb in [0, 5, 8, 10, 15]:
    _g += 1
    RUNS.append(_RB(f"G{_g:03d}", "G", **_base_g, min_keypoint_bins=_mkb))

# G2: min_inliers (vary, min_keypoint_bins fixed at 10)
for _mi in [6, 8, 10, 12, 15, 20]:
    _g += 1
    RUNS.append(_RB(f"G{_g:03d}", "G", **_base_g, min_inliers=_mi))

# G3: 2-way interaction (mkb × mi)
for _mkb in [0, 5, 10]:
    for _mi in [8, 12, 20]:
        _g += 1
        RUNS.append(_RB(f"G{_g:03d}", "G", **_base_g,
            min_keypoint_bins=_mkb, min_inliers=_mi))

assert sum(r["group"] == "G" for r in RUNS) == 20, f"G: {sum(r['group']=='G' for r in RUNS)}"


# ── final count ───────────────────────────────────────────────────────────────
_EXPECTED = 150 + 60 + 45 + 22 + 18 + 20 + 20   # = 335
assert len(RUNS) == _EXPECTED, f"Expected {_EXPECTED} runs, got {len(RUNS)}"


# ─────────────────────────────────────────────────────────────────────────────
# Conversion helper  →  (ExtractionConfig, ReconstructConfig)
# ─────────────────────────────────────────────────────────────────────────────

def to_configs(run: Dict[str, Any]):
    """
    Convert a run dict → (ExtractionConfig, ReconstructConfig).

    Matches the parameter set from build_arena_map_server._build_reconstruct_cfg()
    exactly. sp_ort_provider is hardcoded to "cuda" (4070ti laptop).
    Change to "cpu" if testing on Jetson Orin.

    Usage
    ─────
        from drone_map_grid_gen import (
            ExtractionConfig, ReconstructConfig, ColorRangeMask, reconstruct_from_video
        )
        from run_matrix_v3 import RUNS, to_configs

        for run in RUNS:
            extract_cfg, recon_cfg = to_configs(run)
            stitched = reconstruct_from_video(
                video_path,
                extract_cfg=extract_cfg,
                reconstruct_cfg=recon_cfg,
            )
            # … transfer + consistency + occupancy …
            save_result(run["run_id"], …)
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

        # alignment flags (all confirmed OFF in v2)
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

        # grid (not active, but passed so the config round-trips cleanly)
        grid_match_dist           = run["grid_match_dist"],
        grid_min_intersections    = run["grid_min_intersections"],
        grid_hsv_h_lo             = run["grid_hsv_h_lo"],
        grid_hsv_h_hi             = run["grid_hsv_h_hi"],
        grid_hsv_s_lo             = run["grid_hsv_s_lo"],
        grid_hsv_v_lo             = run["grid_hsv_v_lo"],

        # pose graph (not active)
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
    sift_n = sum(1 for r in RUNS if r["feature_extractor"] == "sift")
    sp_n   = sum(1 for r in RUNS if r["feature_extractor"] == "superpoint")

    print(f"\nTotal runs: {len(RUNS)}  ({sift_n} SIFT, {sp_n} SP+ratio)\n")

    descs = {
        "A": "match_ratio × mad_factor × lookback        [dense 3-way, 6×5×5]",
        "B": "keyframe_interval × lookback               [new dimension, 2×5×6]",
        "C": "SP+ratio_test × match_ratio × mad × lb     [GPU, 5×3×3]",
        "D": "extraction params (fps/movement/blur/…)    [22 individual]",
        "E": "processing_scale × match_ratio             [6×3]",
        "F": "feature_exclude_dilate_px × lookback       [5×4]",
        "G": "min_keypoint_bins + min_inliers            [1-way + 2-way]",
    }

    est_sift = sift_n * 65
    est_sp   = sp_n * 110
    print(f"{'Group':<6} {'Runs':>5}   Description")
    print("─" * 72)
    for g in sorted(gc):
        print(f"  {g:<4} {gc[g]:>5}   {descs.get(g,'')}")
    print("─" * 72)
    print(f"\nEstimated wall time: SIFT {sift_n}×65s + SP {sp_n}×110s"
          f" ≈ {(est_sift+est_sp)//60} min ≈ {(est_sift+est_sp)/3600:.1f} h\n")

    # Verify _BASELINE round-trips cleanly
    print("_BASELINE (v2 rank-1 / L03):")
    keys = ["feature_extractor", "match_ratio", "mad_factor",
            "lookback", "keyframe_interval", "processing_scale",
            "feature_exclude_hsv", "feature_exclude_dilate_px"]
    for k in keys:
        print(f"  {k}: {_BASELINE[k]}")

    print("\nSample — first run of each group:")
    seen: set = set()
    for r in RUNS:
        if r["group"] not in seen:
            seen.add(r["group"])
            print(f"  {r['run_id']}  ext={r['feature_extractor'][:4]}  "
                  f"mr={r['match_ratio']}  mf={r['mad_factor']}  "
                  f"lb={r['lookback']}  ki={r['keyframe_interval']}  "
                  f"ps={r['processing_scale']}  dp={r['feature_exclude_dilate_px']}")