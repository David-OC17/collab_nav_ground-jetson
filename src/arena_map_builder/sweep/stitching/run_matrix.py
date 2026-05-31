"""
run_matrix.py  –  overnight parameter sweep for MapReconstructor
═══════════════════════════════════════════════════════════════════
83 runs  ·  estimated ≈ 7.3 h  ·  safe for a 10-h window
(each run: 3–8 min depending on extractor, grid, and pose-graph)

Groups
──────
  A (24)  architecture × grid × solver × feature-exclusion    ~132 min
  B (12)  match_ratio × mad_factor   [ratio_test paths only]  ~ 54 min
  C  (9)  lookback × keyframe_interval    [sift baseline]     ~ 45 min
  D  (6)  target_fps × min_movement       [sift baseline]     ~ 30 min
  E  (9)  pg_marker_weight × pg_huber_delta  [sift baseline]  ~ 45 min
  F  (4)  feature_exclude_dilate_px       [sift baseline]     ~ 20 min
  G  (4)  min_inliers                     [sift baseline]     ~ 20 min
  H  (3)  cross-backend check with estimated-best params      ~ 18 min
  I  (6)  lookback × keyframe_interval    [sp+ratio baseline] ~ 36 min
  J  (6)  pg_marker_weight × pg_huber_delta [sp+ratio baseline] ~ 36 min
  ───────────────────────────────────────────────────────────────
  Total   83 runs                                              ~436 min

Key encoding
────────────
  feature_exclude_hsv
    "none"       →  []
    "blue_tape"  →  [ColorRangeMask.blue_tape()]
    NOTE: SuperPoint ignores feature_exclude_hsv (its ONNX runner has no mask
    API); only the sift+ratio_test path uses it. Groups A/H set it to "none"
    for superpoint runs to make the asymmetry explicit.

  solver  →  (use_pose_graph, use_fiducials)
    "off"   →  (False, False)   pure online, no global correction
    "pg"    →  (True,  False)   pose graph: odometry + feature loops only
    "full"  →  (True,  True )   pose graph + ArUco fiducial loop closures

Groups C–J assumption
─────────────────────
  These groups fix match_ratio=0.70 and mad_factor=4.0 as an *estimated* best
  value for the sift/sp ratio-test paths (both tighter than the code default
  of 0.80/6.0, which is optimistic for a repetitive grid arena).
  Re-run these groups with the actual Group-B winner once overnight results
  are analysed, if the difference is significant.

Usage
─────
    from run_matrix import RUNS, to_configs
    for run in RUNS:
        extract_cfg, recon_cfg = to_configs(run)
        rec = MapReconstructor(recon_cfg)
        rec.add_video("flight.mp4", extract_cfg=extract_cfg)
        map_img = rec.get_map()
        if recon_cfg.use_pose_graph:
            map_img_corrected = rec.finalize()   # returns corrected map
        save_result(run["run_id"], run["group"], map_img)
"""

from __future__ import annotations
from typing import Any, Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# Code-defaults  (what MapReconstructor uses when you don't override)
# ─────────────────────────────────────────────────────────────────────────────
_D: Dict[str, Any] = {
    # ExtractionConfig
    "target_fps":                 5.0,
    "min_movement":               0.015,
    # ReconstructConfig – feature backend
    "feature_extractor":          "sift",
    "feature_matcher":            "ratio_test",
    # ReconstructConfig – alignment flags
    "use_grid_intersections":     True,
    "use_pose_graph":             True,
    "use_fiducials":              True,
    # ReconstructConfig – feature quality
    "feature_exclude_hsv":        "none",     # "none" | "blue_tape"
    "feature_exclude_dilate_px":  5,
    "match_ratio":                0.80,
    "mad_factor":                 6.0,
    "min_inliers":                10,
    # ReconstructConfig – temporal coverage
    "lookback":                   4,
    "keyframe_interval":          10,
    # ReconstructConfig – pose graph
    "pg_marker_weight":           15.0,
    "pg_odom_weight":             1.0,    # reference weight – never varied
    "pg_loop_weight":             2.0,
    "pg_huber_delta":             4.0,
}


def _R(run_id: str, group: str, **overrides) -> Dict[str, Any]:
    """Build a run dict by applying overrides on top of _D."""
    return {"run_id": run_id, "group": group, **_D, **overrides}


RUNS: List[Dict[str, Any]] = []


# ═════════════════════════════════════════════════════════════════════════════
# GROUP A  –  Architecture × grid × solver × feature-exclusion
#             All continuous params at code defaults (match_ratio=0.80, etc.)
#             sift+ratio:   2 grid × 3 solver × 2 exclude  = 12 runs
#             sp+ratio:     2 grid × 3 solver               =  6 runs
#             sp+lightglue: 2 grid × 3 solver               =  6 runs
#             Total: 24 runs
# ═════════════════════════════════════════════════════════════════════════════

_a = 0

# ── sift + ratio_test ─────────────────────────────────────────────────────────
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
                feature_exclude_hsv=_exc,
            ))

# ── superpoint + ratio_test  (SP ignores feature_exclude_hsv) ────────────────
for _grid in [False, True]:
    for _pg, _fid in [(False, False), (True, False), (True, True)]:
        _a += 1
        RUNS.append(_R(f"A{_a:02d}", "A",
            feature_extractor="superpoint",
            feature_matcher="ratio_test",
            use_grid_intersections=_grid,
            use_pose_graph=_pg,
            use_fiducials=_fid,
            feature_exclude_hsv="none",
        ))

# ── superpoint + lightglue  (LG ignores both match_ratio and feature_exclude) ─
for _grid in [False, True]:
    for _pg, _fid in [(False, False), (True, False), (True, True)]:
        _a += 1
        RUNS.append(_R(f"A{_a:02d}", "A",
            feature_extractor="superpoint",
            feature_matcher="lightglue",
            use_grid_intersections=_grid,
            use_pose_graph=_pg,
            use_fiducials=_fid,
            feature_exclude_hsv="none",
        ))

assert sum(r["group"] == "A" for r in RUNS) == 24


# ═════════════════════════════════════════════════════════════════════════════
# GROUP B  –  match_ratio × mad_factor
#             ratio_test backends only; fixed: grid=True, solver=full
#             sift:   exclude=blue_tape (exclude helps sift)
#             sp:     exclude=none      (SP ignores the mask)
#             Values tested: match_ratio ∈ {0.65, 0.70, 0.75}  (0.80 covered in A)
#                            mad_factor  ∈ {4.0, 6.0}           (6.0 is the default)
#             Total: 2 backends × 3 × 2 = 12 runs
# ═════════════════════════════════════════════════════════════════════════════

_b = 0
for _ext, _exc in [("sift", "blue_tape"), ("superpoint", "none")]:
    for _mr in [0.65, 0.70, 0.75]:
        for _mf in [4.0, 6.0]:
            _b += 1
            RUNS.append(_R(f"B{_b:02d}", "B",
                feature_extractor=_ext,
                feature_matcher="ratio_test",
                use_grid_intersections=True,
                use_pose_graph=True,
                use_fiducials=True,
                feature_exclude_hsv=_exc,
                match_ratio=_mr,
                mad_factor=_mf,
            ))

assert sum(r["group"] == "B" for r in RUNS) == 12


# ─────────────────────────────────────────────────────────────────────────────
# Shared "estimated-best" fixed params for sift groups C–G
#   match_ratio and mad_factor are tighter than defaults based on arena type.
#   Update once Group B results are available.
# ─────────────────────────────────────────────────────────────────────────────
_SIFT_BASE = dict(
    feature_extractor="sift",
    feature_matcher="ratio_test",
    use_grid_intersections=True,
    use_pose_graph=True,
    use_fiducials=True,
    feature_exclude_hsv="blue_tape",
    match_ratio=0.70,
    mad_factor=4.0,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixed params for sp+ratio groups I–J
# ─────────────────────────────────────────────────────────────────────────────
_SP_BASE = dict(
    feature_extractor="superpoint",
    feature_matcher="ratio_test",
    use_grid_intersections=True,
    use_pose_graph=True,
    use_fiducials=True,
    feature_exclude_hsv="none",
    match_ratio=0.70,
    mad_factor=4.0,
)


# ═════════════════════════════════════════════════════════════════════════════
# GROUP C  –  lookback × keyframe_interval
#             Fixed: sift base  (see _SIFT_BASE above)
#             Total: 3 × 3 = 9 runs
# ═════════════════════════════════════════════════════════════════════════════

_c = 0
for _lb in [4, 6, 8]:
    for _ki in [5, 10, 15]:
        _c += 1
        RUNS.append(_R(f"C{_c:02d}", "C",
            **_SIFT_BASE,
            lookback=_lb,
            keyframe_interval=_ki,
        ))

assert sum(r["group"] == "C" for r in RUNS) == 9


# ═════════════════════════════════════════════════════════════════════════════
# GROUP D  –  target_fps × min_movement  (ExtractionConfig)
#             Fixed: sift base
#             Total: 3 × 2 = 6 runs
# ═════════════════════════════════════════════════════════════════════════════

_d = 0
for _fps in [3.0, 5.0, 7.0]:
    for _mm in [0.010, 0.020]:
        _d += 1
        RUNS.append(_R(f"D{_d:02d}", "D",
            **_SIFT_BASE,
            target_fps=_fps,
            min_movement=_mm,
        ))

assert sum(r["group"] == "D" for r in RUNS) == 6


# ═════════════════════════════════════════════════════════════════════════════
# GROUP E  –  pg_marker_weight × pg_huber_delta
#             Fixed: sift base
#             Total: 3 × 3 = 9 runs
# ═════════════════════════════════════════════════════════════════════════════

_e = 0
for _pw in [5.0, 15.0, 25.0]:
    for _hd in [2.0, 4.0, 8.0]:
        _e += 1
        RUNS.append(_R(f"E{_e:02d}", "E",
            **_SIFT_BASE,
            pg_marker_weight=_pw,
            pg_huber_delta=_hd,
        ))

assert sum(r["group"] == "E" for r in RUNS) == 9


# ═════════════════════════════════════════════════════════════════════════════
# GROUP F  –  feature_exclude_dilate_px
#             Fixed: sift base  (only meaningful when feature_exclude_hsv=blue_tape)
#             Total: 4 runs
# ═════════════════════════════════════════════════════════════════════════════

for _f, _dp in enumerate([3, 5, 8, 12], 1):
    RUNS.append(_R(f"F{_f:02d}", "F",
        **_SIFT_BASE,
        feature_exclude_dilate_px=_dp,
    ))

assert sum(r["group"] == "F" for r in RUNS) == 4


# ═════════════════════════════════════════════════════════════════════════════
# GROUP G  –  min_inliers  (RANSAC acceptance gate)
#             Fixed: sift base
#             Total: 4 runs
# ═════════════════════════════════════════════════════════════════════════════

for _g, _mi in enumerate([8, 12, 15, 20], 1):
    RUNS.append(_R(f"G{_g:02d}", "G",
        **_SIFT_BASE,
        min_inliers=_mi,
    ))

assert sum(r["group"] == "G" for r in RUNS) == 4


# ═════════════════════════════════════════════════════════════════════════════
# GROUP H  –  Cross-backend check with estimated-best continuous params
#             grid=True, solver=full, match_ratio=0.70, mad_factor=4.0,
#             lookback=6, keyframe_interval=5
#             Total: 3 runs  (one per backend)
# ═════════════════════════════════════════════════════════════════════════════

for _h, (_ext, _mat, _exc) in enumerate([
    ("sift",       "ratio_test", "blue_tape"),
    ("superpoint", "ratio_test", "none"),
    ("superpoint", "lightglue",  "none"),
], 1):
    RUNS.append(_R(f"H{_h:02d}", "H",
        feature_extractor=_ext,
        feature_matcher=_mat,
        use_grid_intersections=True,
        use_pose_graph=True,
        use_fiducials=True,
        feature_exclude_hsv=_exc,
        match_ratio=0.70,
        mad_factor=4.0,
        lookback=6,
        keyframe_interval=5,
    ))

assert sum(r["group"] == "H" for r in RUNS) == 3


# ═════════════════════════════════════════════════════════════════════════════
# GROUP I  –  lookback × keyframe_interval  (sp+ratio)
#             Fixed: sp base  (see _SP_BASE above)
#             Total: 3 × 2 = 6 runs  (smaller than C; ki=15 dropped)
# ═════════════════════════════════════════════════════════════════════════════

_i = 0
for _lb in [4, 6, 8]:
    for _ki in [5, 10]:
        _i += 1
        RUNS.append(_R(f"I{_i:02d}", "I",
            **_SP_BASE,
            lookback=_lb,
            keyframe_interval=_ki,
        ))

assert sum(r["group"] == "I" for r in RUNS) == 6


# ═════════════════════════════════════════════════════════════════════════════
# GROUP J  –  pg_marker_weight × pg_huber_delta  (sp+ratio)
#             Fixed: sp base
#             Total: 6 runs  (subset of E's grid; corners + midpoints)
# ═════════════════════════════════════════════════════════════════════════════

_j = 0
for _pw, _hd in [
    ( 5.0, 2.0), ( 5.0, 8.0),
    (15.0, 2.0), (15.0, 8.0),
    (25.0, 4.0), (25.0, 8.0),
]:
    _j += 1
    RUNS.append(_R(f"J{_j:02d}", "J",
        **_SP_BASE,
        pg_marker_weight=_pw,
        pg_huber_delta=_hd,
    ))

assert sum(r["group"] == "J" for r in RUNS) == 6

assert len(RUNS) == 83, f"Expected 83 runs, got {len(RUNS)}"


# ─────────────────────────────────────────────────────────────────────────────
# Conversion helper
# ─────────────────────────────────────────────────────────────────────────────

def to_configs(run: Dict[str, Any]):
    """
    Convert a run dict → (ExtractionConfig, ReconstructConfig).

    Assumes drone_map_grid_gen is importable from the pipeline package.
    Adjust the import path to match your project layout.

    Example
    ───────
        from drone_map_grid_gen import (
            ExtractionConfig, ReconstructConfig, ColorRangeMask, MapReconstructor
        )
        from run_matrix import RUNS, to_configs

        for run in RUNS:
            extract_cfg, recon_cfg = to_configs(run)
            rec = MapReconstructor(recon_cfg)
            rec.add_video("flight.mp4", extract_cfg=extract_cfg)
            result = rec.get_map()
            if recon_cfg.use_pose_graph:
                result = rec.finalize()
            save_result(run["run_id"], result)
    """
    from drone_map_grid_gen import ExtractionConfig, ReconstructConfig, ColorRangeMask

    exclude = (
        [ColorRangeMask.blue_tape()]
        if run["feature_exclude_hsv"] == "blue_tape"
        else []
    )

    extract_cfg = ExtractionConfig(
        target_fps   = run["target_fps"],
        min_movement = run["min_movement"],
    )

    recon_cfg = ReconstructConfig(
        feature_extractor         = run["feature_extractor"],
        feature_matcher           = run["feature_matcher"],
        use_grid_intersections    = run["use_grid_intersections"],
        use_pose_graph            = run["use_pose_graph"],
        use_fiducials             = run["use_fiducials"],
        feature_exclude_hsv       = exclude,
        feature_exclude_dilate_px = run["feature_exclude_dilate_px"],
        match_ratio               = run["match_ratio"],
        mad_factor                = run["mad_factor"],
        min_inliers               = run["min_inliers"],
        lookback                  = run["lookback"],
        keyframe_interval         = run["keyframe_interval"],
        pg_marker_weight          = run["pg_marker_weight"],
        pg_odom_weight            = run["pg_odom_weight"],
        pg_loop_weight            = run["pg_loop_weight"],
        pg_huber_delta            = run["pg_huber_delta"],
    )

    return extract_cfg, recon_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Quick summary when run directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from collections import Counter

    groups = Counter(r["group"] for r in RUNS)
    print(f"\nTotal runs: {len(RUNS)}\n")
    print(f"{'Group':<8} {'Runs':>5}   Description")
    print("─" * 60)
    descs = {
        "A": "architecture × grid × solver × feature-exclusion",
        "B": "match_ratio × mad_factor  (ratio_test only)",
        "C": "lookback × keyframe_interval  (sift baseline)",
        "D": "target_fps × min_movement",
        "E": "pg_marker_weight × pg_huber_delta  (sift baseline)",
        "F": "feature_exclude_dilate_px",
        "G": "min_inliers",
        "H": "cross-backend check with estimated-best params",
        "I": "lookback × keyframe_interval  (sp+ratio baseline)",
        "J": "pg_marker_weight × pg_huber_delta  (sp+ratio baseline)",
    }
    for g in sorted(groups):
        print(f"  {g:<6} {groups[g]:>5}   {descs.get(g, '')}")
    print("─" * 60)

    print("\nSample runs (first of each group):")
    seen = set()
    for r in RUNS:
        if r["group"] not in seen:
            seen.add(r["group"])
            print(f"  {r['run_id']}  ext={r['feature_extractor'][:4]}  "
                  f"mat={r['feature_matcher'][:5]}  "
                  f"grid={r['use_grid_intersections']}  "
                  f"pg={r['use_pose_graph']}  fid={r['use_fiducials']}  "
                  f"exc={r['feature_exclude_hsv'][:4]}  "
                  f"mr={r['match_ratio']}  mf={r['mad_factor']}")
