#!/usr/bin/env python3
"""
extract_features.py  –  build XGBoost training data from sweep results
═══════════════════════════════════════════════════════════════════════
For every run directory found in the given sweep result directories:
  1. Finds the stitched map image (transfer_input.png or stitched_map.png).
  2. Re-runs the transfer pipeline (run_pipeline) fresh on that image so
     diagnostics reflect the same pipeline logic, not the saved debug images.
  3. Calls compute_diagnostics() to extract the 46-feature diagnostic vector.
  4. Appends the run's label (pass/fail/unsure/unlabeled) and its config
     parameters, then writes everything to a CSV.

Column groups in the output CSV
────────────────────────────────
  Metadata   : sweep, run_id, label, diag_error, status, n_obstacles,
               mean_consistency, runtime_s
  Diagnostics: 46 columns from MapDiagnostics.to_feature_vector()
               (all float; None → -1.0)
  Config     : cfg_* columns from config.yaml
               (useful for analysis; mark as metadata, not features, in the
               notebook if you don't want config params in the classifier)

Note on Group 6 features (stitcher_* / pg_*)
─────────────────────────────────────────────
metrics.yaml only stores high-level summary metrics; it does not contain
per-run stitcher stats or the pose-graph finalize report.  All Group 6
diagnostic features will therefore be -1.0 in the output CSV.  A future
sweep version could save these fields to metrics.yaml and pass them in.

Usage
─────
  python3 extract_features.py \\
      --results-dirs sweep/results1 sweep/results2 \\
      --background /path/to/background.png \\
      --labels-file labels.yaml \\
      --output features.csv

  # Without labels (features only, useful for a new unlabeled sweep):
  python3 extract_features.py \\
      --results-dirs sweep/results \\
      --background /path/to/background.png \\
      --output features.csv

  # Point to the project source if not running inside the ROS install:
  python3 extract_features.py ... --project-dir /path/to/arena_map_builder/src
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import yaml

# ── project import handling ─────────────────────────────────────────────────
# Parse --project-dir early so imports work before argparse.parse_args().
_pdir_flag = "--project-dir"
if _pdir_flag in sys.argv:
    _idx = sys.argv.index(_pdir_flag)
    _pdir = sys.argv[_idx + 1] if _idx + 1 < len(sys.argv) else None
    if _pdir:
        sys.path.insert(0, _pdir)

try:
    from transfer_obstacles import TransferConfig, run_pipeline
    from map_diagnostics import DiagnosticsConfig, compute_diagnostics
except ImportError as _e:
    print(f"[ERROR] Cannot import project modules: {_e}")
    print("  Run with --project-dir /path/to/project/src  or source the ROS install.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading helpers  (mirror load_all_runs from label.py)
# ══════════════════════════════════════════════════════════════════════════════

def _load_run(run_dir: Path, sweep_name: str) -> Optional[dict]:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    metrics: dict = {}
    met_path = run_dir / "metrics.yaml"
    if met_path.exists():
        with open(met_path) as f:
            metrics = yaml.safe_load(f) or {}
    return {
        "sweep":   sweep_name,
        "run_id":  cfg.get("run_id", run_dir.name),
        "group":   cfg.get("group", "?"),
        "dir":     run_dir,
        "config":  cfg,
        "metrics": metrics,
    }


def load_all_runs(results_dirs: List[Path]) -> List[dict]:
    runs = []
    for rd in sorted(results_dirs, key=lambda p: p.name):
        for d in sorted(rd.iterdir()):
            if not d.is_dir() or not d.name.startswith("run_"):
                continue
            run = _load_run(d, rd.name)
            if run:
                runs.append(run)
    return runs


def run_key(run: dict) -> str:
    return f"{run['sweep']}/{run['run_id']}"


def load_labels(path: Optional[Path]) -> Dict[str, str]:
    if path is None or not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ══════════════════════════════════════════════════════════════════════════════
# Per-run feature extraction
# ══════════════════════════════════════════════════════════════════════════════

_INPUT_CANDIDATES = ["transfer_input.png", "stitched_map.png"]

# Keys in metrics.yaml that belong to base metadata, not the feature vector.
# Everything else in metrics is a diagnostic feature written by sweep.py.
_META_KEYS = frozenset({
    "status", "n_obstacles", "mean_consistency", "runtime_s",
    "server_message", "ssim_vs_gt",
})

# Sentinel: present in metrics.yaml only when sweep._collect_diagnostics ran.
_DIAG_SENTINEL = "stitch_h_line_count"


def _find_input_image(run_dir: Path) -> Optional[Path]:
    for name in _INPUT_CANDIDATES:
        p = run_dir / name
        if p.exists():
            return p
    return None


def extract_row(
    run: dict,
    background_path: Path,
    labels: Dict[str, str],
    diag_cfg: DiagnosticsConfig,
    transfer_cfg: TransferConfig,
) -> dict:
    """Build one CSV row for `run`.

    The row is always returned (never None).  On failure, diag_error=True
    and all diagnostic features are -1.0, so the row can be filtered out
    in the notebook without losing label metadata.
    """
    key     = run_key(run)
    metrics = run["metrics"]

    # Base metadata shared by success and error paths
    base = {
        "sweep":            run["sweep"],
        "run_id":           run["run_id"],
        "group":            run["group"],
        "label":            labels.get(key, "unlabeled"),
        "diag_error":       False,
        "status":           metrics.get("status", ""),
        "n_obstacles":      metrics.get("n_obstacles", -1),
        "mean_consistency": metrics.get("mean_consistency", -1.0),
        "runtime_s":        metrics.get("runtime_s", -1.0),
    }

    cfg_cols = {
        f"cfg_{k}": v
        for k, v in run["config"].items()
        if k not in ("run_id", "group")
    }

    # ── Fast path: sweep already computed all 46 features ─────────────────
    # sweep._collect_diagnostics() writes them flat into metrics.yaml.
    # If the sentinel key is present, use them directly — no pipeline re-run.
    if _DIAG_SENTINEL in metrics:
        features = {
            k: float(v)
            for k, v in metrics.items()
            if k not in _META_KEYS
        }
        return {**base, **features, **cfg_cols}

    # Find the stitched map image
    img_path = _find_input_image(run["dir"])
    if img_path is None:
        names = ", ".join(_INPUT_CANDIDATES)
        print(f"  [SKIP]  {key}: no input image ({names})")
        return {**base, "diag_error": True, **_error_features(), **cfg_cols}

    # Re-run the transfer pipeline (verbose=False — suppress per-stage logs)
    try:
        _, stages = run_pipeline(
            str(img_path),
            str(background_path),
            cfg=transfer_cfg,
            verbose=False,
        )
    except Exception:
        print(f"  [ERROR] {key}: run_pipeline failed")
        traceback.print_exc()
        return {**base, "diag_error": True, **_error_features(), **cfg_cols}

    # Load the stitched map as a numpy array for Group 1 (stitch_*) metrics.
    # stages["input"] is the same image but loaded inside run_pipeline; we
    # reload here so we don't need to carry a reference through the stages dict.
    stitched_map = cv2.imread(str(img_path))

    # Compute diagnostics
    try:
        diag = compute_diagnostics(
            transfer_stages=stages,
            transfer_cfg=transfer_cfg,
            stitched_map=stitched_map,
            # stitcher_stats and finalize_report are NOT available from
            # metrics.yaml; Group 6 features (stitcher_* / pg_*) will be -1.0.
            cfg=diag_cfg,
        )
        features = diag.to_feature_vector()
    except Exception:
        print(f"  [ERROR] {key}: compute_diagnostics failed")
        traceback.print_exc()
        return {**base, "diag_error": True, **_error_features(), **cfg_cols}

    return {**base, **features, **cfg_cols}


def _error_features() -> Dict[str, float]:
    """Sentinel row when diagnostics could not be computed: all -1.0."""
    # Import lazily to avoid a double-import on top-level; DiagnosticsConfig
    # and compute_diagnostics are already imported at module level.
    dummy_cfg  = DiagnosticsConfig()
    dummy_keys = [
        "stitch_h_line_count", "stitch_v_line_count",
        "stitch_h_angle_std_deg", "stitch_v_angle_std_deg",
        "stitch_h_spacing_cv", "stitch_v_spacing_cv",
        "stitch_content_frac", "stitch_convexity_ratio",
        "stitch_bbox_aspect_ratio",
        "dewarp_h_line_count", "dewarp_v_line_count", "dewarp_skipped",
        "green_hull_area_frac", "green_hull_convexity", "green_pixel_frac",
        "blob_count", "blob_mean_area_frac", "blob_std_area_frac",
        "blob_max_area_frac", "blob_mean_consistency", "blob_std_consistency",
        "blob_min_consistency", "blob_count_high_consistency",
        "inter_marker_distance_norm",
        "stitcher_frames_placed", "stitcher_frames_failed",
        "stitcher_placement_rate", "stitcher_keyframes",
        "stitcher_grid_refined_frac", "stitcher_grid_skipped_frac",
        "pg_rms_before", "pg_rms_after", "pg_rms_ratio",
        "pg_marker_count", "pg_edge_count", "pg_frame_count",
    ]
    # Add per-marker keys using the default color map
    for role in dummy_cfg.critical_marker_colors:
        for suffix in ("found", "instance_count", "x_norm", "y_norm", "inside_hull"):
            dummy_keys.append(f"marker_{role}_{suffix}")
    return {k: -1.0 for k in dummy_keys}


# ══════════════════════════════════════════════════════════════════════════════
# CSV writing
# ══════════════════════════════════════════════════════════════════════════════

def write_csv(rows: List[dict], output_path: Path) -> None:
    if not rows:
        print("[WARN] No rows to write.")
        return

    # Column order: metadata first, then diagnostics, then cfg_*
    meta_cols = ["sweep", "run_id", "group", "label", "diag_error",
                 "status", "n_obstacles", "mean_consistency", "runtime_s"]
    all_keys  = list(rows[0].keys())
    diag_cols = [k for k in all_keys if k not in meta_cols and not k.startswith("cfg_")]
    cfg_cols  = sorted(k for k in all_keys if k.startswith("cfg_"))
    fieldnames = meta_cols + diag_cols + cfg_cols

    # Fill any missing keys with -1 (happens when different runs have different
    # marker color sets — unlikely but defensive)
    for row in rows:
        for k in fieldnames:
            if k not in row:
                row[k] = -1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  CSV written: {output_path}  ({len(rows)} rows × {len(fieldnames)} cols)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract diagnostic features from sweep results for XGBoost training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dirs", nargs="+", required=True,
        help="One or more sweep result directories",
    )
    parser.add_argument(
        "--background", required=True,
        help="Path to background.png (used by run_pipeline)",
    )
    parser.add_argument(
        "--labels-file", default=None,
        help="Path to labels.yaml from label.py (optional; unlabeled if omitted)",
    )
    parser.add_argument(
        "--output", default="features.csv",
        help="Output CSV path (default: features.csv)",
    )
    parser.add_argument(
        "--project-dir", default=None,
        help="Directory containing transfer_obstacles.py and map_diagnostics.py "
             "(added to sys.path; for use outside the ROS install)",
    )
    # DiagnosticsConfig overrides
    parser.add_argument(
        "--arena-side-m", type=float, default=3.90,
        help="Known arena side length in metres (default: 3.90)",
    )
    parser.add_argument(
        "--consistency-threshold", type=float, default=0.35,
        help="Consistency threshold for blob_count_high_consistency (default: 0.35)",
    )
    args = parser.parse_args()

    results_dirs = [Path(d) for d in args.results_dirs]
    background   = Path(args.background)
    labels_path  = Path(args.labels_file) if args.labels_file else None
    output_path  = Path(args.output)

    for rd in results_dirs:
        if not rd.is_dir():
            print(f"[ERROR] Not a directory: {rd}")
            sys.exit(1)
    if not background.exists():
        print(f"[ERROR] Background image not found: {background}")
        sys.exit(1)

    diag_cfg     = DiagnosticsConfig(
        arena_side_m=args.arena_side_m,
        blob_consistency_threshold=args.consistency_threshold,
    )
    transfer_cfg = TransferConfig()   # same defaults for all runs

    print(f"\n  Loading runs from {[str(d) for d in results_dirs]} …")
    all_runs = load_all_runs(results_dirs)
    print(f"  {len(all_runs)} run(s) found.")

    labels = load_labels(labels_path)
    n_labeled = sum(1 for r in all_runs if run_key(r) in labels)
    print(f"  {n_labeled}/{len(all_runs)} run(s) have labels.")
    print(f"  Background: {background}")
    print(f"  Output:     {output_path}")
    print(f"\n  Re-running transfer pipeline + diagnostics for each run…\n")

    rows: List[dict] = []
    n_ok = n_err = 0

    for i, run in enumerate(all_runs, 1):
        key = run_key(run)
        print(f"  [{i}/{len(all_runs)}]  {key}", end="  ", flush=True)
        row = extract_row(run, background, labels, diag_cfg, transfer_cfg)
        rows.append(row)
        if row["diag_error"]:
            n_err += 1
            print("→ ERROR")
        else:
            n_ok += 1
            src    = "cached" if _DIAG_SENTINEL in run["metrics"] else "computed"
            blob_c = row.get("blob_count", -1)
            hull_f = row.get("green_hull_area_frac", -1)
            label  = row.get("label", "?")
            print(f"→ ok [{src}]  label={label}  blobs={blob_c}  hull={hull_f:.2f}")

    print(f"\n  Done: {n_ok} ok, {n_err} errors.")
    write_csv(rows, output_path)


if __name__ == "__main__":
    main()