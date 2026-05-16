# python -m drone_map_create.example.sweep_drone_map_params

"""
sweep_drone_map_params.py
─────────────────────────────────────────────────────────────────────────────
Parameter sweep harness for the drone-map reconstruction pipeline.

Runs the full reconstruct_from_video pipeline across a matrix of
ExtractionConfig + ReconstructConfig variations, writes one labelled output
PNG per combination, plus a JSON of the exact config and runtime stats.
After all runs complete it emits two artefacts for review:

  • summary.md         — sorted table of every run with stats and notes
  • contact_sheet.html — single page of thumbnails, all maps side-by-side,
                         each labelled with the params that produced it

Sweep modes
───────────
  curated  : ~12 hand-picked combinations targeting the highest-impact knobs
             (blue-tape feature exclusion on/off, match_ratio, mad_factor).
             Best first run — fastest path to a working baseline.

  axis     : One-at-a-time variation around the baseline.  Each axis is
             swept independently while the others stay at their default.
             Run count = 1 + sum(len(axis) - 1 for each axis).  Use this
             AFTER curated mode picks a baseline, to fine-tune each knob.

  grid     : Cartesian product over a small subset of axes.  Multiplies
             fast — only enable a few axes at a time.

Memory / time
─────────────
Each run is sequential and uses ~300 MB peak (with default canvas size).
Set --scale 0.5 to cut every run's time/memory ~4× — recommended for sweeps.
The baseline below uses scale 0.5; override per-run if needed.

Output layout
─────────────
  sweep_runs/
    001_<slug>/
      map.png
      config.json    — full config dump (the exact values used)
      stats.json     — placed / failed / elapsed / canvas size
    002_<slug>/...
    ...
    summary.md
    contact_sheet.html
"""

import argparse
import itertools
import json
import os
import re
import sys
import time
import traceback
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from drone_map_create.drone_map_gen import (
    ColorRangeMask,
    ExtractionConfig,
    MapReconstructor,
    ReconstructConfig,
)

# ══════════════════════════════════════════════════════════════════════════════
# Baseline — every sweep variation is an OVERRIDE of these values
# ══════════════════════════════════════════════════════════════════════════════

BASELINE_EXTRACT = ExtractionConfig(
    target_fps=8.0,
    blur_thresh=50.0,
    artifact_thresh=1.5,
    min_movement=0.015,
    max_movement=0.55,
)

# Default colour-replacement masks for visual quality of the final map.
# These do NOT affect alignment (alignment uses unmasked frames).
DEFAULT_COLOR_MASKS = [
    ColorRangeMask.yellow(replace_bgr=(255, 0, 255)),
    ColorRangeMask.brown(replace_bgr=(255, 0, 255)),
    ColorRangeMask.orange(replace_bgr=(255, 0, 255)),
]

BASELINE_RECONSTRUCT = ReconstructConfig(
    canvas_margin=2000,
    min_inliers=15,
    lookback=6,
    keyframe_interval=15,
    blend_mode="feather",   # feather for sweeps — pyramid is 3× slower
    pyramid_levels=4,
    processing_scale=0.5,   # half-res by default for sweep speed
    color_masks=DEFAULT_COLOR_MASKS,
    max_keyframes=20,
    max_canvas_px=8000,
    # new knobs from the alignment-fix patch:
    match_ratio=0.70,
    mad_factor=6.0,
    feature_exclude_hsv=[ColorRangeMask.blue_tape()],
    feature_exclude_dilate_px=5,
    min_keypoint_bins=12,
)


# ══════════════════════════════════════════════════════════════════════════════
# Sweep specifications — what to vary
# ══════════════════════════════════════════════════════════════════════════════

# ── CURATED: a small set of high-information runs ───────────────────────────
# Each entry is (label, {"extract": {...}, "reconstruct": {...}}) — keys
# omitted from the inner dicts inherit from the baseline.
CURATED_RUNS: List[Tuple[str, Dict[str, Dict[str, Any]]]] = [
    ("baseline", {}),

    # — alignment / feature stack: which fix actually mattered? —
    ("no_blue_exclusion", {
        "reconstruct": {"feature_exclude_hsv": []},
    }),
    ("no_spread_check", {
        "reconstruct": {"min_keypoint_bins": 0},
    }),
    ("no_blue_no_spread", {
        "reconstruct": {"feature_exclude_hsv": [], "min_keypoint_bins": 0},
    }),

    # — match strictness sweep —
    ("match_ratio_tight_0.65", {
        "reconstruct": {"match_ratio": 0.65},
    }),
    ("match_ratio_loose_0.75", {
        "reconstruct": {"match_ratio": 0.75},
    }),
    ("match_ratio_loose_0.80", {
        "reconstruct": {"match_ratio": 0.80},
    }),

    # — MAD outlier gate sweep —
    ("mad_tight_4", {
        "reconstruct": {"mad_factor": 4.0},
    }),
    ("mad_loose_10", {
        "reconstruct": {"mad_factor": 10.0},
    }),

    # — RANSAC inlier requirement —
    ("inliers_loose_10", {
        "reconstruct": {"min_inliers": 10},
    }),
    ("inliers_strict_25", {
        "reconstruct": {"min_inliers": 25},
    }),

    # — sampling rate (more frames = more overlap but more drift accumulation) —
    ("fps_low_5", {
        "extract": {"target_fps": 5.0},
    }),
    ("fps_high_12", {
        "extract": {"target_fps": 12.0},
    }),

    # — full-resolution comparison (slower; useful as a fidelity reference) —
    ("full_res", {
        "reconstruct": {"processing_scale": 1.0},
    }),

    # — pyramid blending at the chosen baseline (visual quality reference) —
    ("blend_pyramid", {
        "reconstruct": {"blend_mode": "pyramid"},
    }),
]

# ── AXIS: one-at-a-time variation around the baseline ───────────────────────
# Format: {"<extract|reconstruct>.<field>": [values...]}
# Each list entry produces one run with ONLY that one field overridden.
AXIS_SWEEP: Dict[str, List[Any]] = {
    "reconstruct.match_ratio":        [0.60, 0.65, 0.70, 0.75, 0.80],
    "reconstruct.mad_factor":         [3.0, 4.0, 6.0, 8.0, 10.0],
    "reconstruct.min_inliers":        [10, 15, 20, 25, 30],
    "reconstruct.min_keypoint_bins":  [0, 6, 10, 12, 16],
    "reconstruct.feature_exclude_dilate_px": [0, 3, 5, 8, 12],
    "reconstruct.lookback":           [3, 5, 6, 8, 10],
    "reconstruct.keyframe_interval":  [8, 12, 15, 20, 30],
    "extract.target_fps":             [4.0, 6.0, 8.0, 10.0, 12.0],
    "extract.blur_thresh":            [30.0, 50.0, 70.0, 100.0],
    "extract.artifact_thresh":        [1.2, 1.5, 2.0, 3.0],
}

# ── GRID: Cartesian product (keep this SMALL — multiplies fast) ──────────────
# 2 × 3 × 3 = 18 runs by default.
GRID_SWEEP: Dict[str, List[Any]] = {
    "reconstruct.feature_exclude_hsv": [[], [ColorRangeMask.blue_tape()]],
    "reconstruct.match_ratio":         [0.65, 0.70, 0.75],
    "reconstruct.mad_factor":          [4.0, 6.0, 8.0],
}


# ══════════════════════════════════════════════════════════════════════════════
# Config plumbing
# ══════════════════════════════════════════════════════════════════════════════


def _apply_overrides(
    base_extract: ExtractionConfig,
    base_reconstruct: ReconstructConfig,
    overrides: Dict[str, Dict[str, Any]],
) -> Tuple[ExtractionConfig, ReconstructConfig]:
    """Return new (extract, reconstruct) configs with `overrides` applied.

    `overrides` is a dict like {"extract": {...}, "reconstruct": {...}};
    unspecified fields inherit from the baseline.
    """
    # Deepcopy so each run starts clean (and lists like color_masks don't alias).
    ex = deepcopy(base_extract)
    rc = deepcopy(base_reconstruct)
    for k, v in overrides.get("extract", {}).items():
        if not hasattr(ex, k):
            raise KeyError(f"ExtractionConfig has no field {k!r}")
        setattr(ex, k, v)
    for k, v in overrides.get("reconstruct", {}).items():
        if not hasattr(rc, k):
            raise KeyError(f"ReconstructConfig has no field {k!r}")
        setattr(rc, k, v)
    return ex, rc


def _config_to_jsonable(obj: Any) -> Any:
    """Convert a dataclass / nested structure to plain JSON-serialisable types.
    Handles ColorRangeMask (a dataclass) and lists/tuples thereof.
    """
    if is_dataclass(obj):
        return {k: _config_to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_config_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _config_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _slugify(label: str, max_len: int = 50) -> str:
    """Filesystem-safe short tag derived from a human label."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")
    return (s[:max_len] or "run").lower()


def _format_overrides_short(overrides: Dict[str, Dict[str, Any]]) -> str:
    """Compact one-line summary of what differs from baseline."""
    parts: List[str] = []
    for section, kvs in overrides.items():
        for k, v in kvs.items():
            if isinstance(v, list):
                if not v:
                    val = "[]"
                else:
                    # Summarise list of ColorRangeMask by name; otherwise stringify.
                    names = [getattr(x, "name", None) for x in v]
                    if all(n is not None for n in names):
                        val = "[" + ",".join(names) + "]"
                    else:
                        val = "[" + ",".join(str(x) for x in v) + "]"
            else:
                val = str(v)
            parts.append(f"{k}={val}")
    return ", ".join(parts) if parts else "(baseline)"


# ══════════════════════════════════════════════════════════════════════════════
# Build run lists from sweep specs
# ══════════════════════════════════════════════════════════════════════════════


def _split_axis_key(key: str) -> Tuple[str, str]:
    """'reconstruct.match_ratio' → ('reconstruct', 'match_ratio')."""
    section, field = key.split(".", 1)
    if section not in ("extract", "reconstruct"):
        raise ValueError(f"axis key must start with 'extract.' or 'reconstruct.': {key}")
    return section, field


def build_curated() -> List[Tuple[str, Dict[str, Dict[str, Any]]]]:
    return list(CURATED_RUNS)


def build_axis() -> List[Tuple[str, Dict[str, Dict[str, Any]]]]:
    runs: List[Tuple[str, Dict[str, Dict[str, Any]]]] = [("baseline", {})]
    for axis_key, values in AXIS_SWEEP.items():
        section, field = _split_axis_key(axis_key)
        for v in values:
            label = f"{field}={v!r}"
            runs.append((label, {section: {field: v}}))
    return runs


def build_grid() -> List[Tuple[str, Dict[str, Dict[str, Any]]]]:
    keys = list(GRID_SWEEP.keys())
    value_lists = [GRID_SWEEP[k] for k in keys]
    runs: List[Tuple[str, Dict[str, Dict[str, Any]]]] = []
    for combo in itertools.product(*value_lists):
        overrides: Dict[str, Dict[str, Any]] = {"extract": {}, "reconstruct": {}}
        label_parts: List[str] = []
        for key, val in zip(keys, combo):
            section, field = _split_axis_key(key)
            overrides[section][field] = val
            if isinstance(val, list):
                tag = "Y" if val else "N"   # binary for list-valued knobs
                label_parts.append(f"{field}={tag}")
            else:
                label_parts.append(f"{field}={val}")
        # Strip empty sections for tidiness.
        overrides = {k: v for k, v in overrides.items() if v}
        runs.append(("__".join(label_parts), overrides))
    return runs


# ══════════════════════════════════════════════════════════════════════════════
# Run one combination
# ══════════════════════════════════════════════════════════════════════════════


def run_one(
    idx: int,
    label: str,
    overrides: Dict[str, Dict[str, Any]],
    video_path: str,
    out_root: Path,
    output_shape: Optional[Tuple[int, int]],
    skip_existing: bool,
) -> Dict[str, Any]:
    """Run a single reconstruction.  Returns a record for the summary.
    Crashes are caught and logged so the rest of the sweep continues."""
    slug = f"{idx:03d}_{_slugify(label)}"
    run_dir = out_root / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    map_path    = run_dir / "map.png"
    config_path = run_dir / "config.json"
    stats_path  = run_dir / "stats.json"

    record: Dict[str, Any] = {
        "idx":       idx,
        "label":     label,
        "slug":      slug,
        "map_path":  str(map_path),
        "overrides": _format_overrides_short(overrides),
        "status":    "pending",
        "elapsed_s": None,
        "placed":    None,
        "failed":    None,
        "keyframes": None,
        "canvas_mb": None,
        "error":     None,
    }

    # Resume: skip if the output already exists and is non-empty.
    if skip_existing and map_path.exists() and map_path.stat().st_size > 0:
        # Try to reload previous stats so the summary still shows them.
        if stats_path.exists():
            try:
                with open(stats_path) as f:
                    prev = json.load(f)
                record.update({k: prev.get(k) for k in ("elapsed_s", "placed", "failed", "keyframes", "canvas_mb")})
            except Exception:
                pass
        record["status"] = "skipped_existing"
        print(f"  [{idx:03d}] skip (exists)  {label}")
        return record

    try:
        ex, rc = _apply_overrides(BASELINE_EXTRACT, BASELINE_RECONSTRUCT, overrides)

        # Dump the exact config used BEFORE running, so a crashed run still
        # leaves a config.json explaining what was attempted.
        with open(config_path, "w") as f:
            json.dump({
                "label":       label,
                "overrides":   _config_to_jsonable(overrides),
                "extract":     _config_to_jsonable(ex),
                "reconstruct": _config_to_jsonable(rc),
            }, f, indent=2)

        print(f"\n  [{idx:03d}] ▶ {label}")
        print(f"        overrides: {_format_overrides_short(overrides)}")

        t0 = time.perf_counter()
        rec = MapReconstructor(rc)
        rec.add_video(video_path, extract_cfg=ex, verbose=False)
        stats_mid = dict(rec.stats)  # capture BEFORE get_map releases canvas
        result = rec.get_map(output_shape=output_shape)
        elapsed = time.perf_counter() - t0

        # Save the map; cv2.imwrite returns False on failure (don't silently lose data).
        ok = cv2.imwrite(str(map_path), result)
        if not ok:
            raise IOError(f"cv2.imwrite failed for {map_path}")

        run_stats = {
            "label":     label,
            "elapsed_s": round(elapsed, 2),
            "placed":    stats_mid.get("placed"),
            "failed":    stats_mid.get("failed"),
            "keyframes": stats_mid.get("keyframes"),
            "canvas_mb": stats_mid.get("canvas_mb"),
            "out_shape": list(result.shape[:2]),
        }
        with open(stats_path, "w") as f:
            json.dump(run_stats, f, indent=2)

        record.update({
            "status":    "ok",
            "elapsed_s": run_stats["elapsed_s"],
            "placed":    run_stats["placed"],
            "failed":    run_stats["failed"],
            "keyframes": run_stats["keyframes"],
            "canvas_mb": run_stats["canvas_mb"],
        })
        print(f"        ✓  placed={run_stats['placed']}  failed={run_stats['failed']}  "
              f"t={elapsed:.1f}s")

    except Exception as exc:
        tb = traceback.format_exc()
        record["status"] = "error"
        record["error"]  = f"{type(exc).__name__}: {exc}"
        with open(run_dir / "error.txt", "w") as f:
            f.write(tb)
        print(f"        ✗  {type(exc).__name__}: {exc}")

    return record


# ══════════════════════════════════════════════════════════════════════════════
# Post-processing: summary table + visual contact sheet
# ══════════════════════════════════════════════════════════════════════════════


def write_summary_md(out_root: Path, records: List[Dict[str, Any]]) -> Path:
    """Markdown table sorted by placed-frame count (best first)."""
    path = out_root / "summary.md"

    def sort_key(r: Dict[str, Any]):
        # Successful runs first, then by placed desc, then failed asc.
        return (
            r["status"] != "ok",
            -(r["placed"] or -1),
            (r["failed"] or 0),
        )

    rows = sorted(records, key=sort_key)
    lines = [
        "# Drone-map parameter sweep — summary",
        "",
        f"Total runs: **{len(records)}**  ·  successful: "
        f"**{sum(1 for r in records if r['status'] == 'ok')}**  ·  "
        f"errors: **{sum(1 for r in records if r['status'] == 'error')}**",
        "",
        "Sorted by placed-frame count (descending).  Visual inspection of the",
        "contact sheet remains the ground-truth quality check — placed-count",
        "alone can favour a run that placed many frames *poorly*.",
        "",
        "| #  | Label | Placed | Failed | Keyframes | t (s) | Status | Overrides |",
        "|---:|:------|------:|------:|---------:|------:|:------|:----------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['idx']:03d} "
            f"| `{r['label']}` "
            f"| {r['placed'] if r['placed'] is not None else '—'} "
            f"| {r['failed'] if r['failed'] is not None else '—'} "
            f"| {r['keyframes'] if r['keyframes'] is not None else '—'} "
            f"| {r['elapsed_s'] if r['elapsed_s'] is not None else '—'} "
            f"| {r['status']} "
            f"| {r['overrides']} |"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def write_contact_sheet(
    out_root: Path,
    records: List[Dict[str, Any]],
    thumb_px: int = 360,
) -> Path:
    """Generate a single HTML page showing every map as a thumbnail next to
    its label and overrides.  Lets you eyeball all runs side-by-side."""
    # Build thumbnails into a sidecar folder so the main runs stay clean.
    thumbs_dir = out_root / "_thumbs"
    thumbs_dir.mkdir(exist_ok=True)

    cards: List[str] = []
    for r in records:
        thumb_rel = ""
        if r["status"] in ("ok", "skipped_existing"):
            src = Path(r["map_path"])
            if src.exists():
                img = cv2.imread(str(src), cv2.IMREAD_COLOR)
                if img is not None:
                    h, w = img.shape[:2]
                    scale = thumb_px / max(h, w)
                    new_w = max(1, int(round(w * scale)))
                    new_h = max(1, int(round(h * scale)))
                    thumb = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    thumb_path = thumbs_dir / f"{r['slug']}.jpg"
                    cv2.imwrite(str(thumb_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    thumb_rel = f"_thumbs/{thumb_path.name}"

        # Use the original full-size map as the linked target so click-to-zoom works.
        full_rel = ""
        if r["map_path"]:
            try:
                full_rel = str(Path(r["map_path"]).relative_to(out_root))
            except ValueError:
                full_rel = r["map_path"]

        stats_line = (
            f"placed={r['placed']}  failed={r['failed']}  "
            f"kf={r['keyframes']}  t={r['elapsed_s']}s"
            if r["status"] == "ok"
            else f"<i>{r['status']}</i>"
        )
        if r["error"]:
            stats_line += f"  <span class='err'>{r['error']}</span>"

        img_html = (
            f'<a href="{full_rel}"><img src="{thumb_rel}" alt=""></a>'
            if thumb_rel
            else '<div class="noimg">no output</div>'
        )

        cards.append(
            f'<div class="card">'
            f'<div class="head">#{r["idx"]:03d} <b>{r["label"]}</b></div>'
            f'{img_html}'
            f'<div class="meta">{stats_line}</div>'
            f'<div class="ov"><code>{r["overrides"]}</code></div>'
            f'</div>'
        )

    html = f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Drone-map sweep — contact sheet</title>
<style>
  body {{ font: 13px/1.35 system-ui, sans-serif; background:#111; color:#ddd;
         margin: 16px; }}
  h1 {{ font-size: 18px; margin: 0 0 12px; color:#fff; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
           gap: 14px; }}
  .card {{ background:#1c1c1c; border:1px solid #2a2a2a; border-radius:6px;
           padding:10px; }}
  .card .head {{ margin-bottom:6px; color:#fff; }}
  .card img  {{ width:100%; height:auto; display:block; background:#000;
                border-radius:4px; }}
  .noimg {{ height:200px; display:flex; align-items:center; justify-content:center;
            color:#666; background:#000; border-radius:4px; }}
  .meta {{ margin-top:6px; color:#aaa; font-family: ui-monospace, monospace; }}
  .ov   {{ margin-top:4px; color:#888; word-break: break-all; }}
  .err  {{ color:#e88; }}
  code  {{ color:#9cf; }}
</style>
</head><body>
<h1>Drone-map sweep — {len(records)} runs</h1>
<div class="grid">
{''.join(cards)}
</div>
</body></html>
"""
    path = out_root / "contact_sheet.html"
    path.write_text(html)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sweep drone-map reconstruction parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--video",
        default="drone_map_create/data/drone_scans/scan2/scan.mp4",
        help="Input drone video path.",
    )
    ap.add_argument(
        "--out",
        default="drone_map_create/out/sweep_runs",
        help="Output directory root.  Each run gets a subdirectory.",
    )
    ap.add_argument(
        "--mode",
        choices=["curated", "axis", "grid"],
        default="curated",
        help="Which sweep specification to use.",
    )
    ap.add_argument(
        "--output-shape",
        nargs=2, type=int, metavar=("W", "H"),
        default=[2000, 2000],
        help="Resize the final map to this (W, H).  Use 0 0 for native size.",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="If set, only the first N runs are executed (smoke test).",
    )
    ap.add_argument(
        "--no-skip-existing", action="store_true",
        help="Re-run combinations even if their output PNG already exists.",
    )
    ap.add_argument(
        "--summary-only", action="store_true",
        help="Skip execution; just rebuild summary.md and contact_sheet.html "
             "from existing run directories.",
    )
    args = ap.parse_args()

    out_shape = tuple(args.output_shape)
    if out_shape == (0, 0):
        out_shape = None

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if not Path(args.video).exists():
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        return 2

    # Build the run list.
    if args.mode == "curated":
        runs = build_curated()
    elif args.mode == "axis":
        runs = build_axis()
    elif args.mode == "grid":
        runs = build_grid()
    else:
        raise ValueError(args.mode)

    if args.limit is not None:
        runs = runs[: args.limit]

    print(f"\n  Sweep mode: {args.mode}   total runs: {len(runs)}")
    print(f"  Video:      {args.video}")
    print(f"  Output:     {out_root.resolve()}")
    print(f"  Out shape:  {out_shape}")
    print()

    # Execute (unless --summary-only) and collect records.
    records: List[Dict[str, Any]] = []
    if args.summary_only:
        # Rebuild records by reading stats.json from existing dirs.
        for idx, (label, overrides) in enumerate(runs, 1):
            slug = f"{idx:03d}_{_slugify(label)}"
            run_dir = out_root / slug
            stats_path = run_dir / "stats.json"
            rec: Dict[str, Any] = {
                "idx": idx, "label": label, "slug": slug,
                "map_path": str(run_dir / "map.png"),
                "overrides": _format_overrides_short(overrides),
                "status": "missing", "elapsed_s": None,
                "placed": None, "failed": None, "keyframes": None,
                "canvas_mb": None, "error": None,
            }
            if stats_path.exists():
                try:
                    with open(stats_path) as f:
                        prev = json.load(f)
                    rec.update({k: prev.get(k) for k in
                                ("elapsed_s", "placed", "failed", "keyframes", "canvas_mb")})
                    rec["status"] = "ok"
                except Exception as exc:
                    rec["error"] = str(exc)
            records.append(rec)
    else:
        for idx, (label, overrides) in enumerate(runs, 1):
            rec = run_one(
                idx=idx,
                label=label,
                overrides=overrides,
                video_path=args.video,
                out_root=out_root,
                output_shape=out_shape,
                skip_existing=not args.no_skip_existing,
            )
            records.append(rec)

            # Re-emit summary + contact sheet AFTER EACH run so partial results
            # are usable if the sweep is interrupted (Ctrl-C, OOM, etc).
            try:
                write_summary_md(out_root, records)
                write_contact_sheet(out_root, records)
            except Exception as exc:
                print(f"  [warn] partial-summary regen failed: {exc}")

    # Final emit (also covers --summary-only path).
    sm = write_summary_md(out_root, records)
    cs = write_contact_sheet(out_root, records)
    print()
    print(f"  Summary  → {sm}")
    print(f"  Contacts → {cs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
