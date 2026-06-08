#!/usr/bin/env python3
"""
label.py  –  interactive pass / fail / unsure labeler for sweep results
═══════════════════════════════════════════════════════════════════════
Opens each run's transfer_input.png (left) and 08_occupancy_vis.png
(right) side by side.  Press a key to record the label, then the next
unlabeled run appears automatically.  Progress is saved after every
decision — safe to quit and resume at any time.

Controls
────────
  p / 1 / Enter   → PASS    (green header)
  f / 2           → FAIL    (red header)
  s / Space       → UNSURE  (yellow header) — excluded from XGBoost training by default
  u / z           → UNDO last label (goes back one run)
  q / Esc         → quit and save

Usage
─────
  python3 label.py --results-dirs sweep/results
  python3 label.py --results-dirs sweep/results1 sweep/results2 --labels-file labels.yaml
  python3 label.py --results-dirs sweep/results --revisit   # re-label already-labeled runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import yaml

# ══════════════════════════════════════════════════════════════════════════════
# Display settings
# ══════════════════════════════════════════════════════════════════════════════

_WIN          = "label"
_MAX_W        = 1900
_MAX_H        = 980
_SEP_W        = 6       # px width of the divider between panels
_TOP_H        = 42      # header bar height
_BOT_H        = 44      # footer bar height
_IMG_NAMES    = [       # tried in order for each panel
    ("transfer_input.png",   "Stitched map"),
    ("08_occupancy_vis.png", "Occupancy grid"),
]
_FALLBACK     = "stitched_map.png"

_LABEL_COLORS = {
    "pass":   (30,  160,  30),   # green
    "fail":   (20,   20, 180),   # red (BGR)
    "unsure": (20,  160, 200),   # yellow (BGR)
    None:     (50,   50,  50),   # dark grey = unlabeled
}

# Config keys shown in the footer for quick context
_SHOW_CFG = [
    "feature_extractor", "feature_matcher", "use_grid_intersections",
    "use_pose_graph", "use_fiducials", "match_ratio", "mad_factor",
    "lookback", "processing_scale",
]

# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_run(run_dir: Path, sweep_name: str) -> Optional[dict]:
    cfg_path = run_dir / "config.yaml"
    met_path = run_dir / "metrics.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    metrics = {}
    if met_path.exists():
        with open(met_path) as f:
            metrics = yaml.safe_load(f) or {}
    if metrics.get("status") not in ("ok", None, ""):
        # still include runs whose status isn't "ok" — user might want to label
        # them as fail explicitly rather than silently skip them.
        pass
    return {
        "sweep":   sweep_name,
        "run_id":  cfg.get("run_id", run_dir.name),
        "group":   cfg.get("group", "?"),
        "dir":     run_dir,
        "config":  cfg,
        "metrics": metrics,
    }


def load_all_runs(results_dirs: list[Path]) -> list[dict]:
    """Load all run directories across all sweep result directories.

    Sorted by (sweep_name, run_dir_name) for deterministic, resumable order.
    """
    runs = []
    for rd in sorted(results_dirs, key=lambda p: p.name):
        sweep_name = rd.name
        for d in sorted(rd.iterdir()):
            if not d.is_dir() or not d.name.startswith("run_"):
                continue
            run = _load_run(d, sweep_name)
            if run:
                runs.append(run)
    return runs


def run_key(run: dict) -> str:
    """Globally unique label key: '{sweep_basename}/{run_id}'.

    Note: if two sweep directories share the same basename AND the same
    run_id, their labels will collide.  Keep sweep directory names distinct.
    """
    return f"{run['sweep']}/{run['run_id']}"


# ══════════════════════════════════════════════════════════════════════════════
# Label I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_labels(path: Path) -> Dict[str, str]:
    path = Path(path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return data
    return {}


def save_labels(path: Path, labels: Dict[str, str]) -> None:
    with open(path, "w") as f:
        yaml.dump(labels, f, default_flow_style=False, sort_keys=True)


# ══════════════════════════════════════════════════════════════════════════════
# Image loading
# ══════════════════════════════════════════════════════════════════════════════

_img_cache: dict[str, np.ndarray] = {}

def _load_image(path: Path) -> Optional[np.ndarray]:
    key = str(path)
    if key in _img_cache:
        return _img_cache[key]
    img = cv2.imread(key)
    if img is not None:
        _img_cache[key] = img
    return img


def _placeholder(text: str, w: int, h: int) -> np.ndarray:
    img = np.full((h, w, 3), 25, dtype=np.uint8)
    cv2.putText(img, text, (w // 2 - len(text) * 7, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1, cv2.LINE_AA)
    return img


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Scale image to fit within (w, h), maintain aspect ratio, dark-pad."""
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# Display frame construction
# ══════════════════════════════════════════════════════════════════════════════

def _build_frame(
    run: dict,
    pointer: int,
    total: int,
    existing_label: Optional[str],
) -> np.ndarray:
    panel_w = (_MAX_W - _SEP_W) // 2
    panel_h = _MAX_H - _TOP_H - _BOT_H

    # Load the two display images
    imgs = []
    for filename, caption in _IMG_NAMES:
        img = _load_image(run["dir"] / filename)
        if img is None and filename == "transfer_input.png":
            img = _load_image(run["dir"] / _FALLBACK)
        if img is None:
            img = _placeholder(f"{caption}\n(not found)", panel_w, panel_h)
        imgs.append((caption, _fit(img, panel_w, panel_h)))

    # Build each panel
    def _panel(caption: str, img: np.ndarray) -> np.ndarray:
        p = img.copy()
        cv2.putText(p, caption, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
        return p

    left  = _panel(imgs[0][0], imgs[0][1])
    right = _panel(imgs[1][0], imgs[1][1])
    sep   = np.full((panel_h, _SEP_W, 3), 50, dtype=np.uint8)
    body  = np.hstack([left, sep, right])

    # Header bar
    label_color = _LABEL_COLORS.get(existing_label, _LABEL_COLORS[None])
    header = np.full((_TOP_H, _MAX_W, 3), label_color, dtype=np.uint8)
    label_str = existing_label.upper() if existing_label else "UNLABELED"
    htext = (f"[{pointer + 1}/{total}]  "
             f"{run['sweep']} / {run['run_id']}  (group {run['group']})  "
             f"─  {label_str}")
    cv2.putText(header, htext, (10, _TOP_H - 12),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)

    # Footer bar — key config params + controls hint
    footer = np.full((_BOT_H, _MAX_W, 3), 18, dtype=np.uint8)
    cfg = run["config"]
    cfg_str = "  ".join(
        f"{k}={cfg[k]}" for k in _SHOW_CFG if k in cfg
    )
    cv2.putText(footer, cfg_str[:140], (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)
    controls = "p/1/Enter=PASS   f/2=FAIL   s/Space=UNSURE   u/z=UNDO   q/Esc=QUIT"
    cv2.putText(footer, controls, (10, _BOT_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    return np.vstack([header, body, footer])


# ══════════════════════════════════════════════════════════════════════════════
# Main labeling loop
# ══════════════════════════════════════════════════════════════════════════════

_PASS_KEYS   = {ord('p'), ord('1'), 13}    # p, 1, Enter
_FAIL_KEYS   = {ord('f'), ord('2')}
_UNSURE_KEYS = {ord('s'), 32}              # s, Space
_UNDO_KEYS   = {ord('u'), ord('z')}
_QUIT_KEYS   = {ord('q'), 27}              # q, Esc


def run_labeler(
    pending: list[dict],
    labels: Dict[str, str],
    labels_path: Path,
) -> None:
    """Interactive labeling loop. Mutates `labels` in place and saves after
    every decision.

    `pending` is the ordered list of runs to label (pre-filtered by the
    caller; already-labeled runs are excluded unless --revisit was passed).
    """
    if not pending:
        print("  Nothing to label.")
        return

    cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WIN, _MAX_W, _MAX_H)

    # history: list of (pointer_before, key, label_before)
    history: list[tuple[int, str, Optional[str]]] = []
    pointer = 0

    print(f"\n  {len(pending)} run(s) to label. Controls: "
          f"p=PASS  f=FAIL  s=UNSURE  u=UNDO  q=QUIT\n")

    while pointer < len(pending):
        run = pending[pointer]
        key = run_key(run)
        existing = labels.get(key)

        frame = _build_frame(run, pointer, len(pending), existing)
        cv2.imshow(_WIN, frame)

        # Input loop — stay here until a valid key is pressed
        while True:
            k = cv2.waitKey(0) & 0xFF
            # Window closed with X button
            if cv2.getWindowProperty(_WIN, cv2.WND_PROP_VISIBLE) < 1:
                k = ord('q')

            if k in _QUIT_KEYS:
                save_labels(labels_path, labels)
                cv2.destroyAllWindows()
                _print_summary(labels)
                return

            if k in _PASS_KEYS:
                history.append((pointer, key, labels.get(key)))
                labels[key] = "pass"
                save_labels(labels_path, labels)
                print(f"  PASS   {key}")
                pointer += 1
                break

            if k in _FAIL_KEYS:
                history.append((pointer, key, labels.get(key)))
                labels[key] = "fail"
                save_labels(labels_path, labels)
                print(f"  FAIL   {key}")
                pointer += 1
                break

            if k in _UNSURE_KEYS:
                history.append((pointer, key, labels.get(key)))
                labels[key] = "unsure"
                save_labels(labels_path, labels)
                print(f"  UNSURE {key}")
                pointer += 1
                break

            if k in _UNDO_KEYS:
                if history:
                    prev_ptr, prev_key, prev_label = history.pop()
                    if prev_label is None:
                        labels.pop(prev_key, None)
                    else:
                        labels[prev_key] = prev_label
                    save_labels(labels_path, labels)
                    pointer = prev_ptr
                    print(f"  UNDO   {prev_key} → "
                          f"{prev_label or 'unlabeled'}")
                    break  # re-render the current pointer
                else:
                    print("  Nothing to undo.")
                # stay in input loop on failed undo

    cv2.destroyAllWindows()
    print("\n  All runs labeled.")
    _print_summary(labels)
    save_labels(labels_path, labels)


def _print_summary(labels: Dict[str, str]) -> None:
    counts: Dict[str, int] = {}
    for v in labels.values():
        counts[v] = counts.get(v, 0) + 1
    total = sum(counts.values())
    print(f"\n  Labels saved: {total} total  |  " +
          "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive pass/fail/unsure labeler for sweep results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dirs", nargs="+", required=True,
        help="One or more sweep result directories (e.g. sweep/results1 sweep/results2)",
    )
    parser.add_argument(
        "--labels-file", default="labels.yaml",
        help="Path to the global labels YAML file (default: labels.yaml)",
    )
    parser.add_argument(
        "--revisit", action="store_true",
        help="Show already-labeled runs too (re-labeling overwrites)",
    )
    args = parser.parse_args()

    results_dirs = [Path(d) for d in args.results_dirs]
    labels_path  = Path(args.labels_file)

    for rd in results_dirs:
        if not rd.is_dir():
            print(f"[ERROR] Not a directory: {rd}")
            sys.exit(1)

    print(f"\n  Loading runs from {[str(d) for d in results_dirs]} …")
    all_runs = load_all_runs(results_dirs)
    print(f"  {len(all_runs)} run(s) found across {len(results_dirs)} sweep(s).")

    labels = load_labels(labels_path)
    print(f"  {len(labels)} existing label(s) loaded from {labels_path}.")

    if args.revisit:
        pending = all_runs
    else:
        pending = [r for r in all_runs if run_key(r) not in labels]
        skipped = len(all_runs) - len(pending)
        if skipped:
            print(f"  {skipped} already-labeled run(s) skipped "
                  f"(use --revisit to re-label).")

    print(f"  {len(pending)} run(s) to label.\n")

    run_labeler(pending, labels, labels_path)


if __name__ == "__main__":
    main()