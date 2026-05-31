#!/usr/bin/env python3
"""
rank.py  –  interactive binary-comparison ranker for sweep results
══════════════════════════════════════════════════════════════════
Finds the top-N configurations from 80 sweep runs using sorted insertion:
  • Initialize top-N list by insertion-sorting the first N runs  (~7 comparisons)
  • For each remaining run: compare against current Nth place (1 comparison)
    – if it loses → discarded in 1 comparison
    – if it wins  → binary-search insert into the ranked list (2–3 more)
  • Most runs are eliminated in exactly 1 comparison once the list is populated

Expected: ~80–120 total comparisons for top-5 from 80 runs.

Controls (in the image window)
──────────────────────────────
  ←  left arrow   pick LEFT  (or press 1 / a)
  →  right arrow  pick RIGHT (or press 2 / d)
  SPACE / e       both are equal (tied)
  u               undo last comparison
  q / ESC         quit and save

Progress is saved after every comparison — safe to quit and resume.

Usage
─────
  python3 sweep/rank.py
  python3 sweep/rank.py --results-dir sweep/results --top 5
  python3 sweep/rank.py --resume     # same as default: auto-resumes if state exists
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

_HERE    = Path(__file__).resolve().parent
_TIE     = object()   # sentinel returned by _compare when both runs are equal
_DEFAULT = _HERE / "results"

# Images to display for each run (in vertical order within each panel)
# Falls back to the next option if a file is missing.
_DISPLAY_IMAGES = [
    ("transfer_input.png",    "Input"),
    ("transfer_final.png",    "Final"),
]
_FALLBACK_IMAGE = "stitched_map.png"

# Config keys printed in the terminal comparison table
_SHOW_CONFIG = [
    "feature_extractor", "feature_matcher", "use_grid_intersections",
    "use_pose_graph", "use_fiducials", "feature_exclude_hsv",
    "match_ratio", "mad_factor", "lookback", "keyframe_interval",
    "pg_marker_weight", "pg_huber_delta", "target_fps",
]

# Window display settings
_MAX_DISPLAY_W = 1900   # total window width cap (px)
_MAX_DISPLAY_H = 980    # total window height cap (px)
_SEPARATOR_W   = 8      # px width of the dividing bar


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_run_data(run_dir: Path) -> Optional[dict]:
    """Load config + metrics for one run directory. Returns None if invalid."""
    config_path  = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.yaml"
    if not config_path.exists() or not metrics_path.exists():
        return None
    with open(config_path)  as f: config  = yaml.safe_load(f) or {}
    with open(metrics_path) as f: metrics = yaml.safe_load(f) or {}
    if metrics.get("status") != "ok":
        return None
    return {
        "run_id":  config.get("run_id", run_dir.name),
        "group":   config.get("group", "?"),
        "dir":     run_dir,
        "config":  config,
        "metrics": metrics,
    }


def load_all_runs(results_dir: Path) -> list[dict]:
    runs = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("run_"):
            continue
        data = _load_run_data(d)
        if data:
            runs.append(data)
    return runs


# ══════════════════════════════════════════════════════════════════════════════
# Image loading and layout
# ══════════════════════════════════════════════════════════════════════════════

_img_cache: dict[str, np.ndarray] = {}

def _load_image(run: dict, filename: str) -> Optional[np.ndarray]:
    path = str(run["dir"] / filename)
    if path in _img_cache:
        return _img_cache[path]
    img = cv2.imread(path)
    if img is not None:
        _img_cache[path] = img
    return img


def _get_run_images(run: dict) -> list[tuple[str, np.ndarray]]:
    """Return list of (label, image) for display. Falls back gracefully."""
    result = []
    for filename, label in _DISPLAY_IMAGES:
        img = _load_image(run, filename)
        if img is not None:
            result.append((label, img))
    if not result:
        img = _load_image(run, _FALLBACK_IMAGE)
        if img is not None:
            result.append(("Map", img))
        else:
            h, w = 400, 600
            placeholder = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No image", (w//4, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)
            result.append(("Missing", placeholder))
    return result


def _build_comparison_frame(left: dict, right: dict,
                             comp_n: int, total_est: int,
                             ranked: list[dict]) -> np.ndarray:
    """Build the full comparison image for cv2.imshow."""
    li = _get_run_images(left)
    ri = _get_run_images(right)

    n_rows = max(len(li), len(ri))
    panel_w = (_MAX_DISPLAY_W - _SEPARATOR_W) // 2
    row_h   = _MAX_DISPLAY_H // n_rows

    def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
        """Scale image to fit in w×h, maintaining aspect ratio, dark-padded."""
        ih, iw = img.shape[:2]
        scale  = min(w / iw, h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas  = np.zeros((h, w, 3), dtype=np.uint8)
        y0 = (h - nh) // 2
        x0 = (w - nw) // 2
        canvas[y0:y0+nh, x0:x0+nw] = resized
        return canvas

    def _make_panel(run: dict, images: list[tuple[str, np.ndarray]],
                    side: str) -> np.ndarray:
        panel_h = n_rows * row_h
        panel   = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        for row_i in range(n_rows):
            y0 = row_i * row_h
            if row_i < len(images):
                label, img = images[row_i]
                cell = _fit(img, panel_w, row_h)
                panel[y0:y0+row_h] = cell
                # Label overlay
                cv2.putText(panel, label, (10, y0 + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        # Run ID header bar
        bar_h = 36
        cv2.rectangle(panel, (0, 0), (panel_w, bar_h), (40, 40, 80), -1)
        arrow  = "← LEFT" if side == "left" else "RIGHT →"
        run_id = run["run_id"]
        grp    = run["group"]
        header = f"{arrow}  |  {run_id} (group {grp})"
        cv2.putText(panel, header, (10, bar_h - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 220, 100), 1)
        # Bottom hint bar
        hint_y = panel_h - 28
        cv2.rectangle(panel, (0, hint_y - 4), (panel_w, panel_h), (20, 20, 20), -1)
        hint = ("Press  ←  for this run" if side == "left"
                else "Press  →  for this run  |  SPACE = equal")
        cv2.putText(panel, hint, (10, panel_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 150, 150), 1)
        return panel

    left_panel  = _make_panel(left,  li, "left")
    right_panel = _make_panel(right, ri, "right")
    sep  = np.full((n_rows * row_h, _SEPARATOR_W, 3), 60, dtype=np.uint8)

    frame = np.hstack([left_panel, sep, right_panel])

    # Top status bar across full width
    status_h = 30
    bar = np.full((status_h, frame.shape[1], 3), 25, dtype=np.uint8)
    ranked_ids = ", ".join(r["run_id"] for r in ranked[:5]) if ranked else "—"
    status_txt = (f"Comparison {comp_n} / ~{total_est}    "
                  f"Top-{len(ranked)} so far: {ranked_ids}    "
                  f"[u=undo  q=quit]")
    cv2.putText(bar, status_txt, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return np.vstack([bar, frame])


# ══════════════════════════════════════════════════════════════════════════════
# Terminal output
# ══════════════════════════════════════════════════════════════════════════════

def _print_comparison(left: dict, right: dict,
                       comp_n: int, total_est: int,
                       ranked: list[dict], phase: str) -> None:
    W = 70
    print(f"\n{'═'*W}")
    print(f"  Comparison {comp_n} / ~{total_est}   Phase: {phase}")
    ranked_ids = "  ".join(
        f"#{i+1} {r['run_id']}" for i, r in enumerate(ranked)
    )
    print(f"  Top so far: {ranked_ids or '(building…)'}")
    print(f"{'═'*W}")

    lc, lm = left["config"],  left["metrics"]
    rc, rm = right["config"], right["metrics"]

    col = 30
    print(f"\n  {'← LEFT':^{col}}     {'RIGHT →':^{col}}")
    print(f"  {'Run ' + left['run_id']:^{col}}     {'Run ' + right['run_id']:^{col}}")
    print(f"  {'─'*col}     {'─'*col}")

    for key in _SHOW_CONFIG:
        lv = str(lc.get(key, "—"))
        rv = str(rc.get(key, "—"))
        marker = "  " if lv == rv else "* "
        print(f"  {marker}{key:<{col-2}} {lv:<{col}}     {rv}")

    print(f"  {'─'*col}     {'─'*col}")
    for key in ("n_obstacles", "mean_consistency", "runtime_s"):
        lv = str(lm.get(key, "—"))
        rv = str(rm.get(key, "—"))
        print(f"  {key:<{col}} {lv:<{col}}     {rv}")

    print(f"\n  ← left wins   SPACE/e = equal   → right wins   "
          f"u = undo   q = quit\n")


# ══════════════════════════════════════════════════════════════════════════════
# Ranking algorithm: sorted insertion
# ══════════════════════════════════════════════════════════════════════════════

def _run_comparison(left: dict, right: dict,
                    comp_n: int, total_est: int,
                    ranked: list[dict], phase: str) -> Optional[str]:
    """
    Show the comparison and wait for a keypress.
    Returns "left", "right", "undo", or "quit".
    """
    _print_comparison(left, right, comp_n, total_est, ranked, phase)
    frame = _build_comparison_frame(left, right, comp_n, total_est, ranked)
    cv2.imshow("Sweep Ranker  —  pick the better map", frame)

    while True:
        key = cv2.waitKeyEx(0)
        if key in (65361, ord('1'), ord('a')):          # left arrow / 1 / a
            return "left"
        if key in (65363, ord('2'), ord('d')):          # right arrow / 2 / d
            return "right"
        if key in (ord(' '), ord('e'), 13, 10):         # space / e / Enter
            return "equal"
        if key == ord('u'):
            return "undo"
        if key in (ord('q'), 27):                       # q / ESC
            return "quit"


class Ranker:
    """
    Sorted-insertion ranker. Maintains a `ranked` list (best→worst, len ≤ top_n)
    and a `pool` of items not yet fully evaluated.

    State is persisted to a YAML file after every comparison so the session
    can be resumed at any point.
    """

    def __init__(self, runs: list[dict], top_n: int, state_path: Path):
        self.top_n      = top_n
        self.state_path = state_path
        self._all_runs  = {r["run_id"]: r for r in runs}

        # Try to resume from saved state
        if state_path.exists():
            self._load_state()
            print(f"  Resumed: {len(self.ranked)} ranked, "
                  f"{len(self.pool)} in pool, "
                  f"{self.comp_n} comparisons so far.")
        else:
            rng = random.Random(42)
            shuffled = list(runs)
            rng.shuffle(shuffled)
            self.ranked:   list[dict] = []
            self.pool:     list[dict] = shuffled
            self.comp_n:   int        = 0
            self.log:      list[dict] = []
            self._history: list[dict] = []   # for undo

    # ── state persistence ──────────────────────────────────────────────────

    def _save_state(self) -> None:
        state = {
            "ranked":   [r["run_id"] for r in self.ranked],
            "pool":     [r["run_id"] for r in self.pool],
            "comp_n":   self.comp_n,
            "log":      self.log,
        }
        with open(self.state_path, "w") as f:
            yaml.dump(state, f, default_flow_style=False, sort_keys=False)

    def _load_state(self) -> None:
        with open(self.state_path) as f:
            state = yaml.safe_load(f) or {}
        def _get(rid):
            return self._all_runs.get(rid)
        self.ranked  = [r for rid in state.get("ranked", []) if (r := _get(rid))]
        self.pool    = [r for rid in state.get("pool",   []) if (r := _get(rid))]
        self.comp_n  = state.get("comp_n", 0)
        self.log     = state.get("log", [])
        self._history = []

    # ── estimated total comparisons ────────────────────────────────────────

    @property
    def total_est(self) -> int:
        # Initializing top_n: ~top_n*(top_n-1)/2  (average insertion-sort)
        # Remaining: ~len(pool) * 1.3 (most eliminated in 1 comparison)
        init = self.top_n * (self.top_n - 1) // 2
        return init + int(len(self.pool) * 1.3)

    # ── public interface ───────────────────────────────────────────────────

    def run(self) -> list[dict]:
        """
        Drive the full ranking session. Returns the final ranked list.
        Calls _run_comparison() for each head-to-head.
        """
        cv2.namedWindow("Sweep Ranker  —  pick the better map",
                        cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Sweep Ranker  —  pick the better map",
                         _MAX_DISPLAY_W, _MAX_DISPLAY_H + 30)

        # ── Phase 1: fill the ranked list ──────────────────────────────────
        while len(self.ranked) < self.top_n and self.pool:
            challenger = self.pool.pop(0)
            self._insert_into_ranked(challenger)
            self._save_state()
            if len(self.ranked) == self.top_n:
                print(f"\n  Initial top-{self.top_n} established after "
                      f"{self.comp_n} comparisons. Now evaluating remaining "
                      f"{len(self.pool)} runs.\n")

        # ── Phase 2: challenge each remaining item against Nth place ───────
        while self.pool:
            challenger = self.pool.pop(0)
            phase  = f"Challenge  ({len(self.pool)} remaining in pool)"
            result = self._compare(self.ranked[-1], challenger, phase)

            if result is None:       # quit
                break
            elif result is _TIE:
                # Tie with Nth place: both belong in the list; keep the incumbent
                # and binary-search insert the challenger (it will land at the same rank)
                self._insert_into_ranked(challenger)
            elif result is challenger:
                # Challenger beats Nth place: evict incumbent, insert challenger
                self.ranked.pop()
                self._insert_into_ranked(challenger)
            # else: Nth place wins; challenger discarded

            self._save_state()

        cv2.destroyAllWindows()
        return self.ranked

    # ── internal comparison helpers ────────────────────────────────────────

    def _compare(self, a: dict, b: dict, phase: str) -> Optional[dict]:
        """
        Present a vs b; return the winner, None on quit, or replay after undo.
        """
        while True:
            self.comp_n += 1
            decision = _run_comparison(
                a, b, self.comp_n, self.total_est, self.ranked, phase
            )

            if decision == "quit":
                self.comp_n -= 1
                return None

            if decision == "undo":
                self.comp_n -= 1
                if self._undo():
                    self.comp_n -= 1   # undo decrements again for the undone comparison
                    return None        # caller must handle undo by re-running
                print("  Nothing to undo.")
                continue

            if decision == "equal":
                entry = {"comp": self.comp_n, "left": a["run_id"],
                         "right": b["run_id"], "winner": "tie"}
                self.log.append(entry)
                self._history.append({
                    "ranked": list(self.ranked),
                    "pool":   list(self.pool),
                    "comp_n": self.comp_n,
                    "log":    list(self.log),
                })
                print(f"  → {a['run_id']} = {b['run_id']}  (tied)")
                return _TIE

            winner = a if decision == "left" else b
            loser  = b if decision == "left" else a
            entry  = {"comp": self.comp_n, "left": a["run_id"],
                      "right": b["run_id"], "winner": winner["run_id"]}
            self.log.append(entry)
            self._history.append({
                "ranked": list(self.ranked),
                "pool":   list(self.pool),
                "comp_n": self.comp_n,
                "log":    list(self.log),
            })
            print(f"  → {winner['run_id']} wins over {loser['run_id']}")
            return winner

    def _insert_into_ranked(self, new_run: dict) -> None:
        """
        Binary-search insert new_run into self.ranked (sorted best → worst).
        Truncates to top_n after insertion.
        """
        lo, hi = 0, len(self.ranked)
        while lo < hi:
            mid    = (lo + hi) // 2
            phase  = f"Insert #{lo}–#{hi} (finding position in top-{self.top_n})"
            result = self._compare(new_run, self.ranked[mid], phase)
            if result is None:       # quit requested
                self.pool.insert(0, new_run)
                return
            if result is _TIE:       # new_run == ranked[mid]: insert right here
                lo = mid
                break
            if result is new_run:    # new_run beats ranked[mid]
                hi = mid
            else:                    # ranked[mid] beats new_run
                lo = mid + 1
        self.ranked.insert(lo, new_run)
        if len(self.ranked) > self.top_n:
            # Evict the item that got bumped past position top_n, UNLESS the
            # new run tied with the last item specifically (lo >= top_n-1), in
            # which case both belong at the boundary and the list grows by 1.
            if lo < self.top_n - 1:
                evicted = self.ranked.pop()
                print(f"  {evicted['run_id']} evicted from top-{self.top_n}")

    def _undo(self) -> bool:
        if not self._history:
            return False
        prev = self._history.pop()
        self.ranked  = prev["ranked"]
        self.pool    = prev["pool"]
        self.comp_n  = prev["comp_n"]
        self.log     = prev["log"]
        self._save_state()
        print(f"  Undone. Back to comparison {self.comp_n}.")
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Final report
# ══════════════════════════════════════════════════════════════════════════════

def _print_final(ranked: list[dict], comp_n: int, results_dir: Path) -> None:
    W = 70
    print(f"\n{'═'*W}")
    print(f"  FINAL RANKING  ({comp_n} comparisons)")
    print(f"{'═'*W}\n")
    for i, run in enumerate(ranked):
        m = run["metrics"]
        c = run["config"]
        print(f"  #{i+1}  {run['run_id']}  (group {run['group']})")
        print(f"       ext={c.get('feature_extractor','?')}  "
              f"mat={c.get('feature_matcher','?')}  "
              f"grid={c.get('use_grid_intersections','?')}  "
              f"pg={c.get('use_pose_graph','?')}")
        print(f"       mr={c.get('match_ratio','?')}  "
              f"mf={c.get('mad_factor','?')}  "
              f"lb={c.get('lookback','?')}  "
              f"ki={c.get('keyframe_interval','?')}")
        print(f"       obstacles={m.get('n_obstacles','?')}  "
              f"consistency={m.get('mean_consistency', 0.0):.3f}  "
              f"runtime={m.get('runtime_s','?')}s")
        print()

    # Save final ranking to YAML
    out_path = results_dir / "final_ranking.yaml"
    ranked_out = []
    for i, run in enumerate(ranked):
        ranked_out.append({
            "rank":    i + 1,
            "run_id":  run["run_id"],
            "group":   run["group"],
            "config":  {k: v for k, v in run["config"].items()
                        if k not in ("run_id", "group")},
            "metrics": run["metrics"],
        })
    with open(out_path, "w") as f:
        yaml.dump({"total_comparisons": comp_n,
                   "ranking": ranked_out},
                  f, default_flow_style=False, sort_keys=False)
    print(f"  Ranking saved to: {out_path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive binary-comparison ranker for sweep results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", default=str(_DEFAULT),
                        help=f"Results directory (default: {_DEFAULT})")
    parser.add_argument("--top", type=int, default=5,
                        help="Number of runs to rank (default: 5)")
    parser.add_argument("--reset", action="store_true",
                        help="Ignore saved ranking state and start fresh")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    state_path  = results_dir / "ranking_state.yaml"

    if args.reset and state_path.exists():
        state_path.unlink()
        print("  Ranking state cleared. Starting fresh.")

    print(f"\n  Loading runs from {results_dir}…")
    runs = load_all_runs(results_dir)
    print(f"  {len(runs)} successful runs found.\n")

    if len(runs) < args.top:
        print(f"  Need at least {args.top} runs; only {len(runs)} found. Exiting.")
        sys.exit(1)

    ranker = Ranker(runs, top_n=args.top, state_path=state_path)
    ranked = ranker.run()

    _print_final(ranked, ranker.comp_n, results_dir)


if __name__ == "__main__":
    main()
