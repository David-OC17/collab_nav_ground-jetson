#!/usr/bin/env python3
"""
sweep.py  –  overnight parameter sweep for build_arena_map_server
══════════════════════════════════════════════════════════════════
Runs all 83 configurations from run_matrix.py in order.
Progress is saved after every run; the script is safe to kill and restart
at any time.

Prerequisites
─────────────
  source /home/jetson/collab_nav_ground-jetson/install/setup.bash
  # server is started automatically by this script unless --no-server is given

Usage
─────
  python3 sweep/sweep.py                          # start fresh or resume
  python3 sweep/sweep.py --dry-run                # show plan, don't execute
  python3 sweep/sweep.py --no-server              # server already running
  python3 sweep/sweep.py --output-dir /my/dir     # custom results directory
  python3 sweep/sweep.py --retry-failed           # re-run previously failed IDs
  python3 sweep/sweep.py --similarity /path/to/gt.png  # enable SSIM metric
  python3 sweep/sweep.py --timeout 2400           # per-run timeout (seconds)
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from action_msgs.msg import GoalStatus

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from run_matrix_v3 import RUNS

# ── optional diagnostics integration ─────────────────────────────────────────
# Requires transfer_obstacles.py and map_diagnostics.py to be importable.
# If they aren't (e.g. running without the full source tree), diagnostics
# are silently skipped and metrics.yaml only contains the action result fields.
_DIAG_AVAILABLE = False
try:
    from transfer_obstacles import run_pipeline as _run_pipeline
    from transfer_obstacles import TransferConfig as _TransferConfig
    from map_diagnostics import compute_stitcher_diagnostics as _compute_stitcher_diag
    _DIAG_AVAILABLE = True
except ImportError as _diag_import_err:
    print(f"[sweep] diagnostics unavailable ({_diag_import_err}) "
          f"— metrics.yaml will not contain diagnostic features")

from arena_map_builder_msgs.action import BuildArenaMap

# ══════════════════════════════════════════════════════════════════════════════
# Fixed inputs and server name
# ══════════════════════════════════════════════════════════════════════════════

VIDEO_PATH      = "/home/jetson/collab_nav_ground-jetson/src/arena_map_builder/data/drone_scans/scan18/scan.mp4"
BACKGROUND_PATH = "/home/jetson/collab_nav_ground-jetson/src/arena_map_builder/config/background.png"
SERVER_NODE     = "/build_arena_map_server"
PROGRESS_FILE   = "progress.yaml"

# ══════════════════════════════════════════════════════════════════════════════
# ROS param mapping: run-dict key → (ros_param_path, value_type)
# ══════════════════════════════════════════════════════════════════════════════

_PARAM_MAP: Dict[str, tuple] = {
    # ExtractionConfig
    "target_fps":                ("stitch.extract.target_fps",                    "float"),
    "min_movement":              ("stitch.extract.min_movement",                  "float"),
    "max_movement":              ("stitch.extract.max_movement",                  "float"),
    "blur_thresh":               ("stitch.extract.blur_thresh",                   "float"),
    "artifact_thresh":           ("stitch.extract.artifact_thresh",               "float"),
    "static_pixel_thresh":       ("stitch.extract.static_pixel_thresh",           "float"),
    # ReconstructConfig — backend
    "feature_extractor":         ("stitch.reconstruct.feature_extractor",         "str"),
    "feature_matcher":           ("stitch.reconstruct.feature_matcher",           "str"),
    "processing_scale":          ("stitch.reconstruct.processing_scale",          "float"),
    # ReconstructConfig — alignment flags
    "use_grid_intersections":    ("stitch.reconstruct.use_grid_intersections",    "bool"),
    "use_pose_graph":            ("stitch.reconstruct.use_pose_graph",            "bool"),
    "use_fiducials":             ("stitch.reconstruct.use_fiducials",             "bool"),
    # ReconstructConfig — feature quality
    "feature_exclude_dilate_px": ("stitch.reconstruct.feature_exclude_dilate_px", "int"),
    "match_ratio":               ("stitch.reconstruct.match_ratio",               "float"),
    "mad_factor":                ("stitch.reconstruct.mad_factor",                "float"),
    "min_inliers":               ("stitch.reconstruct.min_inliers",               "int"),
    "min_keypoint_bins":         ("stitch.reconstruct.min_keypoint_bins",         "int"),
    # ReconstructConfig — temporal
    "lookback":                  ("stitch.reconstruct.lookback",                  "int"),
    "keyframe_interval":         ("stitch.reconstruct.keyframe_interval",         "int"),
    # ReconstructConfig — grid detection
    "grid_match_dist":           ("stitch.reconstruct.grid_match_dist",           "float"),
    "grid_min_intersections":    ("stitch.reconstruct.grid_min_intersections",    "int"),
    "grid_hsv_h_lo":             ("stitch.reconstruct.grid_hsv_h_lo",             "int"),
    "grid_hsv_h_hi":             ("stitch.reconstruct.grid_hsv_h_hi",             "int"),
    "grid_hsv_s_lo":             ("stitch.reconstruct.grid_hsv_s_lo",             "int"),
    "grid_hsv_v_lo":             ("stitch.reconstruct.grid_hsv_v_lo",             "int"),
    # ReconstructConfig — pose graph
    "pg_marker_weight":          ("stitch.reconstruct.pg_marker_weight",          "float"),
    "pg_odom_weight":            ("stitch.reconstruct.pg_odom_weight",            "float"),
    "pg_loop_weight":            ("stitch.reconstruct.pg_loop_weight",            "float"),
    "pg_huber_delta":            ("stitch.reconstruct.pg_huber_delta",            "float"),
    "pg_iterations":             ("stitch.reconstruct.pg_iterations",             "int"),
}

# ColorRangeMask.blue_tape() encoded as a server string-array entry
_BLUE_TAPE_SPEC = "blue_tape:90,125,60,255,60,255:255,255,255"


# ══════════════════════════════════════════════════════════════════════════════
# SSIM similarity metric  (numpy + cv2, no extra deps)
# ══════════════════════════════════════════════════════════════════════════════

def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM between two same-shape uint8 grayscale images."""
    fa = a.astype(np.float64)
    fb = b.astype(np.float64)
    k, s = (11, 11), 1.5
    mu_a  = cv2.GaussianBlur(fa, k, s)
    mu_b  = cv2.GaussianBlur(fb, k, s)
    mu_a2, mu_b2, mu_ab = mu_a**2, mu_b**2, mu_a * mu_b
    sig_a2 = cv2.GaussianBlur(fa * fa, k, s) - mu_a2
    sig_b2 = cv2.GaussianBlur(fb * fb, k, s) - mu_b2
    sig_ab = cv2.GaussianBlur(fa * fb, k, s) - mu_ab
    C1, C2 = (0.01 * 255)**2, (0.03 * 255)**2
    num = (2 * mu_ab + C1) * (2 * sig_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2)
    return float(np.mean(num / (den + 1e-10)))


def compute_similarity(pred_path: str, gt_path: str) -> Optional[float]:
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt   = cv2.imread(gt_path,   cv2.IMREAD_GRAYSCALE)
    if pred is None or gt is None:
        return None
    if pred.shape != gt.shape:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_AREA)
    return _ssim(pred, gt)


# ══════════════════════════════════════════════════════════════════════════════
# Progress tracking
# ══════════════════════════════════════════════════════════════════════════════

def load_progress(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {"completed": [], "failed": []}


def save_progress(path: Path, progress: dict) -> None:
    with open(path, "w") as f:
        yaml.dump(progress, f, default_flow_style=False, sort_keys=False)


# ══════════════════════════════════════════════════════════════════════════════
# ROS param setting
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(value, typ: str) -> str:
    """Format a value for ros2 param set."""
    if typ == "bool":
        return "true" if value else "false"
    return str(value)


def apply_run_params(run: dict, run_dir: Path) -> bool:
    """Push all params for this run to the server. Returns True on success."""
    params = [
        # Per-run output directory → all debug images land here automatically
        ("debug_dir",               str(run_dir)),
        # Fixed inputs
        ("transfer.background_path", BACKGROUND_PATH),
        # Keep verbose off during the sweep (server logs already go to its terminal)
        ("stitch.verbose",           "false"),
        # Stitcher diagnostics JSON — finalize() writes Groups 1+6 features here
        # so sweep can read them without re-running any pipeline stage.
        # Requires the server to map stitch.reconstruct.diagnostics_path →
        # ReconstructConfig.diagnostics_path (added to drone_map_grid_gen.py).
        ("stitch.reconstruct.diagnostics_path",
         str(run_dir / "stitcher_diagnostics.json")),
    ]

    for key, (ros_param, typ) in _PARAM_MAP.items():
        params.append((ros_param, _fmt(run[key], typ)))

    # feature_exclude_hsv is a string-array param with special encoding
    excl = run.get("feature_exclude_hsv", "none")
    fex_val = f"['{_BLUE_TAPE_SPEC}']" if excl == "blue_tape" else "[]"
    params.append(("stitch.reconstruct.feature_exclude_hsv", fex_val))

    for param, value in params:
        try:
            result = subprocess.run(
                ["ros2", "param", "set", SERVER_NODE, param, value],
                capture_output=True, text=True, timeout=15,
            )
        except subprocess.TimeoutExpired:
            print(f"    [!] ros2 param set {param} timed out")
            return False
        if result.returncode != 0:
            err = (result.stderr or result.stdout).strip()
            print(f"    [!] ros2 param set {param}={value!r} FAILED: {err}")
            return False

    return True


# ══════════════════════════════════════════════════════════════════════════════
# Action client
# ══════════════════════════════════════════════════════════════════════════════

class SweepClient(Node):
    def __init__(self):
        super().__init__("sweep_client")
        self._client = ActionClient(self, BuildArenaMap, "build_arena_map")

    def wait_for_server(self, timeout_s: float = 120.0) -> bool:
        self.get_logger().info("Waiting for action server…")
        return self._client.wait_for_server(timeout_sec=timeout_s)

    def run_action(self, video_path: str, timeout_s: float = 1800.0) -> Optional[dict]:
        """Send one BuildArenaMap goal; block until done, failed, or timeout."""
        goal = BuildArenaMap.Goal()
        goal.video_path = video_path

        # ── send goal ──────────────────────────────────────────────────────
        send_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=30.0)
        if not send_future.done() or send_future.result() is None:
            self.get_logger().error("Goal send timed out")
            return None
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected by server")
            return None

        # ── wait for result ────────────────────────────────────────────────
        result_future = goal_handle.get_result_async()
        deadline = time.monotonic() + timeout_s
        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=1.0)
            if time.monotonic() > deadline:
                self.get_logger().error(f"Goal timed out after {timeout_s:.0f}s — cancelling")
                cancel_future = goal_handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=15.0)
                return None

        result_wrapper = result_future.result()
        if result_wrapper.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(f"Goal did not succeed (status={result_wrapper.status})")
            return None

        r = result_wrapper.result
        return {
            "success":          bool(r.success),
            "n_obstacles":      int(r.n_obstacles),
            "mean_consistency": float(r.mean_consistency),
            "message":          str(r.message),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Server process management
# ══════════════════════════════════════════════════════════════════════════════

def start_server(log_path: Optional[Path] = None) -> subprocess.Popen:
    log_path = log_path or Path("/tmp/build_arena_map_server.log")
    print(f"  Starting build_arena_map_server… (log: {log_path})")
    log_file = open(log_path, "w", buffering=1)
    return subprocess.Popen(
        ["ros2", "run", "arena_map_builder", "build_arena_map_server"],
        stdout=log_file,
        stderr=log_file,
    )


def server_alive(proc: Optional[subprocess.Popen]) -> bool:
    return proc is not None and proc.poll() is None


# ══════════════════════════════════════════════════════════════════════════════
# Per-run execution
# ══════════════════════════════════════════════════════════════════════════════

def execute_run(
    run:        dict,
    client:     SweepClient,
    output_dir: Path,
    timeout_s:  float,
    gt_path:    Optional[str],
) -> str:
    """
    Execute one run. Returns "ok", "failed", or "timeout".

    Sets debug_dir to {output_dir}/run_{run_id} so the server writes all its
    debug images (stitched_map.png, transfer_*.png, 08_occupancy_vis.png, …)
    directly into the run directory — no copying needed.
    """
    run_id  = run["run_id"]
    run_dir = output_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write config immediately — readable even if the run crashes
    _save_config(run, run_dir)

    # Set all server params for this run
    print(f"  Setting params ({len(_PARAM_MAP) + 3} params)…")
    if not apply_run_params(run, run_dir):
        _save_metrics({"status": "param_set_failed", "runtime_s": 0.0}, run_dir)
        return "failed"

    # Send goal and wait
    print(f"  Sending goal (timeout={timeout_s:.0f}s)…")
    t0      = time.monotonic()
    result  = client.run_action(VIDEO_PATH, timeout_s=timeout_s)
    elapsed = time.monotonic() - t0

    metrics: dict = {"runtime_s": round(elapsed, 1)}

    if result is None:
        metrics["status"] = "timeout_or_failed"
        _save_metrics(metrics, run_dir)
        return "timeout"

    metrics["status"]           = "ok" if result["success"] else "server_error"
    metrics["n_obstacles"]      = result["n_obstacles"]
    metrics["mean_consistency"] = result["mean_consistency"]
    metrics["server_message"]   = result["message"]

    # Optional SSIM vs ground truth occupancy image
    if gt_path:
        occ_path = str(run_dir / "08_occupancy_vis.png")
        score    = compute_similarity(occ_path, gt_path)
        metrics["ssim_vs_gt"] = round(score, 4) if score is not None else None

    # Collect all diagnostic features and merge into metrics before the
    # single metrics.yaml write.  Any failure inside _collect_diagnostics
    # is caught internally and the sweep continues.
    _collect_diagnostics(run_dir, metrics)
    _save_metrics(metrics, run_dir)

    status = "ok" if result["success"] else "failed"
    print(f"  → {status}  obstacles={metrics.get('n_obstacles', '?')}  "
          f"consistency={metrics.get('mean_consistency', 0.0):.3f}  "
          f"runtime={elapsed:.0f}s")
    return status


def _collect_diagnostics(run_dir: Path, metrics: dict) -> None:
    """Compute all diagnostic feature values and merge them into `metrics`.

    Called once the ROS action has completed and all outputs are on disk.

    Two-stage collection
    ────────────────────
    Groups 1 + 6 (stitcher side):
        Read from  run_dir/stitcher_diagnostics.json  written by the server
        when ReconstructConfig.diagnostics_path is set.  Falls back to
        computing Group 1 only from the stitched-map image if the JSON is
        absent (e.g. server not yet updated to expose diagnostics_path).

    Groups 2–5 (transfer side):
        Re-run transfer_obstacles.run_pipeline() locally on the saved
        transfer_input.png with return_diagnostics=True.  This is the same
        computation the server already performed; re-running it locally
        (~5–15 s) guarantees the features match what extract_features.py
        would compute and avoids needing the server to expose more fields.
    """
    if not _DIAG_AVAILABLE:
        return

    import json

    # ── Groups 1 + 6: stitcher diagnostics ──────────────────────────────
    stitcher_diag_path = run_dir / "stitcher_diagnostics.json"
    stitcher_diag: dict = {}

    if stitcher_diag_path.exists():
        try:
            with open(stitcher_diag_path) as f:
                stitcher_diag = json.load(f)
            print(f"  Diagnostics: loaded {len(stitcher_diag)} stitcher features")
        except Exception as exc:
            print(f"  [warn] Could not read {stitcher_diag_path.name}: {exc}")

    # ── find the stitched-map image ──────────────────────────────────────
    img_path = None
    for name in ("transfer_input.png", "stitched_map.png"):
        p = run_dir / name
        if p.exists():
            img_path = p
            break

    if img_path is None:
        print(f"  [warn] No stitched-map image in {run_dir.name} "
              f"— diagnostics skipped")
        return

    # ── fallback: compute Group 1 locally if JSON not present ────────────
    if not stitcher_diag:
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                stitcher_diag = _compute_stitcher_diag(
                    stitched_map=img,
                    stitcher_stats=None,     # not available without JSON
                    finalize_report=None,    # Group 6 will be all -1.0
                )
                print(f"  Diagnostics: computed {len(stitcher_diag)} "
                      f"stitcher features from image (Group 6 unavailable)")
        except Exception as exc:
            print(f"  [warn] Stitcher diagnostics computation failed: {exc}")

    # ── Groups 2–5: re-run transfer pipeline locally ─────────────────────
    transfer_diag: dict = {}
    try:
        result = _run_pipeline(
            str(img_path), BACKGROUND_PATH,
            cfg=_TransferConfig(),
            verbose=False,
            return_diagnostics=True,
        )
        # return_diagnostics=True → 3-tuple; else 2-tuple (graceful)
        if len(result) == 3:
            _, _, transfer_diag = result
            print(f"  Diagnostics: computed {len(transfer_diag)} "
                  f"transfer features")
    except Exception as exc:
        print(f"  [warn] Transfer pipeline (diagnostics) failed: {exc}")

    metrics.update(stitcher_diag)
    metrics.update(transfer_diag)
    total = len(stitcher_diag) + len(transfer_diag)
    if total:
        print(f"  Diagnostics: {total} features total saved to metrics.yaml")


def _save_config(run: dict, run_dir: Path) -> None:
    cfg = {k: v for k, v in run.items() if k not in ("run_id", "group")}
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump({"run_id": run["run_id"], "group": run["group"], **cfg},
                  f, default_flow_style=False, sort_keys=False)


def _save_metrics(metrics: dict, run_dir: Path) -> None:
    with open(run_dir / "metrics.yaml", "w") as f:
        yaml.dump(metrics, f, default_flow_style=False, sort_keys=False)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overnight arena map parameter sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", default=str(_HERE / "results"),
                        help="Results directory (default: sweep/results/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the run plan and exit — no execution")
    parser.add_argument("--no-server", action="store_true",
                        help="Skip starting the server (assume it is already running)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-run previously failed/timed-out runs")
    parser.add_argument("--similarity", metavar="GT_PATH",
                        help="Enable SSIM metric: path to reference 08_occupancy_vis.png")
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="Per-run goal timeout in seconds (default: 1800)")
    args = parser.parse_args()

    output_dir    = Path(args.output_dir)
    progress_path = output_dir / PROGRESS_FILE
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load progress
    progress  = load_progress(progress_path)
    completed = set(progress.get("completed", []))
    failed    = set(progress.get("failed", []))

    skip = completed if args.retry_failed else completed | failed
    pending = [r for r in RUNS if r["run_id"] not in skip]

    # ── dry run ────────────────────────────────────────────────────────────
    if args.dry_run:
        _print_plan(pending, completed, failed)
        return

    if not pending:
        print("All runs complete (or marked failed). Use --retry-failed to re-run failures.")
        return

    print(f"\n{'═'*62}")
    print(f"  Sweep: {len(pending)}/{len(RUNS)} runs pending"
          + (f"  ({len(failed)} failed, skipped)" if failed else ""))
    print(f"  Output: {output_dir}")
    print(f"{'═'*62}\n")

    # ── start server ───────────────────────────────────────────────────────
    server_proc: Optional[subprocess.Popen] = None
    if not args.no_server:
        server_proc = start_server(output_dir / "server.log")

    # Graceful shutdown on SIGINT / SIGTERM
    _shutdown = {"requested": False}
    def _handle_signal(sig, frame):
        print("\n  Shutdown requested — finishing current run then stopping.")
        _shutdown["requested"] = True
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ── rclpy setup ────────────────────────────────────────────────────────
    rclpy.init()
    client = SweepClient()

    if not client.wait_for_server(timeout_s=120.0):
        print("[ERROR] Action server not available within 120 s. Exiting.")
        rclpy.shutdown()
        if server_proc:
            server_proc.terminate()
        sys.exit(1)
    print("  Action server ready.\n")

    # ── main loop ──────────────────────────────────────────────────────────
    for i, run in enumerate(pending):
        if _shutdown["requested"]:
            break

        run_id = run["run_id"]
        group  = run["group"]
        print(f"[{i+1}/{len(pending)}]  Run {run_id}  (group {group})  "
              f"ext={run['feature_extractor']}  mat={run['feature_matcher']}  "
              f"grid={run['use_grid_intersections']}  pg={run['use_pose_graph']}")

        # Restart server if it died
        if not args.no_server and not server_alive(server_proc):
            print("  [!] Server process died — restarting…")
            server_proc = start_server(output_dir / "server.log")
            if not client.wait_for_server(timeout_s=120.0):
                print("[ERROR] Server did not recover. Stopping sweep.")
                break

        status = execute_run(
            run        = run,
            client     = client,
            output_dir = output_dir,
            timeout_s  = args.timeout,
            gt_path    = args.similarity,
        )

        # Persist progress immediately after each run
        if status == "ok":
            completed.add(run_id)
            failed.discard(run_id)
        else:
            failed.add(run_id)
        progress["completed"] = sorted(completed)
        progress["failed"]    = sorted(failed)
        save_progress(progress_path, progress)

        print()

    # ── teardown ───────────────────────────────────────────────────────────
    rclpy.shutdown()
    if server_proc:
        print("Stopping server…")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server_proc.kill()

    # ── summary ────────────────────────────────────────────────────────────
    runs_done    = len(completed)
    runs_failed  = len(failed)
    runs_remain  = len(RUNS) - runs_done - runs_failed
    print(f"\n{'═'*62}")
    print(f"  {runs_done} completed  |  {runs_failed} failed  |  {runs_remain} remaining")
    print(f"  Results: {output_dir}")
    print(f"{'═'*62}\n")


def _print_plan(pending: list, completed: set, failed: set) -> None:
    from collections import Counter
    groups = Counter(r["group"] for r in pending)
    print(f"\nTotal runs:     {len(RUNS)}")
    print(f"Completed:      {len(completed)}")
    print(f"Failed/skipped: {len(failed)}")
    print(f"Pending:        {len(pending)}\n")
    descs = {
        "A": "architecture × grid × solver × feature-exclusion",
        "B": "match_ratio × mad_factor  [0.55–0.80 × 2–6]",
        "C": "lookback × keyframe_interval  [lb up to 16, ki up to 20]",
        "D": "processing_scale  [0.25–1.0]",
        "E": "feature_exclude_dilate_px  [3–20 px]",
        "F": "min_keypoint_bins + min_inliers",
        "G": "grid re-exploration: grid_match_dist × HSV recalibration",
        "H": "pose-graph re-exploration: marker_weight × huber_delta",
        "I": "target_fps × min_movement",
        "J": "blur_thresh + max_movement",
        "K": "processing_scale × match_ratio  [interaction]",
        "L": "3-way: match_ratio × mad_factor × lookback",
        "M": "SP × processing_scale  [GPU]",
        "N": "static_pixel_thresh + artifact_thresh",
    }
    print(f"{'Group':<8} {'Pending':>7}   Description")
    print("─" * 55)
    for g in sorted(groups):
        print(f"  {g:<6} {groups[g]:>7}   {descs.get(g, '')}")
    print("─" * 55)
    if pending:
        print("\nFirst 5 pending runs:")
        for r in pending[:5]:
            print(f"  {r['run_id']}  ext={r['feature_extractor'][:4]}  "
                  f"mat={r['feature_matcher'][:9]}  "
                  f"grid={r['use_grid_intersections']}  "
                  f"pg={r['use_pose_graph']}  fid={r['use_fiducials']}  "
                  f"exc={r['feature_exclude_hsv'][:4]}  "
                  f"mr={r['match_ratio']}  mf={r['mad_factor']}")
        if len(pending) > 5:
            print(f"  … and {len(pending) - 5} more")


if __name__ == "__main__":
    main()