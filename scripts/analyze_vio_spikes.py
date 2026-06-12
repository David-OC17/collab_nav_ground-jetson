#!/usr/bin/env python3
"""
Offline VIO spike analysis and filter evaluation.

Reads rosbags from a vo/ directory. Uses only:
  /visual_slam/tracking/odometry   (VIO — nav_msgs/Odometry)
  /optitrack/rigid_body            (ground truth — PoseStamped, offline eval only)

Applies three causal filter approaches to lin_vel_x and compares them
on a shared panel so the final signal quality can be assessed visually.

Filters implemented
───────────────────
  Baseline  — 2nd-order Butterworth LPF + absolute gate  (current approach)
  A         — Causal Hampel identifier
  B         — VIO self-consistency  (position finite-diff vs reported velocity)
  C         — Combined: B → Hampel post-pass

Usage
─────
  python analyze_vio_spikes.py --bag vo_test1
  python analyze_vio_spikes.py --all
  python analyze_vio_spikes.py --all --vo-dir /path/to/vo

Requirements
────────────
  pip install rosbags numpy pandas matplotlib scipy
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import butter, savgol_filter, sosfilt

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

matplotlib.rcParams.update({"figure.dpi": 130, "font.size": 9})

# ── topics ────────────────────────────────────────────────────────────────────
TOPIC_VIO = "/visual_slam/tracking/odometry"
TOPIC_OT  = "/optitrack/rigid_body"

# ══════════════════════════════════════════════════════════════════════════════
# TUNING PARAMETERS  ← edit here
# ══════════════════════════════════════════════════════════════════════════════

# ── Physical limits ───────────────────────────────────────────────────────────
# Hard ceiling on what the AMR can physically do.
# Any |vx| above this is immediately a spike — no model needed.
# Tighten if you know the robot never reaches 0.5 m/s in practice.
AMR_VX_MAX = 0.5          # m/s

# ── Baseline filter (reference only, not used in pipeline) ───────────────────
BASELINE_FC   = 2.0       # Butterworth cutoff frequency [Hz]
BASELINE_FS   = 80.0      # VIO publish rate [Hz]
BASELINE_GATE = 1.0       # absolute velocity gate [m/s]

# ── P1: EMA gate ──────────────────────────────────────────────────────────────
# The primary tuning surface. See parameter guide below.
#
#  min_threshold   most impactful — minimum deviation from EMA to flag a spike.
#                  Lower  (0.10–0.12) → catches smaller spikes, may reject fast starts.
#                  Higher (0.18–0.20) → more permissive, lets small bursts through.
#
#  alpha           EMA time constant in normal mode (~1/alpha samples).
#                  Lower  (0.15) → more stable reference, more lag on real changes.
#                  Higher (0.35) → follows real changes faster, less noise rejection.
#
#  n_sigma         Gate width in units of running std.  Usually secondary to
#                  min_threshold since std is small during smooth motion.
#
#  burst_thresh    Fraction of burst_window that must be natural spikes to enter
#                  burst mode.  Lower → triggers earlier on oscillating noise.
#                  Higher → requires more evidence, less sensitive to brief noise.
#
#  burst_window    Look-back window [samples] for burst rate calculation.
#                  Smaller → reacts faster to burst start/end.
#
#  alpha_burst     EMA drift rate during burst mode (~1/alpha_burst samples to
#                  recover).  Raise if the filter takes too long to follow a
#                  genuine velocity change that happens during a noisy period.

P1_MIN_THRESHOLD = 0.17   # m/s  — primary sensitivity knob
P1_ALPHA         = 0.50   # normal-mode EMA decay
P1_N_SIGMA       = 2.0    # gate width in σ units
P1_BURST_THRESH  = 0.30   # burst trigger fraction
P1_BURST_WINDOW  = 10     # burst look-back [samples]
P1_ALPHA_BURST   = 0.02   # EMA drift rate during burst
P1_WARMUP        = 20     # samples before gating starts

# ── B: Self-consistency ───────────────────────────────────────────────────────
# Flags samples where reported velocity disagrees with position finite-diff.
# Effective when VIO velocity spikes without a matching position jump.
#
#  sc_threshold    Disagreement needed to flag [m/s].
#                  Lower  → more aggressive, may flag legitimate fast moves.
#                  Higher → only catches large inconsistencies.
#
#  sc_sg_window    Savitzky-Golay smoothing window for the FD reference [samples].
#                  Larger → smoother reference, more robust to position noise.

SC_THRESHOLD  = 0.30      # m/s
SC_SG_WINDOW  = 31        # samples (must be odd)

# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# BAG LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _quat_to_yaw(qx, qy, qz, qw) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def load_bag(bag_path: Path, typestore) -> dict:
    """
    Returns {"vio": DataFrame | None, "optitrack": DataFrame | None}.

    VIO columns:      t, vx, wz, px, py, pz, yaw
    OptiTrack columns: t, px, py, yaw
    """
    vio_rows, ot_rows = [], []

    with Reader(bag_path) as reader:
        conns = [c for c in reader.connections if c.topic in (TOPIC_VIO, TOPIC_OT)]
        if not conns:
            return {"vio": None, "optitrack": None}

        for conn, bag_ts_ns, raw in reader.messages(connections=conns):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)

            if conn.topic == TOPIC_VIO:
                t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                q = msg.pose.pose.orientation
                vio_rows.append({
                    "t":   t,
                    "vx":  msg.twist.twist.linear.x,
                    "wz":  msg.twist.twist.angular.z,
                    "px":  msg.pose.pose.position.x,
                    "py":  msg.pose.pose.position.y,
                    "pz":  msg.pose.pose.position.z,
                    "yaw": _quat_to_yaw(q.x, q.y, q.z, q.w),
                })

            elif conn.topic == TOPIC_OT:
                t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                p, q = msg.pose.position, msg.pose.orientation
                ot_rows.append({
                    "t":   t,
                    "px":  p.x,
                    "py":  p.y,
                    "yaw": _quat_to_yaw(q.x, q.y, q.z, q.w),
                })

    def _to_df(rows, t0):
        if not rows:
            return None
        df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
        df["t"] -= t0
        return df

    if not vio_rows:
        return {"vio": None, "optitrack": None}

    t0 = min(r["t"] for r in vio_rows)
    return {
        "vio":       _to_df(vio_rows, t0),
        "optitrack": _to_df(ot_rows,  t0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# GROUND TRUTH VELOCITY  (OptiTrack — offline eval only)
# ══════════════════════════════════════════════════════════════════════════════

def optitrack_velocity(
    df_ot:     pd.DataFrame,
    target_fs: float = 60.0,
    smooth_s:  float = 0.30,
    sg_poly:   int   = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Body-frame forward velocity from OptiTrack ground truth.

    OptiTrack message header stamps are clustered/non-uniform (bursts with
    near-zero inter-sample dt), so per-sample finite differencing of position
    over those raw timestamps explodes the velocity.  Instead the trajectory is
    resampled onto a uniform ``target_fs`` grid by interpolation, then
    differentiated with a constant step and Savitzky-Golay smoothed over a
    ``smooth_s``-second window.  Returns (t_vel, vx_body).
    """
    t   = df_ot["t"].values
    px  = df_ot["px"].values
    py  = df_ot["py"].values
    yaw = df_ot["yaw"].values

    # Strictly increasing time base (drop duplicate/back-stepping stamps).
    order = np.argsort(t, kind="stable")
    t, px, py, yaw = t[order], px[order], py[order], yaw[order]
    keep = np.concatenate([[True], np.diff(t) > 1e-6])
    t, px, py, yaw = t[keep], px[keep], py[keep], yaw[keep]
    if len(t) < 5:
        return np.array([]), np.array([])

    # Uniform resample (unwrap yaw so the interpolation does not jump at ±π).
    dt   = 1.0 / target_fs
    t_u  = np.arange(t[0], t[-1], dt)
    px_u = np.interp(t_u, t, px)
    py_u = np.interp(t_u, t, py)
    yaw_u = np.interp(t_u, t, np.unwrap(yaw))

    vx_w = np.gradient(px_u, dt)
    vy_w = np.gradient(py_u, dt)
    vx_body = vx_w * np.cos(yaw_u) + vy_w * np.sin(yaw_u)

    win = int(smooth_s * target_fs)
    win = win if win % 2 == 1 else win + 1          # must be odd
    if len(vx_body) > win and win > sg_poly:
        vx_body = savgol_filter(vx_body, win, sg_poly)

    return t_u, vx_body


# ══════════════════════════════════════════════════════════════════════════════
# FILTERS  (all causal / online-compatible, VIO-only inputs)
# ══════════════════════════════════════════════════════════════════════════════

# ── Baseline: Butterworth LPF + absolute gate ─────────────────────────────────

def filter_baseline(
    vx: np.ndarray,
    fc:   float = 2.0,
    fs:   float = 80.0,
    gate: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    sos    = butter(2, fc / (0.5 * fs), btype="low", output="sos")
    vx_lpf = sosfilt(sos, vx)                          # causal forward-only
    mask   = np.abs(vx) > gate                         # what the gate removes
    return np.where(mask, 0.0, vx_lpf), mask


# ── Stage 0: Physical validity gate ──────────────────────────────────────────

def filter_physical_gate(
    vx:      np.ndarray,
    vx_max:  float = AMR_VX_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Any |vx| > vx_max is physically impossible for this AMR.
    Flagged samples are replaced with the last accepted value (hold).
    This is the highest-confidence detection possible — no tuning, no model.

    Returns (filtered_vx, spike_mask).
    """
    x      = vx.copy()
    mask   = np.zeros(len(x), dtype=bool)
    last_good = 0.0

    for i, v in enumerate(vx):
        if abs(v) > vx_max:
            mask[i] = True
            x[i]    = last_good        # hold last accepted value
        else:
            last_good = v

    return x, mask




# ── P1: EMA-based adaptive gate with burst detection ─────────────────────────

def filter_ema_gate(
    vx:            np.ndarray,
    alpha:         float = 0.25,   # EMA decay in normal mode (~4-sample time constant)
    alpha_burst:   float = 0.02,   # EMA decay in burst mode (~50-sample time constant)
    n_sigma:       float = 4.0,    # flag if deviation > n_sigma × running std
    min_threshold: float = 0.15,   # absolute floor to avoid gating on noise
    warmup:        int   = 15,     # samples to initialise EMA before gating
    burst_window:  int   = 8,      # sliding window for burst rate calculation
    burst_thresh:  float = 0.30,   # fraction of window spikes required to enter burst
) -> tuple[np.ndarray, np.ndarray]:
    """
    EMA gate with burst detection.

    Normal mode
    ───────────
    Flags a sample if |vx[i] - ema| > max(n_sigma*std, min_threshold).
    EMA and variance update only on accepted samples.

    Burst mode
    ──────────
    Entered when the NATURAL spike rate (what the gate would decide without the
    burst rule) in the last burst_window samples exceeds burst_thresh.

    During burst:
      - All samples are force-held at last_good.
      - EMA drifts very slowly (alpha_burst) toward the clamped raw value.
        This allows the EMA to follow genuine velocity changes during noisy
        periods so the filter can recover — avoids the deadlock of pure freezing.
      - The burst window tracks NATURAL decisions (not forced ones), so burst
        mode exits naturally once the underlying signal stabilises.
    """
    from collections import deque

    x         = vx.copy()
    mask      = np.zeros(len(x), dtype=bool)
    last_good = float(vx[:warmup].mean()) if warmup <= len(vx) else float(vx[0])
    ema       = last_good
    ema_sq    = float(np.mean(vx[:warmup] ** 2)) if warmup <= len(vx) else ema ** 2

    # Track NATURAL spike decisions — not forced ones — to avoid deadlock
    recent_natural: deque = deque([False] * burst_window, maxlen=burst_window)

    for i in range(len(vx)):
        v = vx[i]

        # ── Warmup ────────────────────────────────────────────────────────────
        if i < warmup:
            ema       = alpha * v + (1 - alpha) * ema
            ema_sq    = alpha * v ** 2 + (1 - alpha) * ema_sq
            last_good = v
            recent_natural.append(False)
            continue

        # ── Burst state (from natural history, not forced decisions) ───────────
        in_burst = (sum(recent_natural) / burst_window) >= burst_thresh

        # ── Gate threshold ─────────────────────────────────────────────────────
        var       = max(ema_sq - ema ** 2, 0.0)
        std       = math.sqrt(var)
        threshold = max(n_sigma * std, min_threshold)

        # ── Natural spike decision (gate without burst rule) ───────────────────
        natural_spike = abs(v - ema) > threshold

        # ── Decision ───────────────────────────────────────────────────────────
        if in_burst or natural_spike:
            mask[i] = True
            x[i]    = last_good
            # Slow EMA drift toward clamped raw value — allows recovery if
            # underlying velocity genuinely changed during a noisy period
            v_clamped = max(-AMR_VX_MAX, min(AMR_VX_MAX, v))
            ema    = alpha_burst * v_clamped + (1 - alpha_burst) * ema
            ema_sq = alpha_burst * v_clamped ** 2 + (1 - alpha_burst) * ema_sq
        else:
            last_good = v
            ema    = alpha * v + (1 - alpha) * ema
            ema_sq = alpha * v ** 2 + (1 - alpha) * ema_sq

        # Append NATURAL decision so burst rate can drop when signal recovers
        recent_natural.append(natural_spike)

    return x, mask



# ── B: VIO self-consistency (position FD vs reported velocity) ────────────────

def filter_self_consistency(
    df_vio:     pd.DataFrame,
    threshold:  float = 0.20,
    sg_window:  int   = 9,
    sg_poly:    int   = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Derives an independent velocity estimate by finite-differencing the VIO
    position and rotating into body frame via the quaternion yaw.
    If |vx_reported − vx_fd| > threshold → spike; replace with vx_fd.

    Returns (filtered_vx, spike_mask, vx_fd_smooth).
    """
    t   = df_vio["t"].values
    px  = df_vio["px"].values
    py  = df_vio["py"].values
    yaw = df_vio["yaw"].values
    vx  = df_vio["vx"].values

    dt = np.diff(t)
    dt = np.where(dt < 1e-9, np.nan, dt)

    yaw_mid = 0.5 * (yaw[:-1] + yaw[1:])
    with np.errstate(invalid="ignore"):
        vx_fd = (np.diff(px) * np.cos(yaw_mid) +
                 np.diff(py) * np.sin(yaw_mid)) / dt

    # Pad to original length (prepend first valid value)
    vx_fd_full = np.concatenate([[vx_fd[0]], vx_fd])

    valid = np.isfinite(vx_fd_full)
    vx_fd_smooth = vx_fd_full.copy()
    if valid.sum() >= sg_window:
        vx_fd_smooth[valid] = savgol_filter(vx_fd_full[valid], sg_window, sg_poly)

    mask         = np.abs(vx - vx_fd_smooth) > threshold
    vx_out       = vx.copy()
    vx_out[mask] = vx_fd_smooth[mask]

    return vx_out, mask, vx_fd_smooth


# ── C: Combined — self-consistency → Hampel post-pass ────────────────────────

def filter_combined(
    df_vio:            pd.DataFrame,
    ema_alpha:         float = 0.25,
    ema_alpha_burst:   float = 0.02,
    ema_n_sigma:       float = 4.0,
    ema_min_thr:       float = 0.15,
    ema_burst_window:  int   = 8,
    ema_burst_thresh:  float = 0.30,
) -> tuple[np.ndarray, np.ndarray]:
    """P0 → P1 (EMA gate with burst detection)."""
    vx_raw = df_vio["vx"].values
    vx0, mask0 = filter_physical_gate(vx_raw)
    vx1, mask1 = filter_ema_gate(vx0,
                                  alpha=ema_alpha,
                                  alpha_burst=ema_alpha_burst,
                                  n_sigma=ema_n_sigma,
                                  min_threshold=ema_min_thr,
                                  burst_window=ema_burst_window,
                                  burst_thresh=ema_burst_thresh)
    return vx1, mask0 | mask1


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mae(a, b):
    return float(np.mean(np.abs(a - b)))

def spike_stats(vx_raw: np.ndarray, mask: np.ndarray) -> dict:
    if not mask.any():
        return {"count": 0, "rate_%": 0.0, "mag_mean": 0.0, "mag_max": 0.0,
                "mag_p95": 0.0, "run_len_mean": 0.0, "run_len_max": 0}
    mags = np.abs(vx_raw[mask])
    runs, run_len = [], 0
    for m in mask:
        if m:
            run_len += 1
        elif run_len:
            runs.append(run_len)
            run_len = 0
    if run_len:
        runs.append(run_len)
    return {
        "count":        int(mask.sum()),
        "rate_%":       float(100 * mask.mean()),
        "mag_mean":     float(mags.mean()),
        "mag_max":      float(mags.max()),
        "mag_p95":      float(np.percentile(mags, 95)),
        "run_len_mean": float(np.mean(runs)) if runs else 0.0,
        "run_len_max":  int(max(runs))        if runs else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

# Consistent colours for each method across all panels
METHOD_COLORS = {
    "Raw":      "tab:gray",
    "Baseline": "tab:orange",
    "P0":       "tab:olive",
    "P1":       "tab:blue",
    "B":        "tab:purple",
    "C":        "tab:red",
    "GT":       "tab:green",
}

def _tight_ylim(*arrays, pad: float = 0.15, min_span: float = 0.05):
    """
    Compute Y limits tightly around the provided arrays (ignoring NaN/inf).
    pad:      fractional padding on each side of the data range.
    min_span: minimum visible span in m/s so a flat signal still has room.
    """
    vals = np.concatenate([np.asarray(a).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return -0.1, 0.6
    lo, hi  = float(vals.min()), float(vals.max())
    span    = max(hi - lo, min_span)
    margin  = span * pad
    return lo - margin, hi + margin


def plot_bag(
    bag_name:   str,
    df_vio:     pd.DataFrame,
    results:    dict,           # {"Label": (vx_filtered, mask)}
    vx_fd:      np.ndarray,     # from self-consistency filter
    out_dir:    Path,
    fmt:        str = "pdf",    # "pdf" | "svg" | "png"
    pdf_pages=None,             # matplotlib PdfPages handle (optional)
) -> Path:
    """
    Layout (rows):
      0 — Raw VIO + position FD reference
      1 — COMPARISON: all final signals overlaid on one panel
      2 … N — Per-method detail: raw (grey) + filtered (colour) + spike markers

    Y limits are computed tightly per-panel from the signals actually plotted,
    so filtered outputs fill the axes rather than being squashed by spike outliers.
    """
    t      = df_vio["t"].values
    vx_raw = df_vio["vx"].values

    n_rows = 2 + len(results)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(16, 5.0 * n_rows),
        sharex=True,
        gridspec_kw={"hspace": 0.50},
    )
    fig.suptitle(f"VIO spike analysis — {bag_name}", fontweight="bold", fontsize=11)

    # ── Row 0: raw signals ────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, vx_raw, color=METHOD_COLORS["Raw"], lw=0.6, alpha=0.9, label="VIO raw")
    ax.plot(t, vx_fd,  color="tab:cyan", lw=0.8, alpha=0.8, ls="--",
            label="VIO pos FD (self-consistency ref)")
    ax.axhline( AMR_VX_MAX, color="red", lw=0.7, ls=":", alpha=0.6)
    ax.axhline(-AMR_VX_MAX, color="red", lw=0.7, ls=":", alpha=0.6,
               label=f"±{AMR_VX_MAX} m/s physical limit")
    ax.set_ylabel("vx [m/s]")
    ax.set_ylim(*_tight_ylim(vx_raw, vx_fd))   # raw panel: show everything inc. spikes
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.set_title("Raw VIO + position FD reference", loc="left")
    ax.grid(True, alpha=0.25)

    # ── Row 1: COMPARISON — all final outputs on one panel ────────────────────
    all_filtered = np.concatenate([vx_f for vx_f, _ in results.values()])
    ax = axes[1]
    ax.plot(t, vx_raw, color=METHOD_COLORS["Raw"], lw=0.5, alpha=0.25, label="Raw")
    for label, (vx_f, _) in results.items():
        short = label.split(":")[0].strip()
        c = METHOD_COLORS.get(short, "black")
        ax.plot(t, vx_f, color=c, lw=1.0, alpha=0.85, label=label)
    ax.axhline( AMR_VX_MAX, color="red", lw=0.7, ls=":", alpha=0.5)
    ax.axhline(-AMR_VX_MAX, color="red", lw=0.7, ls=":", alpha=0.5)
    ax.set_ylabel("vx [m/s]")
    ax.set_ylim(*_tight_ylim(all_filtered))     # tight around filtered outputs only
    ax.legend(fontsize=7, loc="upper right", ncol=3)
    ax.set_title("FINAL SIGNAL COMPARISON — all methods overlaid", loc="left",
                 fontweight="bold")
    ax.grid(True, alpha=0.25)

    # ── Rows 2+: per-method detail ────────────────────────────────────────────
    for i, (label, (vx_f, mask)) in enumerate(results.items()):
        ax    = axes[2 + i]
        short = label.split(":")[0].strip()
        c     = METHOD_COLORS.get(short, "black")

        ax.plot(t, vx_raw, color=METHOD_COLORS["Raw"], lw=0.5, alpha=0.35, label="Raw")
        ax.plot(t, vx_f,   color=c, lw=1.0, label=label)

        n_spikes = int(mask.sum()) if mask is not None else 0
        if mask is not None and n_spikes:
            ax.scatter(t[mask], vx_raw[mask], color="red", s=4, zorder=6,
                       label=f"Replaced ({n_spikes})", marker="x", linewidths=0.6)

        parts = [label, f"spikes={n_spikes} ({100*mask.mean():.1f}%)"]
        ax.set_title("  |  ".join(parts), loc="left")

        ax.set_ylabel("vx [m/s]")
        ax.set_ylim(*_tight_ylim(vx_f))        # tight around this method's output
        ax.legend(fontsize=7, loc="upper right", ncol=4)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time [s]")

    out_path = out_dir / f"{bag_name}_spike_analysis.{fmt}"
    fig.savefig(out_path, bbox_inches="tight")
    if pdf_pages is not None:
        pdf_pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# REPORT FIGURES  (clean, single-column, publication-ready)
# ══════════════════════════════════════════════════════════════════════════════
#
# These are NOT the diagnostic multi-panel comparison (plot_bag). They emit the
# two figures the report's EKF section needs:
#
#   IX_D-vio_raw_vs_gated   — raw cuSLAM vx vs the DEPLOYED gated output (C: P0→P1
#                             ema_gate), with the position finite-difference (FD)
#                             self-consistency reference as an independent check
#                             and the replaced samples marked. No OptiTrack GT is
#                             used (not reliable on the presentable runs), so the
#                             FD track is the only cross-reference shown.
#
#   IX_D-vio_gate_ablation  — zoom on the worst burst comparing the rejected
#                             Butterworth LPF baseline against the deployed gate,
#                             evidencing the "LPF transient propagates" claim.
#
# Sized for an IEEE single column (~3.4 in wide) and rendered to PNG so they drop
# straight into report/assets/.

REPORT_FIGSIZE = (3.4, 2.2)          # inches — single IEEE column
REPORT_DPI     = 300


def _worst_burst_window(t: np.ndarray, mask: np.ndarray,
                        half_width_s: float = 3.0) -> tuple[float, float]:
    """Centre a time window on the densest cluster of replaced samples."""
    if not mask.any():
        mid = 0.5 * (t[0] + t[-1])
        return mid - half_width_s, mid + half_width_s
    # Density via a short box convolution over the spike mask.
    k       = max(5, int(0.5 / np.median(np.diff(t))) if len(t) > 1 else 5)
    dens    = np.convolve(mask.astype(float), np.ones(k), mode="same")
    centre  = t[int(np.argmax(dens))]
    return centre - half_width_s, centre + half_width_s


def plot_report_figures(
    bag_name: str,
    df_vio:   pd.DataFrame,
    vx_C:     np.ndarray,      # deployed gated output (P0→P1)
    mask_C:   np.ndarray,      # replaced-sample mask for the deployed gate
    vx_base:  np.ndarray,      # Butterworth LPF baseline (for ablation)
    vx_fd:    np.ndarray,      # position FD self-consistency reference
    out_dir:  Path,
    fmt:      str = "png",
    df_ot:    pd.DataFrame | None = None,   # OptiTrack ground truth (optional)
) -> list[Path]:
    t      = df_vio["t"].values
    vx_raw = df_vio["vx"].values
    paths  = []

    # Ground-truth body-frame velocity from OptiTrack, when available. This is
    # the strongest reference; the FD track is a secondary VIO-internal check.
    t_gt = vx_gt = None
    if df_ot is not None and not df_ot.empty:
        t_gt, vx_gt = optitrack_velocity(df_ot)

    # ── Figure 1: raw vs deployed gate vs ground truth ────────────────────────
    fig, ax = plt.subplots(figsize=REPORT_FIGSIZE)
    ax.plot(t, vx_raw, color="0.65", lw=0.6, alpha=0.9, label="cuSLAM raw")
    if t_gt is not None:
        ax.plot(t_gt, vx_gt, color="tab:green", lw=0.9, alpha=0.85,
                label="OptiTrack truth")
    else:
        ax.plot(t, vx_fd, color="tab:cyan", lw=0.7, ls="--", alpha=0.8,
                label="position FD (consistency ref.)")
    ax.plot(t, vx_C,   color="tab:blue", lw=1.0, label="gated (deployed)")
    if mask_C.any():
        ax.scatter(t[mask_C], vx_raw[mask_C], color="tab:red", s=6, marker="x",
                   linewidths=0.6, zorder=6,
                   label=f"replaced ({int(mask_C.sum())})")
    ax.axhline( AMR_VX_MAX, color="red", lw=0.6, ls=":", alpha=0.5)
    ax.axhline(-AMR_VX_MAX, color="red", lw=0.6, ls=":", alpha=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$v_x$ [m/s]")
    ax.set_ylim(*_tight_ylim(vx_C, vx_gt if vx_gt is not None else vx_fd, pad=0.25))
    ax.legend(fontsize=5.5, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout(pad=0.3)
    p1 = out_dir / f"IX_D-vio_raw_vs_gated.{fmt}"
    fig.savefig(p1, dpi=REPORT_DPI, bbox_inches="tight")
    plt.close(fig)
    paths.append(p1)

    # ── Figure 2: ablation zoom on worst burst (LPF baseline vs deployed) ─────
    t0, t1 = _worst_burst_window(t, mask_C)
    w      = (t >= t0) & (t <= t1)
    fig, ax = plt.subplots(figsize=REPORT_FIGSIZE)
    ax.plot(t[w], vx_raw[w],  color="0.65", lw=0.7, alpha=0.9, label="cuSLAM raw")
    ax.plot(t[w], vx_base[w], color="tab:orange", lw=1.0, alpha=0.9,
            label="Butterworth LPF")
    ax.plot(t[w], vx_C[w],    color="tab:blue", lw=1.2, label="ema_gate (deployed)")
    ax.axhline( AMR_VX_MAX, color="red", lw=0.6, ls=":", alpha=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$v_x$ [m/s]")
    ax.set_ylim(*_tight_ylim(vx_base[w], vx_C[w], pad=0.30))
    ax.legend(fontsize=5.5, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout(pad=0.3)
    p2 = out_dir / f"IX_D-vio_gate_ablation.{fmt}"
    fig.savefig(p2, dpi=REPORT_DPI, bbox_inches="tight")
    plt.close(fig)
    paths.append(p2)

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# PER-BAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def analyze_bag(bag_path: Path, typestore, out_dir: Path,
                fmt: str = "pdf", pdf_pages=None,
                report: bool = False, report_dir: Path | None = None) -> dict | None:
    name = bag_path.name
    sep  = "─" * 60
    print(f"\n{sep}\n  {name}\n{sep}")

    dfs    = load_bag(bag_path, typestore)
    df_vio = dfs["vio"]
    df_ot  = dfs["optitrack"]

    if df_vio is None or df_vio.empty:
        print("  ✗  No VIO data — skipping.")
        return None

    has_ot = df_ot is not None and not df_ot.empty
    vx_raw = df_vio["vx"].values

    print(f"  VIO msgs   : {len(df_vio)}")
    print(f"  vx range   : [{vx_raw.min():.3f}, {vx_raw.max():.3f}] m/s")
    print(f"  OptiTrack  : {'yes (' + str(len(df_ot)) + ' msgs)' if has_ot else 'no'}")

    # ── Run all filters ───────────────────────────────────────────────────────
    vx_base, mask_base = filter_baseline(vx_raw,
                                         fc=BASELINE_FC,
                                         fs=BASELINE_FS,
                                         gate=BASELINE_GATE)
    vx_P0,   mask_P0   = filter_physical_gate(vx_raw,
                                               vx_max=AMR_VX_MAX)
    vx_P1,   mask_P1   = filter_ema_gate(vx_raw,
                                          alpha=P1_ALPHA,
                                          alpha_burst=P1_ALPHA_BURST,
                                          n_sigma=P1_N_SIGMA,
                                          min_threshold=P1_MIN_THRESHOLD,
                                          warmup=P1_WARMUP,
                                          burst_window=P1_BURST_WINDOW,
                                          burst_thresh=P1_BURST_THRESH)
    vx_B,    mask_B, vx_fd = filter_self_consistency(df_vio,
                                                      threshold=SC_THRESHOLD,
                                                      sg_window=SC_SG_WINDOW)
    vx_C,    mask_C    = filter_combined(df_vio,
                                         ema_alpha=P1_ALPHA,
                                         ema_alpha_burst=P1_ALPHA_BURST,
                                         ema_n_sigma=P1_N_SIGMA,
                                         ema_min_thr=P1_MIN_THRESHOLD,
                                         ema_burst_window=P1_BURST_WINDOW,
                                         ema_burst_thresh=P1_BURST_THRESH)

    results = {
        "Baseline: LPF + gate":  (vx_base, mask_base),
        "P0: Physical gate":     (vx_P0,   mask_P0),
        "P1: EMA gate":          (vx_P1,   mask_P1),
        "B: Self-consistency":   (vx_B,    mask_B),
        "C: Combined (P0→P1)":   (vx_C,    mask_C),
    }

    # ── Console statistics ────────────────────────────────────────────────────
    print("\n  Spike stats per method:")
    print(f"  {'Method':<25} {'count':>6}  {'rate%':>6}  "
          f"{'mag_mean':>9}  {'mag_max':>9}  {'run_max':>7}")
    print(f"  {'─'*25} {'─'*6}  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*7}")
    for label, (_, mask) in results.items():
        st = spike_stats(vx_raw, mask)
        print(f"  {label:<25} {st['count']:>6}  {st['rate_%']:>6.2f}  "
              f"{st['mag_mean']:>9.4f}  {st['mag_max']:>9.4f}  "
              f"{st['run_len_max']:>7}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    out_path = plot_bag(name, df_vio, results, vx_fd, out_dir,
                        fmt=fmt, pdf_pages=pdf_pages)
    print(f"\n  → {out_path}")

    # ── Report figures (clean, single-column) ─────────────────────────────────
    if report:
        rdir = report_dir or out_dir
        rdir.mkdir(parents=True, exist_ok=True)
        rpaths = plot_report_figures(name, df_vio, vx_C, mask_C, vx_base, vx_fd,
                                     rdir, fmt="png",
                                     df_ot=df_ot if has_ot else None)
        for rp in rpaths:
            print(f"  report → {rp}")

    return {
        "bag":           name,
        "n_vio":         len(df_vio),
        "has_optitrack": has_ot,
        "vx_max_raw":    float(np.abs(vx_raw).max()),
        "spike_counts":  {k: int(m.sum()) for k, (_, m) in results.items()},
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Analyze VIO spikes in rosbags.")
    parser.add_argument("--vo-dir", type=Path, default=None,
                        help="Path to the vo/ directory")
    parser.add_argument("--bag",  type=str, default=None,
                        help="Single bag directory name to analyze (e.g. vo_test1)")
    parser.add_argument("--all",  action="store_true",
                        help="Analyze every bag in the vo/ directory")
    parser.add_argument("--format", dest="fmt", default="pdf",
                        choices=["pdf", "svg", "png"],
                        help="Output format for plots (default: pdf)")
    parser.add_argument("--report", action="store_true",
                        help="Also emit clean single-column report figures "
                             "(IX_D-vio_raw_vs_gated, IX_D-vio_gate_ablation) as PNG")
    parser.add_argument("--report-dir", type=Path, default=None,
                        help="Directory for report figures "
                             "(default: ../report/assets if it exists, else the analysis dir)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    vo_dir     = args.vo_dir or (script_dir.parent / "vo")
    if not vo_dir.is_dir():
        vo_dir = script_dir / "vo"
    if not vo_dir.is_dir():
        print(f"Error: cannot locate vo/ directory (tried {vo_dir})")
        return

    out_dir = vo_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    # Report figures default to the report's assets dir when present.
    report_dir = args.report_dir
    if args.report and report_dir is None:
        assets = script_dir.parent / "report" / "assets"
        report_dir = assets if assets.is_dir() else out_dir

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    def _valid_bags():
        return sorted(
            p for p in vo_dir.iterdir()
            if p.is_dir()
            and p.name not in ("csv", "analysis")
            and (p / "metadata.yaml").exists()
        )

    if args.bag:
        bag_dirs = [vo_dir / args.bag]
    elif args.all:
        bag_dirs = _valid_bags()
    else:
        bag_dirs = _valid_bags()[:3]
        print(f"No --bag / --all flag — analyzing first {len(bag_dirs)} bag(s).")

    # Combined PDF (only when format is pdf and processing multiple bags)
    use_combined = args.fmt == "pdf" and len(bag_dirs) > 1
    combined_path = out_dir / "spike_analysis_all.pdf"

    from matplotlib.backends.backend_pdf import PdfPages

    summary = []

    def _run(pdf_pages=None):
        for bp in bag_dirs:
            if not (bp / "metadata.yaml").exists():
                print(f"  Skip {bp.name}: missing metadata.yaml")
                continue
            result = analyze_bag(bp, typestore, out_dir,
                                 fmt=args.fmt, pdf_pages=pdf_pages,
                                 report=args.report, report_dir=report_dir)
            if result:
                summary.append(result)

    if use_combined:
        with PdfPages(combined_path) as pdf_pages:
            _run(pdf_pages)
        print(f"\n  Combined PDF → {combined_path}")
    else:
        _run()

    if len(summary) > 1:
        print(f"\n{'═'*60}")
        print(f"  SUMMARY — {len(summary)} bags")
        print(f"{'═'*60}")
        total = sum(r["n_vio"] for r in summary)
        for method in summary[0]["spike_counts"]:
            n = sum(r["spike_counts"].get(method, 0) for r in summary)
            print(f"  {method:<25}: {n:>5} spikes  ({100*n/total:.2f}% of {total} msgs)")
        print(f"\n  Plots → {out_dir}")


if __name__ == "__main__":
    main()