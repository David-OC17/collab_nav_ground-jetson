# python -m drone_map_create.example.drone_map_gen-all

import glob
import os
import time
import traceback

from drone_map_create.drone_map_gen import (
    reconstruct_from_video,
    ExtractionConfig,
    ReconstructConfig,
    ColorRangeMask,
)

# ══════════════════════════════════════════════════════════════════════════════
# Paths
# __file__ = drone_map_create/example/example_drone_map_gen.py
# Go up one level to reach drone_map_create/, where data/ lives.
# ══════════════════════════════════════════════════════════════════════════════
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))    # …/drone_map_create/example/
PKG_DIR     = os.path.dirname(EXAMPLE_DIR)                  # …/drone_map_create/
SCANS_DIR   = os.path.join(PKG_DIR, "data", "drone_scans")  # …/drone_map_create/data/drone_scans/
OUT_DIR     = os.path.join(EXAMPLE_DIR, "out")              # …/drone_map_create/example/out/

os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# Shared configs (applied to every scan)
# ══════════════════════════════════════════════════════════════════════════════
extract_cfg = ExtractionConfig(
    target_fps=8.0,
    blur_thresh=50.0,
    artifact_thresh=1.5,
    min_movement=0.015,
    max_movement=0.55,
)

color_masks = [
    ColorRangeMask(
        name="yellow",
        h_lo=18,  h_hi=38,
        s_lo=80,  s_hi=255,
        v_lo=80,  v_hi=255,
        replace_bgr=(255, 255, 255),
    ),
    ColorRangeMask(
        name="brown",
        h_lo=5,   h_hi=25,
        s_lo=50,  s_hi=255,
        v_lo=40,  v_hi=180,
        replace_bgr=(255, 255, 255),
    ),
    ColorRangeMask(
        name="orange",
        h_lo=5,   h_hi=18,
        s_lo=120, s_hi=255,
        v_lo=80,  v_hi=255,
        replace_bgr=(255, 255, 255),
    ),
]

reconstruct_cfg = ReconstructConfig(
    canvas_margin=2000,
    min_inliers=20,
    lookback=6,
    keyframe_interval=15,
    blend_mode="pyramid",
    pyramid_levels=4,
    processing_scale=1.0,
    color_masks=color_masks,
)

# ══════════════════════════════════════════════════════════════════════════════
# Discover scans
# ══════════════════════════════════════════════════════════════════════════════
scan_videos = sorted(glob.glob(os.path.join(SCANS_DIR, "scan*", "scan.mp4")))

if not scan_videos:
    raise FileNotFoundError(
        f"No scan.mp4 files found under {SCANS_DIR}. "
        "Check that SCANS_DIR points to the right location."
    )

print(f"Found {len(scan_videos)} scan(s):\n" +
      "\n".join(f"  {v}" for v in scan_videos))

# ══════════════════════════════════════════════════════════════════════════════
# Process each scan
# ══════════════════════════════════════════════════════════════════════════════
results  = {}
failures = {}

for video_path in scan_videos:
    # Derive a clean name from the folder, e.g. "scan3"
    scan_name = os.path.basename(os.path.dirname(video_path))
    save_path = os.path.join(OUT_DIR, f"drone_map_{scan_name}.png")

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Processing {scan_name}  ->  {save_path}")
    print(sep)

    t0 = time.time()
    try:
        reconstruct_from_video(
            video_path,
            output_shape=(3900, 3900),
            save_path=save_path,
            extract_cfg=extract_cfg,
            reconstruct_cfg=reconstruct_cfg,
            verbose=True,
        )
        elapsed = time.time() - t0
        results[scan_name] = {"path": save_path, "elapsed": elapsed}
        print(f"  Done in {elapsed:.1f}s  ->  {save_path}")

    except Exception as exc:
        elapsed = time.time() - t0
        failures[scan_name] = str(exc)
        print(f"  [FAILED] {scan_name} after {elapsed:.1f}s: {exc}")
        traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  Completed {len(results)}/{len(scan_videos)} scan(s).")
for name, info in results.items():
    print(f"  [OK]   {name:10s}  {info['elapsed']:5.1f}s  ->  {info['path']}")
for name, err in failures.items():
    print(f"  [FAIL] {name:10s}  {err}")
print("=" * 60)