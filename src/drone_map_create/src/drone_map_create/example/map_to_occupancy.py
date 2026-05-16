# python -m drone_map_create.example.example_map_to_occupancy

import glob
import os
import time
import traceback

import numpy as np

from drone_map_create.map_to_occupancy import MapConfig, process_map

# ══════════════════════════════════════════════════════════════════════════════
# Paths
# __file__ = drone_map_create/example/example_map_to_occupancy.py
# ══════════════════════════════════════════════════════════════════════════════
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))    # …/drone_map_create/example/
OUT_DIR     = os.path.join(EXAMPLE_DIR, "out")              # …/drone_map_create/example/out/
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# Config — applied to every map
# ══════════════════════════════════════════════════════════════════════════════
cfg = MapConfig(
    resolution=0.01,
    correct_perspective=True,

    # Blue grid lines
    blue_h_lo=100,
    blue_h_hi=130,
    blue_s_lo=60,
    blue_s_hi=255,
    blue_v_lo=60,
    blue_v_hi=255,
    hough_threshold=60,
    hough_min_length=40,
    hough_max_gap=20,
    min_lines_per_axis=3,
    blue_edge_min_pixels=8,
    contour_thickness_px=10,

    # White obstacle detection
    white_s_hi=40,
    white_v_lo=200,

    # Black floor
    black_v_hi=65,

    # Morphology
    morph_kernel=5,
    morph_close_iterations=2,

    # Output
    intermediate_cell_value=25,
)

# ══════════════════════════════════════════════════════════════════════════════
# Discover maps produced by example_drone_map_gen.py
# Expects: drone_map_create/example/out/drone_map_scan*.png
# ══════════════════════════════════════════════════════════════════════════════
input_maps = sorted(glob.glob(os.path.join(OUT_DIR, "drone_map_scan*.png")))

if not input_maps:
    raise FileNotFoundError(
        f"No drone_map_scan*.png files found in {OUT_DIR}.\n"
        "Run example_drone_map_gen.py first to generate the stitched maps."
    )

print(f"Found {len(input_maps)} map(s):\n" +
      "\n".join(f"  {p}" for p in input_maps))

# ══════════════════════════════════════════════════════════════════════════════
# Process each map
# ══════════════════════════════════════════════════════════════════════════════
results  = {}
failures = {}

for map_path in input_maps:
    # e.g. "drone_map_scan3.png" → stem "drone_map_scan3"
    stem = os.path.splitext(os.path.basename(map_path))[0]  # drone_map_scan3
    # strip "drone_map_" prefix to get "scan3" for output names
    scan_name    = stem.replace("drone_map_", "")            # scan3

    out_debug    = os.path.join(OUT_DIR, f"occupancy_debug_{scan_name}.png")
    out_grid_png = os.path.join(OUT_DIR, f"occupancy_{scan_name}.png")

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Processing {stem}  ->  occupancy_{scan_name}.png")
    print(sep)

    t0 = time.time()
    try:
        grid, debug_img = process_map(
            map_path,
            cfg,
            save_debug=out_debug,
            save_grid_png=out_grid_png,
            verbose=True,
        )
        elapsed = time.time() - t0

        grid_array = np.array(grid["data"], dtype=np.int8).reshape(
            grid["height"], grid["width"]
        )
        tgt   = cfg.intermediate_cell_value
        total = grid_array.size
        occ_pct  = (grid_array == 100).sum() / total
        free_pct = (grid_array == tgt).sum()  / total

        print(f"  Grid : {grid['width']}x{grid['height']} cells "
              f"@ {grid['resolution']} m/px  "
              f"({grid['width']*grid['resolution']:.2f}x"
              f"{grid['height']*grid['resolution']:.2f} m)")
        print(f"  Free : {free_pct:.1%}   Occupied : {occ_pct:.1%}")
        print(f"  Done in {elapsed:.1f}s")

        results[scan_name] = {
            "grid_png": out_grid_png,
            "debug":    out_debug,
            "elapsed":  elapsed,
            "free":     free_pct,
            "occupied": occ_pct,
        }

    except Exception as exc:
        elapsed = time.time() - t0
        failures[scan_name] = str(exc)
        print(f"  [FAILED] {scan_name} after {elapsed:.1f}s: {exc}")
        traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  Completed {len(results)}/{len(input_maps)} map(s).")
for name, info in results.items():
    print(f"  [OK]   {name:10s}  {info['elapsed']:5.1f}s  "
          f"free={info['free']:.1%}  occ={info['occupied']:.1%}  "
          f"->  {info['grid_png']}")
for name, err in failures.items():
    print(f"  [FAIL] {name:10s}  {err}")
print("=" * 60)