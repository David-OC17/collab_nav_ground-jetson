# python -m drone_map_create.example.map_to_occupancy-single

import os
import time
import traceback

import numpy as np

from drone_map_create.map_to_occupancy import MapConfig, process_map

# ══════════════════════════════════════════════════════════════════════════════
# Config
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
# Single map processing
# ══════════════════════════════════════════════════════════════════════════════
def create_occupancy_map(
    map_path: str,
    save_grid_path: str,
    save_debug_path: str | None = None,
    cfg: MapConfig = cfg,
    verbose: bool = True,
):
    """
    Convert a stitched drone map into an occupancy grid.

    Args:
        map_path: Path to input map image.
        save_grid_path: Path to save occupancy PNG.
        save_debug_path: Optional debug visualization output path.
        cfg: MapConfig instance.
        verbose: Enable verbose processing output.

    Returns:
        dict containing occupancy grid metadata.
    """

    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Input map not found: {map_path}")

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Processing map:")
    print(f"  Input : {map_path}")
    print(f"  Output: {save_grid_path}")
    print(sep)

    t0 = time.time()

    try:
        grid, debug_img = process_map(
            map_path,
            cfg,
            save_debug=save_debug_path,
            save_grid_png=save_grid_path,
            verbose=verbose,
        )

        elapsed = time.time() - t0

        grid_array = np.array(grid["data"], dtype=np.int8).reshape(
            grid["height"], grid["width"]
        )

        tgt = cfg.intermediate_cell_value
        total = grid_array.size

        occ_pct = (grid_array == 100).sum() / total
        free_pct = (grid_array == tgt).sum() / total

        print(f"Grid : {grid['width']}x{grid['height']} cells "
              f"@ {grid['resolution']} m/px "
              f"({grid['width'] * grid['resolution']:.2f} x "
              f"{grid['height'] * grid['resolution']:.2f} m)")

        print(f"Free : {free_pct:.1%}")
        print(f"Occupied : {occ_pct:.1%}")
        print(f"Done in {elapsed:.1f}s")

        return {
            "grid": grid,
            "grid_png": save_grid_path,
            "debug_png": save_debug_path,
            "elapsed": elapsed,
            "free": free_pct,
            "occupied": occ_pct,
        }

    except Exception as exc:
        elapsed = time.time() - t0

        print(f"[FAILED] after {elapsed:.1f}s: {exc}")
        traceback.print_exc()

        raise


# ══════════════════════════════════════════════════════════════════════════════
# Example usage
# ══════════════════════════════════════════════════════════════════════════════
create_occupancy_map(
    map_path="drone_map_create/out/drone_map.png",
    save_grid_path="drone_map_create/out/occupancy_map.png",
    save_debug_path="drone_map_create/out/occupancy_debug.png",
    cfg=cfg,
    verbose=True,
)