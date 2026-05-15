from map_to_occupancy import MapConfig, process_map
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# Full MapConfig — every parameter set explicitly
# ══════════════════════════════════════════════════════════════════════════════
cfg = MapConfig(
    # ── resolution / frame ────────────────────────────────────────────────────
    # Real-world size of one pixel in the output occupancy grid (metres/pixel).
    # Calculated as:  arena_real_width_m / cropped_image_width_px
    # Typical indoor arena at ~3 m altitude: 0.01–0.05 m/px.
    resolution=0.01,

    # ── pipeline control ──────────────────────────────────────────────────────
    # Set False to skip Stage 2 (blue-line affine de-warp) entirely.
    # Useful when the drone was nearly vertical and skew is negligible, or
    # while tuning colour thresholds without waiting for the warp step.
    correct_perspective=True,

    # ── brown border (kept for reference; no longer used in Stage 1) ──────────
    # These ranges are still used by the segment() function (Stage 3) to label
    # brown pixels as walls in the occupancy grid.  They are NOT used for the
    # crop boundary — that is now driven by the blue grid lines below.
    #
    # Hue 5–25 covers warm browns in OpenCV [0, 180] convention.
    # Widen h_hi (up to ~30) if the cardboard looks more orange-brown;
    # narrow s_hi / v_hi if interior brown-ish floor is bleeding into the mask.
    brown_h_lo=5,
    brown_h_hi=22,
    brown_s_lo=50,
    brown_s_hi=190,
    brown_v_lo=40,
    brown_v_hi=160,

    # ── blue grid lines ───────────────────────────────────────────────────────
    # These HSV ranges are used in THREE places:
    #   1. Stage 1 (new): scan inward from each image edge to find crop limits.
    #   2. Stage 2: HoughLinesP input for affine de-rotation.
    #   3. Stage 3: free-space mask (blue floor lines = navigable area).
    #
    # Hue 100–130 covers typical blue in OpenCV convention.
    # Shift both bounds toward 90 if the lines look more cyan;
    # toward 135 if they look more purple/indigo.
    blue_h_lo=100,
    blue_h_hi=130,
    blue_s_lo=60,   # raise if grey/desaturated pixels trigger false blue detections
    blue_s_hi=255,
    blue_v_lo=60,   # raise if dark shadows near lines trigger false detections
    blue_v_hi=255,

    # Hough accumulator vote threshold for Stage 2 line detection.
    # Lower  → more (noisier) lines found.
    # Higher → only the strongest lines, fewer false detections.
    hough_threshold=60,

    # Minimum pixel length for a Hough line segment to be accepted.
    hough_min_length=40,

    # Maximum pixel gap inside a line segment before it is split into two.
    hough_max_gap=20,

    # Minimum number of blue lines per axis (H and V separately) to attempt
    # the affine de-rotation in Stage 2.  If fewer lines are found on an axis
    # the rotation is skipped for that axis.
    min_lines_per_axis=3,

    # ── blue-line boundary detection (Stage 1) ────────────────────────────────
    # Minimum number of blue pixels in a single row (or column) for that
    # row/column to be considered the first grid line when scanning inward
    # from the image edge.
    # Increase if edge noise is triggering a premature boundary;
    # decrease if faint/narrow grid lines are being missed.
    blue_edge_min_pixels=8,

    # Thickness in pixels of the all-black occupied border added around the
    # cropped image after Stage 1.  Acts as a hard wall in the occupancy grid.
    contour_thickness_px=10,

    # ── black floor ───────────────────────────────────────────────────────────
    # Pixels with HSV Value ≤ black_v_hi are treated as the black floor (free).
    # Raise if the floor looks dark grey under shadow (e.g. 65–80);
    # lower if bright reflections on the floor are being misclassified as free.
    black_v_hi=65,

    # ── morphological cleanup ─────────────────────────────────────────────────
    # Kernel size (px) for the open→close morphology applied to every colour
    # mask before segmentation.  Larger = removes more noise but also erodes
    # thin features.  Must be odd and ≥ 3.
    morph_kernel=5,

    # ── occupancy values ──────────────────────────────────────────════════════
    # Internal labels stored in the int8 occupancy array before ROS remapping.
    # Only change these if your downstream code expects different intermediate
    # values; the ROS 2 OccupancyGrid publisher remaps them to 0/100/-1 anyway.
    val_free=0,       # black floor + blue lines  → free
    val_obstacle=75,  # yellow boxes + orange cones → dynamic obstacle
    val_wall=95,      # brown border pixels        → wall
)

# ══════════════════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════════════════
grid_msg, debug_img = process_map(
    "out/drone_map_2.png",
    cfg,
    save_debug="out/debug_tuned.png",         # side-by-side original + occupancy overlay
    save_grid_png="out/occupancy_tuned.png",  # raw grayscale: 255=free, 0=occupied, 128=unknown
    verbose=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# Inspect the grid (works without ROS 2 installed)
# ══════════════════════════════════════════════════════════════════════════════
print(f"Grid size   : {grid_msg['width']} × {grid_msg['height']} cells")
print(f"Resolution  : {grid_msg['resolution']} m/px")
print(f"Real extent : {grid_msg['width']*grid_msg['resolution']:.2f} × "
      f"{grid_msg['height']*grid_msg['resolution']:.2f} m")

# grid_msg['data'] is a flat int8 list in ROS row-major order (bottom row first).
grid_array = np.array(grid_msg["data"], dtype=np.int8).reshape(
    grid_msg["height"], grid_msg["width"]
)

total = grid_array.size
print(f"Free    : {(grid_array == 0).sum() / total:.1%}")
print(f"Occupied: {(grid_array >= 75).sum() / total:.1%}")
print(f"Unknown : {(grid_array == -1).sum() / total:.1%}")