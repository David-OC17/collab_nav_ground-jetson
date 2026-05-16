
# python -m drone_map_create.example.drone_map_gen-single

from drone_map_create.drone_map_gen import (
    reconstruct_from_video,
    ExtractionConfig,
    ReconstructConfig,
    ColorRangeMask,
)

# ══════════════════════════════════════════════════════════════════════════════
# Frame extraction config
# Controls which frames are pulled from the video and accepted for stitching.
# ══════════════════════════════════════════════════════════════════════════════
extract_cfg = ExtractionConfig(
    # How many frames per second to sample from the source video.
    # Higher = more overlap between frames (better stitching, more RAM/CPU).
    # Lower  = faster processing but may miss fast camera movements.
    target_fps=8.0,

    # Laplacian-variance threshold. Frames whose sharpness score is below
    # this are considered blurry and skipped.
    # Raise if motion-blurred frames are still slipping through;
    # lower if too many valid frames are being dropped.
    blur_thresh=50.0,

    # DCT block-artifact ratio threshold. Frames above this value show
    # heavy JPEG/codec compression artifacts and are discarded.
    # Lower = stricter (rejects more compressed frames).
    artifact_thresh=1.5,

    # Minimum normalised mean feature displacement between consecutive kept
    # frames. Frames with less movement than this are considered static
    # (drone hovering) and skipped to avoid redundant canvas coverage.
    # Range: 0.0–1.0 (fraction of the frame diagonal).
    min_movement=0.015,

    # Maximum normalised mean feature displacement. Frames above this are
    # considered jerky or tracking failures (too much camera shake) and dropped.
    # Range: 0.0–1.0 (fraction of the frame diagonal).
    max_movement=0.55,
)

# ══════════════════════════════════════════════════════════════════════════════
# Pre-stitch color masking
# Each ColorRangeMask defines one HSV range whose pixels are replaced with
# a solid colour BEFORE a frame is blended onto the canvas.
# Feature detection for homography always uses the original unmasked frame,
# so alignment quality is not affected by masking.
# ══════════════════════════════════════════════════════════════════════════════
color_masks = [
    # ── yellow obstacles ────────────────────────────────────────────────────
    # Hue 18–38 covers most saturated yellows in OpenCV [0,180] convention.
    # Raise yellow_h_hi if warm yellows are missed; lower yellow_h_lo if
    # orange cones bleed into this range (prefer the orange mask below).
    ColorRangeMask(
        name="yellow",
        h_lo=18,  h_hi=38,    # hue  [0, 180]
        s_lo=80,  s_hi=255,   # saturation — high floor rejects pale/grey yellows
        v_lo=80,  v_hi=255,   # value      — excludes very dark yellow-brown
        replace_bgr=(255, 0, 255),   # → bright pink
    ),

    # ── brown border / cardboard ─────────────────────────────────────────────
    # Hue 5–25 captures warm brown tones.
    # Narrow s_hi / v_hi if interior brown-ish floor texture is being masked;
    # widen if the border cardboard has uneven lighting.
    ColorRangeMask(
        name="brown",
        h_lo=5,   h_hi=25,
        s_lo=50,  s_hi=255,
        v_lo=40,  v_hi=180,   # v_hi=180 keeps out very bright reflective patches
        replace_bgr=(255, 0, 255),
    ),

    # ── orange cones ─────────────────────────────────────────────────────────
    # Hue 5–18 — narrower than brown, higher saturation floor to avoid skin tones.
    ColorRangeMask(
        name="orange",
        h_lo=5,   h_hi=18,
        s_lo=120, s_hi=255,   # high s_lo separates orange from tan/beige
        v_lo=80,  v_hi=255,
        replace_bgr=(255, 0, 255),
    ),

    # ── example: add more ranges as needed ───────────────────────────────────
    # ColorRangeMask(
    #     name="lime_green",
    #     h_lo=35, h_hi=75,
    #     s_lo=60, s_hi=255,
    #     v_lo=60, v_hi=255,
    #     replace_bgr=(255, 0, 255),
    # ),
]

# ══════════════════════════════════════════════════════════════════════════════
# Reconstruction / stitching config
# ══════════════════════════════════════════════════════════════════════════════
reconstruct_cfg = ReconstructConfig(
    # Initial blank padding (px) added around the first frame on all four sides.
    # Increase if the canvas needs to expand very often during early frames
    # (costs a small amount of extra RAM at startup, saves repeated reallocs).
    canvas_margin=2000,

    # Minimum number of RANSAC inliers required to accept a homography.
    # Raise for stricter alignment (fewer bad placements, more dropped frames);
    # lower if the drone moves fast and features are sparse.
    min_inliers=20,

    # Number of recently placed frames kept in the ring buffer for re-matching.
    # Higher = better recovery from drift but more matching work per frame.
    lookback=6,

    # A global keyframe is archived every N successfully placed frames.
    # Keyframes act as long-range anchors for loop closure / re-localisation.
    # Lower = more anchors (better drift recovery, more RAM).
    keyframe_interval=15,

    # Blending strategy for overlapping frame regions:
    #   "feather"  – distance-weighted linear blend  (fast, good for clean maps)
    #   "pyramid"  – Laplacian multi-band blend      (best quality, ~3× slower)
    #   "flat"     – simple 50/50 average            (debug / benchmarking only)
    blend_mode="pyramid",

    # Laplacian pyramid depth; only used when blend_mode="pyramid".
    # Higher = smoother seams at fine detail; also slightly slower.
    pyramid_levels=4,

    # Downsample every incoming frame by this factor before processing.
    # 1.0 = full resolution.  0.5 = half (4× smaller canvas and warp buffers).
    # Use 0.5 for large arenas or high-resolution video to save RAM.
    processing_scale=1.0,

    # Pre-stitch color masks defined above.
    # Set to [] (or omit) to disable all masking.
    color_masks=color_masks,
)

# ══════════════════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════════════════
result = reconstruct_from_video(
    "drone_map_create/data/drone_scans/scan6/scan.mp4",
    output_shape=(2000, 2000),   # (W, H) to resize the final map; None = native
    save_path="drone_map_create/out/drone_map-single.png",
    extract_cfg=extract_cfg,
    reconstruct_cfg=reconstruct_cfg,
    verbose=True,
)