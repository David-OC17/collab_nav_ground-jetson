# python -m drone_map_create.example.drone_map_grid_gen

from drone_map_create.drone_map_grid_gen import (
    reconstruct_from_video,
    ExtractionConfig,
    ReconstructConfig,
    ColorRangeMask,
)

extract_cfg = ExtractionConfig(
    target_fps=8.0,
    blur_thresh=50.0,
    artifact_thresh=2.0,
    min_movement=0.015,
    max_movement=0.55,
)

color_masks = [
    ColorRangeMask(
        name="yellow",
        h_lo=35,  h_hi=45,
        s_lo=50,  s_hi=255,
        v_lo=80,  v_hi=255,
        replace_bgr=(0, 255, 0),
    ),
    ColorRangeMask(
        name="brown",
        h_lo=15,   h_hi=30,
        s_lo=50,  s_hi=255,
        v_lo=125,  v_hi=255,
        replace_bgr=(255, 0, 255),
    ),
    ColorRangeMask(
        name="orange",
        h_lo=10,   h_hi=30,
        s_lo=50, s_hi=255,
        v_lo=20,  v_hi=125,
        replace_bgr=(0, 255, 0),
    ),
]

reconstruct_cfg = ReconstructConfig(
    canvas_margin=2000,

    # ── Alignment thresholds ─────────────────────────────────────────────────
    min_inliers=10,
    lookback=15,
    keyframe_interval=8,
    max_keyframes=60,

    # ── Blending ─────────────────────────────────────────────────────────────
    blend_mode="pyramid",
    pyramid_levels=4,
    processing_scale=1.0,
    color_masks=color_masks,

    # ── Feature detection ────────────────────────────────────────────────────
    feature_exclude_hsv=[],

    # ── Blue grid intersection alignment ───────────────────────────
    use_grid_intersections=True,
    grid_match_dist=25.0,
    grid_min_intersections=4,
    grid_hsv_h_lo=90,
    grid_hsv_h_hi=130,
    grid_hsv_s_lo=50,
    grid_hsv_v_lo=50,
)

result = reconstruct_from_video(
    "drone_map_create/data/square_cut/manual_scan4-square.mp4",
    # "drone_map_create/data/drone_scans/scan6/scan.mp4",
    output_shape=(2000, 2000),
    save_path="drone_map_create/out/drone_map.png",
    extract_cfg=extract_cfg,
    reconstruct_cfg=reconstruct_cfg,
    verbose=True,
)