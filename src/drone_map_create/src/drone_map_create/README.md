# Drone map create

## Dependencies

With Python 3.10:

```bash
pip install opencv-python numpy transformers torch pillow einops timm
```

## Usage

Invoke creation of map from drone video (stitching):
```Python


```

Invoke post processing of map image into a flat grid:
```bash
# 1) Simplest: default config (heuristic only, bbox projection mode)
python transfer_obstacles.py drone_map-sample2.png \
    --background background.png \
    --out result.png

# 2) Common case: write per-stage debug images, more closing passes,
#    use the blue-grid projection mode
python transfer_obstacles.py drone_map-sample2.png \
    --background background.png \
    --out result.png \
    --close-iters 5 \
    --project-mode grid \
    --debug-dir debug/

# 3) Strict pipeline: verify every blob with Florence-2 and keep
#    blobs the heuristic couldn't classify (drawn in gray)
python transfer_obstacles.py drone_map-sample2.png \
    -b background.png \
    -o result.png \
    --use-florence2 \
    --florence2-device cuda \
    --keep-unknown
```

```Python
# python -m drone_map_create.example.transfer_obstacles

from drone_map_create.transfer_obstacles import (
    TransferConfig,
    ExpectedShape,
    run_pipeline,
)
import cv2

# 1) Quick-start with defaults
final_img, stages = run_pipeline(
    "drone_map-sample2.png",
    "background.png",
)
cv2.imwrite("result.png", final_img)

# 2) Manual control: customize config, capture intermediate stages
cfg = TransferConfig(
    close_iterations=5,  # more aggressive blob fill
    project_mode="grid",  # piecewise-linear via blue grid
    drop_unknown=False,  # render gray for unclassifiable blobs
    min_blob_area_frac=0.0008,  # be more permissive on small objects
)

final_img, stages = run_pipeline(
    "drone_map-sample2.png",
    "background.png",
    cfg=cfg,
    debug_dir="debug/",  # write per-stage PNGs
    verbose=True,
)

cv2.imwrite("result.png", final_img)
cv2.imwrite("debug_cleaned.png", stages["wall_masked"])  # stage-2b output
cv2.imwrite("debug_overlay.png", stages["blob_overlay"])  # stage-4 labels

# 3) Custom expected-shapes list + Florence-2 verification
cfg = TransferConfig(
    use_florence2=True,
    florence2_device="cuda",  # "auto" / "cpu" / "cuda"
    expected_shapes=[
        ExpectedShape(
            name="box",
            descriptions=[
                "a cardboard box",
                "a rectangular box seen from above",
                "a square crate",
            ],
            draw_color_bgr=(200, 120, 40),  # blue-ish (BGR)
        ),
        ExpectedShape(
            name="cone",
            descriptions=[
                "a traffic cone",
                "an orange safety cone",
            ],
            draw_color_bgr=(0, 140, 255),  # orange (BGR)
        ),
        ExpectedShape(  # add a third class
            name="barrel",
            descriptions=["a cylindrical barrel", "an oil drum"],
            draw_color_bgr=(50, 200, 50),  # green (BGR)
        ),
    ],
)

final_img, _ = run_pipeline("drone_map.png", "background.png", cfg=cfg)
cv2.imwrite("result.png", final_img)
```

Utilize the ROS2 wrappers implemented as actions for these two tasks.

```bash
ros2 param set /build_arena_map_server transfer.background_path /home/david/Documents/UNI_S.8/Robo/project/collab_nav_ground-jetson/src/arena_map_builder/data/background.png

python example_client.py /home/david/Documents/UNI_S.8/Robo/project/collab_nav_ground-jetson/src/arena_map_builder/data/drone_scans/scan6/scan.mp4
```