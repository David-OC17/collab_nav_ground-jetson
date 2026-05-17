# python -m drone_map_create.example.transfer_obstacles

from drone_map_create.transfer_obstacles import (
    TransferConfig,
    ExpectedShape,
    run_pipeline,
)
import cv2

cfg = TransferConfig(
    use_florence2=False,
    florence2_device="cuda",  # "auto" / "cpu" / "cuda"
    expected_shapes=[
        ExpectedShape(
            name="box",
            descriptions=[
                "a cardboard box",
                "a rectangular box seen from above",
                "a square crate",
            ],
            draw_color_bgr=(80, 190, 240),  # yellow
        ),
        ExpectedShape(
            name="cone",
            descriptions=[
                "a traffic cone",
                "an orange safety cone",
            ],
            draw_color_bgr=(0, 140, 255),  # orange
        ),
        ExpectedShape(  # add a third class
            name="barrel",
            descriptions=["a cylindrical barrel", "an oil drum"],
            draw_color_bgr=(50, 200, 50),  # green (BGR)
        ),
    ],
)

final_img, _ = run_pipeline("drone_map_create/out/drone_map.png", "drone_map_create/data/background.png", cfg=cfg)
cv2.imwrite("drone_map_create/out/final_map.png", final_img)
