# python -m drone_map_create.example.drone_map_csv_gen

from drone_map_create.drone_map_csv_gen import (
    reconstruct_from_csv,
    CSVStitchConfig,
    CameraIntrinsics,
    CoordinateConfig,
    ColorRangeMask,
)

intr = CameraIntrinsics(
    fx=1370.5,
    fy=1369.8,
    cx=961.2,
    cy=541.7,
    dist_coeffs=(-0.243, 0.087, 0.0001, -0.0002, 0.0),  # k1, k2, p1, p2, k3
)

cfg = CSVStitchConfig(
    intrinsics=intr, undistort_frames=False, refine_yaw=False, calibrate_yaw_offset=True
)

reconstruct_from_csv(
    "drone_map_create/data/drone_scans/scan1/scan.mp4",
    "drone_map_create/data/drone_scans/scan1/telemetry.csv",
    cfg,
    output_shape=(2000, 2000),
    save_path="out_1_pure.png",
)
