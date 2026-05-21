"""
arena_marker_localizer.debug_image
─────────────────────────────────────────────────────────────────────────────
Draws a top-down reference image of the arena showing:

  * a scale grid matching the OccupancyGrid cells,
  * the drone's XY trajectory from the OptiTrack CSV,
  * a cross + ID number at each aggregated marker position.

World/coordinate convention matches the OccupancyGrid produced by
arena_map_builder:
  - origin at the bottom-left of the arena bbox
  - +X right, +Y up (so we flip the image vertically before saving)

The image is sized as
    width_px  = grid_width_cells  * px_per_cell
    height_px = grid_height_cells * px_per_cell
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import os
import math
import numpy as np
import cv2

from .pipeline import MarkerResult
from .optitrack import DronePose


# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class DebugImageConfig:
    grid_width_cells:  int   = 80
    grid_height_cells: int   = 80
    resolution_m_per_cell: float = 0.05
    px_per_cell:       int   = 12   # tunable resolution of the debug image

    # Visual tuning
    background_bgr:   Tuple[int, int, int] = (255, 255, 255)
    grid_minor_bgr:   Tuple[int, int, int] = (220, 220, 220)
    grid_major_bgr:   Tuple[int, int, int] = (170, 170, 170)
    grid_major_every: int = 10        # bold line every N cells (≈ every 0.5m at 0.05 m/cell)
    trajectory_bgr:   Tuple[int, int, int] = (60, 180, 60)
    trajectory_thickness_px: int = 2
    marker_cross_bgr: Tuple[int, int, int] = (40, 40, 220)
    marker_cross_radius_px: int = 16
    marker_label_bgr: Tuple[int, int, int] = (40, 40, 220)
    marker_label_scale: float = 0.7

    # Full T_map_from_opti @ flip matrix (4×4, same as pipeline.py builds).
    # Applied to each drone OptiTrack position to get map-frame coordinates.
    T_opti_to_map: np.ndarray = field(default_factory=lambda: np.eye(4))


# ─────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────

def _world_to_px(x_m: float, y_m: float, cfg: DebugImageConfig,
                 H_px: int) -> Tuple[int, int]:
    """Map a world (metres, bottom-left origin) point to image pixel
    coords (top-left origin)."""
    px_per_m = cfg.px_per_cell / cfg.resolution_m_per_cell
    u = int(round(x_m * px_per_m))
    v = int(round(H_px - 1 - y_m * px_per_m))    # flip Y for image coords
    return u, v


def _draw_grid(img: np.ndarray, cfg: DebugImageConfig):
    H, W = img.shape[:2]
    step = cfg.px_per_cell
    # Minor lines first, major on top
    for i in range(0, cfg.grid_width_cells + 1):
        x_px = i * step
        if x_px >= W:
            continue
        col = (cfg.grid_major_bgr if i % cfg.grid_major_every == 0
               else cfg.grid_minor_bgr)
        cv2.line(img, (x_px, 0), (x_px, H - 1), col, 1, cv2.LINE_AA)
    for j in range(0, cfg.grid_height_cells + 1):
        y_px = j * step
        if y_px >= H:
            continue
        col = (cfg.grid_major_bgr if j % cfg.grid_major_every == 0
               else cfg.grid_minor_bgr)
        cv2.line(img, (0, y_px), (W - 1, y_px), col, 1, cv2.LINE_AA)

    # Axis labels at major intersections (every grid_major_every cells)
    label_color = (130, 130, 130)
    px_per_m = cfg.px_per_cell / cfg.resolution_m_per_cell
    for i in range(0, cfg.grid_width_cells + 1, cfg.grid_major_every):
        x_m = i * cfg.resolution_m_per_cell
        u = i * step
        if u + 4 < W:
            cv2.putText(img, f"{x_m:.1f}", (u + 4, H - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        label_color, 1, cv2.LINE_AA)
    for j in range(0, cfg.grid_height_cells + 1, cfg.grid_major_every):
        y_m = j * cfg.resolution_m_per_cell
        v = H - 1 - j * step
        if v - 4 > 0:
            cv2.putText(img, f"{y_m:.1f}", (4, v - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        label_color, 1, cv2.LINE_AA)


def _draw_trajectory(img: np.ndarray, drone_poses: List[DronePose],
                     cfg: DebugImageConfig):
    if not drone_poses:
        return
    H, W = img.shape[:2]
    pts: List[Tuple[int, int]] = []
    for dp in drone_poses:
        p_opti = np.array([dp.pos_xyz[0], dp.pos_xyz[1], dp.pos_xyz[2], 1.0])
        p_map = cfg.T_opti_to_map @ p_opti
        u, v = _world_to_px(p_map[0], p_map[1], cfg, H)
        pts.append((u, v))

    if len(pts) >= 2:
        pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            img, [pts_arr], isClosed=False,
            color=cfg.trajectory_bgr,
            thickness=cfg.trajectory_thickness_px,
            lineType=cv2.LINE_AA,
        )
    # Mark start and end of the trajectory
    if pts:
        cv2.circle(img, pts[0], 5, (60, 180, 60), -1, cv2.LINE_AA)
        cv2.circle(img, pts[-1], 5, (30, 100, 30), -1, cv2.LINE_AA)


def _draw_marker(img: np.ndarray, m: MarkerResult, cfg: DebugImageConfig):
    H, W = img.shape[:2]
    u, v = _world_to_px(m.position_m[0], m.position_m[1], cfg, H)
    r = cfg.marker_cross_radius_px
    col = cfg.marker_cross_bgr
    # Filled small circle in the centre + cross arms
    cv2.line(img, (u - r, v), (u + r, v), col, 2, cv2.LINE_AA)
    cv2.line(img, (u, v - r), (u, v + r), col, 2, cv2.LINE_AA)
    cv2.circle(img, (u, v), 4, col, -1, cv2.LINE_AA)

    # Yaw arrow — short tick showing the marker's facing direction.
    arrow_len = r + 6
    yaw = m.yaw_rad
    end_x = u + int(round(arrow_len * math.cos(yaw)))
    # screen-y axis points down, so flip the sin component
    end_y = v - int(round(arrow_len * math.sin(yaw)))
    cv2.arrowedLine(img, (u, v), (end_x, end_y), col, 2,
                    line_type=cv2.LINE_AA, tipLength=0.35)

    # ID label
    label = f"id={m.id}" if hasattr(m, "id") else f"id={m.marker_id}"
    # Render with outline so it's readable on a busy background.
    cv2.putText(img, label, (u + r + 4, v - 4),
                cv2.FONT_HERSHEY_SIMPLEX, cfg.marker_label_scale,
                (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, label, (u + r + 4, v - 4),
                cv2.FONT_HERSHEY_SIMPLEX, cfg.marker_label_scale,
                cfg.marker_label_bgr, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────

def render_debug_image(
    markers:     Iterable[MarkerResult],
    drone_poses: List[DronePose],
    cfg:         DebugImageConfig,
    output_path: str,
) -> str:
    """Render the debug image and write it to `output_path`. Returns the
    path written (same as the input). The destination directory is
    created if missing."""
    W_px = cfg.grid_width_cells  * cfg.px_per_cell
    H_px = cfg.grid_height_cells * cfg.px_per_cell
    img = np.full((H_px, W_px, 3),
                  fill_value=np.array(cfg.background_bgr, dtype=np.uint8),
                  dtype=np.uint8)

    _draw_grid(img, cfg)
    _draw_trajectory(img, drone_poses, cfg)
    for m in markers:
        _draw_marker(img, m, cfg)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)
    return output_path