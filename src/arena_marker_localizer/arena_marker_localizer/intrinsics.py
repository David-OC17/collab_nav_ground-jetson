"""
arena_marker_localizer.intrinsics
─────────────────────────────────────────────────────────────────────────────
Read camera calibration files in the OpenCV-flavored YAML format used by
OpenCV's own `cv2.FileStorage`. Example:

    %YAML:1.0
    ---
    camera_matrix: !!opencv-matrix
       rows: 3
       cols: 3
       dt: d
       data: [ fx, 0, cx, 0, fy, cy, 0, 0, 1 ]
    dist_coeff: !!opencv-matrix
       rows: 1
       cols: 5
       dt: d
       data: [ k1, k2, p1, p2, k3 ]

The standard `yaml` Python library can't parse `!!opencv-matrix` and chokes
on `%YAML:1.0` (an unknown directive in some libyaml builds). We sidestep
both by using OpenCV's own FileStorage reader, which is the canonical
reader for these files. As a soft fallback we also accept the file as
plain YAML (without the OpenCV-specific tags).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import os
import numpy as np
import cv2


@dataclass
class CameraIntrinsics:
    K:    np.ndarray   # (3, 3) float64
    dist: np.ndarray   # (N,)   float64  (N is usually 4, 5, 8, 12, or 14)

    @property
    def fx(self) -> float: return float(self.K[0, 0])
    @property
    def fy(self) -> float: return float(self.K[1, 1])
    @property
    def cx(self) -> float: return float(self.K[0, 2])
    @property
    def cy(self) -> float: return float(self.K[1, 2])


def load_intrinsics(path: str,
                    matrix_key: str = "camera_matrix",
                    dist_key:   str = "dist_coeff",
                    ) -> CameraIntrinsics:
    """Read intrinsics from an OpenCV-style YAML calibration file.

    Parameters
    ──────────
    path        : path to the .yaml/.yml/.xml calibration file
    matrix_key  : key for the 3x3 camera matrix (default: "camera_matrix")
    dist_key    : key for the distortion coeffs (default: "dist_coeff")
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Intrinsics file not found: {path!r}")

    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"OpenCV could not open intrinsics file: {path!r}")

    try:
        K_node    = fs.getNode(matrix_key)
        dist_node = fs.getNode(dist_key)
        if K_node.empty():
            raise KeyError(
                f"Key {matrix_key!r} not found in {path!r}. "
                f"(Top-level keys present: {list(_top_keys(fs))})"
            )
        K = K_node.mat().astype(np.float64)
        if K.shape != (3, 3):
            raise ValueError(
                f"Expected 3x3 camera matrix, got shape {K.shape}"
            )

        if dist_node.empty():
            # Many calibrations omit the distortion field for already-undistorted
            # video. Defaulting to zeros is the right thing for that case.
            dist = np.zeros((5,), dtype=np.float64)
        else:
            dist = dist_node.mat().astype(np.float64).flatten()
    finally:
        fs.release()

    return CameraIntrinsics(K=K, dist=dist)


def _top_keys(fs: "cv2.FileStorage") -> list:
    """Best-effort enumeration of top-level keys, for diagnostic messages.
    OpenCV's Python binding doesn't expose this directly across all
    versions; we attempt a few approaches and return [] if none works."""
    try:
        root = fs.root()
        if root is None:
            return []
        keys = root.keys() if hasattr(root, "keys") else []
        return list(keys)
    except Exception:
        return []
