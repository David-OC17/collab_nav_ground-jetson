"""
arena_marker_localizer.marker_detection
─────────────────────────────────────────────────────────────────────────────
Multi-dictionary ArUco detection + per-marker pose estimation in the
camera frame using `solvePnP`. Returns `(dict_name, marker_id,
T_cam_from_marker)` per detected marker.

OpenCV ArUco API note
─────────────────────
This module uses the modern ArUco API (`ArucoDetector` + `getPredefinedDictionary`)
available from OpenCV 4.7+. It falls back to the legacy `Dictionary_get` +
`detectMarkers(image, dictionary)` API when only that's available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from .intrinsics import CameraIntrinsics


# ─────────────────────────────────────────────────────────────────────────
# Dictionary helpers
# ─────────────────────────────────────────────────────────────────────────

def aruco_dict_id(name: str) -> int:
    """Map a string like 'DICT_4X4_50' to OpenCV's int constant."""
    cleaned = name.strip().upper()
    if not hasattr(cv2.aruco, cleaned):
        raise ValueError(
            f"Unknown ArUco dictionary {name!r}. "
            f"Examples: DICT_4X4_50, DICT_5X5_100, DICT_6X6_250, "
            f"DICT_ARUCO_ORIGINAL."
        )
    return int(getattr(cv2.aruco, cleaned))


@dataclass
class DictionaryConfig:
    """One ArUco dictionary to scan with, and its expected marker size."""
    name: str                  # e.g. "DICT_4X4_50"
    marker_size_m: float       # physical side length of the printed marker


@dataclass
class Detection:
    dict_name:    str
    marker_id:    int
    T_cam_marker: np.ndarray   # 4x4, marker pose in the camera frame
    corners_px:   np.ndarray   # (4, 2) — pixel coords of the marker corners
    reproj_err:   float        # mean reprojection error in pixels


# ─────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────

class MultiDictDetector:
    """Holds one OpenCV detector per configured dictionary, plus the
    pre-built object-point templates per dictionary (since marker_size
    can vary by dictionary)."""

    def __init__(self, dictionaries: List[DictionaryConfig]):
        if not dictionaries:
            raise ValueError("Need at least one DictionaryConfig.")
        self._dicts = list(dictionaries)
        self._has_new_api = hasattr(cv2.aruco, "ArucoDetector")

        self._detectors = []   # one per dict (for the modern API)
        self._legacy_dicts = []
        self._obj_pts: List[np.ndarray] = []
        for dc in self._dicts:
            d = cv2.aruco.getPredefinedDictionary(aruco_dict_id(dc.name))
            if self._has_new_api:
                params = cv2.aruco.DetectorParameters()
                detector = cv2.aruco.ArucoDetector(d, params)
                self._detectors.append(detector)
            else:
                self._legacy_dicts.append(d)
            half = dc.marker_size_m / 2.0
            self._obj_pts.append(np.array([
                [-half,  half, 0.0],
                [ half,  half, 0.0],
                [ half, -half, 0.0],
                [-half, -half, 0.0],
            ], dtype=np.float64))

    def detect(self, frame_bgr: np.ndarray,
               intrinsics: CameraIntrinsics) -> List[Detection]:
        """Run every configured detector on the frame and return all
        successfully pose-estimated markers."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        out: List[Detection] = []
        for i, dc in enumerate(self._dicts):
            if self._has_new_api:
                corners, ids, _ = self._detectors[i].detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray, self._legacy_dicts[i]
                )
            if ids is None or len(ids) == 0:
                continue

            obj_pts = self._obj_pts[i]

            for j, marker_id in enumerate(ids.flatten()):
                img_pts = corners[j].reshape(4, 2).astype(np.float64)

                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts,
                    intrinsics.K, intrinsics.dist,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                if not ok:
                    continue

                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3,  3] = tvec.reshape(3)

                # Reprojection error (sanity gate downstream)
                reproj, _ = cv2.projectPoints(
                    obj_pts, rvec, tvec, intrinsics.K, intrinsics.dist
                )
                err = float(np.linalg.norm(
                    reproj.reshape(-1, 2) - img_pts, axis=1
                ).mean())

                out.append(Detection(
                    dict_name=dc.name,
                    marker_id=int(marker_id),
                    T_cam_marker=T,
                    corners_px=img_pts,
                    reproj_err=err,
                ))
        return out
