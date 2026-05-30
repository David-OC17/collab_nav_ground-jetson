"""
fiducials.py
─────────────────────────────────────────────────────────────────────────────
ArUco marker detection for the stitching pose graph.

Markers move between runs and aren't surveyed, so they are NOT absolute world
anchors.  But within a single run the scene is static and each ArUco ID is
unique, so a marker seen in frame 50 and again in frame 220 is provably the
SAME physical point — a zero-ambiguity loop-closure landmark.  This module
just turns a BGR frame into a list of (marker_id, 4 subpixel corner points)
that pose_graph.PoseGraph.add_marker_obs() consumes directly.

OpenCV API compatibility
─────────────────────────
The aruco API changed at OpenCV 4.7 (the ArucoDetector class replaced the
module-level detectMarkers).  This module detects which is available at
runtime and uses it, so the same code runs on the target's 4.5.4 (old API)
and on newer builds.

Corner ordering
───────────────
cv2.aruco returns the 4 corners in a consistent order (clockwise from the
marker's top-left in its own frame), so "corner j" is the same physical corner
in every frame that sees the marker.  That consistency is all the pose graph
needs — the marker landmark template only has to be a square in the same order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

import cv2
import numpy as np


def _dict_const(name: str) -> int:
    """Resolve a dictionary name like 'DICT_4X4_50' to its cv2.aruco constant."""
    key = name.upper()
    if not key.startswith("DICT_"):
        key = "DICT_" + key
    if not hasattr(cv2.aruco, key):
        raise ValueError(
            f"Unknown ArUco dictionary {name!r}. Expected something like "
            f"'DICT_4X4_50', 'DICT_5X5_1000', 'DICT_APRILTAG_36h11'."
        )
    return getattr(cv2.aruco, key)


@dataclass
class FiducialConfig:
    dictionary: str = "DICT_4X4_50"
    """ArUco dictionary name (target uses DICT_4X4_50)."""

    subpix_refine: bool = True
    """Refine ArUco corners to subpixel accuracy with cornerSubPix.
    Worth it — the corners become the most accurate landmarks in the scene,
    which is exactly why we weight marker constraints above feature edges."""

    subpix_win: int = 5
    """Half-window (px) for cornerSubPix. Keep small so the window stays on
    the marker's black border and doesn't wander onto background texture."""
    subpix_iters: int = 30
    subpix_eps: float = 0.01

    allowed_ids: Optional[Set[int]] = None
    """If set, only these marker IDs are returned. Use it to ignore spurious
    detections when you know which IDs are physically in the arena."""


@dataclass
class MarkerObs:
    """One marker sighting in one frame."""
    marker_id: int
    corners_px: np.ndarray   # (4, 2) float32, aruco corner order


class FiducialDetector:
    """Reusable detector. Build once, call detect() per frame.

    Thread-safety: detect() reads only immutable state, so it is safe to call
    from the producer thread used by MapReconstructor.add_video().
    """

    def __init__(self, cfg: Optional[FiducialConfig] = None):
        self.cfg = cfg or FiducialConfig()
        dict_id = _dict_const(self.cfg.dictionary)

        # Dictionary — getPredefinedDictionary exists in both old and new APIs.
        self._dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

        # Detector parameters — class ctor (new) vs factory (old).
        if hasattr(cv2.aruco, "DetectorParameters"):
            try:
                self._params = cv2.aruco.DetectorParameters()
            except TypeError:                       # some 4.5.x expose it oddly
                self._params = cv2.aruco.DetectorParameters_create()
        else:
            self._params = cv2.aruco.DetectorParameters_create()

        # New API: a stateful ArucoDetector object. Old API: module function.
        self._detector = None
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._detector = cv2.aruco.ArucoDetector(self._dictionary, self._params)

        self._criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            self.cfg.subpix_iters,
            self.cfg.subpix_eps,
        )

    def detect(self, img: np.ndarray) -> List[MarkerObs]:
        """Detect markers in a BGR (or gray) frame. Returns [] if none found."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        if self._detector is not None:                       # new API (>=4.7)
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:                                                # old API (4.5.4)
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self._dictionary, parameters=self._params
            )

        if ids is None or len(ids) == 0:
            return []

        allowed = self.cfg.allowed_ids
        out: List[MarkerObs] = []
        for c, i in zip(corners, ids.flatten()):
            mid = int(i)
            if allowed is not None and mid not in allowed:
                continue
            pts = np.asarray(c, dtype=np.float32).reshape(4, 2)
            if self.cfg.subpix_refine:
                refined = pts.reshape(-1, 1, 2).copy()
                cv2.cornerSubPix(
                    gray, refined,
                    (self.cfg.subpix_win, self.cfg.subpix_win), (-1, -1),
                    self._criteria,
                )
                pts = refined.reshape(4, 2)
            out.append(MarkerObs(marker_id=mid, corners_px=pts))
        return out
