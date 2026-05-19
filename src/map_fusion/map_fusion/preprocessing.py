"""Stages 1 & 2 of the pipeline: threshold an OccupancyGrid to binary, extract
obstacle contours via a morphological gradient, and build a metric point cloud.
"""

import cv2
import numpy as np

from .grid_utils import cell_centers_to_metric


def _edge_image(binary, kernel_radius):
    """Morphological gradient: ``edges = binary XOR erode(binary)``.

    ``kernel_radius`` is treated as the structuring-element *radius*, so the
    actual kernel is ``(2r + 1) x (2r + 1)``. A literal 1x1 kernel would be the
    identity transform and yield no edges at all, so the spec parameter
    ``edge_kernel_size`` is interpreted as a radius here (default 1 -> 3x3).
    """
    r = max(1, int(kernel_radius))
    k = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    eroded = cv2.erode(binary, kernel)
    return cv2.bitwise_xor(binary, eroded)


def preprocess_grid(arr, info, occupied_threshold, kernel_radius,
                    unknown_is_obstacle=False):
    """Threshold and edge-extract an OccupancyGrid array.

    Parameters
    ----------
    arr : (H, W) int8 array
        Raw occupancy values (-1 unknown, 0..100).
    info : GridInfo
        Geometry of the grid.
    occupied_threshold : int
        Cells with value >= this become obstacle (1).
    kernel_radius : int
        Radius of the morphological structuring element.
    unknown_is_obstacle : bool
        If False (default) unknown cells are treated as free for the purpose
        of edge extraction, which keeps blob contours clean.

    Returns
    -------
    dict with keys:
        ``edge_points`` : (N, 2) metric coords of edge cells in the grid frame
        ``edge_image``  : (H, W) uint8 binary edge mask
        ``known_mask``  : (H, W) uint8, 1 where the cell is not -1
        ``binary``      : (H, W) uint8 thresholded occupancy
    """
    known_mask = (arr != -1).astype(np.uint8)
    binary = (arr >= occupied_threshold).astype(np.uint8)
    if not unknown_is_obstacle:
        binary[arr == -1] = 0

    edges = _edge_image(binary, kernel_radius)
    rows, cols = np.where(edges == 1)
    pts = cell_centers_to_metric(rows, cols, info)

    return {
        'edge_points': pts,
        'edge_image': edges,
        'known_mask': known_mask,
        'binary': binary,
    }
