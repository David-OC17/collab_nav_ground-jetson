"""SE(2) transform utilities for map fusion.

A 2D rigid transform is represented throughout this package as a 3-tuple
``T = (tx, ty, theta)`` meaning: a point ``p`` expressed in the child frame
maps into the parent frame as ``R(theta) @ p + (tx, ty)``.
"""

import math

import numpy as np


def se2_matrix(tx, ty, theta):
    """Return the 3x3 homogeneous matrix for ``T = (tx, ty, theta)``."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, tx],
                     [s,  c, ty],
                     [0,  0,  1]], dtype=float)


def matrix_to_se2(m):
    """Inverse of :func:`se2_matrix`."""
    return (float(m[0, 2]), float(m[1, 2]), math.atan2(m[1, 0], m[0, 0]))


def apply_se2(t, points):
    """Apply ``t = (tx, ty, theta)`` to an (N, 2) array of points."""
    tx, ty, th = t
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, 2)
    c, s = math.cos(th), math.sin(th)
    rot = np.array([[c, -s], [s, c]])
    return pts @ rot.T + np.array([tx, ty])


def compose(t_a, t_b):
    """Return ``t_a o t_b`` (apply ``t_b`` first, then ``t_a``)."""
    return matrix_to_se2(se2_matrix(*t_a) @ se2_matrix(*t_b))


def invert(t):
    """Return the inverse transform of ``t``."""
    return matrix_to_se2(np.linalg.inv(se2_matrix(*t)))


def angle_diff(a, b):
    """Smallest signed difference ``a - b`` wrapped to [-pi, pi]."""
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(x, y, z, w):
    """Extract the z-axis yaw from a quaternion."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_from_yaw(yaw):
    """Return (x, y, z, w) for a pure z-axis rotation."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def rigid_fit_2d(src, dst):
    """Least-squares SE(2) aligning source points onto target points.

    ``src`` and ``dst`` are (N, 2) arrays of corresponding points. Returns
    ``(tx, ty, theta)`` minimising ``sum ||R p_i + t - q_i||^2`` via the
    Kabsch/Umeyama solution (with a reflection guard).
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)
    h = (src - c_src).T @ (dst - c_dst)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, d]) @ u.T
    t = c_dst - rot @ c_src
    return (float(t[0]), float(t[1]), math.atan2(rot[1, 0], rot[0, 0]))
