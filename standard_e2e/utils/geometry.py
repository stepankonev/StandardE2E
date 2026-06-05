"""Shared SE(3) / quaternion / camera-intrinsics helpers for dataset processors.

These small operations were otherwise reimplemented per dataset (comma2k19,
WayveScenes/COLMAP, NAVSIM/nuPlan, Argoverse 2). Quaternions follow the
**Hamilton, scalar-first ``(w, x, y, z)``** convention every ingested dataset
uses; the scalar-last reorder that :class:`scipy.spatial.transform.Rotation`
expects lives here once instead of at each call site. The active rotation a
quaternion represents is returned as-is — callers attach the frame semantics
(``ecef_from_device``, ``cam_from_world``, ...).
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from scipy.spatial.transform import Rotation


def quats_wxyz_to_rotmats(quats_wxyz: ArrayLike) -> np.ndarray:
    """Convert ``(N, 4)`` Hamilton ``(w, x, y, z)`` quaternions to ``(N, 3, 3)``
    rotation matrices (float64). Quaternions are normalised by scipy."""
    q = np.asarray(quats_wxyz, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quats must have shape (N, 4); got {q.shape}")
    # scipy uses scalar-last (x, y, z, w); reorder from our scalar-first wxyz.
    rotmats = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    return cast(np.ndarray, rotmats)


def quat_wxyz_to_rotmat(quat_wxyz: ArrayLike) -> np.ndarray:
    """Convert a single Hamilton ``(w, x, y, z)`` quaternion to ``(3, 3)`` (float64)."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(1, 4)
    return cast(np.ndarray, quats_wxyz_to_rotmats(q)[0])


def se3(
    rotation: ArrayLike, translation: ArrayLike, dtype: DTypeLike = np.float64
) -> np.ndarray:
    """Assemble a 4x4 homogeneous transform from a 3x3 rotation + 3-vector."""
    transform = np.eye(4, dtype=dtype)
    transform[:3, :3] = np.asarray(rotation, dtype=dtype)
    transform[:3, 3] = np.asarray(translation, dtype=dtype).reshape(3)
    return transform


def intrinsics_matrix(
    fx: float, fy: float, cx: float, cy: float, dtype: DTypeLike = np.float32
) -> np.ndarray:
    """Build a pinhole camera matrix ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``."""
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=dtype)


def transform_points(transform: ArrayLike, points: ArrayLike) -> np.ndarray:
    """Apply a 4x4 SE(3) ``transform`` to ``(N, 3)`` ``points`` -> ``(N, 3)``.

    Computed in the points' dtype (so float32 in stays float32 out).
    """
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3); got {pts.shape}")
    mat = np.asarray(transform, dtype=pts.dtype)
    homog = np.concatenate([pts, np.ones((len(pts), 1), dtype=pts.dtype)], axis=1)
    return cast(np.ndarray, (homog @ mat.T)[:, :3])
