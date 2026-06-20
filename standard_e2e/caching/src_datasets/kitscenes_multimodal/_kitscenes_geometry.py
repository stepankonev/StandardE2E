"""Calibration / pose geometry for KITScenes Multimodal.

KITScenes expresses every sensor's pose in the **base_frame** reference -- a
REP-103 FLU frame (x-forward, y-left, z-up) whose origin sits at the top LiDAR
(``lidar_top``'s extrinsic is the identity). StandardE2E uses that base_frame as
the ego frame, so a sensor's ``T_to_reference`` (sensor -> base_frame) is exactly
the ``T_ego_from_sensor`` extrinsic the data structures expect.

* :func:`parse_calibration` reads ``calibration/calib.json`` into the six
  ingested ring cameras' pinhole ``K`` + ``T_ego_from_camera`` extrinsics and the
  ``lidar_top`` -> ego extrinsic.
* :func:`pose_from_tum` turns one ``poses.txt`` row (TUM
  ``ts tx ty tz qx qy qz qw``, quaternion scalar-last, translation in the
  Lanelet2 map-local frame) into ``T_maplocal_from_ego`` -- the ego pose used for
  ``global_position`` / the trajectory aggregator and to place the HD map.

Only the six surround ``camera_ring_*`` views map onto the canonical
:class:`CameraDirection` members; KITScenes' three additional "base" cameras (the
high-resolution front-center and the rectified stereo pair) are not ingested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from standard_e2e.enums import CameraDirection
from standard_e2e.utils import intrinsics_matrix, quat_wxyz_to_rotmat, se3

# The six surround "ring" cameras -> canonical CameraDirection members. The
# three "base" cameras (camera_base_front_center and the
# camera_base_front_{left,right}_rect stereo pair) have no canonical slot and are
# not ingested by this integration.
RING_CAMERA_TO_DIRECTION: dict[str, CameraDirection] = {
    "camera_ring_front": CameraDirection.FRONT,
    "camera_ring_front_left": CameraDirection.FRONT_LEFT,
    "camera_ring_front_right": CameraDirection.FRONT_RIGHT,
    "camera_ring_rear": CameraDirection.REAR,
    "camera_ring_rear_left": CameraDirection.REAR_LEFT,
    "camera_ring_rear_right": CameraDirection.REAR_RIGHT,
}

_LIDAR_TOP_KEY = "lidar_top"


@dataclass(frozen=True)
class RingCameraCalibration:
    """One ring camera's pinhole ``K`` + ``T_ego_from_camera`` extrinsics."""

    camera_name: str
    direction: CameraDirection
    intrinsics: np.ndarray  # (3, 3) float32
    extrinsics: np.ndarray  # (4, 4) float64, T_ego_from_camera


def _intrinsics_from_entry(entry: dict) -> np.ndarray:
    """Build a ``(3, 3)`` pinhole ``K`` from a KITScenes camera calib entry."""
    intr = entry["intrinsics"]
    focal = float(intr["focal_length"])
    cu = float(intr["principal_point_u"])
    cv = float(intr["principal_point_v"])
    return intrinsics_matrix(focal, focal, cu, cv)


def _transform_to_reference(entry: dict) -> np.ndarray:
    """Read a sensor's ``T_to_reference`` (sensor -> ego) as ``(4, 4)`` float64."""
    transform = np.asarray(entry["T_to_reference"], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"T_to_reference must be 4x4; got {transform.shape}")
    return transform


def parse_calibration(
    calib: dict,
) -> tuple[dict[CameraDirection, RingCameraCalibration], np.ndarray]:
    """Parse ``calib.json`` into ring-camera calibrations + the LiDAR extrinsic.

    Returns ``(cameras, lidar_top_extrinsic)`` where ``cameras`` maps each present
    ring camera's :class:`CameraDirection` to its
    :class:`RingCameraCalibration`, and ``lidar_top_extrinsic`` is the ``(4, 4)``
    ``T_ego_from_lidar`` (identity for KITScenes, applied for generality).

    Raises ``KeyError`` if ``lidar_top`` is missing from the calibration.
    """
    cameras: dict[CameraDirection, RingCameraCalibration] = {}
    for camera_name, direction in RING_CAMERA_TO_DIRECTION.items():
        entry = calib.get(camera_name)
        if entry is None:
            continue
        cameras[direction] = RingCameraCalibration(
            camera_name=camera_name,
            direction=direction,
            intrinsics=_intrinsics_from_entry(entry),
            extrinsics=_transform_to_reference(entry),
        )

    if _LIDAR_TOP_KEY not in calib:
        raise KeyError(
            f"calibration missing {_LIDAR_TOP_KEY!r}; found keys "
            f"{sorted(k for k in calib if k != 'reference')}"
        )
    lidar_top_extrinsic = _transform_to_reference(calib[_LIDAR_TOP_KEY])
    return cameras, lidar_top_extrinsic


def pose_from_tum(row: np.ndarray) -> np.ndarray:
    """Convert one TUM ``poses.txt`` row to ``(4, 4)`` float64 ``T_maplocal_from_ego``.

    Args:
        row: ``(8,)`` ``[timestamp, tx, ty, tz, qx, qy, qz, qw]`` -- translation
            in the Lanelet2 map-local frame, quaternion scalar-last (xyzw).
    """
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    if row.shape[0] != 8:
        raise ValueError(f"TUM pose row must have 8 values; got {row.shape[0]}")
    translation = row[1:4]
    qx, qy, qz, qw = row[4:8]
    # utils uses Hamilton scalar-first (w, x, y, z); reorder from TUM's xyzw.
    rotation = quat_wxyz_to_rotmat([qw, qx, qy, qz])
    return se3(rotation, translation)
