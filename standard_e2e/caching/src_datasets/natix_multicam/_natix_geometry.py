"""Calibration / pose geometry for the NATIX Multi-Camera Driving Dataset.

Calibration (``fixed_metadata.json``) expresses every device in the
``ground_nominal`` reference frame -- an FLU vehicle frame (x-forward, y-left,
z-up; verified against the data: the left repeater sits at ``ty=+90``, the
right at ``ty=-90``) with its origin on the ground plane. StandardE2E uses
``ground_nominal`` as the ego frame directly.

* The per-camera ``r11..r33`` (row-major) is ``R_ego_from_body`` -- it maps the
  camera's **body** frame (FLU: x out the lens, y-left, z-up) into the ego
  frame. Verified against the facings: the left repeater's body x-axis maps to
  ``(-0.848, +0.530, 0)`` (backward-left) and the left pillar's to
  ``(+0.341, +0.936, +0.087)`` (forward-left); the transposed reading would
  swap left/right. ``tx, ty, tz`` are in **centimetres** (the front camera
  sits at ``(182, 0, 130)`` -- windshield height on a Model 3).
* :func:`parse_fixed_metadata` converts each camera entry to the pinhole ``K``,
  the Brown-Conrady distortion (reordered from NATIX's ``k1,k2,k3,p1,p2`` to
  the container's ``k1,k2,p1,p2,k3``) and the ``T_ego_from_camera`` extrinsic
  in the **optical** camera frame (z out the lens, x-right, y-down -- the
  nuScenes / KITScenes convention), metres.

Ego pose comes from the per-frame GPS fixes (consumer-grade; no altitude):

* :class:`LocalMetricProjection` maps WGS84 lat/lon to a local metric
  east/north plane (``pyproj`` azimuthal-equidistant, anchored at the
  segment's first fix, so within-segment coordinates stay small enough for the
  float32 round-trip in ``Trajectory``).
* ``heading_deg`` is clockwise from north; :func:`headings_rad_from_deg` turns
  it into the FLU yaw (counter-clockwise from the local +x/east axis).
  :func:`resolve_headings` handles the documented degradation -- heading may
  be missing (``na``) or zeroed together with the speed when GPS updates
  arrive too quickly, and is GPS-noise-dominated when nearly stationary -- by
  holding the last reliable heading (a vehicle that is not moving does not
  yaw).
* :func:`poses_world_from_xy_heading` assembles the per-frame ``T_world_ego``
  (yaw-only; z = 0 -- the GPS stream carries no altitude) used for
  ``global_position`` and ``aux_data["pose_matrix"]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer

from standard_e2e.enums import CameraDirection
from standard_e2e.utils import intrinsics_matrix, se3

# Calibration translations are centimetres (the front camera sits at
# tz=130 -- 1.30 m, windshield height); StandardE2E works in metres.
_CM_TO_M = 0.01

# Speeds at or below this are GPS-noise-dominated: the fix-to-fix displacement
# that NATIX derives the heading from is comparable to the fix accuracy, so
# the heading is unreliable and the last moving heading is held instead.
MIN_SPEED_FOR_HEADING_MPS = 0.5

# Camera body frame (FLU: x out the lens) -> optical frame (z out the lens,
# x-right, y-down): columns are the optical axes expressed in body axes.
BODY_FROM_OPTICAL = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

# Calibration device names -> the canonical CameraDirection matching the
# facing encoded in the calibration rotation (all rigs are Tesla dashcams).
# In 4-camera trips ``camera_left`` / ``camera_right`` ARE the repeaters --
# same mounts and the same backward-side facing as the 6-camera trips'
# ``camera_*_repeater`` (body x-axis (-0.848, +/-0.530, 0)) -- so they map to
# REAR_LEFT / REAR_RIGHT, not SIDE_*. The pillar cameras face forward-side.
DEVICE_NAME_TO_DIRECTION: dict[str, CameraDirection] = {
    "camera_front": CameraDirection.FRONT,
    "camera_rear": CameraDirection.REAR,
    "camera_left": CameraDirection.REAR_LEFT,
    "camera_right": CameraDirection.REAR_RIGHT,
    "camera_left_repeater": CameraDirection.REAR_LEFT,
    "camera_right_repeater": CameraDirection.REAR_RIGHT,
    "camera_left_pillar": CameraDirection.FRONT_LEFT,
    "camera_right_pillar": CameraDirection.FRONT_RIGHT,
}

_ROTATION_KEYS = ("r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33")
# NATIX order: k1, k2, k3, p1, p2 -> CameraData's Brown-Conrady order.
_DISTORTION_KEYS_CONTAINER_ORDER = ("k1", "k2", "p1", "p2", "k3")


@dataclass(frozen=True)
class NatixCameraCalibration:
    """One camera's pinhole ``K``, distortion and ``T_ego_from_camera``.

    ``folder_key`` is the camera-folder prefix the trip's clip directories use
    for this device (``camera_left_repeater`` -> ``LEFT_REPEATER_FOLDER`` /
    ``LEFT_REPEATER_*.mp4``).
    """

    device_name: str
    folder_key: str
    direction: CameraDirection
    intrinsics: np.ndarray  # (3, 3) float32
    extrinsics: np.ndarray  # (4, 4) float64, T_ego_from_camera (optical frame)
    distortion: np.ndarray  # (5,) float32, Brown-Conrady (k1, k2, p1, p2, k3)


def folder_key_from_device_name(device_name: str) -> str:
    """``camera_left_repeater`` -> ``LEFT_REPEATER`` (the on-disk prefix)."""
    return device_name.removeprefix("camera_").upper()


def _camera_calibration_from_entry(entry: dict) -> NatixCameraCalibration:
    device_name = str(entry["device_name"])
    direction = DEVICE_NAME_TO_DIRECTION[device_name]
    rotation_body = np.array(
        [float(entry[key]) for key in _ROTATION_KEYS], dtype=np.float64
    ).reshape(3, 3)
    translation_m = (
        np.array(
            [float(entry["tx"]), float(entry["ty"]), float(entry["tz"])],
            dtype=np.float64,
        )
        * _CM_TO_M
    )
    extrinsics = se3(rotation_body @ BODY_FROM_OPTICAL, translation_m)
    intrinsics = intrinsics_matrix(
        float(entry["fx"]), float(entry["fy"]), float(entry["cx"]), float(entry["cy"])
    )
    distortion = np.array(
        [float(entry.get(key, 0.0)) for key in _DISTORTION_KEYS_CONTAINER_ORDER],
        dtype=np.float32,
    )
    return NatixCameraCalibration(
        device_name=device_name,
        folder_key=folder_key_from_device_name(device_name),
        direction=direction,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        distortion=distortion,
    )


def parse_fixed_metadata(
    fixed_metadata: dict,
) -> dict[CameraDirection, NatixCameraCalibration]:
    """Parse ``fixed_metadata.json`` into per-direction camera calibrations.

    Non-camera devices (``phone``, ``vx360`` -- the GPS unit) and camera names
    outside :data:`DEVICE_NAME_TO_DIRECTION` are skipped; two devices mapping
    to the same direction (never observed -- the 4- and 6-camera rigs use
    disjoint repeater names) raise ``ValueError``.
    """
    calibrations: dict[CameraDirection, NatixCameraCalibration] = {}
    for entry in fixed_metadata.get("device_extrinsics", []):
        device_name = str(entry.get("device_name", ""))
        if not device_name.startswith("camera"):
            continue
        if device_name not in DEVICE_NAME_TO_DIRECTION:
            raise KeyError(
                f"unknown NATIX camera device {device_name!r}; extend "
                "DEVICE_NAME_TO_DIRECTION with its facing"
            )
        calibration = _camera_calibration_from_entry(entry)
        if calibration.direction in calibrations:
            raise ValueError(
                f"two devices map to {calibration.direction}: "
                f"{calibrations[calibration.direction].device_name!r} "
                f"and {device_name!r}"
            )
        calibrations[calibration.direction] = calibration
    return calibrations


class LocalMetricProjection:  # pylint: disable=too-few-public-methods
    """WGS84 lat/lon -> local metric east/north, anchored at ``(lat0, lon0)``.

    Azimuthal equidistant keeps distances from the anchor exact and needs no
    UTM-zone bookkeeping; over a trip-piece's extent (a few km) the distortion
    is far below the GPS fix accuracy. Built lazily per worker (inside the
    trip cache), so it is never pickled into the pool.
    """

    def __init__(self, lat0: float, lon0: float):
        self._lat0 = float(lat0)
        self._lon0 = float(lon0)
        self._transformer = Transformer.from_crs(
            "EPSG:4326",
            f"+proj=aeqd +lat_0={self._lat0} +lon_0={self._lon0} "
            "+datum=WGS84 +units=m +no_defs",
            always_xy=True,
        )

    def to_local_xy(
        self, latitudes: np.ndarray, longitudes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(N,)`` lat / lon degrees -> ``(N,)`` east / north metres."""
        x_east, y_north = self._transformer.transform(
            np.asarray(longitudes, dtype=np.float64),
            np.asarray(latitudes, dtype=np.float64),
        )
        return np.asarray(x_east, dtype=np.float64), np.asarray(
            y_north, dtype=np.float64
        )


def headings_rad_from_deg(headings_deg: np.ndarray) -> np.ndarray:
    """Compass headings (degrees, clockwise, 0 = north) -> FLU yaw (radians,
    counter-clockwise from the local +x/east axis), wrapped to ``[-pi, pi]``."""
    yaw = np.deg2rad(90.0 - np.asarray(headings_deg, dtype=np.float64))
    return np.arctan2(np.sin(yaw), np.cos(yaw))


def resolve_headings(headings_deg: np.ndarray, speeds_mps: np.ndarray) -> np.ndarray:
    """Fill unreliable compass headings by holding the last reliable one.

    A heading is *reliable* when it is present (not ``na``) and the fix speed
    exceeds :data:`MIN_SPEED_FOR_HEADING_MPS` -- below that the displacement
    NATIX derives the heading from is GPS noise (the README documents heading
    and speed zeroing out together on too-frequent updates). Unreliable
    entries take the previous reliable heading (a stationary vehicle keeps
    its yaw); entries before the first reliable one take the first reliable
    heading. With no reliable heading at all, returns zeros (north).
    """
    headings = np.asarray(headings_deg, dtype=np.float64).copy()
    speeds = np.asarray(speeds_mps, dtype=np.float64)
    reliable = (
        np.isfinite(headings)
        & np.isfinite(speeds)
        & (speeds > MIN_SPEED_FOR_HEADING_MPS)
    )
    if not reliable.any():
        return np.zeros_like(headings)
    reliable_indices = np.flatnonzero(reliable)
    # For every frame, the index of the latest reliable fix at or before it;
    # frames before the first reliable fix borrow the first one.
    take = np.searchsorted(reliable_indices, np.arange(len(headings)), side="right")
    take = np.clip(take - 1, 0, len(reliable_indices) - 1)
    return headings[reliable_indices[take]]


def poses_world_from_xy_heading(
    x_east: np.ndarray, y_north: np.ndarray, headings_rad: np.ndarray
) -> np.ndarray:
    """Assemble per-frame ``T_world_ego`` (ego-FLU -> local east/north world).

    Yaw-only rotations about +z; ``z = 0`` throughout (the GPS metadata
    carries no altitude).

    Returns:
        ``(N, 4, 4)`` float64 homogeneous transforms.
    """
    x = np.asarray(x_east, dtype=np.float64).reshape(-1)
    y = np.asarray(y_north, dtype=np.float64).reshape(-1)
    yaw = np.asarray(headings_rad, dtype=np.float64).reshape(-1)
    if not len(x) == len(y) == len(yaw):
        raise ValueError(
            f"x, y and heading lengths differ: {len(x)}, {len(y)}, {len(yaw)}"
        )
    cos, sin = np.cos(yaw), np.sin(yaw)
    transforms = np.tile(np.eye(4, dtype=np.float64), (len(x), 1, 1))
    transforms[:, 0, 0] = cos
    transforms[:, 0, 1] = -sin
    transforms[:, 1, 0] = sin
    transforms[:, 1, 1] = cos
    transforms[:, 0, 3] = x
    transforms[:, 1, 3] = y
    return transforms
