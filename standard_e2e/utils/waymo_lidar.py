"""Waymo lidar range-image -> ego-frame point cloud helper.

Pure-NumPy port of the math in the public Waymo Open Dataset SDK at
``waymo_open_dataset/utils/range_image_utils.py`` (Apache-2.0). We keep
the conventions the SDK uses so our output matches the reference
decode on real Waymo data:

  * Range image columns scan from azimuth ``+pi`` (column 0, "rear") to
    ``-pi`` (last column, "near rear from the other side"). The exact
    formula:

        ratios = 1 - (col + 0.5) / W                   # in (0, 1)
        azimuth = (ratios * 2 - 1) * pi - az_correction

    where ``az_correction = atan2(extrinsic[1, 0], extrinsic[0, 0])``
    factors the sensor-mount yaw out of the rotation. With the yaw
    factored out, the conversion to ego frame is ``+ extrinsic[:3, 3]``
    (translation only), matching the SDK.

  * Beam inclinations are stored as a 1D vector of length H, indexed
    bottom-to-top (the SDK's convention). When a calibration only
    provides ``beam_inclination_min/max``, callers should linspace
    between the two; this helper takes pre-computed inclinations.

The output is in **ego frame at the current frame's timestamp** as
required by ``LidarData`` (see ``data_structures/containers.py``);
``Frame.pose`` (world<-ego) is not composed here.
"""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd

from standard_e2e.data_structures import LidarData

if TYPE_CHECKING:
    # pylint: disable=no-name-in-module
    from standard_e2e.third_party.waymo_open_dataset.dataset_pb2 import (
        Frame as WaymoFrame,
    )


class LaserSpec(NamedTuple):
    """Inputs for one laser's range-image -> ego-frame decode.

    laser_id is the integer Waymo ``LaserName.Name`` enum value (e.g.
    1=TOP, 2=FRONT). Carried through into the combined DataFrame so
    consumers can filter per-laser without re-decoding.
    """

    laser_id: int
    range_image: np.ndarray
    extrinsic: np.ndarray
    beam_inclinations: np.ndarray


def range_image_to_points_in_ego(
    range_image: np.ndarray,
    extrinsic: np.ndarray,
    beam_inclinations: np.ndarray,
) -> np.ndarray:
    """Decode one Waymo lidar range image into ego-frame XYZ.

    Args:
        range_image: 2D float array ``(H, W)`` of range values in meters.
            If a 3D ``(H, W, C)`` array is passed, only channel 0 (range)
            is used.
        extrinsic: ``(4, 4)`` sensor->ego transform. The yaw component is
            factored out into the azimuth correction (matching the
            Waymo SDK); only the translation column is added back.
        beam_inclinations: ``(H,)`` per-row beam inclination angles in
            radians (bottom-to-top).

    Returns:
        ``(N, 3)`` float32 ego-frame XYZ for valid (range > 0) pixels,
        flattened in row-major order.
    """
    if range_image.ndim == 3:
        range_image = range_image[..., 0]
    if range_image.ndim != 2:
        raise ValueError(f"range_image must be 2D or 3D; got ndim={range_image.ndim}")
    if extrinsic.shape != (4, 4):
        raise ValueError(f"extrinsic must be (4,4); got {extrinsic.shape}")
    height, width = range_image.shape
    if beam_inclinations.shape != (height,):
        raise ValueError(
            f"beam_inclinations must have shape ({height},); "
            f"got {beam_inclinations.shape}"
        )

    az_correction = float(np.arctan2(extrinsic[1, 0], extrinsic[0, 0]))
    col = np.arange(width)
    ratios = 1.0 - (col + 0.5) / width
    azimuth = (ratios * 2.0 - 1.0) * np.pi - az_correction  # (W,)

    az = azimuth[None, :]
    inc = beam_inclinations[:, None]
    cos_inc = np.cos(inc)
    x = range_image * cos_inc * np.cos(az)
    y = range_image * cos_inc * np.sin(az)
    z = range_image * np.sin(inc)

    valid = range_image > 0
    pts = np.stack([x[valid], y[valid], z[valid]], axis=1).astype(
        np.float32, copy=False
    )
    pts = pts + extrinsic[:3, 3].astype(np.float32, copy=False)
    return pts


def lasers_to_lidar_data(specs: list[LaserSpec]) -> LidarData:
    """Decode multiple laser specs into one combined ego-frame LidarData.

    The returned DataFrame has columns ``x, y, z, laser_id``. Lasers
    are concatenated in the order given; pixels with range <= 0 are
    dropped per laser.
    """
    frames: list[pd.DataFrame] = []
    for spec in specs:
        pts = range_image_to_points_in_ego(
            spec.range_image, spec.extrinsic, spec.beam_inclinations
        )
        frames.append(
            pd.DataFrame(
                {
                    "x": pts[:, 0],
                    "y": pts[:, 1],
                    "z": pts[:, 2],
                    "laser_id": np.full(len(pts), int(spec.laser_id), dtype=np.int32),
                }
            )
        )
    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(
            {
                "x": np.array([], dtype=np.float32),
                "y": np.array([], dtype=np.float32),
                "z": np.array([], dtype=np.float32),
                "laser_id": np.array([], dtype=np.int32),
            }
        )
    return LidarData(points=combined)


def _decode_compressed_range_image(blob: bytes) -> np.ndarray:
    """Inflate + parse a Waymo ``range_image_compressed`` payload."""
    # pylint: disable=no-name-in-module
    from standard_e2e.third_party.waymo_open_dataset.dataset_pb2 import MatrixFloat

    decompressed = zlib.decompress(blob)
    matrix = MatrixFloat()
    matrix.ParseFromString(decompressed)
    dims = list(matrix.shape.dims)
    return np.asarray(matrix.data, dtype=np.float32).reshape(dims)


def frame_lasers_to_lidar_data(frame: "WaymoFrame") -> LidarData:
    """Decode all 5 lasers of a Waymo Perception ``Frame`` into one
    ego-frame ``LidarData`` (DataFrame columns: x, y, z, laser_id).

    Each laser's ``ri_return1`` range image is decoded; ``ri_return2``
    is intentionally ignored for now (a second-return field is straight-
    forward to add later under the same DataFrame schema with a
    ``return_idx`` column).
    """
    cal_by_name = {cal.name: cal for cal in frame.context.laser_calibrations}
    specs: list[LaserSpec] = []
    for laser in frame.lasers:
        cal = cal_by_name.get(laser.name)
        if cal is None:
            raise ValueError(
                f"Frame missing laser_calibration for laser name={laser.name}"
            )
        ri = _decode_compressed_range_image(laser.ri_return1.range_image_compressed)
        height = ri.shape[0]
        if cal.beam_inclinations:
            beam_inclinations = np.asarray(cal.beam_inclinations, dtype=np.float32)
            if beam_inclinations.shape[0] != height:
                raise ValueError(
                    f"beam_inclinations length {beam_inclinations.shape[0]} "
                    f"does not match range image height {height}"
                )
        else:
            beam_inclinations = np.linspace(
                cal.beam_inclination_min,
                cal.beam_inclination_max,
                height,
                dtype=np.float32,
            )
        extrinsic = np.asarray(cal.extrinsic.transform, dtype=np.float32).reshape(4, 4)
        specs.append(
            LaserSpec(
                laser_id=int(laser.name),
                range_image=ri,
                extrinsic=extrinsic,
                beam_inclinations=beam_inclinations,
            )
        )
    return lasers_to_lidar_data(specs)
