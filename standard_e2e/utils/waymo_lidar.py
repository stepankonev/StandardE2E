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

  * Range image channel layout (per Waymo SDK): C=0 range, C=1 intensity,
    C=2 elongation, C=3 NLZ flag. We currently extract range + intensity;
    elongation / NLZ / second-return / range_image_pose are deferred.

The output is in **ego frame at the current frame's timestamp** as
required by ``LidarData`` (see ``data_structures/containers.py``);
``Frame.pose`` (world<-ego) is not composed here.
"""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING, NamedTuple, Optional

import numpy as np
import pandas as pd

from standard_e2e.data_structures import LidarData

if TYPE_CHECKING:
    # pylint: disable=no-name-in-module
    from standard_e2e.third_party.waymo_open_dataset.dataset_pb2 import (
        Frame as WaymoFrame,
    )


# Waymo TOP lidar rotates at 10 Hz -> 100 ms sweep period. Non-spinning
# lasers (FRONT, SIDE_*, REAR) flash all pixels at frame_timestamp.
WAYMO_TOP_SWEEP_PERIOD_NS = 100_000_000
# Waymo LaserName.Name enum: TOP=1; the four corner lasers are the
# fixed flash units that share the frame timestamp.
WAYMO_TOP_LASER_ID = 1


class LaserSpec(NamedTuple):
    """Inputs for one laser's range-image -> ego-frame decode.

    laser_id is the integer Waymo ``LaserName.Name`` enum value (e.g.
    1=TOP, 2=FRONT). Carried through into the combined DataFrame so
    consumers can filter per-laser without re-decoding.

    is_spinning marks rotating lidars (TOP) where each azimuth column is
    captured at a slightly different time within the sweep; for fixed
    flash lidars the per-point timestamp collapses to the frame
    timestamp. Used by ``lasers_to_lidar_data`` when computing the
    optional ``timestamp_ns`` column.
    """

    laser_id: int
    range_image: np.ndarray
    extrinsic: np.ndarray
    beam_inclinations: np.ndarray
    is_spinning: bool = False


class _DecodedPoints(NamedTuple):
    """Internal: per-pixel decode output for one laser.

    ``intensity`` is ``None`` when the input range image is 2D (range
    channel only); it is populated when the input is 3D with at least
    two channels (range + intensity).
    """

    xyz: np.ndarray  # (N, 3) float32, ego frame
    intensity: Optional[np.ndarray]  # (N,) float32 or None
    col_index: np.ndarray  # (N,) int32 — azimuth column of each point
    width: int  # range image W (azimuth column count)


def range_image_to_points_in_ego(
    range_image: np.ndarray,
    extrinsic: np.ndarray,
    beam_inclinations: np.ndarray,
) -> np.ndarray:
    """Decode one Waymo lidar range image into ego-frame XYZ.

    Args:
        range_image: 2D float array ``(H, W)`` of range values in meters.
            If a 3D ``(H, W, C)`` array is passed, only channel 0 (range)
            is used here; ``_decode_range_image_full`` extracts the
            additional channels.
        extrinsic: ``(4, 4)`` sensor->ego transform. The yaw component is
            factored out into the azimuth correction (matching the
            Waymo SDK); only the translation column is added back.
        beam_inclinations: ``(H,)`` per-row beam inclination angles in
            radians (bottom-to-top).

    Returns:
        ``(N, 3)`` float32 ego-frame XYZ for valid (range > 0) pixels,
        flattened in row-major order.
    """
    return _decode_range_image_full(range_image, extrinsic, beam_inclinations).xyz


def _decode_range_image_full(
    range_image: np.ndarray,
    extrinsic: np.ndarray,
    beam_inclinations: np.ndarray,
) -> _DecodedPoints:
    """Like ``range_image_to_points_in_ego`` but returns intensity + column index too."""
    if range_image.ndim == 3:
        range_2d = range_image[..., 0]
        intensity_2d = range_image[..., 1] if range_image.shape[2] >= 2 else None
    elif range_image.ndim == 2:
        range_2d = range_image
        intensity_2d = None
    else:
        raise ValueError(f"range_image must be 2D or 3D; got ndim={range_image.ndim}")
    if extrinsic.shape != (4, 4):
        raise ValueError(f"extrinsic must be (4,4); got {extrinsic.shape}")
    height, width = range_2d.shape
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
    x = range_2d * cos_inc * np.cos(az)
    y = range_2d * cos_inc * np.sin(az)
    z = range_2d * np.sin(inc)

    valid = range_2d > 0
    pts = np.stack([x[valid], y[valid], z[valid]], axis=1).astype(
        np.float32, copy=False
    )
    pts = pts + extrinsic[:3, 3].astype(np.float32, copy=False)

    intensity_valid: Optional[np.ndarray] = None
    if intensity_2d is not None:
        intensity_valid = intensity_2d[valid].astype(np.float32, copy=False)

    # Column index per valid pixel (row-major scan matches the boolean
    # mask iteration above).
    col_grid = np.broadcast_to(np.arange(width, dtype=np.int32), (height, width))
    col_index = col_grid[valid].astype(np.int32, copy=False)

    return _DecodedPoints(
        xyz=pts,
        intensity=intensity_valid,
        col_index=col_index,
        width=int(width),
    )


def lasers_to_lidar_data(
    specs: list[LaserSpec],
    *,
    frame_timestamp_ns: Optional[int] = None,
    sweep_period_ns: int = WAYMO_TOP_SWEEP_PERIOD_NS,
) -> LidarData:
    """Decode multiple laser specs into one combined ego-frame LidarData.

    The DataFrame always contains ``x, y, z, laser_id``. ``intensity``
    is added when at least one spec carries a 3D range image with the
    intensity channel. ``timestamp_ns`` is added when
    ``frame_timestamp_ns`` is provided:

    * For ``spec.is_spinning=True`` (TOP lidar), the per-point time
      offset is ``((col + 0.5) / W - 0.5) * sweep_period_ns`` so the
      range maps to ``[-T/2, +T/2]`` centered on ``frame_timestamp_ns``.
    * For non-spinning lasers, every point shares the frame timestamp.

    Lasers are concatenated in the order given; pixels with range <= 0
    are dropped per laser.
    """
    frames: list[pd.DataFrame] = []
    any_intensity = False
    for spec in specs:
        decoded = _decode_range_image_full(
            spec.range_image, spec.extrinsic, spec.beam_inclinations
        )
        cols: dict[str, np.ndarray] = {
            "x": decoded.xyz[:, 0],
            "y": decoded.xyz[:, 1],
            "z": decoded.xyz[:, 2],
            "laser_id": np.full(len(decoded.xyz), int(spec.laser_id), dtype=np.int32),
        }
        if decoded.intensity is not None:
            cols["intensity"] = decoded.intensity
            any_intensity = True
        if frame_timestamp_ns is not None:
            cols["timestamp_ns"] = _per_point_timestamp_ns(
                col_index=decoded.col_index,
                width=decoded.width,
                is_spinning=spec.is_spinning,
                frame_timestamp_ns=int(frame_timestamp_ns),
                sweep_period_ns=int(sweep_period_ns),
            )
        frames.append(pd.DataFrame(cols))
    if frames:
        # If only some specs had intensity, concat would fill missing
        # values with NaN; that breaks the int/bool downstream stack.
        # All Waymo lasers carry intensity, so this assertion catches
        # an upstream input shape regression rather than user error.
        if any_intensity and not all("intensity" in f.columns for f in frames):
            raise ValueError(
                "Mixed intensity presence across lasers — pass 3D range "
                "images for all specs or none."
            )
        combined = pd.concat(frames, ignore_index=True)
    else:
        empty_cols: dict[str, np.ndarray] = {
            "x": np.array([], dtype=np.float32),
            "y": np.array([], dtype=np.float32),
            "z": np.array([], dtype=np.float32),
            "laser_id": np.array([], dtype=np.int32),
        }
        combined = pd.DataFrame(empty_cols)
    return LidarData(points=combined)


def _per_point_timestamp_ns(
    *,
    col_index: np.ndarray,
    width: int,
    is_spinning: bool,
    frame_timestamp_ns: int,
    sweep_period_ns: int,
) -> np.ndarray:
    """Compute per-point ``timestamp_ns`` (int64).

    Spinning lasers (TOP) get a column-fraction offset centered on the
    frame timestamp. Fixed-flash lasers get the frame timestamp
    uniformly. The midpoint convention matches nuScenes / AV2 patterns:
    a sweep centered on the frame timestamp keeps ego-motion compensation
    error symmetric.
    """
    if not is_spinning:
        return np.full(col_index.shape, frame_timestamp_ns, dtype=np.int64)
    # offset_ns in [-T/2, +T/2]: ((col + 0.5) / W - 0.5) * T
    fraction = (col_index.astype(np.float64) + 0.5) / float(width) - 0.5
    offset_ns = (fraction * sweep_period_ns).astype(np.int64)
    return offset_ns + np.int64(frame_timestamp_ns)


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
    """Decode all lasers of a Waymo Perception ``Frame`` into one
    ego-frame ``LidarData``.

    DataFrame columns: ``x, y, z, laser_id, intensity, timestamp_ns``.

    Each laser's ``ri_return1`` range image is decoded; ``ri_return2``
    is intentionally ignored for now (a second-echo field is straight-
    forward to add later under the same DataFrame schema with a
    ``return_idx`` column).

    Per-point ``timestamp_ns`` is computed from
    ``frame.timestamp_micros``: the TOP lidar (laser_id == 1) gets a
    column-fraction offset within a 100 ms sweep centered on the frame
    timestamp; the four fixed-flash lasers share the frame timestamp.
    """
    # Channels 2 (elongation) and 3 (NLZ) and ri_return2 are currently
    # dropped; range_image_pose_compressed (per-pixel sensor pose for
    # rolling-shutter compensation) is also dropped. Adding any of
    # those is a column-extension under this same schema.
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
                is_spinning=int(laser.name) == WAYMO_TOP_LASER_ID,
            )
        )
    frame_timestamp_ns = int(frame.timestamp_micros) * 1000
    return lasers_to_lidar_data(specs, frame_timestamp_ns=frame_timestamp_ns)
