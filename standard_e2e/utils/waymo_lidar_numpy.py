"""Pure-numpy equivalents of the Waymo lidar decode hot path.

The upstream ``waymo_open_dataset.utils.frame_utils`` runs:
  - ``parse_range_image_and_camera_projection``: zlib-decode + proto parse
  - ``convert_range_image_to_point_cloud``: spherical -> cartesian, applies
    laser extrinsic, plus per-pixel pose for the TOP laser.

Both go through TF ops. Under a 32-process worker pool the combined TF
runtime cost (eager kernel dispatch + intra-op threads + serialization
overhead) makes each frame ~485 ms in workers, dominating the pipeline.

The numpy versions here use ``zlib.decompress`` for the decompression and
plain ndarray math for the geometry, avoiding the TF runtime entirely
inside workers. Output is bit-exact for non-TOP lasers and within
float32 tolerance for the TOP laser (where the original uses
``tf.linalg.inv``; numpy's inverse is the same algorithm but compiled
differently).
"""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np

# pylint: disable=no-name-in-module
from standard_e2e.third_party.waymo_open_dataset.dataset_pb2 import (
    MatrixFloat,
    MatrixInt32,
)

# LaserName enum value for TOP (only that laser uses per-pixel pose).
LASER_TOP = 1


def numpy_parse_range_image_and_camera_projection(
    frame: Any,
) -> tuple[dict[int, list[Any]], dict[int, list[Any]], dict[int, list[Any]], Any]:
    """Numpy-only equivalent of ``frame_utils.parse_range_image_and_camera_projection``.

    Replaces ``tf.io.decode_compressed(..., 'ZLIB')`` with ``zlib.decompress``.
    Everything else (proto parse) is already pure-Python.
    """
    range_images: dict[int, list[Any]] = {}
    camera_projections: dict[int, list[Any]] = {}
    seg_labels: dict[int, list[Any]] = {}
    range_image_top_pose = MatrixFloat()

    for laser in frame.lasers:
        if len(laser.ri_return1.range_image_compressed) > 0:
            ri = MatrixFloat()
            ri.ParseFromString(zlib.decompress(laser.ri_return1.range_image_compressed))
            range_images[laser.name] = [ri]

            if laser.name == LASER_TOP:
                range_image_top_pose = MatrixFloat()
                range_image_top_pose.ParseFromString(
                    zlib.decompress(laser.ri_return1.range_image_pose_compressed)
                )

            cp = MatrixInt32()
            cp.ParseFromString(
                zlib.decompress(laser.ri_return1.camera_projection_compressed)
            )
            camera_projections[laser.name] = [cp]

            if len(laser.ri_return1.segmentation_label_compressed) > 0:
                seg_label = MatrixInt32()
                seg_label.ParseFromString(
                    zlib.decompress(laser.ri_return1.segmentation_label_compressed)
                )
                seg_labels[laser.name] = [seg_label]

        if len(laser.ri_return2.range_image_compressed) > 0:
            ri = MatrixFloat()
            ri.ParseFromString(zlib.decompress(laser.ri_return2.range_image_compressed))
            range_images[laser.name].append(ri)

            cp = MatrixInt32()
            cp.ParseFromString(
                zlib.decompress(laser.ri_return2.camera_projection_compressed)
            )
            camera_projections[laser.name].append(cp)

            if len(laser.ri_return2.segmentation_label_compressed) > 0:
                seg_label = MatrixInt32()
                seg_label.ParseFromString(
                    zlib.decompress(laser.ri_return2.segmentation_label_compressed)
                )
                seg_labels[laser.name].append(seg_label)

    return range_images, camera_projections, seg_labels, range_image_top_pose


def _rotation_from_rpy(
    roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray
) -> np.ndarray:
    """Build R = R_yaw @ R_pitch @ R_roll, matching ``transform_utils.get_rotation_matrix``.

    Inputs are arrays of the same shape; output is shape (... , 3, 3).
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # R = R_yaw @ R_pitch @ R_roll. Expanded form:
    # | cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr |
    # | sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr |
    # |   -sp,  cp*sr,             cp*cr            |
    shape = roll.shape
    rot = np.empty(shape + (3, 3), dtype=roll.dtype)
    rot[..., 0, 0] = cy * cp
    rot[..., 0, 1] = cy * sp * sr - sy * cr
    rot[..., 0, 2] = cy * sp * cr + sy * sr
    rot[..., 1, 0] = sy * cp
    rot[..., 1, 1] = sy * sp * sr + cy * cr
    rot[..., 1, 2] = sy * sp * cr - cy * sr
    rot[..., 2, 0] = -sp
    rot[..., 2, 1] = cp * sr
    rot[..., 2, 2] = cp * cr
    return rot


def _proto_floats_to_ndarray(scalar_container: Any) -> np.ndarray:
    """Convert a proto ``repeated float`` field to a float32 ndarray.

    ``np.array(container)`` and ``np.asarray(container)`` iterate the proto
    container element-by-element through Python, which is ~40 ms for a
    typical 678 k-float range image. ``np.fromiter`` with a preallocated
    output is ~1.7× faster (~22 ms) on the same data.
    """
    return np.fromiter(
        scalar_container, dtype=np.float32, count=len(scalar_container)
    )


def numpy_convert_range_image_to_point_cloud(
    frame: Any,
    range_images: dict[int, list[Any]],
    range_image_top_pose: Any,
    ri_index: int = 0,
) -> list[np.ndarray]:
    """Numpy-only equivalent of ``frame_utils.convert_range_image_to_point_cloud``.

    Returns a list of (N, 3) float32 ndarrays in vehicle frame, one per
    laser, ordered by ``laser.name`` (matching the upstream sort).
    Polar → cartesian → laser-extrinsic → (TOP only) per-pixel pose →
    world → current-vehicle-frame inverse pose.

    Only the first return (``ri_index=0``) is supported. ``keep_polar_features``
    is not supported here — none of the StandardE2E adapters consume them.
    """
    if ri_index != 0:
        raise NotImplementedError("Only first return is supported")

    # Frame pose: vehicle -> world (4x4)
    frame_pose = np.array(frame.pose.transform, dtype=np.float32).reshape(4, 4)

    # Per-pixel pose for the TOP laser (roll, pitch, yaw, tx, ty, tz) -> 4x4 each
    top_pose_data = _proto_floats_to_ndarray(range_image_top_pose.data).reshape(
        range_image_top_pose.shape.dims
    )
    H_top, W_top = top_pose_data.shape[:2]
    pp_rot = _rotation_from_rpy(
        top_pose_data[..., 0], top_pose_data[..., 1], top_pose_data[..., 2]
    )  # [H, W, 3, 3]
    pp_trans = top_pose_data[..., 3:6].astype(np.float32)  # [H, W, 3]

    world_to_vehicle = np.linalg.inv(frame_pose).astype(np.float32)
    wv_rot = world_to_vehicle[:3, :3]
    wv_trans = world_to_vehicle[:3, 3]

    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points_per_laser: list[np.ndarray] = []

    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        ri = _proto_floats_to_ndarray(range_image.data).reshape(
            range_image.shape.dims
        )
        H, W = ri.shape[:2]

        # Beam inclinations: per-row angle (rad). When the proto doesn't
        # ship explicit values (side lasers), recreate them with pixel-
        # center spacing exactly as ``range_image_utils.compute_inclination``:
        #   inc[i] = ((i + 0.5) / H) * (max - min) + min
        # (NOT ``linspace``, which would put samples at min and max — the
        # half-pixel offset matters for ~7 cm at typical sensor ranges.)
        if len(c.beam_inclinations) == 0:
            diff = c.beam_inclination_max - c.beam_inclination_min
            beam_inclinations = (
                (0.5 + np.arange(H, dtype=np.float32)) / np.float32(H) * diff
                + c.beam_inclination_min
            ).astype(np.float32)
        else:
            beam_inclinations = np.array(c.beam_inclinations, dtype=np.float32)
        # Upstream reverses (axis=-1) so the topmost row corresponds to
        # the largest inclination.
        beam_inclinations = beam_inclinations[::-1]

        # Sensor extrinsic (sensor -> vehicle), 4x4
        extrinsic = np.array(c.extrinsic.transform, dtype=np.float32).reshape(4, 4)
        rot_e = extrinsic[:3, :3]
        trans_e = extrinsic[:3, 3]

        # Azimuth correction = atan2(R[1,0], R[0,0])
        az_correction = np.arctan2(extrinsic[1, 0], extrinsic[0, 0])

        # Per-column azimuth, matching the upstream's ((W-i-0.5)/W*2-1)*pi - az_correction
        ratios = (np.arange(W, 0, -1, dtype=np.float32) - 0.5) / np.float32(W)
        azimuth = (ratios * 2.0 - 1.0) * np.pi - az_correction  # [W]

        # Polar -> cartesian in sensor frame
        range_vals = ri[..., 0]  # [H, W]
        cos_az = np.cos(azimuth)[np.newaxis, :]  # [1, W]
        sin_az = np.sin(azimuth)[np.newaxis, :]
        cos_incl = np.cos(beam_inclinations)[:, np.newaxis]  # [H, 1]
        sin_incl = np.sin(beam_inclinations)[:, np.newaxis]

        x = cos_az * cos_incl * range_vals
        y = sin_az * cos_incl * range_vals
        z = sin_incl * range_vals
        sensor_pts = np.stack([x, y, z], axis=-1)  # [H, W, 3]

        # Sensor -> vehicle frame (apply extrinsic). einsum 'ij,hwj->hwi'.
        vehicle_pts = np.einsum("ij,hwj->hwi", rot_e, sensor_pts) + trans_e

        # For TOP laser, apply per-pixel pose to undo per-row motion
        # within the scan, then convert from world back to the frame's
        # vehicle frame.
        if c.name == LASER_TOP:
            world_pts = np.einsum("hwij,hwj->hwi", pp_rot, vehicle_pts) + pp_trans
            vehicle_pts = np.einsum("ij,hwj->hwi", wv_rot, world_pts) + wv_trans

        # Keep only pixels where range > 0
        mask = range_vals > 0
        points_per_laser.append(vehicle_pts[mask].astype(np.float32))

    return points_per_laser
