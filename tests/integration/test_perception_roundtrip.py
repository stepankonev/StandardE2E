"""Numerical round-trip gates for the Waymo Perception pipeline.

The PR2 cross-modality test only checks that lidar's 99th-percentile
radius and detection-center radii sit in the same ballpark - that is a
*necessary* but insufficient condition. These tests close that gap with
two end-to-end numerical checks that tie multiple modalities together
into a single assertion:

* Trajectory round-trip: ``T_world_ego[t] @ past_states[t-Δ]`` should
  recover ``global_position[t-Δ]``. A frame-of-reference bug (e.g. the
  aggregator emitting positions in world frame instead of ego frame, or
  applying ``inv()`` once too many) produces detections that look
  visually plausible but are numerically off — only this round-trip
  catches it.

* Detection projection: ``inv(extrinsics) -> intrinsics`` projects
  close-range 3D box centers into the image plane. If extrinsics drift,
  if the intrinsic matrix transposes, or if the Waymo
  X-forward/Y-left/Z-up convention silently flips, no close-range
  detection lands inside the front camera and this test fires.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from standard_e2e.caching.adapters import (
    CamerasIdentityAdapter,
    Detections3DIdentityAdapter,
    LidarPCIdentityAdapter,
)
from standard_e2e.caching.segment_context import (
    FuturePastStatesFromMatricesAggregator,
)
from standard_e2e.data_structures import (
    CameraData,
    FrameDetections3D,
    TransformedFrameData,
)
from standard_e2e.enums import Modality
from standard_e2e.enums import TrajectoryComponent as TC

WAYMO_PERCEPTION_VAL = Path(
    "/mnt/bigdisk/datasets/waymo/waymo_open_dataset_v_1_4_3"
    "/individual_files/validation"
)
WAYMO_PERCEPTION_SHARD = (
    WAYMO_PERCEPTION_VAL
    / "segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord"
)

NUM_FRAMES = 5
# Detections this close to ego-vehicle should always project into a
# camera unless the extrinsic / intrinsic / convention is broken.
DETECTION_CLOSE_RANGE_M = 30.0
# Linear-algebra round-trip tolerance: pose composition is
# floating-point-stable; sub-millimetre is achievable but allow a few
# centimetres for safety.
TRAJECTORY_ROUND_TRIP_TOL_M = 0.05

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def perception_segment_frames(tmp_path_factory) -> list[TransformedFrameData]:
    """Process N frames + run the pose-matrix state aggregator only.

    We deliberately skip ``FutureDetectionsAggregator`` so each frame's
    ``Modality.DETECTIONS_3D`` payload remains a ``FrameDetections3D``
    (with **current-frame** box centers in ego coords) instead of the
    aggregated future-trajectory list. Both round-trip tests need the
    raw current-frame geometry, not future trajectories.
    """
    if not WAYMO_PERCEPTION_SHARD.exists():
        pytest.skip(f"Waymo Perception segment not found at {WAYMO_PERCEPTION_SHARD}")

    import tensorflow as tf

    from standard_e2e.caching.src_datasets.waymo_perception import (
        WaymoPerceptionDatasetProcessor,
    )

    cache_dir = tmp_path_factory.mktemp("perception_roundtrip")
    adapters = [
        CamerasIdentityAdapter(),
        Detections3DIdentityAdapter(),
        LidarPCIdentityAdapter(),
    ]
    aggregators = [FuturePastStatesFromMatricesAggregator(str(cache_dir))]
    processor = WaymoPerceptionDatasetProcessor(
        common_output_path=str(cache_dir),
        split="validation",
        adapters=adapters,
        context_aggregators=aggregators,
    )

    index_records: list[dict] = []
    raw_iter = tf.data.TFRecordDataset(str(WAYMO_PERCEPTION_SHARD), compression_type="")
    for raw in raw_iter.take(NUM_FRAMES):
        frame_data, _ = processor.process_frame(raw)
        target = Path(cache_dir) / frame_data.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        frame_data.to_npz(str(target))
        index_records.append(
            {
                "segment_id": frame_data.segment_id,
                "timestamp": frame_data.timestamp,
                "filename": frame_data.filename,
            }
        )

    index_df = (
        pd.DataFrame(index_records).sort_values(by="timestamp").reset_index(drop=True)
    )
    for aggregator in processor.context_aggregators:
        aggregator.process(index_df)

    return [
        TransformedFrameData.from_npz(str(Path(cache_dir) / r["filename"]))
        for r in index_records
    ]


def _extract_pose_matrix(frame: TransformedFrameData) -> np.ndarray:
    """Pull the 4x4 ego->world pose set by the Perception processor."""
    assert frame.aux_data is not None, "aux_data missing - did processor strip it?"
    pose = np.asarray(frame.aux_data["pose_matrix"], dtype=np.float64)
    assert pose.shape == (4, 4)
    return pose


def test_past_states_world_round_trip(
    perception_segment_frames: list[TransformedFrameData],
):
    """``T_world_ego[i] @ past_states[i][k]`` == ``global_position[k]``.

    For each frame ``i > 0``, the aggregator emits ``past_states`` of
    length ``i + 1`` containing ego positions at frames ``0..i`` in
    frame ``i``'s ego frame. Composing with ``T_world_ego[i]`` should
    recover the world position recorded for frame ``k`` from its own
    ``global_position`` (via the pose-matrix translation column).
    """
    assert len(perception_segment_frames) >= 2, "need at least 2 frames to round-trip"

    world_positions = []
    for frame in perception_segment_frames:
        pose = _extract_pose_matrix(frame)
        world_positions.append(pose[:3, 3])
    world_positions_np = np.stack(world_positions, axis=0)  # (N, 3)

    for i in range(1, len(perception_segment_frames)):
        frame_i = perception_segment_frames[i]
        past = frame_i.get_modality_data(Modality.PAST_STATES)
        assert (
            past is not None and past.length == i + 1
        ), f"frame {i}: past_states length {past.length if past else None} != {i + 1}"

        past_xyz = past.get([TC.X, TC.Y, TC.Z])  # (i+1, 3) in ego frame at time i
        T_world_ego_i = _extract_pose_matrix(frame_i)

        # Compose to world frame for each historical timestep.
        ones = np.ones((past_xyz.shape[0], 1), dtype=np.float64)
        past_h = np.concatenate([past_xyz.astype(np.float64), ones], axis=1)
        recovered_world_h = (T_world_ego_i @ past_h.T).T
        recovered_world = recovered_world_h[:, :3]

        # Last past_states entry is the current frame itself - position
        # 0 in ego frame round-trips to ego's own world position.
        np.testing.assert_allclose(
            recovered_world[-1],
            world_positions_np[i],
            atol=TRAJECTORY_ROUND_TRIP_TOL_M,
            err_msg=f"frame {i}: self-roundtrip drift",
        )

        for k in range(i):
            np.testing.assert_allclose(
                recovered_world[k],
                world_positions_np[k],
                atol=TRAJECTORY_ROUND_TRIP_TOL_M,
                err_msg=(
                    f"frame {i}: past_states[{k}] -> world drift exceeds "
                    f"{TRAJECTORY_ROUND_TRIP_TOL_M} m. Indicates ego/world "
                    "frame confusion in FuturePastStatesFromMatricesAggregator."
                ),
            )


def _project_ego_to_image(
    points_ego_xyz: np.ndarray, intrinsics_K: np.ndarray, extrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project ego-frame 3D points to pixels using Waymo's convention.

    Waymo extrinsic semantics (per our ``CameraData`` docstring): a point
    in camera frame maps to ego via ``p_ego = extrinsics @ p_cam``.
    Vehicle and camera frames both use the X-forward, Y-left, Z-up
    convention. We invert to map ego -> camera, then permute to standard
    image-plane axes (u = -Y_cam, v = -Z_cam, depth = X_cam) before
    applying ``K``.
    """
    n = points_ego_xyz.shape[0]
    p_ego_h = np.concatenate(
        [points_ego_xyz, np.ones((n, 1), dtype=np.float64)], axis=1
    )
    T_camera_ego = np.linalg.inv(extrinsics.astype(np.float64))
    p_cam = (T_camera_ego @ p_ego_h.T).T  # (N, 4)
    # Waymo cam: X-forward, Y-left, Z-up
    # Image axes:  u = -Y_cam, v = -Z_cam, depth = X_cam
    axes = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    p_img_h = (axes @ p_cam.T).T  # (N, 3): (u_unscaled, v_unscaled, depth)
    depth = p_img_h[:, 2]
    in_front = depth > 1e-3
    pixels_h = (intrinsics_K.astype(np.float64) @ p_img_h.T).T
    # Avoid divide-by-zero for points behind the camera; we'll mask them out.
    safe_depth = np.where(np.abs(pixels_h[:, 2]) < 1e-9, 1.0, pixels_h[:, 2])
    pixels = pixels_h[:, :2] / safe_depth[:, None]
    return pixels, in_front


def test_close_range_detections_project_into_some_camera(
    perception_segment_frames: list[TransformedFrameData],
):
    """At least one close-range detection projects into at least one of
    the surround cameras. Catches:

    * extrinsic drift / inversion bugs (a wrong-handed extrinsic puts
      every box behind every camera or off-image),
    * intrinsic transposition (focal lengths swapped with principal
      point),
    * convention flips (X-forward becoming Z-forward without the
      corresponding axes-permutation getting updated).

    We sweep all 5 Perception cameras because the FRONT camera's
    horizontal FOV is narrow (~50 deg) - close-range agents to the side
    or rear are visible in the side cameras, not FRONT, and we don't
    want a perfectly fine pipeline to fail just because the picked
    segment happened to have no agents in the narrow FRONT cone. The
    PR2 cross-modality test compares lidar radius vs detection-centre
    radius - it can pass even when cameras are misaligned with the
    world. This is the missing complementary check.
    """
    found_inside = False
    debug_per_frame: list[str] = []
    for i, frame in enumerate(perception_segment_frames):
        cameras = frame.get_modality_data(Modality.CAMERAS)
        detections = frame.get_modality_data(Modality.DETECTIONS_3D)
        assert isinstance(detections, FrameDetections3D), (
            f"frame {i}: expected raw FrameDetections3D, got {type(detections)} - "
            "FutureDetectionsAggregator should NOT have run for this fixture."
        )

        centers: list[np.ndarray] = []
        for det in detections.detections:
            xyz = det.trajectory.get([TC.X, TC.Y, TC.Z])
            assert xyz.shape == (1, 3)
            radius = float(np.linalg.norm(xyz[0]))
            if radius <= DETECTION_CLOSE_RANGE_M:
                centers.append(xyz[0].astype(np.float64))
        if not centers:
            debug_per_frame.append(f"frame {i}: no close-range detections")
            continue

        centers_np = np.stack(centers, axis=0)
        per_camera_inside = {}
        for direction, cam in cameras.items():
            cam_data: CameraData = cam
            pixels, in_front = _project_ego_to_image(
                centers_np, cam_data.intrinsics, cam_data.extrinsics
            )
            h, w = cam_data.H, cam_data.W
            # 5% margin: a box's *center* can sit just outside the
            # visible rectangle while the rendered box is still in-frame
            # (Waymo labels the full 3D extent, not just the centroid).
            margin = 0.05
            u_lo, u_hi = -margin * w, (1 + margin) * w
            v_lo, v_hi = -margin * h, (1 + margin) * h
            in_image = (
                (pixels[:, 0] >= u_lo)
                & (pixels[:, 0] <= u_hi)
                & (pixels[:, 1] >= v_lo)
                & (pixels[:, 1] <= v_hi)
            )
            valid = in_front & in_image
            per_camera_inside[direction] = int(valid.sum())
            if valid.any():
                found_inside = True

        debug_per_frame.append(
            f"frame {i}: {len(centers)} close-range; "
            f"per-camera inside={per_camera_inside}"
        )

    assert found_inside, (
        "No close-range (<{:.0f} m) detection projected inside ANY surround "
        "camera across {} frames. Per-frame stats: {}"
    ).format(DETECTION_CLOSE_RANGE_M, len(perception_segment_frames), debug_per_frame)
