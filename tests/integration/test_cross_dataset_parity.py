"""Cross-dataset schema parity gate.

Builds a ``TransformedFrameDataBatch`` from both Waymo E2E and Waymo
Perception fixtures and asserts identical dtypes / shapes / devices for
the modalities they both produce (cameras, future/past states). Catches
silent adapter drift the moment one dataset starts emitting
``float32 CHW`` while the other still emits ``uint8 HWC`` (or similar).

Per-dataset PR gates only catch per-dataset breakage; this test catches
cross-dataset symmetry breakage that downstream model trainers cannot
recover from.

The Waymo Perception default adapter set ships ``PanoImageAdapter`` (a
single stitched panorama for ``Modality.CAMERAS``). For an honest schema
comparison against Waymo E2E (which emits ``dict[CameraDirection,
CameraData]`` via ``CamerasIdentityAdapter``), we explicitly pass the
``CamerasIdentityAdapter`` to the Perception processor here. If the
default ever changes such that both datasets do agree on a single
``Modality.CAMERAS`` schema, this override can be dropped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from standard_e2e.caching.adapters import (
    CamerasIdentityAdapter,
    Detections3DIdentityAdapter,
    LidarPCIdentityAdapter,
)
from standard_e2e.caching.segment_context import (
    FutureDetectionsAggregator,
    FuturePastStatesFromMatricesAggregator,
)
from standard_e2e.data_structures import (
    BatchedCameraData,
    TransformedFrameData,
    TransformedFrameDataBatch,
)
from standard_e2e.data_structures.trajectory_data import BatchedTrajectory
from standard_e2e.enums import CameraDirection, Modality

WAYMO_E2E_ROOT = Path(
    "/mnt/bigdisk/datasets/waymo/waymo_open_dataset_end_to_end_camera_v_1_0_0"
)
WAYMO_E2E_SHARD = (
    WAYMO_E2E_ROOT / "training_202504031202_202504151040.tfrecord-00000-of-00263"
)
WAYMO_PERCEPTION_VAL = Path(
    "/mnt/bigdisk/datasets/waymo/waymo_open_dataset_v_1_4_3"
    "/individual_files/validation"
)
WAYMO_PERCEPTION_SHARD = (
    WAYMO_PERCEPTION_VAL
    / "segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord"
)

NUM_FRAMES = 3

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def e2e_batch(tmp_path_factory) -> TransformedFrameDataBatch:
    if not WAYMO_E2E_SHARD.exists():
        pytest.skip(f"Waymo E2E shard not found at {WAYMO_E2E_SHARD}")

    import tensorflow as tf

    from standard_e2e.caching.src_datasets.waymo_e2e.waymo_e2e_dataset_processor import (  # noqa: E501
        WaymoE2EDatasetProcessor,
    )

    out_dir = tmp_path_factory.mktemp("parity_e2e")
    processor = WaymoE2EDatasetProcessor(
        common_output_path=str(out_dir), split="training"
    )
    frames: list[TransformedFrameData] = []
    raw_iter = tf.data.TFRecordDataset(str(WAYMO_E2E_SHARD), compression_type="")
    for raw in raw_iter.take(NUM_FRAMES):
        frame_data, _ = processor.process_frame(raw)
        target = Path(out_dir) / frame_data.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        frame_data.to_npz(str(target))
        frames.append(TransformedFrameData.from_npz(str(target)))
    return TransformedFrameDataBatch(frames)


@pytest.fixture(scope="module")
def perception_batch(tmp_path_factory) -> TransformedFrameDataBatch:
    if not WAYMO_PERCEPTION_SHARD.exists():
        pytest.skip(f"Waymo Perception segment not found at {WAYMO_PERCEPTION_SHARD}")

    import tensorflow as tf

    from standard_e2e.caching.src_datasets.waymo_perception import (
        WaymoPerceptionDatasetProcessor,
    )

    cache_dir = tmp_path_factory.mktemp("parity_perception")
    # Force per-direction CameraData (matches the Waymo E2E default schema)
    # rather than the panorama default. HD-map adapter is omitted so the
    # cross-dataset parity test does not require segment-level HD-map I/O.
    adapters = [
        CamerasIdentityAdapter(),
        Detections3DIdentityAdapter(),
        LidarPCIdentityAdapter(),
    ]
    aggregators = [
        FuturePastStatesFromMatricesAggregator(str(cache_dir)),
        FutureDetectionsAggregator(str(cache_dir)),
    ]
    processor = WaymoPerceptionDatasetProcessor(
        common_output_path=str(cache_dir),
        split="validation",
        adapters=adapters,
        context_aggregators=aggregators,
    )

    frames: list[TransformedFrameData] = []
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

    # DETECTIONS_3D is a ``list[Detection3D]`` after aggregation - variable
    # length across frames makes the default torch collate raise "each
    # element in list of batch should be of equal size". Schema parity
    # checks only the modalities both datasets emit (cameras + states),
    # so we filter the loaded frames to those + LIDAR_PC.
    keep = [
        Modality.CAMERAS,
        Modality.LIDAR_PC,
        Modality.FUTURE_STATES,
        Modality.PAST_STATES,
    ]
    for record in index_records:
        path = Path(cache_dir) / record["filename"]
        frames.append(
            TransformedFrameData.from_npz(str(path), required_modalities=keep)
        )
    return TransformedFrameDataBatch(frames)


def test_camera_schema_parity_on_shared_directions(
    e2e_batch: TransformedFrameDataBatch,
    perception_batch: TransformedFrameDataBatch,
):
    """Cameras shared by both datasets must agree on dtype + ndim + device.

    Image (H, W) intentionally differs across Waymo's per-direction sensor
    resolutions, so we do NOT lock spatial dims - dtype/ndim/device drift
    is the failure mode that breaks downstream training.
    """
    e2e_cams = e2e_batch.get_modality_data(Modality.CAMERAS)
    perception_cams = perception_batch.get_modality_data(Modality.CAMERAS)
    assert isinstance(e2e_cams, BatchedCameraData)
    assert isinstance(perception_cams, BatchedCameraData)

    shared = set(e2e_cams.directions) & set(perception_cams.directions)
    # Waymo Perception ships 5 surround cameras (no rear); intersection
    # against E2E's 8 must therefore be exactly the front + side ring.
    assert shared == {
        CameraDirection.FRONT,
        CameraDirection.FRONT_LEFT,
        CameraDirection.FRONT_RIGHT,
        CameraDirection.SIDE_LEFT,
        CameraDirection.SIDE_RIGHT,
    }, f"unexpected shared camera set: {shared}"

    for direction in shared:
        e2e_img = e2e_cams.images[direction]
        perception_img = perception_cams.images[direction]
        assert e2e_img.dtype == perception_img.dtype, (
            f"image dtype drift on {direction}: "
            f"E2E={e2e_img.dtype} vs Perception={perception_img.dtype}"
        )
        assert e2e_img.ndim == perception_img.ndim == 4, (
            f"image ndim drift on {direction}: "
            f"E2E={e2e_img.ndim} vs Perception={perception_img.ndim}"
        )
        # Channel-count check guards against accidental CHW vs HWC swap
        # in only one of the adapters. Last axis must be 3 (HWC).
        assert e2e_img.shape[-1] == perception_img.shape[-1] == 3, (
            f"channel-axis drift on {direction}: "
            f"E2E={e2e_img.shape} vs Perception={perception_img.shape}"
        )
        assert e2e_img.device == perception_img.device, (
            f"device drift on {direction}: "
            f"E2E={e2e_img.device} vs Perception={perception_img.device}"
        )

        # Calibration tensors are fixed-shape and must match exactly.
        for label, e2e_t, perception_t, expected_shape in (
            (
                "intrinsics",
                e2e_cams.intrinsics[direction],
                perception_cams.intrinsics[direction],
                (NUM_FRAMES, 3, 3),
            ),
            (
                "extrinsics",
                e2e_cams.extrinsics[direction],
                perception_cams.extrinsics[direction],
                (NUM_FRAMES, 4, 4),
            ),
        ):
            assert e2e_t.shape == perception_t.shape == expected_shape, (
                f"{label} shape drift on {direction}: "
                f"E2E={e2e_t.shape} vs Perception={perception_t.shape}"
            )
            assert e2e_t.dtype == perception_t.dtype == torch.float32, (
                f"{label} dtype drift on {direction}: "
                f"E2E={e2e_t.dtype} vs Perception={perception_t.dtype}"
            )
            assert e2e_t.device == perception_t.device, (
                f"{label} device drift on {direction}: "
                f"E2E={e2e_t.device} vs Perception={perception_t.device}"
            )


def test_trajectory_schema_parity(
    e2e_batch: TransformedFrameDataBatch,
    perception_batch: TransformedFrameDataBatch,
):
    """``FUTURE_STATES`` / ``PAST_STATES`` must be ``BatchedTrajectory`` on
    both datasets with identical tensor dtype + device for the shared
    components. Sequence length intentionally differs (Perception derives
    states from pose matrices over the segment; E2E ships them per-frame
    from the proto), so we only lock dtype, batch size, and device.
    """
    for modality in (Modality.FUTURE_STATES, Modality.PAST_STATES):
        e2e_traj = e2e_batch.get_modality_data(modality)
        perception_traj = perception_batch.get_modality_data(modality)
        assert isinstance(
            e2e_traj, BatchedTrajectory
        ), f"E2E {modality} is not BatchedTrajectory: {type(e2e_traj)}"
        assert isinstance(
            perception_traj, BatchedTrajectory
        ), f"Perception {modality} is not BatchedTrajectory: {type(perception_traj)}"

        assert e2e_traj.batch_size == perception_traj.batch_size == NUM_FRAMES
        assert e2e_traj.device == perception_traj.device

        # Both adapters must populate the canonical pose components for
        # downstream models to share an embedding head.
        from standard_e2e.enums import TrajectoryComponent as TC

        for component in (TC.X, TC.Y):
            assert e2e_traj.has(component), f"E2E {modality} missing {component}"
            assert perception_traj.has(
                component
            ), f"Perception {modality} missing {component}"
            e2e_t = e2e_traj.get(component)
            perception_t = perception_traj.get(component)
            assert e2e_t.dtype == perception_t.dtype == torch.float32, (
                f"{modality}.{component} dtype drift: "
                f"E2E={e2e_t.dtype} vs Perception={perception_t.dtype}"
            )


def test_batch_metadata_parity(
    e2e_batch: TransformedFrameDataBatch,
    perception_batch: TransformedFrameDataBatch,
):
    """Top-level batch metadata (timestamp tensor, filename list, device)
    must agree on dtype/length/device across datasets."""
    assert (
        e2e_batch.timestamp.dtype == perception_batch.timestamp.dtype == torch.float32
    )
    assert (
        e2e_batch.timestamp.shape == perception_batch.timestamp.shape == (NUM_FRAMES,)
    )
    assert e2e_batch.timestamp.device == perception_batch.timestamp.device
    assert len(e2e_batch.filename) == len(perception_batch.filename) == NUM_FRAMES
