"""GPU forward-pass smoke gate.

Tiny ``nn.Sequential(Conv2d -> AdaptiveAvgPool2d -> Linear)`` fed batched
camera tensors on CUDA, once per dataset. Catches device/dtype/
contiguity bugs that CPU collate tests miss:

* HWC vs CHW permutation drift (Conv2d would silently consume the wrong
  axis if both adapters happen to share an [B, H, W, C] -> [B, C, H, W]
  bug).
* uint8-not-floated images (cuBLAS GEMM does not accept uint8 inputs).
* Non-contiguous slices that crash on the GPU but not on CPU.

Same module exercised on both datasets is the strongest cheap
generalization signal: if either dataset's adapter starts emitting an
incompatible tensor layout, this test fires.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from standard_e2e.caching.adapters import (
    CamerasIdentityAdapter,
    Detections3DIdentityAdapter,
    LidarPCIdentityAdapter,
)
from standard_e2e.caching.segment_context import (
    FuturePastStatesFromMatricesAggregator,
)
from standard_e2e.data_structures import (
    BatchedCameraData,
    TransformedFrameData,
    TransformedFrameDataBatch,
)
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
EMBED_DIM = 16
NUM_CLASSES = 4

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _build_camera_encoder(in_channels: int = 3) -> nn.Module:
    """Tiny image encoder: Conv2d -> AdaptiveAvgPool2d -> Linear.

    Deliberately minimal so this test stays sub-second on a 4090. The
    point is to *touch* cuBLAS / cuDNN with the cached tensors, not to
    train anything.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, EMBED_DIM, kernel_size=3, stride=2, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(EMBED_DIM, NUM_CLASSES),
    )


def _images_to_chw_float(images_hwc_uint8: torch.Tensor) -> torch.Tensor:
    """Convert uint8 ``[B, H, W, C]`` to contiguous float32 ``[B, C, H, W]``."""
    assert (
        images_hwc_uint8.dtype == torch.uint8
    ), f"expected uint8 HWC; got dtype={images_hwc_uint8.dtype}"
    assert (
        images_hwc_uint8.ndim == 4
    ), f"expected [B, H, W, C]; got shape={images_hwc_uint8.shape}"
    return (
        images_hwc_uint8.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32)
        / 255.0
    )


def _forward_through_front_camera(
    batch: TransformedFrameDataBatch, device: torch.device
) -> torch.Tensor:
    cameras = batch.get_modality_data(Modality.CAMERAS)
    assert isinstance(cameras, BatchedCameraData)
    images_hwc = cameras.images[CameraDirection.FRONT].to(device)
    images_chw = _images_to_chw_float(images_hwc)
    encoder = _build_camera_encoder().to(device)
    with torch.no_grad():
        logits = encoder(images_chw)
    return logits


@pytest.fixture(scope="module")
def e2e_batch(tmp_path_factory) -> TransformedFrameDataBatch:
    if not WAYMO_E2E_SHARD.exists():
        pytest.skip(f"Waymo E2E shard not found at {WAYMO_E2E_SHARD}")
    import tensorflow as tf

    from standard_e2e.caching.src_datasets.waymo_e2e.waymo_e2e_dataset_processor import (  # noqa: E501
        WaymoE2EDatasetProcessor,
    )

    out_dir = tmp_path_factory.mktemp("gpu_smoke_e2e")
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

    cache_dir = tmp_path_factory.mktemp("gpu_smoke_perception")
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

    keep = [Modality.CAMERAS, Modality.FUTURE_STATES, Modality.PAST_STATES]
    for record in index_records:
        path = Path(cache_dir) / record["filename"]
        frames.append(
            TransformedFrameData.from_npz(str(path), required_modalities=keep)
        )
    return TransformedFrameDataBatch(frames)


def test_e2e_cameras_forward_on_cuda(
    e2e_batch: TransformedFrameDataBatch, cuda_device: torch.device
):
    logits = _forward_through_front_camera(e2e_batch, cuda_device)
    assert logits.device.type == "cuda"
    assert logits.shape == (NUM_FRAMES, NUM_CLASSES)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all(), "non-finite outputs from forward pass"


def test_perception_cameras_forward_on_cuda(
    perception_batch: TransformedFrameDataBatch, cuda_device: torch.device
):
    logits = _forward_through_front_camera(perception_batch, cuda_device)
    assert logits.device.type == "cuda"
    assert logits.shape == (NUM_FRAMES, NUM_CLASSES)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all(), "non-finite outputs from forward pass"


def test_shared_encoder_runs_on_both_datasets(
    e2e_batch: TransformedFrameDataBatch,
    perception_batch: TransformedFrameDataBatch,
    cuda_device: torch.device,
):
    """Same instantiated module weights consume both datasets' tensors.

    Cross-dataset generalization signal: if one dataset starts emitting a
    layout the encoder cannot accept, this test fires before any model
    trainer hits the same crash. Using one shared module (instead of two
    independent ones) is what makes this stronger than the per-dataset
    smokes above.
    """
    encoder = _build_camera_encoder().to(cuda_device).eval()

    for label, batch in (("e2e", e2e_batch), ("perception", perception_batch)):
        cameras = batch.get_modality_data(Modality.CAMERAS)
        assert isinstance(cameras, BatchedCameraData)
        images_hwc = cameras.images[CameraDirection.FRONT].to(cuda_device)
        images_chw = _images_to_chw_float(images_hwc)
        with torch.no_grad():
            logits = encoder(images_chw)
        assert logits.shape == (
            NUM_FRAMES,
            NUM_CLASSES,
        ), f"{label}: unexpected logit shape {logits.shape}"
        assert torch.isfinite(logits).all(), f"{label}: non-finite logits"
