"""PR 1 secondary gate: training-split shard exercises non-empty trajectories.

Mirrors the test-split gate but on a ``training`` shard so future_states /
past_states are populated. Adds a DataLoader smoke that pulls cached frames
through ``UnifiedE2EDataset.collate_fn``, validating the PR 1 collate path
end-to-end (BatchedCameraData + BatchedTrajectory + scalar tensors).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from standard_e2e.data_structures import (
    BatchedCameraData,
    TransformedFrameData,
    TransformedFrameDataBatch,
)
from standard_e2e.data_structures.trajectory_data import BatchedTrajectory
from standard_e2e.enums import CameraDirection, Intent, Modality
from standard_e2e.unified_dataset import UnifiedE2EDataset

WAYMO_E2E_ROOT = Path(
    "/mnt/bigdisk/datasets/waymo/waymo_open_dataset_end_to_end_camera_v_1_0_0"
)
SHARD = WAYMO_E2E_ROOT / "training_202504031202_202504151040.tfrecord-00000-of-00263"

REPO_ROOT = Path(__file__).resolve().parents[2]
RENDER_SCRIPT = REPO_ROOT / "scripts" / "render_frame.py"
VISUAL_OUT = REPO_ROOT / "tests" / "visual_inspection" / "pr1_training"

NUM_FRAMES = 4
RENDER_MODALITIES = ["combined"]


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def render_module():
    spec = importlib.util.spec_from_file_location("_render_frame", RENDER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load render script from {RENDER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_render_frame"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def processed_frames(tmp_path_factory) -> list[Path]:
    if not SHARD.exists():
        pytest.skip(f"Waymo E2E training shard not found at {SHARD}")

    import tensorflow as tf

    from standard_e2e.caching.src_datasets.waymo_e2e.waymo_e2e_dataset_processor import (  # noqa: E501
        WaymoE2EDatasetProcessor,
    )

    out_dir = tmp_path_factory.mktemp("waymo_e2e_pr1_training")
    processor = WaymoE2EDatasetProcessor(
        common_output_path=str(out_dir), split="training"
    )

    written: list[Path] = []
    raw_iter = tf.data.TFRecordDataset(str(SHARD), compression_type="")
    for raw in raw_iter.take(NUM_FRAMES):
        frame_data, _ = processor.process_frame(raw)
        target = Path(out_dir) / frame_data.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        frame_data.to_npz(str(target))
        written.append(target)

    assert len(written) == NUM_FRAMES, "expected NUM_FRAMES processed frames"
    return written


def test_training_split_has_populated_trajectories(processed_frames):
    """Training split exposes non-empty past_states + future_states."""
    for path in processed_frames:
        frame = TransformedFrameData.from_npz(str(path))
        intent = frame.get_modality_data(Modality.INTENT)
        assert intent is not None
        assert int(intent) in {i.value for i in Intent}

        future = frame.get_modality_data(Modality.FUTURE_STATES)
        past = frame.get_modality_data(Modality.PAST_STATES)
        assert (
            future is not None and future.length > 0
        ), f"training-split future_states should be non-empty in {path}"
        assert (
            past is not None and past.length > 0
        ), f"training-split past_states should be non-empty in {path}"


def test_training_split_render_combined(processed_frames, render_module):
    """Combined render on training frames shows the populated trajectory."""
    VISUAL_OUT.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    for i, npz_path in enumerate(processed_frames[:3]):
        for modality in RENDER_MODALITIES:
            out = VISUAL_OUT / f"frame{i}_{modality}.png"
            render_module.render_frame(
                npz_path=str(npz_path), modality=modality, out_path=str(out)
            )
            assert out.exists()
            size = out.stat().st_size
            assert (
                5 * 1024 <= size <= 2 * 1024 * 1024
            ), f"PNG {out} size out of band: {size} bytes"
            produced.append(out)
    assert len(produced) == 3 * len(RENDER_MODALITIES)


class _FrameListDataset(Dataset):
    """Minimal Dataset that yields {'frame': TransformedFrameData} samples.

    Mirrors ``UnifiedE2EDataset.__getitem__`` shape so we can exercise the
    same collate_fn without bootstrapping an index file.
    """

    def __init__(self, paths: list[Path]):
        self._paths = list(paths)

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> dict[str, TransformedFrameData]:
        return {"frame": TransformedFrameData.from_npz(str(self._paths[idx]))}


def test_dataloader_collate_smoke(processed_frames):
    """Pull cached frames through DataLoader + UnifiedE2EDataset.collate_fn.

    Validates the full PR 1 collate path: BatchedCameraData per direction,
    BatchedTrajectory for past/future, scalar intent stacked to a tensor.
    """
    dataset = _FrameListDataset(processed_frames)
    loader = DataLoader(
        dataset,
        batch_size=NUM_FRAMES,
        shuffle=False,
        collate_fn=UnifiedE2EDataset.collate_fn,
    )
    batches = list(loader)
    assert len(batches) == 1
    batch = batches[0]
    assert "frame" in batch
    fb = batch["frame"]
    assert isinstance(fb, TransformedFrameDataBatch)

    cameras = fb.get_modality_data(Modality.CAMERAS)
    assert isinstance(cameras, BatchedCameraData)
    assert cameras.batch_size == NUM_FRAMES
    expected_dirs = set(CameraDirection)
    assert set(cameras.directions) == expected_dirs
    front = cameras.images[CameraDirection.FRONT]
    assert front.shape[0] == NUM_FRAMES
    assert front.dtype == torch.uint8
    assert front.ndim == 4  # [B, H, W, C]
    intr = cameras.intrinsics[CameraDirection.FRONT]
    extr = cameras.extrinsics[CameraDirection.FRONT]
    assert intr.shape == (NUM_FRAMES, 3, 3)
    assert extr.shape == (NUM_FRAMES, 4, 4)

    intent = fb.get_modality_data(Modality.INTENT)
    assert isinstance(intent, torch.Tensor)
    assert intent.shape == (NUM_FRAMES,)

    future = fb.get_modality_data(Modality.FUTURE_STATES)
    past = fb.get_modality_data(Modality.PAST_STATES)
    assert isinstance(future, BatchedTrajectory)
    assert isinstance(past, BatchedTrajectory)
    assert future.batch_size == NUM_FRAMES
    assert past.batch_size == NUM_FRAMES

    # Lidar must NOT be present in this E2E v1.0.0 batch.
    present = set(fb._modality_data.keys())  # noqa: SLF001
    assert Modality.LIDAR_PC not in present


def test_camera_direction_mapping_matches_proto():
    """``WAYMO_CAMERAS_ORDER`` must mirror Waymo's ``CameraName.Name`` enum.

    Sanity check that PR 1's per-direction mapping matches the canonical
    proto enum without needing the full waymo-open-dataset SDK installed.
    """
    from standard_e2e.third_party.waymo_open_dataset import (
        dataset_pb2,  # type: ignore[attr-defined]
    )
    from standard_e2e.utils.image_utils import WAYMO_CAMERAS_ORDER

    proto_enum = dataset_pb2.CameraName.Name  # pylint: disable=no-member
    expected = {
        CameraDirection.FRONT: proto_enum.Value("FRONT"),
        CameraDirection.FRONT_LEFT: proto_enum.Value("FRONT_LEFT"),
        CameraDirection.FRONT_RIGHT: proto_enum.Value("FRONT_RIGHT"),
        CameraDirection.SIDE_LEFT: proto_enum.Value("SIDE_LEFT"),
        CameraDirection.SIDE_RIGHT: proto_enum.Value("SIDE_RIGHT"),
        CameraDirection.REAR_LEFT: proto_enum.Value("REAR_LEFT"),
        CameraDirection.REAR: proto_enum.Value("REAR"),
        CameraDirection.REAR_RIGHT: proto_enum.Value("REAR_RIGHT"),
    }
    assert WAYMO_CAMERAS_ORDER == expected
