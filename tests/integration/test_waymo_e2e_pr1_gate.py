"""PR 1 gate: Waymo E2E integration smoke + visual rendering.

Runs a small slice of one real shard through ``WaymoE2EDatasetProcessor``,
asserts shape invariants on the surround-camera output, and renders 9 PNGs
(3 modalities x 3 frames) into ``tests/visual_inspection/pr1/`` for review.

E2E shuffles frames across shards, so we assert per-frame invariants only,
not per-segment continuity.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from standard_e2e.data_structures import TransformedFrameData
from standard_e2e.enums import CameraDirection, Intent, Modality

WAYMO_E2E_ROOT = Path(
    "/mnt/bigdisk/datasets/waymo/waymo_open_dataset_end_to_end_camera_v_1_0_0"
)
SHARD = WAYMO_E2E_ROOT / "test_202504211836-202504220845.tfrecord-00000-of-00266"

REPO_ROOT = Path(__file__).resolve().parents[2]
RENDER_SCRIPT = REPO_ROOT / "scripts" / "render_frame.py"
VISUAL_OUT = REPO_ROOT / "tests" / "visual_inspection" / "pr1"

NUM_FRAMES = 3
RENDER_MODALITIES = ["cameras", "combined", "intent"]


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
        pytest.skip(f"Waymo E2E shard not found at {SHARD}")

    import tensorflow as tf

    from standard_e2e.caching.src_datasets.waymo_e2e.waymo_e2e_dataset_processor import (  # noqa: E501
        WaymoE2EDatasetProcessor,
    )

    out_dir = tmp_path_factory.mktemp("waymo_e2e_pr1")
    processor = WaymoE2EDatasetProcessor(common_output_path=str(out_dir), split="test")

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


def test_processed_frame_has_full_surround_cameras(processed_frames):
    """8 cameras populated; uint8 HWC; intrinsics 3x3, extrinsics 4x4."""
    expected_dirs = set(CameraDirection)
    for path in processed_frames:
        frame = TransformedFrameData.from_npz(str(path))
        cameras = frame.get_modality_data(Modality.CAMERAS)
        assert cameras is not None
        assert set(cameras.keys()) == expected_dirs, (
            f"missing camera directions in {path}: "
            f"{expected_dirs - set(cameras.keys())}"
        )
        for direction, cam in cameras.items():
            assert cam.image.ndim == 3
            assert cam.image.dtype.name == "uint8"
            assert cam.intrinsics.shape == (3, 3)
            assert cam.extrinsics.shape == (4, 4)


def test_processed_frame_has_intent_and_trajectories(processed_frames):
    """Intent in domain; future/past trajectory objects present.

    The Waymo E2E *test* split intentionally hides labels for some frames
    (empty future_states is normal there); we assert presence of the
    Trajectory objects but not non-zero length.
    """
    for path in processed_frames:
        frame = TransformedFrameData.from_npz(str(path))
        intent = frame.get_modality_data(Modality.INTENT)
        assert intent is not None
        assert int(intent) in {i.value for i in Intent}

        future = frame.get_modality_data(Modality.FUTURE_STATES)
        past = frame.get_modality_data(Modality.PAST_STATES)
        assert future is not None
        assert past is not None


def test_no_lidar_in_waymo_e2e_v1_0_0(processed_frames):
    """E2E v1.0.0 ships zero lasers; lidar_pc must NOT be present.

    Confirms the PR 1 boundary: lidar scaffold is wired but the dataset
    populates no lidar yet (extraction lands in PR 2 against Perception).
    """
    for path in processed_frames:
        frame = TransformedFrameData.from_npz(str(path))
        present = set(frame.get_present_modality_keys())
        assert Modality.LIDAR_PC not in present


def test_render_visual_gate_pngs(processed_frames, render_module):
    """Render 9 PNGs (3 modalities x 3 frames) into visual_inspection dir."""
    VISUAL_OUT.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    for i, npz_path in enumerate(processed_frames):
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
    assert len(produced) == NUM_FRAMES * len(RENDER_MODALITIES)
