"""PR 1 step 1.3: surround camera mode + ``BatchedCameraData`` collation.

Covers:
    * ``CamerasIdentityAdapter`` is the default in
      ``WaymoE2EDatasetProcessor._get_default_adapters`` (pano stays opt-in).
    * ``BatchedCameraData`` keeps a ``dict[CameraDirection, Tensor]`` shape
      (per-direction, NOT stacked across cameras) so missing cams remain a
      missing key and PR 3's stereo / 9-cam case stays compatible.
    * ``collate_cameras_fn`` produces ``BatchedCameraData`` from a batch of
      ``dict[CameraDirection, CameraData]`` payloads.
    * ``configs/surround.yaml`` parses to the surround adapter set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from standard_e2e.caching.adapters import get_adapters_from_config
from standard_e2e.data_structures import CameraData
from standard_e2e.data_structures.containers import BatchedCameraData
from standard_e2e.data_structures.frame_data import collate_modalities
from standard_e2e.enums import CameraDirection

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_camera(
    direction: CameraDirection, h: int = 16, w: int = 24, seed: int = 0
) -> CameraData:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    return CameraData(
        camera_direction=direction,
        image=image,
        intrinsics=np.eye(3, dtype=np.float32),
        extrinsics=np.eye(4, dtype=np.float32),
    )


def _make_8_cam_dict(seed: int) -> dict[CameraDirection, CameraData]:
    return {
        d: _make_camera(d, seed=seed * 100 + i) for i, d in enumerate(CameraDirection)
    }


def test_waymo_e2e_default_adapter_is_cameras_identity():
    """Pano was the default; surround camera identity replaces it."""
    from standard_e2e.caching.src_datasets.waymo_e2e.waymo_e2e_dataset_processor import (  # noqa: E501
        WaymoE2EDatasetProcessor,
    )

    proc = WaymoE2EDatasetProcessor("/tmp/_unused", split="training")
    names = {type(a).__name__ for a in getattr(proc, "_adapters")}
    assert "CamerasIdentityAdapter" in names
    assert "PanoImageAdapter" not in names


def test_surround_yaml_config_parses_to_cameras_identity():
    config_path = REPO_ROOT / "configs" / "surround.yaml"
    assert config_path.exists(), f"missing {config_path}"
    cfg = yaml.safe_load(config_path.read_text())
    adapter_cfgs = cfg["preprocessing"]["adapters"]
    adapters = get_adapters_from_config(adapter_cfgs)
    types = {type(a).__name__ for a in adapters}
    assert "CamerasIdentityAdapter" in types
    assert "PanoImageAdapter" not in types


def test_batched_camera_data_holds_per_direction_dict():
    batch = [_make_8_cam_dict(seed=i) for i in range(2)]
    out = collate_modalities(batch)
    assert isinstance(out, BatchedCameraData)
    assert out.batch_size == 2
    # Per-direction dict, NOT a single stacked [B, N, H, W, C] tensor.
    for direction in CameraDirection:
        images = out.images[direction]
        assert isinstance(images, torch.Tensor)
        # [B, H, W, C]
        assert images.shape == (2, 16, 24, 3)
        intrinsics = out.intrinsics[direction]
        extrinsics = out.extrinsics[direction]
        assert intrinsics.shape == (2, 3, 3)
        assert extrinsics.shape == (2, 4, 4)


def test_batched_camera_data_directions_property():
    batch = [_make_8_cam_dict(seed=i) for i in range(2)]
    out = collate_modalities(batch)
    assert set(out.directions) == set(CameraDirection)


def test_batched_camera_data_handles_partial_directions():
    """Direction missing in one frame -> drop that direction (missing key)."""
    full = _make_8_cam_dict(seed=0)
    partial = {
        CameraDirection.FRONT: _make_camera(CameraDirection.FRONT, seed=99),
        CameraDirection.REAR: _make_camera(CameraDirection.REAR, seed=98),
    }
    out = collate_modalities([full, partial])
    assert isinstance(out, BatchedCameraData)
    # Only the directions present in EVERY frame are stacked.
    assert set(out.directions) == {CameraDirection.FRONT, CameraDirection.REAR}
    assert out.images[CameraDirection.FRONT].shape == (2, 16, 24, 3)


def test_collate_cameras_in_mixed_batch_doesnt_break_other_modalities():
    cams_a = _make_8_cam_dict(seed=0)
    cams_b = _make_8_cam_dict(seed=1)
    batch = [
        {"cameras": cams_a, "speed": 5.0},
        {"cameras": cams_b, "speed": 7.0},
    ]
    out = collate_modalities(batch)
    assert isinstance(out["cameras"], BatchedCameraData)
    assert out["cameras"].batch_size == 2
    assert isinstance(out["speed"], torch.Tensor)
    assert out["speed"].tolist() == [5.0, 7.0]


def test_batched_camera_data_image_dtype_and_shape():
    """Stacked images must keep uint8 dtype and HWC ordering."""
    batch = [_make_8_cam_dict(seed=i) for i in range(3)]
    out = collate_modalities(batch)
    front = out.images[CameraDirection.FRONT]
    assert front.dtype == torch.uint8
    assert front.shape[0] == 3
    assert front.shape[-1] == 3  # channels last


def test_cameras_npz_roundtrip_byte_equal(tmp_path):
    """``dict[CameraDirection, CameraData]`` survives ``to_npz`` -> ``from_npz``."""
    from standard_e2e.data_structures import TransformedFrameData
    from standard_e2e.enums import Modality

    cams = _make_8_cam_dict(seed=3)
    frame = TransformedFrameData(
        dataset_name="ds",
        segment_id="seg0",
        frame_id=0,
        timestamp=0.0,
        split="train",
    )
    frame.set_modality_data(Modality.CAMERAS, cams)

    out = tmp_path / "frame_cams.npz"
    frame.to_npz(str(out))

    loaded = TransformedFrameData.from_npz(str(out))
    loaded_cams = loaded.get_modality_data(Modality.CAMERAS)
    assert isinstance(loaded_cams, dict)
    assert set(loaded_cams.keys()) == set(CameraDirection)
    for direction, original in cams.items():
        restored = loaded_cams[direction]
        assert isinstance(restored, CameraData)
        assert restored.camera_direction == direction
        np.testing.assert_array_equal(restored.image, original.image)
        np.testing.assert_array_equal(restored.intrinsics, original.intrinsics)
        np.testing.assert_array_equal(restored.extrinsics, original.extrinsics)
        assert restored.image.dtype == np.uint8
