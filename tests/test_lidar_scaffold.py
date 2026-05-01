"""PR 1 step 1.2: lidar plumbing scaffold tests.

Covers:
    * ``LidarPCIdentityAdapter`` registered + maps ``StandardFrameData.lidar``
      to ``Modality.LIDAR_PC``.
    * ``collate_lidar_fn`` returns ``list[LidarData]`` (passthrough, no padding,
      per ADR 0008).
    * ``TransformedFrameData.to_npz`` / ``from_npz`` round-trips a
      ``LidarData`` payload byte-equal at the DataFrame level (Variant A).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from standard_e2e.caching.adapters import (
    LidarPCIdentityAdapter,
    get_adapters_from_config,
)
from standard_e2e.data_structures import (
    LidarData,
    StandardFrameData,
    TransformedFrameData,
)
from standard_e2e.data_structures.frame_data import collate_modalities
from standard_e2e.enums import Modality


def _make_lidar(seed: int = 0, n: int = 128) -> LidarData:
    rng = np.random.default_rng(seed)
    points = rng.normal(scale=8.0, size=(n, 3)).astype(np.float32)
    df = pd.DataFrame(
        {
            "x": points[:, 0],
            "y": points[:, 1],
            "z": points[:, 2],
            "intensity": rng.uniform(0.0, 1.0, size=n).astype(np.float32),
        }
    )
    return LidarData(points=df)


def test_lidar_pc_identity_adapter_maps_lidar_attr():
    lidar = _make_lidar()
    frame = StandardFrameData(
        timestamp=0.0,
        frame_id=0,
        segment_id="seg0",
        dataset_name="ds",
        split="train",
        lidar=lidar,
    )
    adapter = LidarPCIdentityAdapter()
    out = adapter.transform(frame)
    assert set(out.keys()) == {Modality.LIDAR_PC}
    # Identity behavior: returns the same LidarData (Pydantic model, not
    # values/ndarray) so downstream sees a typed object.
    assert isinstance(out[Modality.LIDAR_PC], LidarData)
    assert out[Modality.LIDAR_PC] is lidar


def test_lidar_pc_identity_adapter_missing_returns_empty():
    frame = StandardFrameData(
        timestamp=0.0,
        frame_id=0,
        segment_id="seg0",
        dataset_name="ds",
        split="train",
    )  # no lidar
    adapter = LidarPCIdentityAdapter()
    assert adapter.transform(frame) == {}


def test_lidar_pc_adapter_registered_via_config():
    adapters = get_adapters_from_config([{"name": "lidar_pc_identity_adapter"}])
    assert len(adapters) == 1
    assert isinstance(adapters[0], LidarPCIdentityAdapter)


def test_collate_lidar_returns_list_of_lidar_data():
    batch = [_make_lidar(seed=0), _make_lidar(seed=1), _make_lidar(seed=2)]
    out = collate_modalities(batch)
    assert isinstance(out, list)
    assert len(out) == 3
    assert all(isinstance(item, LidarData) for item in out)
    # Lossless: row counts preserved per frame.
    for src, dst in zip(batch, out):
        assert len(src.points) == len(dst.points)
        # Same DataFrame contents passthrough.
        pd.testing.assert_frame_equal(src.points, dst.points)


def test_collate_lidar_variable_length_per_frame_supported():
    """ADR 0008: per-frame variable-length is allowed; no premature padding."""
    batch = [_make_lidar(seed=0, n=10), _make_lidar(seed=1, n=200)]
    out = collate_modalities(batch)
    assert isinstance(out, list)
    assert len(out[0].points) == 10
    assert len(out[1].points) == 200


def test_lidar_npz_roundtrip_byte_equal(tmp_path: Path):
    lidar = _make_lidar(seed=42)
    frame = TransformedFrameData(
        dataset_name="ds",
        segment_id="seg0",
        frame_id=0,
        timestamp=0.0,
        split="train",
    )
    frame.set_modality_data(Modality.LIDAR_PC, lidar)

    out = tmp_path / "frame_lidar.npz"
    frame.to_npz(str(out))

    loaded = TransformedFrameData.from_npz(str(out))
    loaded_lidar = loaded.get_modality_data(Modality.LIDAR_PC)
    assert isinstance(loaded_lidar, LidarData)
    pd.testing.assert_frame_equal(loaded_lidar.points, lidar.points)


def test_lidar_npz_roundtrip_through_required_modalities(tmp_path: Path):
    lidar = _make_lidar(seed=7)
    frame = TransformedFrameData(
        dataset_name="ds",
        segment_id="seg0",
        frame_id=0,
        timestamp=0.0,
        split="train",
    )
    frame.set_modality_data(Modality.LIDAR_PC, lidar)
    frame.set_modality_data(Modality.INTENT, 1)

    out = tmp_path / "frame_lidar_required.npz"
    frame.to_npz(str(out))

    loaded = TransformedFrameData.from_npz(
        str(out), required_modalities=[Modality.LIDAR_PC]
    )
    assert set(loaded.get_present_modality_keys()) == {Modality.LIDAR_PC}
    pd.testing.assert_frame_equal(
        loaded.get_modality_data(Modality.LIDAR_PC).points, lidar.points
    )


def test_collate_lidar_in_mixed_batch_doesnt_break_other_modalities():
    """Mixed batch (lidar + scalar). Lidar collates as list, scalars as tensor."""
    import torch

    batch = [
        {"lidar": _make_lidar(seed=0, n=4), "speed": 10.0},
        {"lidar": _make_lidar(seed=1, n=4), "speed": 20.0},
    ]
    out = collate_modalities(batch)
    assert isinstance(out["lidar"], list)
    assert all(isinstance(x, LidarData) for x in out["lidar"])
    assert isinstance(out["speed"], torch.Tensor)
    assert out["speed"].tolist() == [10.0, 20.0]


def test_lidar_pc_identity_adapter_subclass():
    from standard_e2e.caching.adapters.abstract_adapter import AbstractAdapter

    assert issubclass(LidarPCIdentityAdapter, AbstractAdapter)
