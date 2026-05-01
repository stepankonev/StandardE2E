"""Tests for ``BatchedLidarData`` (pad-and-stack lidar collate).

Covers construction shape, ragged batches, optional column handling,
empty-frame edge case, dtype preservation, device transfer, and the
end-to-end path through ``collate_modalities``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from standard_e2e.data_structures import BatchedLidarData, LidarData
from standard_e2e.data_structures.frame_data import collate_modalities


def _make_lidar(
    n: int = 32,
    *,
    seed: int = 0,
    intensity: bool = False,
    timestamp_ns: bool = False,
    laser_id: bool = False,
) -> LidarData:
    rng = np.random.default_rng(seed)
    cols: dict[str, np.ndarray] = {
        "x": rng.normal(scale=8.0, size=n).astype(np.float32),
        "y": rng.normal(scale=8.0, size=n).astype(np.float32),
        "z": rng.normal(scale=4.0, size=n).astype(np.float32),
    }
    if intensity:
        cols["intensity"] = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    if timestamp_ns:
        cols["timestamp_ns"] = rng.integers(
            0, 1_000_000_000, size=n, dtype=np.int64
        )
    if laser_id:
        cols["laser_id"] = rng.integers(0, 5, size=n, dtype=np.int8)
    return LidarData(points=pd.DataFrame(cols))


def test_construct_homogeneous_n():
    batch = [_make_lidar(n=32, seed=i) for i in range(3)]
    bl = BatchedLidarData(batch)
    assert bl.batch_size == 3
    assert bl.points.shape == (3, 32, 3)
    assert bl.points.dtype == torch.float32
    assert bl.valid_mask.shape == (3, 32)
    assert bl.valid_mask.dtype == torch.bool
    assert bl.valid_mask.all().item() is True
    assert bl.num_points.tolist() == [32, 32, 32]


def test_construct_ragged_n():
    batch = [
        _make_lidar(n=10, seed=0),
        _make_lidar(n=50, seed=1),
        _make_lidar(n=23, seed=2),
    ]
    bl = BatchedLidarData(batch)
    assert bl.points.shape == (3, 50, 3)
    assert bl.num_points.tolist() == [10, 50, 23]
    # Per-frame mask sums match num_points.
    assert int(bl.valid_mask[0].sum().item()) == 10
    assert int(bl.valid_mask[1].sum().item()) == 50
    assert int(bl.valid_mask[2].sum().item()) == 23
    # Padded entries are zero.
    assert torch.all(bl.points[0, 10:] == 0.0).item() is True
    assert torch.all(bl.points[2, 23:] == 0.0).item() is True


def test_optional_columns_common_only_dropped_when_missing():
    # Frame A and B have intensity; frame C does not. Result: no extras stacked.
    batch = [
        _make_lidar(n=20, seed=0, intensity=True),
        _make_lidar(n=20, seed=1, intensity=True),
        _make_lidar(n=20, seed=2, intensity=False),
    ]
    bl = BatchedLidarData(batch)
    assert "intensity" not in bl.extra_columns


def test_optional_columns_common_to_all_are_stacked():
    batch = [
        _make_lidar(n=20, seed=0, intensity=True, timestamp_ns=True),
        _make_lidar(n=20, seed=1, intensity=True, timestamp_ns=True),
        _make_lidar(n=20, seed=2, intensity=True, timestamp_ns=True),
    ]
    bl = BatchedLidarData(batch)
    assert set(bl.extra_columns) == {"intensity", "timestamp_ns"}
    assert bl.extra_columns["intensity"].shape == (3, 20)
    assert bl.extra_columns["intensity"].dtype == torch.float32
    assert bl.extra_columns["timestamp_ns"].shape == (3, 20)
    assert bl.extra_columns["timestamp_ns"].dtype == torch.int64


def test_optional_columns_pad_with_zeros_in_short_frames():
    batch = [
        _make_lidar(n=5, seed=0, intensity=True),
        _make_lidar(n=15, seed=1, intensity=True),
    ]
    bl = BatchedLidarData(batch)
    assert bl.extra_columns["intensity"].shape == (2, 15)
    # Frame 0 pads positions 5..15 with zero.
    assert torch.all(bl.extra_columns["intensity"][0, 5:] == 0.0).item() is True
    # Frame 0 valid positions 0..5 are nonzero (probability 1 with this RNG).
    assert torch.any(bl.extra_columns["intensity"][0, :5] != 0.0).item() is True


def test_dtype_preservation():
    batch = [
        _make_lidar(
            n=4, seed=0, intensity=True, timestamp_ns=True, laser_id=True
        ),
        _make_lidar(
            n=4, seed=1, intensity=True, timestamp_ns=True, laser_id=True
        ),
    ]
    bl = BatchedLidarData(batch)
    assert bl.points.dtype == torch.float32
    assert bl.valid_mask.dtype == torch.bool
    assert bl.num_points.dtype == torch.int64
    assert bl.extra_columns["intensity"].dtype == torch.float32
    assert bl.extra_columns["timestamp_ns"].dtype == torch.int64
    assert bl.extra_columns["laser_id"].dtype == torch.int8


def test_to_device_cpu_idempotent():
    batch = [_make_lidar(n=8, seed=0), _make_lidar(n=12, seed=1)]
    bl = BatchedLidarData(batch)
    cpu = torch.device("cpu")
    out = bl.to(cpu)
    assert out is bl
    assert bl.points.device.type == "cpu"
    assert bl.valid_mask.device.type == "cpu"
    assert bl.num_points.device.type == "cpu"


def test_empty_frame_in_batch():
    empty_df = pd.DataFrame({"x": [], "y": [], "z": []}).astype(np.float32)
    batch = [
        LidarData(points=empty_df),
        _make_lidar(n=7, seed=1),
    ]
    bl = BatchedLidarData(batch)
    assert bl.points.shape == (2, 7, 3)
    assert bl.num_points.tolist() == [0, 7]
    assert int(bl.valid_mask[0].sum().item()) == 0
    assert int(bl.valid_mask[1].sum().item()) == 7


def test_collate_through_collate_modalities():
    """The collate entry point yields BatchedLidarData for a list of LidarData."""
    batch = [_make_lidar(n=10, seed=0), _make_lidar(n=20, seed=1)]
    out = collate_modalities(batch)
    assert isinstance(out, BatchedLidarData)
    assert out.batch_size == 2
    assert out.points.shape == (2, 20, 3)


def test_rejects_empty_batch():
    with pytest.raises(ValueError):
        BatchedLidarData([])


def test_rejects_non_lidar_items():
    with pytest.raises(TypeError):
        BatchedLidarData([_make_lidar(n=4, seed=0), "not a lidar"])  # type: ignore[list-item]


def test_xyz_values_match_source_at_valid_positions():
    """Stacking must preserve xyz values verbatim at unmasked positions."""
    f0 = _make_lidar(n=6, seed=0)
    f1 = _make_lidar(n=11, seed=1)
    bl = BatchedLidarData([f0, f1])
    expected_0 = torch.from_numpy(
        np.ascontiguousarray(f0.points[["x", "y", "z"]].to_numpy(np.float32))
    )
    expected_1 = torch.from_numpy(
        np.ascontiguousarray(f1.points[["x", "y", "z"]].to_numpy(np.float32))
    )
    assert torch.allclose(bl.points[0, :6], expected_0)
    assert torch.allclose(bl.points[1, :11], expected_1)
