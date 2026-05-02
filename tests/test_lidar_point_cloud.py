import numpy as np
import pytest
import torch

from standard_e2e.data_structures.containers import (
    BatchedLidarPointCloud,
    LidarPointCloud,
)
from standard_e2e.enums import LidarComponent

XYZ = [LidarComponent.X, LidarComponent.Y, LidarComponent.Z]


def _pc(n: int) -> LidarPointCloud:
    return LidarPointCloud(np.arange(n * 3, dtype=np.float32).reshape(n, 3), XYZ)


# ---------------------- LidarPointCloud ----------------------


def test_lidar_point_cloud_happy_path():
    pc = _pc(4)
    assert pc.num_points == 4
    assert len(pc) == 4
    assert pc.components == XYZ
    assert pc.points.shape == (4, 3)
    assert pc.points.dtype == np.float32


def test_lidar_point_cloud_dtype_coercion():
    pc = LidarPointCloud(np.zeros((2, 3), dtype=np.float64), XYZ)
    assert pc.points.dtype == np.float32


def test_lidar_point_cloud_empty_is_valid():
    pc = LidarPointCloud(np.zeros((0, 3), dtype=np.float32), XYZ)
    assert pc.num_points == 0


def test_lidar_point_cloud_bad_ndim_raises():
    with pytest.raises(ValueError):
        LidarPointCloud(np.zeros((6,), dtype=np.float32), XYZ)


def test_lidar_point_cloud_columns_components_mismatch_raises():
    with pytest.raises(ValueError):
        LidarPointCloud(np.zeros((2, 4), dtype=np.float32), XYZ)


def test_lidar_point_cloud_duplicate_components_raises():
    with pytest.raises(ValueError):
        LidarPointCloud(
            np.zeros((2, 3), dtype=np.float32),
            [LidarComponent.X, LidarComponent.X, LidarComponent.Z],
        )


def test_lidar_point_cloud_non_array_raises():
    with pytest.raises(TypeError):
        LidarPointCloud([[0.0, 1.0, 2.0]], XYZ)  # type: ignore[arg-type]


def test_lidar_point_cloud_get_single_and_multi():
    pc = _pc(3)
    x = pc.get(LidarComponent.X)
    assert x.shape == (3, 1)
    assert np.allclose(x[:, 0], pc.points[:, 0])
    xy = pc.get([LidarComponent.X, LidarComponent.Y])
    assert xy.shape == (3, 2)
    assert np.allclose(xy, pc.points[:, :2])


def test_lidar_point_cloud_get_missing_component_raises():
    pc = LidarPointCloud(
        np.zeros((1, 2), dtype=np.float32), [LidarComponent.X, LidarComponent.Y]
    )
    with pytest.raises(KeyError):
        pc.get(LidarComponent.Z)


# ---------------------- BatchedLidarPointCloud ----------------------


def test_batched_lidar_point_cloud_concat_with_batch_idx():
    pcs = [_pc(2), _pc(3), _pc(0)]
    batched = BatchedLidarPointCloud(pcs)
    assert batched.batch_size == 3
    assert batched.points.shape == (5, 3)
    assert batched.batch_idx.tolist() == [0, 0, 1, 1, 1]
    assert batched.components == XYZ


def test_batched_lidar_point_cloud_all_empty_samples():
    pcs = [LidarPointCloud(np.zeros((0, 3), dtype=np.float32), XYZ) for _ in range(2)]
    batched = BatchedLidarPointCloud(pcs)
    assert batched.batch_size == 2
    assert batched.points.shape == (0, 3)
    assert batched.batch_idx.shape == (0,)


def test_batched_lidar_point_cloud_get_columns():
    pcs = [_pc(2), _pc(1)]
    batched = BatchedLidarPointCloud(pcs)
    xy = batched.get([LidarComponent.X, LidarComponent.Y])
    assert xy.shape == (3, 2)
    assert isinstance(xy, torch.Tensor)


def test_batched_lidar_point_cloud_empty_input_raises():
    with pytest.raises(ValueError):
        BatchedLidarPointCloud([])


def test_batched_lidar_point_cloud_components_mismatch_raises():
    pc_xyz = _pc(2)
    pc_xy = LidarPointCloud(
        np.zeros((1, 2), dtype=np.float32), [LidarComponent.X, LidarComponent.Y]
    )
    with pytest.raises(ValueError):
        BatchedLidarPointCloud([pc_xyz, pc_xy])


def test_batched_lidar_point_cloud_to_same_device_noop():
    pcs = [_pc(1)]
    batched = BatchedLidarPointCloud(pcs)
    same = batched.to(torch.device("cpu"))
    assert same is batched
