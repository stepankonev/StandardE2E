"""Tests for WaymoPerceptionDatasetProcessor (construction, defaults, HD map).

We don't parse real protobuf frames here to keep the test lightweight; the
HD-map test builds small synthetic ones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# TensorFlow is installed in the Waymo environments only (datasets/waymo_*/).
tf = pytest.importorskip("tensorflow")

from standard_e2e.caching.adapters import (
    Detections3DIdentityAdapter,
    HDMapBEVAdapter,
    LidarAdapter,
    PanoImageAdapter,
)
from standard_e2e.caching.src_datasets.waymo_perception import (
    waymo_perception_dataset_processor as _wpdp,
)

# pylint: disable=no-name-in-module
from standard_e2e.third_party.waymo_open_dataset.dataset_pb2 import Frame as WaymoFrame


def test_waymo_perception_defaults(tmp_path: Path):
    proc = _wpdp.WaymoPerceptionDatasetProcessor(str(tmp_path), split="training")
    assert proc.dataset_name == "waymo_perception"
    assert set(["training", "validation", "testing"]).issubset(set(proc.allowed_splits))
    adapters = getattr(proc, "_adapters")  # Access internal for test
    assert len(adapters) == 4
    assert isinstance(adapters[0], PanoImageAdapter)
    assert isinstance(adapters[1], LidarAdapter)
    assert isinstance(adapters[2], HDMapBEVAdapter)
    assert isinstance(adapters[3], Detections3DIdentityAdapter)


def _yaw_pose(yaw: float, translation: list[float]) -> np.ndarray:
    pose = np.eye(4)
    pose[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    pose[:3, 3] = translation
    return pose


def _frame(timestamp_micros: int, pose: np.ndarray) -> WaymoFrame:
    frame = WaymoFrame()
    frame.context.name = "segment"
    frame.timestamp_micros = timestamp_micros
    frame.pose.transform.extend(pose.flatten().tolist())
    return frame


def test_hd_map_applies_map_pose_offset(tmp_path: Path):
    """Each frame's ``map_pose_offset`` aligns it with the segment's map.

    Waymo's dataset.proto: the offset "must be added to lidar points from this
    frame to compensate for pose drift and align with the map features"; the
    SDK's tutorial_maps.ipynb adds it after the pose has moved the points into
    the world frame. A map feature at ``pose @ p + offset`` must therefore come
    out at the vehicle-frame point ``p``.
    """
    proc = _wpdp.WaymoPerceptionDatasetProcessor(
        str(tmp_path), split="training", adapters=[HDMapBEVAdapter()]
    )
    # Yawed, so an offset added in the vehicle frame would land elsewhere.
    pose = _yaw_pose(np.pi / 3, [130.0, 215.0, 11.0])
    offset = np.array([0.3, -0.2, 0.8])
    curb_vehicle = np.array([[5.0, 2.0, -1.5], [15.0, 2.5, -1.4]])
    curb_map = curb_vehicle @ pose[:3, :3].T + pose[:3, 3] + offset

    # Only a segment's first frame carries the map; later frames reuse it.
    first = _frame(1_000_000, _yaw_pose(0.0, [100.0, 200.0, 10.0]))
    road_edge = first.map_features.add()
    road_edge.id = 7
    for x, y, z in curb_map:
        point = road_edge.road_edge.polyline.add()
        point.x, point.y, point.z = x, y, z
    later = _frame(2_000_000, pose)
    later.map_pose_offset.x, later.map_pose_offset.y, later.map_pose_offset.z = offset

    proc._prepare_standardized_frame_data(tf.constant(first.SerializeToString()))
    frame_data = proc._prepare_standardized_frame_data(
        tf.constant(later.SerializeToString())
    )

    assert frame_data.hd_map is not None
    (element,) = frame_data.hd_map.elements
    np.testing.assert_allclose(element.points, curb_vehicle, atol=1e-4)
