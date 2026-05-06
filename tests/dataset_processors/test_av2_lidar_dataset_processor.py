"""Tests for ``Av2LidarDatasetProcessor``.

Two layers, mirroring the AV2 sensor tests:

* **Construction defaults** — adapters / context aggregators wired correctly
  (lidar + HD map only, no cameras, no detections).
* **Override sanity** — the camera and detection helpers short-circuit to
  empty containers regardless of input, and the inherited HD-map builder
  still produces the unified element types when run on a real AV2 lidar log
  (skipped automatically when the dataset isn't mounted).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from av2.map.map_api import ArgoverseStaticMap
from scipy.spatial.transform import Rotation

from standard_e2e.caching.adapters import HDMapBEVAdapter, LidarAdapter
from standard_e2e.caching.segment_context import (
    FuturePastStatesFromMatricesAggregator,
)
from standard_e2e.caching.src_datasets.av2_lidar.av2_lidar_dataset_processor import (
    Av2LidarDatasetProcessor,
)
from standard_e2e.enums import MapElementType

AV2_LIDAR_ROOT = Path("/mnt/bigdisk/datasets/argoverse2/lidar/train")
SAMPLE_LOG = "000VFSWWAAkobywItdrErpC6fedKDWg4"


# --- construction defaults ---------------------------------------------------


def test_av2_lidar_defaults(tmp_path: Path):
    proc = Av2LidarDatasetProcessor(str(tmp_path), split="train")
    assert proc.dataset_name == "av2_lidar"
    assert set(("train", "val", "test")).issubset(set(proc.allowed_splits))
    adapters = getattr(proc, "_adapters")
    assert len(adapters) == 2
    assert isinstance(adapters[0], LidarAdapter)
    assert isinstance(adapters[1], HDMapBEVAdapter)


def test_av2_lidar_default_aggregators_exclude_future_detections(tmp_path: Path):
    proc = Av2LidarDatasetProcessor(str(tmp_path), split="train")
    aggregators = proc.context_aggregators
    assert len(aggregators) == 1
    assert isinstance(aggregators[0], FuturePastStatesFromMatricesAggregator)


# --- override behaviours -----------------------------------------------------


def test_camera_dict_is_always_empty(tmp_path: Path):
    proc = Av2LidarDatasetProcessor(str(tmp_path), split="train")
    assert proc._build_camera_dict(Path("/dev/null"), 0) == {}


def test_detections_are_always_empty(tmp_path: Path):
    proc = Av2LidarDatasetProcessor(str(tmp_path), split="train")
    assert proc._build_detections(0, 0.0) == []


# --- inherited HD-map builder still works ------------------------------------


def _ego_pose_at_first_frame(log_path: Path) -> np.ndarray:
    df = pd.read_feather(log_path / "city_SE3_egovehicle.feather")
    row = df.iloc[0]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Rotation.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix()
    T[:3, 3] = (row.tx_m, row.ty_m, row.tz_m)
    return T


@pytest.fixture
def lidar_log_processor() -> Av2LidarDatasetProcessor:
    if not (AV2_LIDAR_ROOT / SAMPLE_LOG / "map").exists():
        pytest.skip("AV2 lidar sample log not available locally")
    proc = Av2LidarDatasetProcessor.__new__(Av2LidarDatasetProcessor)
    proc._map = ArgoverseStaticMap.from_map_dir(
        AV2_LIDAR_ROOT / SAMPLE_LOG / "map", build_raster=False
    )
    return proc


@pytest.fixture
def lidar_log_pose() -> np.ndarray:
    if not (AV2_LIDAR_ROOT / SAMPLE_LOG).exists():
        pytest.skip("AV2 lidar sample log not available locally")
    return _ego_pose_at_first_frame(AV2_LIDAR_ROOT / SAMPLE_LOG)


def test_hd_map_builder_emits_unified_elements_on_real_lidar_log(
    lidar_log_processor, lidar_log_pose
):
    """The HD-map taxonomy reuses Av2SensorDatasetProcessor's logic verbatim,
    so the lidar split must produce the same map element types we expect:
    LANE_CENTER per lane segment, LANE_BOUNDARY for painted edges, plus
    DRIVABLE_AREA and CROSSWALK polygons."""
    hd_map = lidar_log_processor._build_hd_map(lidar_log_pose)
    assert hd_map is not None
    by_type = {t: 0 for t in MapElementType}
    for elem in hd_map.elements:
        by_type[elem.type] += 1
    # Every AV2 log has at least lane segments and drivable area.
    assert by_type[MapElementType.LANE_CENTER] > 0
    assert by_type[MapElementType.DRIVABLE_AREA] > 0
    # Detections-side types must stay zero — no source primitives map to them.
    assert by_type[MapElementType.STOP_SIGN] == 0
    assert by_type[MapElementType.WALKWAY] == 0
    assert by_type[MapElementType.TRAFFIC_LIGHT] == 0
