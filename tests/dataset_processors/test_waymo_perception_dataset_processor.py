"""Tests for WaymoPerceptionDatasetProcessor (construction + defaults).

We don't parse real protobuf frames here to keep the test lightweight.
"""

from __future__ import annotations

from pathlib import Path

from standard_e2e.caching.adapters import (
    Detections3DIdentityAdapter,
    HDMapBEVAdapter,
    LidarAdapter,
    PanoImageAdapter,
)
from standard_e2e.caching.src_datasets.waymo_perception import (
    waymo_perception_dataset_processor as _wpdp,
)


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
