"""Tests for WaymoE2EDatasetProcessor (construction + defaults)."""

from __future__ import annotations

from pathlib import Path

from standard_e2e.caching.adapters import (
    FutureStatesIdentityAdapter,
    IntentIdentityAdapter,
    PanoImageAdapter,
    PastStatesIdentityAdapter,
    PreferenceTrajectoryAdapter,
)
from standard_e2e.caching.src_datasets.waymo_e2e.waymo_e2e_dataset_processor import (
    WaymoE2EDatasetProcessor,
)


def test_waymo_e2e_defaults(tmp_path: Path):
    proc = WaymoE2EDatasetProcessor(str(tmp_path), split="training")
    assert proc.dataset_name == "waymo_e2e"
    assert set(["training", "val", "test"]).issubset(set(proc.allowed_splits))
    names = {type(a).__name__ for a in getattr(proc, "_adapters")}
    expected = {
        PanoImageAdapter.__name__,
        IntentIdentityAdapter.__name__,
        PastStatesIdentityAdapter.__name__,
        FutureStatesIdentityAdapter.__name__,
        PreferenceTrajectoryAdapter.__name__,
    }
    assert expected.issubset(names)
