"""Tests for WaymoE2EDatasetProcessor (construction + defaults)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from standard_e2e.caching.adapters import (
    CamerasIdentityAdapter,
    FutureStatesIdentityAdapter,
    IntentIdentityAdapter,
    PastStatesIdentityAdapter,
    PreferenceTrajectoryAdapter,
)
from standard_e2e.caching.segment_context.future_past_states_from_matrices import (
    FuturePastStatesFromMatricesAggregator,
)
from standard_e2e.caching.src_datasets.waymo_e2e.waymo_e2e_dataset_processor import (
    WaymoE2EDatasetProcessor,
)
from standard_e2e.data_structures import TransformedFrameData


def test_waymo_e2e_defaults(tmp_path: Path):
    proc = WaymoE2EDatasetProcessor(str(tmp_path), split="training")
    assert proc.dataset_name == "waymo_e2e"
    assert set(["training", "val", "test"]).issubset(set(proc.allowed_splits))
    names = {type(a).__name__ for a in getattr(proc, "_adapters")}
    expected = {
        CamerasIdentityAdapter.__name__,
        IntentIdentityAdapter.__name__,
        PastStatesIdentityAdapter.__name__,
        FutureStatesIdentityAdapter.__name__,
        PreferenceTrajectoryAdapter.__name__,
    }
    assert expected.issubset(names)


def test_e2e_frames_cannot_be_paired_with_matrices_aggregator(tmp_path: Path):
    """The E2E processor doesn't populate ``aux_data['pose_matrix']``.
    Running the matrices-based aggregator on its output must raise a
    clear precondition error rather than an opaque ``KeyError``,
    matching the safety net ``HDMapEgoCropAggregator`` already provides.
    """
    frame = TransformedFrameData(
        dataset_name="waymo_e2e",
        split="training",
        segment_id="e2e-seg",
        frame_id=0,
        timestamp=0.0,
    )
    out = tmp_path / frame.filename
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_npz(str(out))
    index_df = pd.DataFrame(
        {
            "segment_id": [frame.segment_id],
            "timestamp": [frame.timestamp],
            "filename": [frame.filename],
        }
    )
    aggr = FuturePastStatesFromMatricesAggregator(str(tmp_path))
    with pytest.raises(ValueError, match="pose_matrix"):
        aggr.process(index_df)
