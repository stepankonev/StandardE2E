"""Tests for ``FutureDetectionsAggregator``.

These pin three behaviors that came out of the PR1+PR2 review:

1. The aggregated trajectory carries the full 8-component shape produced
   by source dataset processors (TIMESTAMP, X, Y, Z, HEADING, LENGTH,
   WIDTH, HEIGHT) — not the {X, Y, HEADING}-only subset.
2. LENGTH/WIDTH/HEIGHT are pose-invariant and propagate from the seed.
3. ``detection_type`` drift across the future window raises rather than
   silently inheriting whichever class the last frame happened to bind.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from standard_e2e.caching.segment_context.future_detections import (
    FutureDetectionsAggregator,
)
from standard_e2e.data_structures import (
    Detection3D,
    FrameDetections3D,
    Trajectory,
    TransformedFrameData,
)
from standard_e2e.enums import DetectionType, Modality
from standard_e2e.enums import TrajectoryComponent as TC


def _detection(
    agent_id: str,
    detection_type: DetectionType,
    timestamp: float,
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    heading: float = 0.0,
    length: float = 4.0,
    width: float = 2.0,
    height: float = 1.5,
) -> Detection3D:
    return Detection3D(
        unique_agent_id=agent_id,
        detection_type=detection_type,
        trajectory=Trajectory(
            {
                TC.TIMESTAMP: [timestamp],
                TC.X: [x],
                TC.Y: [y],
                TC.Z: [z],
                TC.HEADING: [heading],
                TC.LENGTH: [length],
                TC.WIDTH: [width],
                TC.HEIGHT: [height],
            }
        ),
    )


def _frame(
    seg_id: str,
    frame_id: int,
    timestamp: float,
    detections: list[Detection3D],
    global_xyzh: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> TransformedFrameData:
    gx, gy, gz, gh = global_xyzh
    frame = TransformedFrameData(
        dataset_name="test_ds",
        segment_id=seg_id,
        frame_id=frame_id,
        timestamp=timestamp,
        split="train",
        global_position=Trajectory(
            {
                TC.TIMESTAMP: [timestamp],
                TC.X: [gx],
                TC.Y: [gy],
                TC.Z: [gz],
                TC.HEADING: [gh],
            }
        ),
    )
    frame.set_modality_data(
        Modality.DETECTIONS_3D, FrameDetections3D(detections=detections)
    )
    return frame


def _persist(tmp_path: Path, frames: list[TransformedFrameData]) -> pd.DataFrame:
    for f in frames:
        out = tmp_path / f.filename
        out.parent.mkdir(parents=True, exist_ok=True)
        f.to_npz(str(out))
    return pd.DataFrame(
        {
            "segment_id": [f.segment_id for f in frames],
            "timestamp": [f.timestamp for f in frames],
            "filename": [f.filename for f in frames],
        }
    )


def test_preserves_all_eight_components(tmp_path: Path):
    frames = [
        _frame(
            "segP",
            i,
            i * 0.1,
            [_detection("agent-0", DetectionType.VEHICLE, i * 0.1, x=10.0 + i)],
        )
        for i in range(3)
    ]
    index_df = _persist(tmp_path, frames)
    FutureDetectionsAggregator(str(tmp_path)).process(index_df)

    reloaded = TransformedFrameData.from_npz(os.path.join(tmp_path, frames[0].filename))
    aggregated = reloaded.get_modality_data(Modality.DETECTIONS_3D)
    assert isinstance(aggregated, list)
    assert len(aggregated) == 1
    full = aggregated[0].trajectory.get(
        [
            TC.TIMESTAMP,
            TC.X,
            TC.Y,
            TC.Z,
            TC.HEADING,
            TC.LENGTH,
            TC.WIDTH,
            TC.HEIGHT,
        ]
    )
    # Two future frames; eight components per row.
    assert full.shape == (2, 8)


def test_pose_invariant_components_match_seed(tmp_path: Path):
    frames = [
        _frame(
            "segS",
            i,
            i * 0.1,
            [
                _detection(
                    "agent-0",
                    DetectionType.VEHICLE,
                    i * 0.1,
                    x=float(i),
                    length=5.5,
                    width=2.2,
                    height=1.7,
                )
            ],
        )
        for i in range(3)
    ]
    index_df = _persist(tmp_path, frames)
    FutureDetectionsAggregator(str(tmp_path)).process(index_df)

    agent = TransformedFrameData.from_npz(
        os.path.join(tmp_path, frames[0].filename)
    ).get_modality_data(Modality.DETECTIONS_3D)[0]

    assert np.all(agent.trajectory.get(TC.LENGTH).reshape(-1) == 5.5)
    assert np.all(agent.trajectory.get(TC.WIDTH).reshape(-1) == 2.2)
    assert np.all(agent.trajectory.get(TC.HEIGHT).reshape(-1) == 1.7)


def test_timestamp_is_relative_delta_t(tmp_path: Path):
    frames = [
        _frame(
            "segT",
            0,
            100.0,
            [_detection("agent-0", DetectionType.VEHICLE, 100.0, x=1.0)],
        ),
        _frame(
            "segT",
            1,
            100.5,
            [_detection("agent-0", DetectionType.VEHICLE, 100.5, x=2.0)],
        ),
        _frame(
            "segT",
            2,
            101.0,
            [_detection("agent-0", DetectionType.VEHICLE, 101.0, x=3.0)],
        ),
    ]
    index_df = _persist(tmp_path, frames)
    FutureDetectionsAggregator(str(tmp_path)).process(index_df)

    agent = TransformedFrameData.from_npz(
        os.path.join(tmp_path, frames[0].filename)
    ).get_modality_data(Modality.DETECTIONS_3D)[0]
    deltas = agent.trajectory.get(TC.TIMESTAMP).reshape(-1)
    assert np.allclose(deltas, [0.5, 1.0])


def test_detection_type_drift_raises(tmp_path: Path):
    frames = [
        _frame(
            "segD", 0, 0.0, [_detection("agent-0", DetectionType.VEHICLE, 0.0, x=1.0)]
        ),
        _frame(
            "segD", 1, 0.1, [_detection("agent-0", DetectionType.VEHICLE, 0.1, x=2.0)]
        ),
        _frame(
            "segD", 2, 0.2, [_detection("agent-0", DetectionType.UNKNOWN, 0.2, x=3.0)]
        ),
    ]
    index_df = _persist(tmp_path, frames)
    aggr = FutureDetectionsAggregator(str(tmp_path))
    with pytest.raises(AssertionError, match="detection_type drift"):
        aggr.process(index_df)


def test_rotation_math_unchanged_when_ego_stationary(tmp_path: Path):
    """Stationary ego (zero global_position diff, zero heading) means the
    aggregated XY equals the seed XY pre-rotation -> post-rotation
    identity. Pins the math against accidental regressions while
    preserving the new component shape.
    """
    frames = [
        _frame(
            "segR",
            i,
            i * 0.1,
            [
                _detection(
                    "agent-0",
                    DetectionType.VEHICLE,
                    i * 0.1,
                    x=10.0 + i,
                    y=5.0,
                    heading=0.0,
                )
            ],
        )
        for i in range(3)
    ]
    index_df = _persist(tmp_path, frames)
    FutureDetectionsAggregator(str(tmp_path)).process(index_df)

    agent = TransformedFrameData.from_npz(
        os.path.join(tmp_path, frames[0].filename)
    ).get_modality_data(Modality.DETECTIONS_3D)[0]
    xy = agent.trajectory.get([TC.X, TC.Y])
    # Future frames carried x=11, x=12 (stationary ego, zero heading).
    assert np.allclose(xy[:, 0], [11.0, 12.0])
    assert np.allclose(xy[:, 1], [5.0, 5.0])
