import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from standard_e2e.caching.segment_context.future_past_states_from_matrices import (
    FuturePastStatesFromMatricesAggregator,
)
from standard_e2e.data_structures import Trajectory, TransformedFrameData
from standard_e2e.enums import Modality
from standard_e2e.enums import TrajectoryComponent as TC


def _pose_tx(x: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[0, 3] = x
    return T


def _make_frame(seg: str, frame_id: int, ts: float, x: float) -> TransformedFrameData:
    return TransformedFrameData(
        dataset_name="ds",
        split="train",
        segment_id=seg,
        frame_id=frame_id,
        timestamp=ts,
        aux_data={"pose_matrix": _pose_tx(x)},
    )


def _write_frames(
    tmp_path: Path, segment_id: str, timestamps: list[float], xs: list[float]
):
    frames = []
    for i, (ts, x) in enumerate(zip(timestamps, xs)):
        f = _make_frame(segment_id, i, ts, x)
        out_path = tmp_path / f.filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        f.to_npz(str(out_path))
        frames.append(f)
    return frames


def _index_from_frames(frames: list[TransformedFrameData]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": [f.segment_id for f in frames],
            "timestamp": [f.timestamp for f in frames],
            "filename": [f.filename for f in frames],
        }
    )


def test_constructor_param_exclusivity():
    with pytest.raises(ValueError):
        FuturePastStatesFromMatricesAggregator(
            "unused", max_history_length=2, max_history_delta_t=1
        )
    with pytest.raises(ValueError):
        FuturePastStatesFromMatricesAggregator(
            "unused", max_future_length=2, max_future_delta_t=1
        )
    # Negative values
    with pytest.raises(ValueError):
        FuturePastStatesFromMatricesAggregator("unused", max_history_length=-1)


def test_fetch_value_shape_validation(tmp_path: Path):
    # Write a frame with wrong pose shape
    bad = TransformedFrameData(
        dataset_name="ds",
        split="train",
        segment_id="segX",
        frame_id=0,
        timestamp=0.0,
        aux_data={"pose_matrix": np.zeros((3, 3), dtype=np.float32)},
    )
    out = tmp_path / bad.filename
    out.parent.mkdir(parents=True, exist_ok=True)
    bad.to_npz(str(out))
    idx = _index_from_frames([bad])
    aggr = FuturePastStatesFromMatricesAggregator(str(tmp_path))
    with pytest.raises(ValueError):
        aggr.process(idx)


def test_process_adds_past_future_states_with_limits(tmp_path: Path):
    # 4 frames moving along x; timestamps increasing by 1
    frames = _write_frames(tmp_path, "segM", [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0])
    index_df = _index_from_frames(frames)
    aggr = FuturePastStatesFromMatricesAggregator(
        str(tmp_path),
        max_history_length=2,  # only last 2 including current
        max_future_length=1,  # only immediate next future
    )
    aggr.process(index_df)

    for i, f in enumerate(frames):
        reloaded = TransformedFrameData.from_npz(os.path.join(tmp_path, f.filename))
        future_traj: Trajectory = reloaded.get_modality_data(Modality.FUTURE_STATES)
        past_traj: Trajectory = reloaded.get_modality_data(Modality.PAST_STATES)
        assert isinstance(future_traj, Trajectory)
        assert isinstance(past_traj, Trajectory)
        # Past length capped at 2 (or fewer for early frames)
        assert past_traj.length == min(2, i + 1)
        # Future length capped at 1 (or zero for last frame)
        expected_future_len = 0 if i == len(frames) - 1 else 1
        assert future_traj.length == expected_future_len
        # Delta t signs: past <=0, future >0
        if past_traj.length:
            past_ts = past_traj.get(TC.TIMESTAMP).reshape(-1)
            assert np.all(past_ts <= 0.0)
            # Last element (current frame) delta_t should be 0
            assert past_ts[-1] == 0.0
        if future_traj.length:
            future_ts = future_traj.get(TC.TIMESTAMP).reshape(-1)
            assert np.all(future_ts > 0.0)
            # Because of max_future_length=1 future timestamp should equal 1.0
            assert future_ts[0] == 1.0


def test_process_history_future_delta_t_limits(tmp_path: Path):
    # 5 frames at 0..4 seconds, x position 0..4
    frames = _write_frames(tmp_path, "segD", [0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    index_df = _index_from_frames(frames)
    # Keep only history within 2 sec and future within 1 sec.
    aggr = FuturePastStatesFromMatricesAggregator(
        str(tmp_path),
        max_history_delta_t=2,  # include timestamps with delta_t > -2
        max_future_delta_t=1,  # include timestamps with delta_t < 1
    )
    aggr.process(index_df)

    # Check frame with timestamp=2 (index 2)
    mid_frame = frames[2]
    reloaded = TransformedFrameData.from_npz(os.path.join(tmp_path, mid_frame.filename))
    past_traj: Trajectory = reloaded.get_modality_data(Modality.PAST_STATES)
    future_traj: Trajectory = reloaded.get_modality_data(Modality.FUTURE_STATES)
    # History delta_t values should be in {-2,-1,0}
    past_ts = past_traj.get(TC.TIMESTAMP).reshape(-1)
    assert set(past_ts.tolist()) == {-2.0, -1.0, 0.0}
    # Future delta_t values should be {1}
    future_ts = future_traj.get(TC.TIMESTAMP).reshape(-1)
    assert set(future_ts.tolist()) == {1.0}
