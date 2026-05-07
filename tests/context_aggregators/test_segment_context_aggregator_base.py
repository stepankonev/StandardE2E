import os
from pathlib import Path

import pandas as pd
import pytest

from standard_e2e.caching.segment_context.segment_context_aggregator import (
    SegmentContextAggregator,
)
from standard_e2e.data_structures import TransformedFrameData


def _make_frame(
    dataset_name: str,
    split: str,
    segment_id: str,
    frame_id: int,
    timestamp: float,
) -> TransformedFrameData:
    return TransformedFrameData(
        dataset_name=dataset_name,
        split=split,
        segment_id=segment_id,
        frame_id=frame_id,
        timestamp=timestamp,
        aux_data={"dummy": 1},
    )


class _TestAggregator(SegmentContextAggregator):
    """Concrete test subclass adding counts of history & future to aux_data."""

    def _fetch_value_from_transformed_frame(
        self, transformed_frame: TransformedFrameData
    ):  # noqa: D401
        # Return the timestamp as the value (simple scalar)
        return transformed_frame.timestamp

    def _update_frame_with_context(
        self, transformed_frame, history_context, future_context
    ):  # noqa: D401
        transformed_frame.aux_data = transformed_frame.aux_data or {}
        transformed_frame.aux_data["history_count"] = len(history_context)
        transformed_frame.aux_data["future_count"] = len(future_context)
        return transformed_frame


class _BadReturnAggregator(_TestAggregator):
    def _update_frame_with_context(
        self, transformed_frame, history_context, future_context
    ):  # noqa: D401
        return None  # type: ignore[return-value]


def _write_frames(tmp_path: Path, segment_id: str, timestamps: list[float]):
    frames = []
    for i, ts in enumerate(timestamps):
        f = _make_frame("ds", "train", segment_id, i, ts)
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


# pylint: disable=protected-access
def test_validate_segment_frame_errors(tmp_path):
    aggr = _TestAggregator(str(tmp_path))
    # Not a DataFrame
    with pytest.raises(TypeError):
        aggr._validate_segment_frame("not df")  # type: ignore[arg-type]
    # Missing columns
    with pytest.raises(ValueError):
        aggr._validate_segment_frame(
            pd.DataFrame({"segment_id": ["s1"], "filename": ["a.npz"]})
        )
    with pytest.raises(ValueError):
        aggr._validate_segment_frame(pd.DataFrame({"timestamp": [0.0]}))
    # Multiple segments
    with pytest.raises(ValueError):
        aggr._validate_segment_frame(
            pd.DataFrame({"segment_id": ["s1", "s2"], "timestamp": [0.0, 1.0]})
        )
    # Non-increasing timestamps
    with pytest.raises(ValueError):
        aggr._validate_segment_frame(
            pd.DataFrame({"segment_id": ["s1", "s1"], "timestamp": [1.0, 1.0]})
        )


def test_process_segment_updates_history_future_counts(tmp_path: Path):
    frames = _write_frames(tmp_path, "segA", [0.0, 0.5, 1.0])
    index_df = _index_from_frames(frames)
    aggr = _TestAggregator(str(tmp_path))
    aggr.process(index_df)

    # Reload and check counts
    for i, f in enumerate(frames):
        reloaded = TransformedFrameData.from_npz(os.path.join(tmp_path, f.filename))
        # history includes current => size i+1, future excludes current => remaining
        # pylint: disable=unsubscriptable-object
        assert reloaded.aux_data["history_count"] == i + 1
        assert reloaded.aux_data["future_count"] == len(frames) - (i + 1)


def test_process_raises_on_bad_return_type(tmp_path: Path):
    frames = _write_frames(tmp_path, "segB", [0.0, 1.0])
    aggr = _BadReturnAggregator(str(tmp_path))
    with pytest.raises(TypeError):
        aggr.process(_index_from_frames(frames))


def test_process_raises_on_timestamp_mismatch(tmp_path: Path):
    frames = _write_frames(tmp_path, "segC", [0.0, 1.0])
    bad_index = _index_from_frames(frames).copy()
    # Corrupt second timestamp in index so it no longer matches actual frame content
    bad_index.loc[1, "timestamp"] = 1.5
    aggr = _TestAggregator(str(tmp_path))
    with pytest.raises(ValueError):
        aggr.process(bad_index)


def test_process_parallel_matches_sequential(tmp_path: Path):
    """Per-segment fan-out across a process pool must match the sequential path
    npz-for-npz; segments are independent so order is irrelevant."""
    seg_specs = [
        ("segP1", [0.0, 0.5, 1.0]),
        ("segP2", [0.0, 1.0, 2.0]),
        ("segP3", [0.0, 0.4, 0.8, 1.2]),
    ]
    seq_root = tmp_path / "seq"
    par_root = tmp_path / "par"
    all_frames = []
    for sid, ts in seg_specs:
        all_frames.extend(_write_frames(seq_root, sid, ts))
        _write_frames(par_root, sid, ts)
    index_df = _index_from_frames(all_frames)

    _TestAggregator(str(seq_root)).process(index_df, do_parallel=False)
    # ``fork`` for the test: the helper ``_TestAggregator`` lives in this test
    # module which has no ``__init__.py`` package path, so a ``spawn`` worker
    # would not be able to re-import it for unpickling. Production code runs
    # ``spawn`` (the default) where the aggregator class lives in an installed
    # package — see the ``start_method`` docstring on ``process``.
    _TestAggregator(str(par_root)).process(
        index_df, num_workers=2, do_parallel=True, start_method="fork"
    )

    for f in all_frames:
        seq_reload = TransformedFrameData.from_npz(os.path.join(seq_root, f.filename))
        par_reload = TransformedFrameData.from_npz(os.path.join(par_root, f.filename))
        assert seq_reload.aux_data == par_reload.aux_data


def test_process_single_segment_skips_pool(tmp_path: Path):
    """Single-segment indices fall through to the sequential branch even when
    ``do_parallel`` is requested — guards against spawning a 1-worker pool."""
    frames = _write_frames(tmp_path, "segOnlyOne", [0.0, 1.0, 2.0])
    aggr = _TestAggregator(str(tmp_path))
    aggr.process(_index_from_frames(frames), num_workers=8, do_parallel=True)
    for i, f in enumerate(frames):
        reloaded = TransformedFrameData.from_npz(os.path.join(tmp_path, f.filename))
        aux = reloaded.aux_data
        assert aux is not None
        assert aux["history_count"] == i + 1
        assert aux["future_count"] == len(frames) - (i + 1)
