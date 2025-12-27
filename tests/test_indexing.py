import pandas as pd
import pytest

from standard_e2e.data_structures import TransformedFrameData
from standard_e2e.indexing import IndexDataGenerator
from standard_e2e.indexing.filters import (
    FrameFilterByBooleanColumn,
    FrameFilterByTimeDelta,
    IndexFilter,
)


def make_frame(**overrides) -> TransformedFrameData:
    kwargs = dict(
        dataset_name="ds",
        segment_id="segA",
        frame_id=1,
        timestamp=10.0,
        split="train",
    )
    kwargs.update(overrides)
    return TransformedFrameData(**kwargs)


# ---------------------- IndexDataGenerator tests ----------------------


def test_index_data_generator_basic():
    f = make_frame(frame_id=42, segment_id="S1", timestamp=123.4)
    f.extra_index_data = {"foo": 7}
    gen = IndexDataGenerator()
    idx = gen.generate_index_data(f)
    assert idx.dataset_name == f.dataset_name
    assert idx.segment_id == f.segment_id
    assert idx.frame_id == f.frame_id
    assert idx.timestamp == f.timestamp
    assert idx.split == f.split
    assert idx.filename == f.filename
    assert idx.extra_index_data == {"foo": 7}


# ---------------------- FrameFilterByBooleanColumn tests ----------------------


def test_frame_filter_by_boolean_basic_selection():
    df = pd.DataFrame(
        {
            "segment_id": ["S1", "S1", "S2"],
            "timestamp": [0.0, 1.0, 0.0],
            "keep": [True, False, True],
        }
    )
    flt = FrameFilterByBooleanColumn("keep")
    out = flt.filter(df.copy())
    mask_col = IndexFilter.FILTERING_COL_NAME
    assert mask_col in out.columns
    # True rows should remain True, False row becomes False
    assert out[mask_col].tolist() == [True, False, True]


def test_frame_filter_by_boolean_missing_allowed():
    df = pd.DataFrame({"segment_id": ["S1"], "timestamp": [0.0]})
    flt = FrameFilterByBooleanColumn("missing", allow_missing_column=True)
    out = flt.filter(df.copy())
    assert out[IndexFilter.FILTERING_COL_NAME].tolist() == [True]


def test_frame_filter_by_boolean_missing_not_allowed():
    df = pd.DataFrame({"segment_id": ["S1"], "timestamp": [0.0]})
    flt = FrameFilterByBooleanColumn("missing", allow_missing_column=False)
    with pytest.raises(ValueError):
        flt.filter(df.copy())


def test_frame_filter_by_boolean_non_boolean_column_type_error():
    df = pd.DataFrame(
        {"segment_id": ["S1", "S1"], "timestamp": [0.0, 1.0], "keep": [1, 0]}
    )
    flt = FrameFilterByBooleanColumn("keep")
    with pytest.raises(TypeError):
        flt.filter(df.copy())


def test_frame_filter_by_boolean_all_filtered_out_results_in_zero_size():
    df = pd.DataFrame(
        {
            "segment_id": ["S1", "S1"],
            "timestamp": [0.0, 1.0],
            "keep": [False, False],
        }
    )
    flt = FrameFilterByBooleanColumn("keep")
    out = flt.filter(df.copy())
    assert out[IndexFilter.FILTERING_COL_NAME].sum() == 0


# ---------------------- FrameFilterByTimeDelta tests ----------------------


def _make_time_df():
    # Two segments with different lengths
    return pd.DataFrame(
        {
            "segment_id": ["A", "A", "A", "B", "B"],
            "timestamp": [0.0, 1.0, 2.0, 5.0, 6.0],
        }
    )


def test_frame_filter_by_time_delta_positive_excludes_trailing():
    df = _make_time_df()
    flt = FrameFilterByTimeDelta(1.0)  # exclude last second in each segment
    out = flt.filter(df.copy())
    mask = out[IndexFilter.FILTERING_COL_NAME]
    # Segment A last ts = 2.0 -> keep timestamps <= 1.0
    # Segment B last ts = 6.0 -> keep timestamps <= 5.0
    assert mask.tolist() == [True, True, False, True, False]


def test_frame_filter_by_time_delta_negative_excludes_leading():
    df = _make_time_df()
    flt = FrameFilterByTimeDelta(-1.0)  # exclude first second in each segment
    out = flt.filter(df.copy())
    mask = out[IndexFilter.FILTERING_COL_NAME]
    # Segment A first ts = 0.0 -> keep >= 1.0 => [False, True, True]
    # Segment B first ts = 5.0 -> keep >= 6.0 => [False, True]
    assert mask.tolist() == [False, True, True, False, True]


def test_frame_filter_by_time_delta_zero_no_change():
    df = _make_time_df()
    flt = FrameFilterByTimeDelta(0.0)
    out = flt.filter(df.copy())
    assert out[IndexFilter.FILTERING_COL_NAME].all()  # all True


def test_index_filter_chain_and_column_and_logic():
    df = pd.DataFrame(
        {
            "segment_id": ["S1", "S1", "S1", "S2"],
            "timestamp": [0.0, 1.0, 2.0, 0.0],
            "keep": [True, False, True, True],
        }
    )
    bool_filter = FrameFilterByBooleanColumn("keep")
    time_filter = FrameFilterByTimeDelta(1.0)  # exclude last ts per segment
    out = time_filter.filter(bool_filter.filter(df.copy()))
    mask = out[IndexFilter.FILTERING_COL_NAME]
    # For S1 timestamps [0,1,2]; time filter keeps <=1 (since last=2, delta=1) -> [0,1]
    # Combined with keep -> [True & True, False & True] => [True, False]
    # For S2 only one ts=0, last=0 -> keep <= -1?
    # Condition is ts <= last - delta => 0 <= -1 -> False
    # So it gets filtered out regardless of keep True
    assert mask.tolist() == [True, False, False, False]


def test_index_filter_empty_dataframe_returns_empty_unchanged():
    df = pd.DataFrame(columns=["segment_id", "timestamp"])
    flt = FrameFilterByTimeDelta(1.0)
    out = flt.filter(df)
    assert out.empty
    # FILTERING_COL should not be added to truly empty df per implementation
    assert IndexFilter.FILTERING_COL_NAME not in out.columns
