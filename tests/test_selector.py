import pandas as pd
import pytest

from standard_e2e.dataset_utils.selector import (
    ClosestTimestampSelector,
    CurrentSelector,
    FrameSelector,
    create_frame_selector_from_config,
)


def _make_index_df():
    # Sorted by segment_id then timestamp
    return pd.DataFrame(
        {
            "segment_id": ["A", "A", "A", "B", "B"],
            "timestamp": [0.0, 1.0, 3.0, 10.0, 10.5],
        }
    )


# ---------------------- Base FrameSelector tests ----------------------


def test_frame_selector_invalid_location_raises():
    class Dummy(FrameSelector):  # minimal concrete implementation
        @property
        def name(self):
            return "dummy"

        def select_frame(self, current_frame_iloc: int, index_data=None) -> int:  # noqa
            return current_frame_iloc

    with pytest.raises(ValueError):
        Dummy(location="other")  # invalid location


# ---------------------- CurrentSelector tests ----------------------


def test_current_selector_returns_same_index():
    sel = CurrentSelector(location="labels")
    assert sel.select_frame(5) == 5


# ------------------ ClosestTimestampSelector param validation ------------------


def test_closest_timestamp_selector_labels_requires_non_negative():
    """
    Prohibit negative delta_t (past) for labels.
    """
    with pytest.raises(ValueError):
        ClosestTimestampSelector(
            location="labels", delta_t=-0.1, index_data=_make_index_df()
        )


def test_closest_timestamp_selector_features_requires_non_positive():
    """
    Prohibit positive delta_t (future) for features.
    """
    with pytest.raises(ValueError):
        ClosestTimestampSelector(
            location="features", delta_t=0.1, index_data=_make_index_df()
        )


def test_closest_timestamp_selector_zero_delta_allowed_both_locations():
    df = _make_index_df()
    sel_labels = ClosestTimestampSelector(location="labels", delta_t=0.0, index_data=df)
    sel_features = ClosestTimestampSelector(
        location="features", delta_t=0.0, index_data=df
    )
    assert sel_labels.select_frame(0) == 0
    assert sel_features.select_frame(1) == 1


# ---------------------- ClosestTimestampSelector mapping logic ----------------------


def test_closest_timestamp_selector_positive_delta_mapping():
    df = _make_index_df()
    sel = ClosestTimestampSelector(location="labels", delta_t=0.4, index_data=df)
    # Expected mapping (see analysis): [0,1,2,4,4]
    expected = [0, 1, 2, 4, 4]
    got = [sel.select_frame(i) for i in range(len(df))]
    assert got == expected


def test_closest_timestamp_selector_negative_delta_mapping():
    df = _make_index_df()
    sel = ClosestTimestampSelector(location="features", delta_t=-0.6, index_data=df)
    # Expected mapping (see analysis): [0,0,2,3,3]
    expected = [0, 0, 2, 3, 3]
    got = [sel.select_frame(i) for i in range(len(df))]
    assert got == expected


def test_closest_timestamp_selector_tie_prefers_left():
    df = pd.DataFrame({"segment_id": ["S", "S"], "timestamp": [0.0, 2.0]})
    # delta=1.0 causes target at 1.0 for first row -> tie between 0.0 and 2.0
    sel = ClosestTimestampSelector(location="labels", delta_t=1.0, index_data=df)
    assert sel.select_frame(0) == 0  # prefers left on tie


def test_closest_timestamp_selector_set_index_data_updates_mapping():
    df1 = pd.DataFrame({"segment_id": ["S", "S"], "timestamp": [0.0, 1.0]})
    sel = ClosestTimestampSelector(location="labels", delta_t=0.4, index_data=df1)
    first = sel.select_frame(0)
    df2 = pd.DataFrame({"segment_id": ["S", "S", "S"], "timestamp": [0.0, 1.0, 1.2]})
    sel.set_index_data(df2)
    second = sel.select_frame(0)
    # With new data, the closest to 0.0+0.4 is still 0.0 (index 0)
    assert first == 0 and second == 0


# ---------------------- Factory function tests ----------------------


def test_create_frame_selector_from_config_closest():
    df = _make_index_df()
    cfg = {"name": "closest_timestamp", "parameters": {"delta_t": 0.0}}
    sel = create_frame_selector_from_config(cfg, location="labels", index_data=df)
    assert isinstance(sel, ClosestTimestampSelector)
    assert sel.name == "closest_timestamp_selector"


def test_create_frame_selector_from_config_current():
    cfg = {"name": "current", "parameters": {}}
    sel = create_frame_selector_from_config(cfg, location="features", index_data=None)
    assert isinstance(sel, CurrentSelector)
    assert sel.name == "current_selector"


def test_create_frame_selector_from_config_unknown_raises():
    cfg = {"name": "unknown_kind", "parameters": {}}
    with pytest.raises(ValueError):
        create_frame_selector_from_config(cfg, location="labels", index_data=None)
