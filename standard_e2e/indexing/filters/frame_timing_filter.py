import logging
from typing import Literal

import pandas as pd

from standard_e2e.indexing.filters.index_filter import IndexFilter


class FrameFilterByTimeDelta(IndexFilter):
    """
    Filter to select frames based on timing criteria.
    Parameter delta_t ensures the margin from the edges of the segment.
    If delta_t is positive (future) it ensures that the frame is not \
        too close to the end of the segment.
    If delta_t is negative (past) it ensures that the frame is not too
    close to the beginning of the segment.
    """

    def __init__(self, time_delta: float):
        self._time_delta = time_delta
        if self._time_delta == 0:
            logging.warning("Time delta is set to 0, this filter is needless.")
        # Narrow type so pandas transform string literal overload is accepted
        self._aggregation_mode: Literal["first", "last"] = (
            "last" if self._time_delta > 0 else "first"
        )

    @property
    def name(self) -> str:
        return "frame_timing_filter"

    def _get_binary_mask(self, index_data: pd.DataFrame) -> pd.Series:
        edge_ts_of_segment = index_data.groupby("segment_id")["timestamp"].transform(
            self._aggregation_mode
        )
        if self._time_delta > 0:
            mask = index_data["timestamp"] <= (edge_ts_of_segment - self._time_delta)
        else:
            mask = index_data["timestamp"] >= (
                edge_ts_of_segment + abs(self._time_delta)
            )
        if not isinstance(mask, pd.Series):
            raise ValueError("Mask must be a pandas Series.")
        return mask
