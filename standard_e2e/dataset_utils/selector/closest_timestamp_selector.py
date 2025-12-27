import numpy as np
import pandas as pd

from standard_e2e.dataset_utils.selector.frame_selector import FrameSelector


class ClosestTimestampSelector(FrameSelector):
    """
    Selector that chooses the frame with
    the closest timestamp to a given time delta.
    """

    def __init__(
        self, location: str, delta_t: float, index_data: pd.DataFrame | None = None
    ):
        """Initialize selector.

        Args:
            location: "labels" or "features"; constrains the sign of ``delta_t``.
            delta_t: Time offset (seconds) relative to the current frame.
            index_data: Optional index dataframe to precompute mappings eagerly.
        """
        self._delta_t = delta_t
        super().__init__(location, index_data=index_data)
        if index_data is not None:
            self._mapping = self._precompute_closest_timestamp_frame_idx(index_data)

    def set_index_data(self, index_data: pd.DataFrame):
        """Set index data and recompute closest-frame mapping."""
        self._index_data = index_data
        self._mapping = self._precompute_closest_timestamp_frame_idx(index_data)

    @property
    def name(self) -> str:
        return "closest_timestamp_selector"

    def _validate_params(self):
        if self._location == "labels" and self._delta_t < 0:
            raise ValueError(
                "For 'labels' location, 'delta_t' must be non-negative."
                "(not coming from the past)"
            )
        if self._location == "features" and self._delta_t > 0:
            raise ValueError(
                "For 'features' location, 'delta_t' must be non-positive."
                "(not coming from the future)"
            )

    def _precompute_closest_timestamp_frame_idx(
        self, index_data: pd.DataFrame
    ) -> pd.Series:
        """Precompute the iloc of the frame whose timestamp best matches target.

        For each row ``i`` in ``index_data`` (sorted by ``segment_id`` and
        ``timestamp``), finds the frame in the same segment closest to
        ``timestamp[i] + delta_t``. Ties prefer the earlier frame.
        """
        n = len(index_data)
        out = np.empty(n, dtype=np.int64)

        seg = index_data["segment_id"].to_numpy()
        ts = index_data["timestamp"].to_numpy()
        delta = float(self._delta_t)

        # segment boundaries (contiguous because DF is sorted by segment_id)
        starts = np.r_[0, 1 + np.flatnonzero(seg[1:] != seg[:-1])]
        ends = np.r_[starts[1:], n]

        for start, end in zip(starts, ends):
            ts_seg = ts[start:end]
            target = ts_seg + delta
            m = end - start

            j = 0  # moving pointer in ts_seg
            for i in range(m):
                t = target[i]
                # advance j while next timestamp is still <= target
                while j + 1 < m and ts_seg[j + 1] <= t:
                    j += 1

                if j == m - 1:
                    best_local = j
                else:
                    # pick closer of j and j+1 (<= prefers left on exact tie)
                    best_local = (
                        j if abs(t - ts_seg[j]) <= abs(ts_seg[j + 1] - t) else j + 1
                    )

                out[start + i] = start + best_local
        return pd.Series(out, index=index_data.index)

    def select_frame(
        self, current_frame_iloc: int, index_data: pd.DataFrame | None = None
    ) -> int:
        """
        Select a frame from the index data based on the closest
        timestamp to the current frame within given segment.
        :param index_data: DataFrame containing index data
        with 'timestamp' and 'segment_id' columns.
        """
        if self._mapping is None:
            raise ValueError("Index data is not set. Please set index data first.")
        return int(self._mapping.iloc[current_frame_iloc])
