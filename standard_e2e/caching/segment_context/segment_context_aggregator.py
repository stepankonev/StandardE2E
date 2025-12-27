import os
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from tqdm import tqdm

from standard_e2e.data_structures import TransformedFrameData
from standard_e2e.utils import _assert_strictly_increasing


class SegmentContextAggregator(ABC):
    """Abstract pass to add history/future context back into transformed frames."""

    def __init__(self, data_path: str):
        self._data_path = data_path

    @abstractmethod
    def _fetch_value_from_transformed_frame(
        self, transformed_frame: TransformedFrameData
    ) -> Any:
        """Extract the context value stored for a single frame."""
        pass

    @abstractmethod
    def _update_frame_with_context(
        self,
        transformed_frame: TransformedFrameData,
        history_context: pd.DataFrame,
        future_context: pd.DataFrame,
    ) -> TransformedFrameData:
        """Write derived context back into ``transformed_frame``."""
        pass

    def _validate_segment_frame(self, segment_index_data: pd.DataFrame) -> None:
        if not isinstance(segment_index_data, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame")
        if "segment_id" not in segment_index_data.columns:
            raise ValueError("Missing column: segment_id")
        if "timestamp" not in segment_index_data.columns:
            raise ValueError("Missing column: timestamp")
        n_unique_segments = len(segment_index_data["segment_id"].unique())
        if n_unique_segments == 0:
            raise ValueError("Segment index data is empty")
        if n_unique_segments > 1:
            raise ValueError("segment_index_data contains multiple segments")
        _assert_strictly_increasing(segment_index_data["timestamp"].to_numpy())

    def _prepare_segment_context(
        self, segment_index_data: pd.DataFrame
    ) -> pd.DataFrame:
        result_values = []
        for _, row in segment_index_data.iterrows():
            transformed_frame = TransformedFrameData.from_npz(
                os.path.join(self._data_path, row["filename"])
            )
            value = self._fetch_value_from_transformed_frame(transformed_frame)
            result_values.append(
                {
                    "segment_id": row["segment_id"],
                    "timestamp": row["timestamp"],
                    "value": value,
                }
            )
        return pd.DataFrame(result_values)

    def _filter_context_for_frame(
        self,
        history_context: pd.DataFrame,
        future_context: pd.DataFrame,
        frame: TransformedFrameData,  # pylint: disable=unused-argument
    ):
        return history_context, future_context

    def _process_segment(self, segment_index_data: pd.DataFrame) -> None:
        self._validate_segment_frame(segment_index_data)
        prepared_context = self._prepare_segment_context(segment_index_data)
        for idx in range(len(prepared_context)):
            frame = TransformedFrameData.from_npz(
                os.path.join(self._data_path, segment_index_data.iloc[idx]["filename"])
            )
            if frame.timestamp != prepared_context["timestamp"][idx]:
                raise ValueError(
                    "Timestamps do not match between index and prepared context"
                )
            prepared_context = prepared_context.copy()
            prepared_context["delta_t"] = (
                prepared_context["timestamp"] - frame.timestamp
            )
            history_context = prepared_context[: idx + 1]
            future_context = prepared_context[idx + 1 :]
            if history_context["delta_t"].max() > 0:
                raise ValueError("History context contains future timestamps")
            if future_context["delta_t"].min() <= 0:
                raise ValueError("Future context contains past / current timestamps")
            history_context, future_context = self._filter_context_for_frame(
                history_context, future_context, frame
            )
            frame = self._update_frame_with_context(
                frame, history_context, future_context
            )
            if not isinstance(frame, TransformedFrameData):
                raise TypeError(
                    "_update_frame_with_context must return TransformedFrameData, "
                    f"got {type(frame)}"
                )
            frame.to_npz(
                os.path.join(self._data_path, segment_index_data.iloc[idx]["filename"])
            )

    def process(self, index_df: pd.DataFrame):
        """Run the aggregator for each segment described in ``index_df``."""
        if not isinstance(index_df, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame")
        pbar = tqdm(index_df.groupby("segment_id"), desc="Processing segments")
        for _, segment_data in pbar:
            self._process_segment(segment_data)
