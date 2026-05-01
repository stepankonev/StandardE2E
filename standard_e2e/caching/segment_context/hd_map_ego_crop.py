"""Abstract per-source HD-map ego-crop aggregator (per ADR 0007).

Each per-source subclass implements ``_parse_world_segment_map`` to
produce a world-frame ``RawSegmentHDMap`` from the source-native HD map
(Waymo proto, AV2 SDK, ...). This base handles the per-segment
lifecycle:

1. Parse the world-frame segment map **once**.
2. For every frame in the segment, read the frame's
   ``aux_data["pose_matrix"]``, call
   ``crop_hd_map_ego_relative`` to lift the world map into ego frame
   for this timestamp, and persist the resulting ``HDMapData`` under
   ``Modality.HD_MAP``.
3. Discard the world-frame map at end-of-segment so multi-segment runs
   do not balloon memory.

This aggregator overrides ``_process_segment`` because the standard
history/future context lookup does not apply: HD-map data is segment-
constant, not per-frame.
"""

from __future__ import annotations

import os
from abc import abstractmethod
from typing import Any

import pandas as pd

from standard_e2e.caching.segment_context.segment_context_aggregator import (
    SegmentContextAggregator,
)
from standard_e2e.data_structures import (
    RawSegmentHDMap,
    TransformedFrameData,
)
from standard_e2e.enums import Modality
from standard_e2e.utils.hd_map import crop_hd_map_ego_relative


class HDMapEgoCropAggregator(SegmentContextAggregator):
    """Abstract base; one concrete subclass per source dataset."""

    def __init__(
        self,
        data_path: str,
        source_data_path: str,
        x_range: float,
        y_range: float,
    ):
        super().__init__(data_path)
        self._source_data_path = source_data_path
        self._x_range = float(x_range)
        self._y_range = float(y_range)

    @property
    def crop_extent(self) -> tuple[float, float]:
        return self._x_range, self._y_range

    @abstractmethod
    def _parse_world_segment_map(self, segment_id: str) -> RawSegmentHDMap:
        """Per-source parse of the segment's world-frame HD map.

        Implementations should NOT cache the returned object on the
        instance; this base class keeps it as a local in
        ``_process_segment`` and discards it when the segment finishes.
        """

    def _fetch_value_from_transformed_frame(
        self, transformed_frame: TransformedFrameData
    ) -> Any:
        if (
            transformed_frame.aux_data is None
            or "pose_matrix" not in transformed_frame.aux_data
        ):
            raise ValueError(
                "HDMapEgoCropAggregator requires aux_data['pose_matrix'] on "
                "every frame; the source processor must populate it."
            )
        return transformed_frame.aux_data["pose_matrix"]

    def _update_frame_with_context(
        self,
        transformed_frame: TransformedFrameData,
        history_context: pd.DataFrame,
        future_context: pd.DataFrame,
    ) -> TransformedFrameData:  # pragma: no cover - never called
        raise NotImplementedError(
            "HDMapEgoCropAggregator overrides _process_segment; this hook is "
            "unused. Crops are applied directly in _process_segment."
        )

    def _process_segment(self, segment_index_data: pd.DataFrame) -> None:
        self._validate_segment_frame(segment_index_data)
        segment_id = str(segment_index_data["segment_id"].iloc[0])
        raw = self._parse_world_segment_map(segment_id)
        if not isinstance(raw, RawSegmentHDMap):
            raise TypeError(
                "_parse_world_segment_map must return RawSegmentHDMap; "
                f"got {type(raw).__name__}"
            )
        try:
            for _, row in segment_index_data.iterrows():
                path = os.path.join(self._data_path, row["filename"])
                frame = TransformedFrameData.from_npz(path)
                ego_pose = self._fetch_value_from_transformed_frame(frame)
                ego_map = crop_hd_map_ego_relative(
                    raw, ego_pose, self._x_range, self._y_range
                )
                frame.set_modality_data(Modality.HD_MAP, ego_map)
                frame.to_npz(path)
        finally:
            del raw
