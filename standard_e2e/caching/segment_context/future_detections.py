from typing import Any, List, cast

import numpy as np
import pandas as pd

from standard_e2e.data_structures import (
    Array1DNP,
    Detection3D,
    Trajectory,
    TransformedFrameData,
)
from standard_e2e.enums import Modality
from standard_e2e.enums import TrajectoryComponent as TC

from .segment_context_aggregator import SegmentContextAggregator


def get_rotation_matrix(theta: float) -> np.ndarray:
    """Returns a 2D rotation matrix for a given angle in radians."""
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


class FutureDetectionsAggregator(SegmentContextAggregator):
    """Aggregate future detections into the current frame's modality payload."""

    def _fetch_value_from_transformed_frame(
        self, transformed_frame: TransformedFrameData
    ) -> Any:
        # global_position is set to default zeroes if missing
        return (
            transformed_frame.get_modality_data(Modality.DETECTIONS_3D),
            (
                transformed_frame.global_position.get(  # type: ignore[union-attr]
                    [TC.X, TC.Y, TC.HEADING]
                )[-1]
            ),
        )

    def _update_frame_with_context(
        self,
        transformed_frame: TransformedFrameData,
        history_context: pd.DataFrame,
        future_context: pd.DataFrame,
    ) -> TransformedFrameData:
        current_xyh = history_context["value"].iloc[-1][1]
        map_id_to_detections: dict[str, list[tuple]] = {}
        processed_frame_detections: list[Detection3D] = []
        for frame_detections, frame_xyh in future_context["value"]:
            for detection in frame_detections.detections:
                map_id_to_detections.setdefault(detection.unique_agent_id, []).append(
                    (detection, frame_xyh - current_xyh)
                )
        for detection_id, values in map_id_to_detections.items():
            aggregated_traj_dict: dict[TC, List[float]] = {
                TC.X: [],
                TC.Y: [],
                TC.HEADING: [],
            }
            for detection, pos_diff_xyh in values:
                if not isinstance(detection, Detection3D):
                    raise ValueError(
                        "Expected Detection3D instances in future detections."
                    )
                traj = detection.trajectory
                xyh = traj.get([TC.X, TC.Y, TC.HEADING])
                xyh = xyh + pos_diff_xyh
                xy = xyh[:, :2] @ get_rotation_matrix(xyh[0, 2]).T
                aggregated_traj_dict[TC.X].append(xy[0, 0])
                aggregated_traj_dict[TC.Y].append(xy[0, 1])
                aggregated_traj_dict[TC.HEADING].append(xyh[0, 2])
                assert detection.unique_agent_id == detection_id
            processed_frame_detections.append(
                Detection3D(
                    trajectory=Trajectory(
                        cast(dict[TC, Array1DNP], aggregated_traj_dict)
                    ),
                    unique_agent_id=detection.unique_agent_id,
                    detection_type=detection.detection_type,
                )
            )
        transformed_frame.set_modality_data(
            Modality.DETECTIONS_3D, processed_frame_detections
        )
        return transformed_frame
