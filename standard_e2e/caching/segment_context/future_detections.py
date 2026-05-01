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


# Components carried through the future-window aggregation. Mirrors the
# seed Detection3D shape produced by source dataset processors so the
# downstream renderer / batched view see the full pose-invariant +
# pose-dependent set rather than {X, Y, HEADING} only.
_AGGREGATED_COMPONENTS: tuple[TC, ...] = (
    TC.TIMESTAMP,
    TC.X,
    TC.Y,
    TC.Z,
    TC.HEADING,
    TC.LENGTH,
    TC.WIDTH,
    TC.HEIGHT,
)


class FutureDetectionsAggregator(SegmentContextAggregator):
    """Aggregate future detections into the current frame's modality payload."""

    def _fetch_value_from_transformed_frame(
        self, transformed_frame: TransformedFrameData
    ) -> Any:
        if transformed_frame.global_position is None:
            raise ValueError(
                "FutureDetectionsAggregator requires global_position on every "
                "frame; the source processor must populate it."
            )
        return (
            transformed_frame.get_modality_data(Modality.DETECTIONS_3D),
            transformed_frame.global_position.get([TC.X, TC.Y, TC.Z, TC.HEADING])[-1],
        )

    def _update_frame_with_context(
        self,
        transformed_frame: TransformedFrameData,
        history_context: pd.DataFrame,
        future_context: pd.DataFrame,
    ) -> TransformedFrameData:
        current_xyzh = history_context["value"].iloc[-1][1]
        map_id_to_detections: dict[str, list[tuple[Detection3D, np.ndarray, float]]] = (
            {}
        )
        for frame_value, delta_t in zip(
            future_context["value"], future_context["delta_t"]
        ):
            frame_detections, frame_xyzh = frame_value
            pos_diff_xyzh = frame_xyzh - current_xyzh
            for detection in frame_detections.detections:
                map_id_to_detections.setdefault(detection.unique_agent_id, []).append(
                    (detection, pos_diff_xyzh, float(delta_t))
                )
        processed_frame_detections: list[Detection3D] = []
        for detection_id, values in map_id_to_detections.items():
            seed_detection = values[0][0]
            seed_type = seed_detection.detection_type
            # LENGTH/WIDTH/HEIGHT are pose-invariant — read once from the seed.
            seed_traj = seed_detection.trajectory
            length = float(seed_traj.get(TC.LENGTH).reshape(-1)[0])
            width = float(seed_traj.get(TC.WIDTH).reshape(-1)[0])
            height = float(seed_traj.get(TC.HEIGHT).reshape(-1)[0])
            aggregated_traj_dict: dict[TC, List[float]] = {
                comp: [] for comp in _AGGREGATED_COMPONENTS
            }
            for detection, pos_diff_xyzh, delta_t in values:
                if not isinstance(detection, Detection3D):
                    raise ValueError(
                        "Expected Detection3D instances in future detections."
                    )
                assert detection.unique_agent_id == detection_id
                assert detection.detection_type == seed_type, (
                    f"detection_type drift across future window for agent "
                    f"{detection_id}: seed={seed_type} vs "
                    f"frame={detection.detection_type}"
                )
                traj = detection.trajectory
                xyzh = traj.get([TC.X, TC.Y, TC.Z, TC.HEADING])
                xyzh = xyzh + pos_diff_xyzh
                xy = xyzh[:, :2] @ get_rotation_matrix(xyzh[0, 3]).T
                aggregated_traj_dict[TC.TIMESTAMP].append(delta_t)
                aggregated_traj_dict[TC.X].append(float(xy[0, 0]))
                aggregated_traj_dict[TC.Y].append(float(xy[0, 1]))
                aggregated_traj_dict[TC.Z].append(float(xyzh[0, 2]))
                aggregated_traj_dict[TC.HEADING].append(float(xyzh[0, 3]))
                aggregated_traj_dict[TC.LENGTH].append(length)
                aggregated_traj_dict[TC.WIDTH].append(width)
                aggregated_traj_dict[TC.HEIGHT].append(height)
            processed_frame_detections.append(
                Detection3D(
                    trajectory=Trajectory(
                        cast(dict[TC, Array1DNP], aggregated_traj_dict)
                    ),
                    unique_agent_id=detection_id,
                    detection_type=seed_type,
                )
            )
        transformed_frame.set_modality_data(
            Modality.DETECTIONS_3D, processed_frame_detections
        )
        return transformed_frame
