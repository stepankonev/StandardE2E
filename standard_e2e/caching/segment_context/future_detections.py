from typing import Any, List, cast

import numpy as np
import pandas as pd

from standard_e2e.data_structures import (
    Array1DNP,
    Detection3D,
    FrameDetections3D,
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
        # Per-frame detection trajectories carry (X, Y, HEADING) in the ego
        # frame at *that frame's* timestamp (egoF). To express them in the
        # current ego frame (egoC) we need the SE(2) composition
        #   det_egoC = R(H_F - H_C) @ det_egoF + R(-H_C) @ (P_F - P_C)
        # where (P_F, H_F) and (P_C, H_C) are the world-frame ego poses at
        # the future and current frames. ``pos_diff_xyh`` is precomputed as
        # ``frame_xyh - current_xyh`` (world deltas) by the loop below.
        current_xyh = history_context["value"].iloc[-1][1]
        H_C = float(current_xyh[2])
        R_neg_HC = get_rotation_matrix(-H_C)

        map_id_to_detections: dict[str, list[tuple]] = {}
        processed_frame_detections: list[Detection3D] = []
        for frame_detections, frame_xyh in future_context["value"]:
            for detection in frame_detections.detections:
                map_id_to_detections.setdefault(detection.unique_agent_id, []).append(
                    (detection, frame_xyh - current_xyh)
                )
        for detection_id, values in map_id_to_detections.items():
            # Box dimensions (LENGTH/WIDTH/HEIGHT) and altitude (Z) are rigid-body
            # invariants under the SE(2) ego-pose change, so we copy them per future
            # step from the source detection. Downstream consumers can then draw
            # rotated 3D boxes from the aggregated trajectory alone.
            aggregated_traj_dict: dict[TC, List[float]] = {
                TC.X: [],
                TC.Y: [],
                TC.Z: [],
                TC.HEADING: [],
                TC.LENGTH: [],
                TC.WIDTH: [],
                TC.HEIGHT: [],
            }
            for detection, pos_diff_xyh in values:
                if not isinstance(detection, Detection3D):
                    raise ValueError(
                        "Expected Detection3D instances in future detections."
                    )
                traj = detection.trajectory
                xyh = traj.get([TC.X, TC.Y, TC.HEADING])
                z_lwh = traj.get([TC.Z, TC.LENGTH, TC.WIDTH, TC.HEIGHT], strict=False)
                delta_h = float(pos_diff_xyh[2])
                R_dh = get_rotation_matrix(delta_h)
                delta_xy_egoC = R_neg_HC @ pos_diff_xyh[:2]
                det_xy_egoC = xyh[:, :2] @ R_dh.T + delta_xy_egoC
                det_h_egoC = xyh[:, 2] + delta_h
                aggregated_traj_dict[TC.X].append(float(det_xy_egoC[0, 0]))
                aggregated_traj_dict[TC.Y].append(float(det_xy_egoC[0, 1]))
                aggregated_traj_dict[TC.Z].append(float(z_lwh[0, 0]))
                aggregated_traj_dict[TC.HEADING].append(float(det_h_egoC[0]))
                aggregated_traj_dict[TC.LENGTH].append(float(z_lwh[0, 1]))
                aggregated_traj_dict[TC.WIDTH].append(float(z_lwh[0, 2]))
                aggregated_traj_dict[TC.HEIGHT].append(float(z_lwh[0, 3]))
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
        # Wrap in FrameDetections3D so the registered collate handler picks
        # it up; a bare list bypasses the override and falls through to
        # torch's default collate, which raises on variable-length batches.
        transformed_frame.set_modality_data(
            Modality.DETECTIONS_3D,
            FrameDetections3D(detections=processed_frame_detections),
        )
        return transformed_frame
