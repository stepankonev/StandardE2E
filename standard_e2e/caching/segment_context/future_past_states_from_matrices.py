import numpy as np
import pandas as pd

from standard_e2e.data_structures import Trajectory, TransformedFrameData
from standard_e2e.enums import Modality
from standard_e2e.enums import TrajectoryComponent as TC

from .segment_context_aggregator import SegmentContextAggregator


def _check_not_both(param1, param2, name1, name2):
    if param1 is not None and param2 is not None:
        raise ValueError(f"Specify only one of {name1} or {name2}")


def _check_if_not_none(value, name):
    if value is not None and value < 0:
        raise ValueError(f"{name} must be non-negative")


class FuturePastStatesFromMatricesAggregator(SegmentContextAggregator):
    """Derive past/future ego-relative trajectories from pose matrices."""

    def __init__(
        self,
        data_path: str,
        max_history_length: int | None = None,
        max_future_length: int | None = None,
        max_history_delta_t: int | None = None,
        max_future_delta_t: int | None = None,
    ):
        super().__init__(data_path)
        _check_not_both(
            max_history_length,
            max_history_delta_t,
            "max_history_length",
            "max_history_delta_t",
        )
        _check_not_both(
            max_future_length,
            max_future_delta_t,
            "max_future_length",
            "max_future_delta_t",
        )
        _check_if_not_none(max_history_length, "max_history_length")
        _check_if_not_none(max_history_delta_t, "max_history_delta_t")
        _check_if_not_none(max_future_length, "max_future_length")
        _check_if_not_none(max_future_delta_t, "max_future_delta_t")
        self._max_history_length = max_history_length
        self._max_future_length = max_future_length
        self._max_history_delta_t = max_history_delta_t
        self._max_future_delta_t = max_future_delta_t

    def _fetch_value_from_transformed_frame(self, transformed_frame):
        matrix = transformed_frame.aux_data["pose_matrix"]
        if not matrix.shape == (4, 4):
            raise ValueError("Invalid pose_matrix shape")
        return matrix

    def _filter_context_for_frame(
        self,
        history_context: pd.DataFrame,
        future_context: pd.DataFrame,
        frame: TransformedFrameData,
    ):
        if self._max_future_length is not None:
            future_context = future_context.iloc[: self._max_future_length]
        if self._max_history_length is not None:
            history_context = history_context.iloc[-self._max_history_length :]
        if self._max_future_delta_t is not None:
            future_context = future_context[
                future_context["delta_t"] <= self._max_future_delta_t
            ]
        if self._max_history_delta_t is not None:
            history_context = history_context[
                history_context["delta_t"] >= -abs(self._max_history_delta_t)
            ]
        return history_context, future_context

    def _get_states(
        self, matrices: list[np.ndarray], reference: np.ndarray
    ) -> dict[str, list[float]]:
        states: dict[str, list[float]] = {"x": [], "y": [], "z": [], "heading": []}
        for T in matrices:
            # Relative transform: T_rel = inv(reference) @ T
            T_rel = np.linalg.inv(reference) @ T
            # Extract x, y, z position
            x, y, z = T_rel[0, 3], T_rel[1, 3], T_rel[2, 3]
            # Extract heading (yaw angle from rotation part, assuming Z-up)
            heading = np.arctan2(T_rel[1, 0], T_rel[0, 0])
            states["x"].append(x)
            states["y"].append(y)
            states["z"].append(z)
            states["heading"].append(heading)
        return states

    def _extract_states(self, history_pos_matrices, future_pos_matrices):
        # Current reference frame (last in history)
        current_T = history_pos_matrices[-1]
        history_states = self._get_states(history_pos_matrices, current_T)
        future_states = self._get_states(future_pos_matrices, current_T)
        return history_states, future_states

    def _update_frame_with_context(
        self, transformed_frame, history_context, future_context
    ):
        history_states, future_states = self._extract_states(
            history_context["value"].values, future_context["value"].values
        )
        transformed_frame.set_modality_data(
            Modality.FUTURE_STATES,
            Trajectory(
                {
                    TC.X: future_states["x"],
                    TC.Y: future_states["y"],
                    TC.Z: future_states["z"],
                    TC.HEADING: future_states["heading"],
                    TC.TIMESTAMP: future_context["delta_t"].values,
                }
            ),
        )
        transformed_frame.set_modality_data(
            Modality.PAST_STATES,
            Trajectory(
                {
                    TC.X: history_states["x"],
                    TC.Y: history_states["y"],
                    TC.Z: history_states["z"],
                    TC.HEADING: history_states["heading"],
                    TC.TIMESTAMP: history_context["delta_t"].values,
                }
            ),
        )
        return transformed_frame
