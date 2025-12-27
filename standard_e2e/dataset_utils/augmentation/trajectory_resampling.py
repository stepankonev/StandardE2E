from typing import List

from standard_e2e.data_structures import TransformedFrameData
from standard_e2e.data_structures.trajectory_data import Trajectory
from standard_e2e.enums import Modality

from .augmentation import FrameAugmentation


class TrajectoryResampling(FrameAugmentation):
    """Resample past/future/preference trajectories onto target timestamps."""

    def __init__(
        self,
        history_target_timestamps: list[float] | None = None,
        future_target_timestamps: list[float] | None = None,
        target_frame_names: list[str] | None = None,
    ):
        """Configure resampling targets for trajectories.

        Args:
            history_target_timestamps: New timestamps for ``PAST_STATES``.
            future_target_timestamps: New timestamps for ``FUTURE_STATES`` and
                ``PREFERENCE_TRAJECTORY`` (when present).
            target_frame_names: Optional subset of frame keys to resample; defaults
                to all frames provided to ``augment``.
        """
        super().__init__()
        self._history_target_timestamps = history_target_timestamps
        self._future_target_timestamps = future_target_timestamps
        self._target_frame_names = target_frame_names

    def _update_trajectory(
        self,
        frame: TransformedFrameData,
        modality: Modality,
        target_timestamps: list[float],
    ) -> TransformedFrameData:
        assert modality in [Modality.FUTURE_STATES, Modality.PAST_STATES]
        trajectory: Trajectory = frame.get_modality_data(modality)
        resampled_trajectory = trajectory.resample(target_timestamps)
        frame.set_modality_data(modality, resampled_trajectory)
        return frame

    def _update_preference_trajectories(
        self,
        frame: TransformedFrameData,
        target_timestamps: list[float],
    ) -> TransformedFrameData:
        preference_trajectories: List[Trajectory] | None = frame.get_modality_data(
            Modality.PREFERENCE_TRAJECTORY
        )
        if preference_trajectories is not None:
            resampled_preference_trajectories = [
                preference_trajectory.resample(target_timestamps)
                for preference_trajectory in preference_trajectories
            ]
            frame.set_modality_data(
                Modality.PREFERENCE_TRAJECTORY, resampled_preference_trajectories
            )
        return frame

    def _augment(
        self, frames: dict[str, TransformedFrameData], regime: str
    ) -> dict[str, TransformedFrameData]:
        target_frame_names = (
            self._target_frame_names
            if self._target_frame_names is not None
            else list(frames.keys())
        )
        for frame_name in target_frame_names:
            if self._history_target_timestamps is not None:
                frames[frame_name] = self._update_trajectory(
                    frames[frame_name],
                    Modality.PAST_STATES,
                    self._history_target_timestamps,
                )
            if self._future_target_timestamps is not None:
                frames[frame_name] = self._update_trajectory(
                    frames[frame_name],
                    Modality.FUTURE_STATES,
                    self._future_target_timestamps,
                )
                frames[frame_name] = self._update_preference_trajectories(
                    frames[frame_name], self._future_target_timestamps
                )
        return frames
