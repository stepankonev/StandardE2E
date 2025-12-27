import logging

import numpy as np

from standard_e2e.caching import SourceDatasetProcessor
from standard_e2e.caching.adapters import (
    AbstractAdapter,
    FutureStatesIdentityAdapter,
    IntentIdentityAdapter,
    PanoImageAdapter,
    PastStatesIdentityAdapter,
    PreferenceTrajectoryAdapter,
)
from standard_e2e.constants import PREFERENCE_TRAJECTORIES_KEY
from standard_e2e.data_structures import (
    StandardFrameData,
    Trajectory,
)
from standard_e2e.enums import CameraDirection, Intent, Modality, TrajectoryComponent
from standard_e2e.third_party.waymo_open_dataset.protos import (
    end_to_end_driving_data_pb2 as wod_e2ed_pb2,
)
from standard_e2e.utils.image_utils import waymo_fetch_images_from_frame


class WaymoE2EDatasetProcessor(SourceDatasetProcessor):
    """Processor for the Waymo E2E dataset."""

    DATASET_NAME = "waymo_e2e"
    PREFERENCE_TRAJECTORIES_LENGTH = 21
    TRAJECTORY_DELTA_T = 0.25  # seconds, time between trajectory points
    FRAME_DELTA_T = 0.1  # seconds, time between frames
    CAMERAS_ORDER = {
        CameraDirection.FRONT: 1,
        CameraDirection.FRONT_LEFT: 2,
        CameraDirection.FRONT_RIGHT: 3,
        CameraDirection.SIDE_LEFT: 4,
        CameraDirection.SIDE_RIGHT: 5,
        CameraDirection.REAR_LEFT: 6,
        CameraDirection.REAR: 7,
        CameraDirection.REAR_RIGHT: 8,
    }

    @property
    def allowed_splits(self) -> list[str]:
        """Return the list of allowed splits for the Waymo E2E dataset."""
        return ["training", "val", "test"]

    def _get_default_adapters(self) -> list[AbstractAdapter]:
        """Get the adapters for the Waymo E2E dataset."""
        return [
            PanoImageAdapter(),
            IntentIdentityAdapter(),
            PastStatesIdentityAdapter(),
            FutureStatesIdentityAdapter(),
            PreferenceTrajectoryAdapter(),
        ]

    def _prepare_standardized_frame_data(self, raw_frame_data) -> StandardFrameData:
        data = wod_e2ed_pb2.E2EDFrame()  # pylint: disable=no-member
        data.ParseFromString(raw_frame_data.numpy())
        segment_id, frame_id = data.frame.context.name.split("-")
        frame_id = int(frame_id)
        future_states = Trajectory(
            data={
                TrajectoryComponent.X: np.array(data.future_states.pos_x),
                TrajectoryComponent.Y: np.array(data.future_states.pos_y),
                TrajectoryComponent.Z: np.array(data.future_states.pos_z),
                TrajectoryComponent.TIMESTAMP: self.TRAJECTORY_DELTA_T
                * np.arange(1, len(data.future_states.pos_x) + 1),
            }
        )
        past_states = Trajectory(
            data={
                TrajectoryComponent.X: np.array(data.past_states.pos_x),
                TrajectoryComponent.Y: np.array(data.past_states.pos_y),
                TrajectoryComponent.VELOCITY_X: np.array(data.past_states.vel_x),
                TrajectoryComponent.VELOCITY_Y: np.array(data.past_states.vel_y),
                TrajectoryComponent.ACCELERATION_X: np.array(data.past_states.accel_x),
                TrajectoryComponent.ACCELERATION_Y: np.array(data.past_states.accel_y),
                TrajectoryComponent.TIMESTAMP: self.TRAJECTORY_DELTA_T
                * np.arange(-len(data.past_states.pos_x) + 1, 1),
            }
        )
        preference_scores_sum = sum(
            trajectory.preference_score for trajectory in data.preference_trajectories
        )
        # If preference scores / trajectories are actually there
        preference_trajectories: list[Trajectory] | None
        if preference_scores_sum > 0:
            preference_trajectories = []
            for preference_trajectory in data.preference_trajectories:
                preference_trajectories.append(
                    Trajectory(
                        data={
                            TrajectoryComponent.X: np.array(
                                preference_trajectory.pos_x
                            ),
                            TrajectoryComponent.Y: np.array(
                                preference_trajectory.pos_y
                            ),
                            TrajectoryComponent.TIMESTAMP: self.TRAJECTORY_DELTA_T
                            * np.arange(1, len(preference_trajectory.pos_x) + 1),
                        },
                        score=float(preference_trajectory.preference_score),
                    )
                )
                if (
                    preference_trajectories[-1].length
                    != self.PREFERENCE_TRAJECTORIES_LENGTH
                ):
                    logging.warning(
                        "Preference trajectory length mismatch: %d != %d",
                        preference_trajectories[-1].length,
                        self.PREFERENCE_TRAJECTORIES_LENGTH,
                    )
        else:
            preference_trajectories = None

        standard_frame_data = StandardFrameData(
            dataset_name=self.dataset_name,
            segment_id=segment_id,
            frame_id=frame_id,
            timestamp=frame_id * self.FRAME_DELTA_T,
            split=self._split,
            cameras=waymo_fetch_images_from_frame(data.frame),
            intent=Intent(data.intent),
            future_states=future_states,
            past_states=past_states,
            aux_data={PREFERENCE_TRAJECTORIES_KEY: preference_trajectories},
            extra_index_data={
                f"has_{PREFERENCE_TRAJECTORIES_KEY}": preference_scores_sum > 0,
                Modality.INTENT: Intent(data.intent),
            },
        )
        return standard_frame_data

    @property
    def dataset_name(self) -> str:
        """Return the name of the dataset."""
        return self.DATASET_NAME
