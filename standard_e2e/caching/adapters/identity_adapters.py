from typing import Any

from standard_e2e.caching.adapters.abstract_adapter import AbstractAdapter
from standard_e2e.constants import PREFERENCE_TRAJECTORIES_KEY
from standard_e2e.data_structures import StandardFrameData
from standard_e2e.enums import Modality


class IdentityAdapter(AbstractAdapter):
    """
    Identity adapter that maps modalities to their corresponding attributes.
    """

    DEFAULT_MAPPING = {
        Modality.CAMERAS: "cameras",
        Modality.LIDAR_PC: "lidar",
        Modality.FUTURE_STATES: "future_states",
        Modality.PAST_STATES: "past_states",
        Modality.DETECTIONS_3D: "detections_3d",
    }

    @property
    def name(self) -> str:
        return f"IdentityAdapter({self._modality.name})"

    def __init__(self, modality, attr=None):
        self._modality = modality
        self._attr = attr
        if self._attr is None:
            self._attr = self.DEFAULT_MAPPING.get(modality)
        if self._attr is None:
            raise ValueError(
                f"No default mapping for modality {modality}, must provide attr"
            )

    def _transform(self, standard_frame_data: StandardFrameData) -> dict[Modality, Any]:
        if (
            not hasattr(standard_frame_data, self._attr)
            or getattr(standard_frame_data, self._attr) is None
        ):
            return {}
        return {self._modality: getattr(standard_frame_data, self._attr)}


class CamerasIdentityAdapter(IdentityAdapter):
    """Identity adapter for camera data."""

    def __init__(self):
        super().__init__(Modality.CAMERAS, "cameras")

    @property
    def name(self) -> str:
        return "CamerasIdentityAdapter"


class Detections3DIdentityAdapter(IdentityAdapter):
    """Identity adapter for 3D detections data."""

    def __init__(self):
        super().__init__(Modality.DETECTIONS_3D, "frame_detections_3d")

    @property
    def name(self) -> str:
        return "Detections3DIdentityAdapter"


class PreferenceTrajectoryAdapter(AbstractAdapter):
    """Adapter for preference trajectory data."""

    @property
    def name(self) -> str:
        return "PreferenceTrajectoryAdapter"

    def _transform(self, standard_frame_data: StandardFrameData) -> dict[Modality, Any]:
        if (
            standard_frame_data.aux_data is None
            or standard_frame_data.aux_data.get(PREFERENCE_TRAJECTORIES_KEY) is None
        ):
            return {}
        return {
            Modality.PREFERENCE_TRAJECTORY: standard_frame_data.aux_data[
                PREFERENCE_TRAJECTORIES_KEY
            ]
        }


# class LidarIdentityAdapter(IdentityAdapter):
#     """Identity adapter for lidar data."""

#     def __init__(self):
#         super().__init__(Modality.LIDAR_PC, "lidar")

#     @property
#     def name(self) -> str:
#         return "LidarIdentityAdapter"

#     def transform(
#         self, standard_frame_data: StandardFrameData) -> dict[Modality, any]:
#         """
#         Transform lidar data (stored as pd.DataFrame) to ndarray
#         """
#         return super().transform(standard_frame_data).values


class FutureStatesIdentityAdapter(IdentityAdapter):
    """Identity adapter for future states data."""

    def __init__(self):
        super().__init__(Modality.FUTURE_STATES, "future_states")

    @property
    def name(self) -> str:
        return "FutureStatesIdentityAdapter"


class PastStatesIdentityAdapter(IdentityAdapter):
    """Identity adapter for past states data."""

    def __init__(self):
        super().__init__(Modality.PAST_STATES, "past_states")

    @property
    def name(self) -> str:
        return "PastStatesIdentityAdapter"


class IntentIdentityAdapter(IdentityAdapter):
    """Identity adapter for intent data."""

    def __init__(self):
        super().__init__(Modality.INTENT, "intent")

    @property
    def name(self) -> str:
        return "IntentIdentityAdapter"
