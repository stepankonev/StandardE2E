from typing import Any

from standard_e2e.caching.adapters.abstract_adapter import AbstractAdapter
from standard_e2e.constants import PREFERENCE_TRAJECTORIES_KEY
from standard_e2e.data_structures import StandardFrameData
from standard_e2e.enums import Modality, StandardFrameDataField


class IdentityAdapter(AbstractAdapter):
    """Pass a single ``StandardFrameData`` field through unchanged as a modality.

    ``modality`` is the output key written to ``TransformedFrameData``;
    ``attr`` is the input ``StandardFrameData`` field it reads. They are
    related but distinct (e.g. ``Modality.LIDAR_PC`` ←
    ``StandardFrameDataField.LIDAR``), so both are passed explicitly.
    """

    @property
    def name(self) -> str:
        return f"IdentityAdapter({self._modality.name})"

    def __init__(self, modality: Modality, attr: StandardFrameDataField):
        self._modality = modality
        self._attr = attr

    @property
    def consumes_attrs(self) -> set[StandardFrameDataField]:
        return {self._attr}

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
        super().__init__(Modality.CAMERAS, StandardFrameDataField.CAMERAS)

    @property
    def name(self) -> str:
        return "CamerasIdentityAdapter"


class Detections3DIdentityAdapter(IdentityAdapter):
    """Identity adapter for 3D detections data."""

    def __init__(self):
        super().__init__(
            Modality.DETECTIONS_3D, StandardFrameDataField.FRAME_DETECTIONS_3D
        )

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


class FutureStatesIdentityAdapter(IdentityAdapter):
    """Identity adapter for future states data."""

    def __init__(self):
        super().__init__(Modality.FUTURE_STATES, StandardFrameDataField.FUTURE_STATES)

    @property
    def name(self) -> str:
        return "FutureStatesIdentityAdapter"


class PastStatesIdentityAdapter(IdentityAdapter):
    """Identity adapter for past states data."""

    def __init__(self):
        super().__init__(Modality.PAST_STATES, StandardFrameDataField.PAST_STATES)

    @property
    def name(self) -> str:
        return "PastStatesIdentityAdapter"


class IntentIdentityAdapter(IdentityAdapter):
    """Identity adapter for intent data."""

    def __init__(self):
        super().__init__(Modality.INTENT, StandardFrameDataField.INTENT)

    @property
    def name(self) -> str:
        return "IntentIdentityAdapter"
