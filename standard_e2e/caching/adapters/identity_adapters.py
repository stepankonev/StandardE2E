from typing import Any

import cv2
import numpy as np

from standard_e2e.caching.adapters.abstract_adapter import AbstractAdapter
from standard_e2e.constants import PREFERENCE_TRAJECTORIES_KEY
from standard_e2e.data_structures import CameraData, StandardFrameData
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


def _downscale_camera(camera: CameraData, max_size: int) -> CameraData:
    """Downscale a camera image so its longest side is at most ``max_size`` px,
    scaling the intrinsics by the same per-axis ratio so projection holds."""
    height, width = camera.image.shape[:2]
    if max(height, width) <= max_size:
        return camera
    scale = max_size / max(height, width)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    image = np.ascontiguousarray(
        cv2.resize(camera.image, (new_w, new_h), interpolation=cv2.INTER_AREA),
        dtype=np.uint8,
    )
    intrinsics = camera.intrinsics.copy()
    intrinsics[0, :] *= new_w / width  # fx, skew, cx
    intrinsics[1, :] *= new_h / height  # fy, cy
    return CameraData(
        camera_direction=camera.camera_direction,
        image=image,
        intrinsics=intrinsics,
        extrinsics=camera.extrinsics,
        distortion=camera.distortion,
        is_fisheye=camera.is_fisheye,
    )


class CamerasIdentityAdapter(IdentityAdapter):
    """Identity adapter for camera data.

    With ``max_size`` set, each camera image is downscaled so its longest side
    is at most ``max_size`` pixels and its intrinsics are scaled to match (so
    projection still holds). ``None`` (the default) passes the cameras through
    unchanged.
    """

    def __init__(self, max_size: int | None = None):
        super().__init__(Modality.CAMERAS, StandardFrameDataField.CAMERAS)
        self._max_size = max_size

    @property
    def name(self) -> str:
        return "CamerasIdentityAdapter"

    def _transform(self, standard_frame_data: StandardFrameData) -> dict[Modality, Any]:
        transformed = super()._transform(standard_frame_data)
        if self._max_size is None or Modality.CAMERAS not in transformed:
            return transformed
        cameras = transformed[Modality.CAMERAS]
        return {
            Modality.CAMERAS: {
                direction: _downscale_camera(camera, self._max_size)
                for direction, camera in cameras.items()
            }
        }


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
