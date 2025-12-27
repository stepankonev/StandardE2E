from typing import Any

import albumentations as A
import numpy as np

from standard_e2e.caching.adapters.abstract_adapter import AbstractAdapter
from standard_e2e.data_structures.frame_data import StandardFrameData
from standard_e2e.enums import CameraDirection, Modality
from standard_e2e.utils.image_utils import CropTop


class PanoImageAdapter(AbstractAdapter):
    """Image adapter for Waymo E2E dataset."""

    DEFAULT_CAMERAS_ORDER = [
        CameraDirection.FRONT_LEFT,
        CameraDirection.FRONT,
        CameraDirection.FRONT_RIGHT,
    ]

    def __init__(
        self,
        top_cut_frac: float = 0.0,
        max_size: int = 640,
        cameras_order: list[CameraDirection] | None = None,
    ):
        super().__init__()
        self._image_transform = A.Compose(
            [
                CropTop(top_cut_frac=top_cut_frac),
                A.LongestMaxSize(max_size=max_size, p=1.0),
            ]
        )
        self._cameras_order = cameras_order or self.DEFAULT_CAMERAS_ORDER

    @property
    def name(self) -> str:
        return "pano_image_adapter"

    def _transform(self, standard_frame_data: StandardFrameData) -> dict[Modality, Any]:
        """Transform cameras data to a single panoramic image."""
        image_list = [
            standard_frame_data.cameras[camera_direction].image
            for camera_direction in self._cameras_order
        ]
        concatenated_image = np.concatenate(image_list, axis=1)
        adapted_image = self._image_transform(image=concatenated_image)["image"]
        return {Modality.CAMERAS: adapted_image}
