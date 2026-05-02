from typing import Any

import numpy as np

from standard_e2e.caching.adapters.abstract_adapter import AbstractAdapter
from standard_e2e.data_structures import LidarPointCloud, StandardFrameData
from standard_e2e.enums import LidarComponent, Modality


class LidarAdapter(AbstractAdapter):
    """Lidar adapter: extracts XYZ from ``StandardFrameData.lidar`` into a
    :class:`LidarPointCloud`, with optional deterministic stride subsampling.

    Args:
        max_points: optional cap on the number of output points. When set and
            the input cloud has more points, a stride subsample is taken
            (``points[::N // max_points][:max_points]``) producing at most
            ``max_points`` rows. No randomness is involved, so preprocessing
            is fully reproducible without seeding. Otherwise the full cloud
            is passed through.
    """

    _COMPONENTS = [LidarComponent.X, LidarComponent.Y, LidarComponent.Z]

    def __init__(self, max_points: int | None = None) -> None:
        super().__init__()
        if max_points is not None and max_points <= 0:
            raise ValueError("max_points must be a positive integer when set")
        self._max_points = max_points

    @property
    def name(self) -> str:
        return "LidarAdapter"

    def _transform(self, standard_frame_data: StandardFrameData) -> dict[Modality, Any]:
        if standard_frame_data.lidar is None:
            return {}
        df = standard_frame_data.lidar.points
        points = df[[c.value for c in self._COMPONENTS]].to_numpy(dtype=np.float32)
        if self._max_points is not None and points.shape[0] > self._max_points:
            step = points.shape[0] // self._max_points
            points = points[::step][: self._max_points]
        return {
            Modality.LIDAR_PC: LidarPointCloud(
                points=points, components=self._COMPONENTS
            )
        }
