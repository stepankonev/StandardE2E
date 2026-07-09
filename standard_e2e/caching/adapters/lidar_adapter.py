from typing import Any

import numpy as np

from standard_e2e.caching.adapters.abstract_adapter import AbstractAdapter
from standard_e2e.data_structures import LidarPointCloud, StandardFrameData
from standard_e2e.enums import LidarComponent, Modality, StandardFrameDataField

LIDAR_BEV_CHANNELS_AUX_KEY = "lidar_bev_channels"
LIDAR_BEV_GRID_AUX_KEY = "lidar_bev_grid"


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

    @property
    def consumes_attrs(self) -> set[StandardFrameDataField]:
        return {StandardFrameDataField.LIDAR}

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


class LidarBEVAdapter(AbstractAdapter):
    """Top-down 2-D histogram of lidar points in vehicle frame.

    Implements the Transfuser / CARLA-Garage BEV recipe: drop points above
    ``max_height``, optionally split into ground / obstacle channels at
    ``split_height``, then 2-D bin into a regular grid over
    ``[min_x, max_x] x [min_y, max_y]`` at ``pixels_per_meter``. Each
    channel is the per-cell point count, clipped at ``count_cap`` and
    divided into ``[0, 1]``.

    Output (``Modality.LIDAR_BEV``): ``np.float32`` array of shape
    ``(C, H, W)`` where ``C = 2`` if ``use_ground_plane`` else ``1``.
    Axis order matches ``np.histogramdd((x, y), ...)`` so ``H`` is the
    number of x-bins and ``W`` the number of y-bins; with the ground
    plane channel order is ``(below, above)``.

    Args:
        min_x, max_x: BEV x extent in meters (vehicle x is forward).
        min_y, max_y: BEV y extent in meters (vehicle y is left).
        pixels_per_meter: grid resolution; cast to ``int`` internally
            (matches Transfuser).
        max_height: drop points with ``z >= max_height``.
        split_height: ground / obstacle threshold (``z <=`` -> ground).
        use_ground_plane: include the ground channel.
        count_cap: per-cell saturation; histogram clipped at this value
            before being divided into ``[0, 1]``.
    """

    _COMPONENTS = [LidarComponent.X, LidarComponent.Y, LidarComponent.Z]

    def __init__(
        self,
        min_x: float = -32.0,
        max_x: float = 32.0,
        min_y: float = -32.0,
        max_y: float = 32.0,
        pixels_per_meter: float = 4.0,
        max_height: float = 100.0,
        split_height: float = 0.2,
        use_ground_plane: bool = False,
        count_cap: int = 5,
    ) -> None:
        super().__init__()
        if max_x <= min_x or max_y <= min_y:
            raise ValueError("max_{x,y} must be greater than min_{x,y}")
        if int(pixels_per_meter) < 1:
            raise ValueError("pixels_per_meter must be >= 1 after int cast")
        if count_cap <= 0:
            raise ValueError("count_cap must be a positive integer")
        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y
        self._pixels_per_meter = pixels_per_meter
        self._max_height = max_height
        self._split_height = split_height
        self._use_ground_plane = use_ground_plane
        self._count_cap = count_cap

    @property
    def name(self) -> str:
        return "LidarBEVAdapter"

    @property
    def consumes_attrs(self) -> set[StandardFrameDataField]:
        return {StandardFrameDataField.LIDAR}

    @property
    def output_shape(self) -> tuple[int, int, int]:
        """Shape ``(C, H, W)`` of the BEV tensor this adapter emits."""
        ppm = int(self._pixels_per_meter)
        h = int((self._max_x - self._min_x) * ppm)
        w = int((self._max_y - self._min_y) * ppm)
        c = 2 if self._use_ground_plane else 1
        return (c, h, w)

    @property
    def metadata(self) -> dict[str, Any]:
        """Expose the BEV channel order + grid so the .npz / dataset_info.yaml is
        self-describing (pixels map back to meters; channels name the height
        bands)."""
        channels = ["below", "above"] if self._use_ground_plane else ["above"]
        return {
            LIDAR_BEV_CHANNELS_AUX_KEY: channels,
            LIDAR_BEV_GRID_AUX_KEY: {
                "min_x": self._min_x,
                "max_x": self._max_x,
                "min_y": self._min_y,
                "max_y": self._max_y,
                "pixels_per_meter": self._pixels_per_meter,
            },
        }

    def _splat(self, xy: np.ndarray) -> np.ndarray:
        ppm = int(self._pixels_per_meter)
        n_x_edges = int((self._max_x - self._min_x) * ppm) + 1
        n_y_edges = int((self._max_y - self._min_y) * ppm) + 1
        xbins = np.linspace(self._min_x, self._max_x, n_x_edges)
        ybins = np.linspace(self._min_y, self._max_y, n_y_edges)
        hist, _ = np.histogramdd(xy, bins=(xbins, ybins))
        np.clip(hist, 0, self._count_cap, out=hist)
        return (hist / self._count_cap).astype(np.float32)

    def _transform(self, standard_frame_data: StandardFrameData) -> dict[Modality, Any]:
        if standard_frame_data.lidar is None:
            return {}
        df = standard_frame_data.lidar.points
        pts = df[[c.value for c in self._COMPONENTS]].to_numpy(dtype=np.float32)
        pts = pts[pts[:, 2] < self._max_height]
        above = self._splat(pts[pts[:, 2] > self._split_height][:, :2])
        if self._use_ground_plane:
            below = self._splat(pts[pts[:, 2] <= self._split_height][:, :2])
            bev = np.stack([below, above])
        else:
            bev = above[None]
        return {Modality.LIDAR_BEV: bev}
