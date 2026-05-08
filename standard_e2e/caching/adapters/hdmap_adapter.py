from typing import Any, Sequence, Union

import cv2
import numpy as np

from standard_e2e.caching.adapters.abstract_adapter import AbstractAdapter
from standard_e2e.data_structures import StandardFrameData
from standard_e2e.enums import MapElementType, Modality

HD_MAP_BEV_CHANNELS_AUX_KEY = "hd_map_bev_channels"

ChannelSpec = Union[MapElementType, str]


def _resolve_channels(
    channels: Sequence[ChannelSpec] | None,
) -> list[MapElementType]:
    """Coerce a YAML/string-friendly channel list to ``list[MapElementType]``.

    ``None`` → all members in declaration order. String entries must match a
    member ``value`` (e.g. ``"lane_center"``); unknown strings raise
    ``ValueError`` listing the valid options so misconfigurations fail fast at
    adapter construction rather than silently producing empty channels.
    """
    if channels is None:
        return list(MapElementType)
    out: list[MapElementType] = []
    valid_values = {t.value for t in MapElementType}
    for entry in channels:
        if isinstance(entry, MapElementType):
            out.append(entry)
        elif isinstance(entry, str):
            try:
                out.append(MapElementType(entry))
            except ValueError as exc:
                raise ValueError(
                    f"Unknown HD-map channel '{entry}'. "
                    f"Valid: {sorted(valid_values)}"
                ) from exc
        else:
            raise TypeError(
                "channels entries must be MapElementType or str, "
                f"got {type(entry).__name__}"
            )
    return out


class HDMapBEVAdapter(AbstractAdapter):
    """Rasterizes ``StandardFrameData.hd_map`` to a multi-channel BEV image.

    One channel per element type listed in ``channels`` (default: every member
    of :class:`MapElementType` in declaration order). Polygons are drawn
    filled; polylines are drawn with ``polyline_thickness``; points
    (single-row ``MapElement.points``) are drawn as filled circles of radius
    ``polyline_thickness``. Output is ``np.float32`` in ``[0, 1]``.

    Axis convention matches :class:`LidarBEVAdapter`: rows correspond to the
    vehicle x axis (forward), columns to the y axis (left). Output shape is
    ``(C, H, W)`` with ``C = len(channels)``,
    ``H = (max_x - min_x) * pixels_per_meter``,
    ``W = (max_y - min_y) * pixels_per_meter``.

    Args:
        min_x, max_x: BEV x extent in meters (vehicle x is forward).
        min_y, max_y: BEV y extent in meters (vehicle y is left).
        pixels_per_meter: grid resolution; cast to ``int`` internally.
        channels: ordered list of ``MapElementType`` (or their string ``value``
            for YAML configs) determining channel composition and order.
            Defaults to all enum members in declaration order. The resolved
            list is exposed via :attr:`metadata` so downstream consumers can
            interpret the BEV without re-deriving the channel order.
        polyline_thickness: line thickness in pixels for polyline elements;
            also the radius of point-element circles.
    """

    def __init__(
        self,
        min_x: float = -32.0,
        max_x: float = 32.0,
        min_y: float = -32.0,
        max_y: float = 32.0,
        pixels_per_meter: float = 4.0,
        channels: Sequence[ChannelSpec] | None = None,
        polyline_thickness: int = 1,
    ) -> None:
        super().__init__()
        if max_x <= min_x or max_y <= min_y:
            raise ValueError("max_{x,y} must be greater than min_{x,y}")
        if int(pixels_per_meter) < 1:
            raise ValueError("pixels_per_meter must be >= 1 after int cast")
        if polyline_thickness < 1:
            raise ValueError("polyline_thickness must be a positive integer")
        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y
        self._pixels_per_meter = pixels_per_meter
        self._channels = _resolve_channels(channels)
        self._type_to_channel = {t: i for i, t in enumerate(self._channels)}
        self._polyline_thickness = polyline_thickness

    @property
    def name(self) -> str:
        return "HDMapBEVAdapter"

    @property
    def consumes_attrs(self) -> set[str]:
        return {"hd_map"}

    @property
    def channels(self) -> list[MapElementType]:
        """Resolved ordered channel list (read-only view)."""
        return list(self._channels)

    @property
    def metadata(self) -> dict[str, Any]:
        """Expose the BEV channel order so the .npz remains self-describing."""
        return {HD_MAP_BEV_CHANNELS_AUX_KEY: [t.value for t in self._channels]}

    @property
    def output_shape(self) -> tuple[int, int, int]:
        ppm = int(self._pixels_per_meter)
        h = int((self._max_x - self._min_x) * ppm)
        w = int((self._max_y - self._min_y) * ppm)
        return (len(self._channels), h, w)

    def _world_to_pixel(self, points_xy: np.ndarray) -> np.ndarray:
        """Map (N, 2) vehicle-frame xy to (N, 2) cv2 pixel coords (col, row).

        cv2 expects each point as ``(x_pixel, y_pixel)`` where ``x_pixel`` is
        the column index and ``y_pixel`` is the row index. We map vehicle x
        (forward) to row and vehicle y (left) to column to match the
        :class:`LidarBEVAdapter` axis convention.
        """
        ppm = int(self._pixels_per_meter)
        col = ((points_xy[:, 1] - self._min_y) * ppm).astype(np.int32)
        row = ((points_xy[:, 0] - self._min_x) * ppm).astype(np.int32)
        return np.stack([col, row], axis=1)

    def _transform(self, standard_frame_data: StandardFrameData) -> dict[Modality, Any]:
        if standard_frame_data.hd_map is None:
            return {}
        c, h, w = self.output_shape
        canvas = np.zeros((c, h, w), dtype=np.uint8)
        for element in standard_frame_data.hd_map.elements:
            channel = self._type_to_channel.get(element.type)
            if channel is None:
                continue
            pts = self._world_to_pixel(element.points[:, :2])
            if element.is_closed:
                cv2.fillPoly(canvas[channel], [pts], color=255, lineType=cv2.LINE_8)
            elif pts.shape[0] == 1:
                cv2.circle(
                    canvas[channel],
                    (int(pts[0, 0]), int(pts[0, 1])),
                    radius=self._polyline_thickness,
                    color=255,
                    thickness=-1,
                )
            else:
                cv2.polylines(
                    canvas[channel],
                    [pts],
                    isClosed=False,
                    color=255,
                    thickness=self._polyline_thickness,
                    lineType=cv2.LINE_8,
                )
        return {Modality.HD_MAP_BEV: canvas.astype(np.float32) / 255.0}
