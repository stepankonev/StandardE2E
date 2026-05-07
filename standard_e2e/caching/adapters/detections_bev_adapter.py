from typing import Any, Sequence, Union

import cv2
import numpy as np

from standard_e2e.caching.adapters.abstract_adapter import AbstractAdapter
from standard_e2e.data_structures import StandardFrameData
from standard_e2e.enums import DetectionType, Modality
from standard_e2e.enums import TrajectoryComponent as TC

DETECTIONS_3D_BEV_CHANNELS_AUX_KEY = "detections_3d_bev_channels"

ClassSpec = Union[DetectionType, str]


def _resolve_classes(
    classes: Sequence[ClassSpec] | None,
) -> list[DetectionType]:
    """Coerce a YAML/string-friendly class list to ``list[DetectionType]``.

    ``None`` → all members in declaration order. String entries must match a
    member ``value`` (e.g. ``"vehicle"``); unknown strings raise ``ValueError``
    listing valid options so misconfigurations fail fast at adapter
    construction rather than silently producing empty channels.
    """
    if classes is None:
        return list(DetectionType)
    out: list[DetectionType] = []
    valid_values = {t.value for t in DetectionType}
    for entry in classes:
        if isinstance(entry, DetectionType):
            out.append(entry)
        elif isinstance(entry, str):
            try:
                out.append(DetectionType(entry))
            except ValueError as exc:
                raise ValueError(
                    f"Unknown detection class '{entry}'. "
                    f"Valid: {sorted(valid_values)}"
                ) from exc
        else:
            raise TypeError(
                "classes entries must be DetectionType or str, "
                f"got {type(entry).__name__}"
            )
    return out


class Detections3DBEVAdapter(AbstractAdapter):
    """Rasterizes ``StandardFrameData.frame_detections_3d`` to a multi-channel
    BEV image. One channel per element of ``classes`` (default: every member
    of :class:`DetectionType` in declaration order); each detection is drawn
    as a filled oriented rectangle (length × width) at the box centroid,
    rotated by the detection's heading. Output is ``np.float32`` in
    ``[0, 1]`` (0 outside boxes, 1 inside).

    Axis convention matches :class:`HDMapBEVAdapter` and
    :class:`LidarBEVAdapter`: rows correspond to the vehicle x axis (forward),
    columns to the y axis (left). Output shape is ``(C, H, W)`` with
    ``C = len(classes)``,
    ``H = (max_x - min_x) * pixels_per_meter``,
    ``W = (max_y - min_y) * pixels_per_meter``.

    Args:
        min_x, max_x: BEV x extent in meters (vehicle x is forward).
        min_y, max_y: BEV y extent in meters (vehicle y is left).
        pixels_per_meter: grid resolution; cast to ``int`` internally.
        classes: ordered list of ``DetectionType`` (or their string ``value``
            for YAML configs) determining channel composition and order.
            Defaults to all enum members in declaration order. The resolved
            list is exposed via :attr:`metadata` so downstream consumers can
            interpret the BEV without re-deriving the channel order.
    """

    def __init__(
        self,
        min_x: float = -32.0,
        max_x: float = 32.0,
        min_y: float = -32.0,
        max_y: float = 32.0,
        pixels_per_meter: float = 4.0,
        classes: Sequence[ClassSpec] | None = None,
    ) -> None:
        super().__init__()
        if max_x <= min_x or max_y <= min_y:
            raise ValueError("max_{x,y} must be greater than min_{x,y}")
        if int(pixels_per_meter) < 1:
            raise ValueError("pixels_per_meter must be >= 1 after int cast")
        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y
        self._pixels_per_meter = pixels_per_meter
        self._classes = _resolve_classes(classes)
        self._type_to_channel = {t: i for i, t in enumerate(self._classes)}

    @property
    def name(self) -> str:
        return "Detections3DBEVAdapter"

    @property
    def classes(self) -> list[DetectionType]:
        """Resolved ordered class list (read-only view)."""
        return list(self._classes)

    @property
    def metadata(self) -> dict[str, Any]:
        """Expose the BEV channel order so the .npz remains self-describing."""
        return {DETECTIONS_3D_BEV_CHANNELS_AUX_KEY: [t.value for t in self._classes]}

    @property
    def output_shape(self) -> tuple[int, int, int]:
        ppm = int(self._pixels_per_meter)
        h = int((self._max_x - self._min_x) * ppm)
        w = int((self._max_y - self._min_y) * ppm)
        return (len(self._classes), h, w)

    def _world_to_pixel(self, points_xy: np.ndarray) -> np.ndarray:
        """Map (N, 2) vehicle-frame xy to (N, 2) cv2 pixel coords (col, row).

        Mirrors :meth:`HDMapBEVAdapter._world_to_pixel`: vehicle x (forward)
        maps to row, vehicle y (left) maps to column.
        """
        ppm = int(self._pixels_per_meter)
        col = ((points_xy[:, 1] - self._min_y) * ppm).astype(np.int32)
        row = ((points_xy[:, 0] - self._min_x) * ppm).astype(np.int32)
        return np.stack([col, row], axis=1)

    @staticmethod
    def _box_corners(
        cx: float, cy: float, heading: float, length: float, width: float
    ) -> np.ndarray:
        """Compute the four BEV-plane corners of an oriented box.

        Returns ``(4, 2)`` float64 in vehicle-frame xy order: front-left,
        front-right, rear-right, rear-left. ``heading`` is in radians measured
        from the vehicle x axis (CCW positive); ``length`` aligns with the
        box's local x axis and ``width`` with the local y axis.
        """
        half_l = 0.5 * length
        half_w = 0.5 * width
        local = np.array(
            [
                [+half_l, +half_w],  # front-left
                [+half_l, -half_w],  # front-right
                [-half_l, -half_w],  # rear-right
                [-half_l, +half_w],  # rear-left
            ],
            dtype=np.float64,
        )
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float64)
        return (local @ rot.T) + np.array([cx, cy], dtype=np.float64)

    def _transform(self, standard_frame_data: StandardFrameData) -> dict[Modality, Any]:
        if standard_frame_data.frame_detections_3d is None:
            return {}
        c, h, w = self.output_shape
        canvas = np.zeros((c, h, w), dtype=np.uint8)
        detections = standard_frame_data.frame_detections_3d.detections
        for det in detections:
            channel = self._type_to_channel.get(det.detection_type)
            if channel is None:
                continue
            box = det.trajectory.get(
                [TC.X, TC.Y, TC.HEADING, TC.LENGTH, TC.WIDTH], strict=False
            )
            if box is None or len(box) == 0:
                continue
            # Use the current-frame row (index 0). Aggregator-emitted future
            # detections also place the closest-to-current sample at row 0.
            cx, cy, heading, length, width = (float(v) for v in box[0])
            if length <= 0.0 or width <= 0.0:
                continue
            corners_xy = self._box_corners(cx, cy, heading, length, width)
            pts = self._world_to_pixel(corners_xy)
            cv2.fillPoly(canvas[channel], [pts], color=255, lineType=cv2.LINE_8)
        return {Modality.DETECTIONS_3D_BEV: canvas.astype(np.float32) / 255.0}
