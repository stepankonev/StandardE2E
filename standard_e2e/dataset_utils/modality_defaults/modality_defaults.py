from abc import ABC, abstractmethod
from typing import Any, final

import numpy as np
import pandas as pd

from standard_e2e.data_structures.containers import (
    Detection3D,
    FrameDetections3D,
    HDMapData,
    LidarData,
)
from standard_e2e.data_structures.trajectory_data import Trajectory
from standard_e2e.enums import DetectionType, Intent, Modality
from standard_e2e.enums import TrajectoryComponent as TC


class ModalityDefaults(ABC):
    """
    Base class for modality-specific default value handling on the fly.
    """

    @final
    def normalize(self, raw_value: Any, modality: Modality) -> Any:
        """Normalize a raw modality payload using the subclass implementation.

        Args:
            raw_value: Raw modality payload (may be ``None``).
            modality: Modality to normalize; must be present in ``allowed_modalities``.

        Returns:
            Any: The normalized payload produced by ``_normalize``.

        Raises:
            ValueError: If ``modality`` is not a ``Modality`` enum member or is not
                allowed for this defaults handler.
        """
        if not isinstance(modality, Modality):
            raise ValueError("modality must be an instance of Modality enum")
        if modality not in self.allowed_modalities:
            raise ValueError(
                f"modality {modality} is not allowed for this defaults handler"
            )
        return self._normalize(raw_value, modality)

    @abstractmethod
    def _normalize(self, raw_value: Any, modality: Modality) -> Any: ...

    @property
    @abstractmethod
    def allowed_modalities(self) -> list[Modality]:
        """Return a list of allowed modalities for this defaults handler."""


class PreferredTrajectoryDefaults(ModalityDefaults):
    """Substitute missing preference trajectories with ``n`` empty trajectories."""

    def __init__(self, n: int = 3) -> None:
        if n <= 0:
            raise ValueError("n must be positive")
        self._n = int(n)

    def _normalize(self, raw_value: Any, modality: Modality) -> Any:
        if modality is Modality.PREFERENCE_TRAJECTORY:
            if raw_value is None or (
                isinstance(raw_value, (list, tuple)) and len(raw_value) == 0
            ):
                return [Trajectory() for _ in range(self._n)]
        return raw_value

    @property
    def allowed_modalities(self) -> list[Modality]:
        return [Modality.PREFERENCE_TRAJECTORY]


class IntentDefaults(ModalityDefaults):
    """Provide default handling for the ``INTENT`` modality (fallback UNKNOWN)."""

    def _normalize(self, raw_value: Any, modality: Modality) -> Any:
        if modality is Modality.INTENT:
            if raw_value is None:
                return Intent.UNKNOWN
        return raw_value

    @property
    def allowed_modalities(self) -> list[Modality]:
        return [Modality.INTENT]


class EmptyLidarDefaults(ModalityDefaults):
    """Substitute missing lidar with an empty point cloud.

    ``BatchedLidarData`` already pads ragged frames; an empty
    ``LidarData`` collates to a per-frame ``num_points = 0`` slot.

    The placeholder carries zero rows for *every* canonical optional
    column (``intensity``, ``range``, ``timestamp_ns``, ``laser_id``)
    with the right dtype. ``BatchedLidarData`` only stacks columns
    present in *every* frame of the batch, so a bare ``x/y/z``
    placeholder mixed with real frames would silently drop those
    optional columns from the whole batch.
    """

    def _normalize(self, raw_value: Any, modality: Modality) -> Any:
        if modality is Modality.LIDAR_PC and raw_value is None:
            return LidarData(
                points=pd.DataFrame(
                    {
                        "x": np.array([], dtype=np.float32),
                        "y": np.array([], dtype=np.float32),
                        "z": np.array([], dtype=np.float32),
                        "intensity": np.array([], dtype=np.float32),
                        "range": np.array([], dtype=np.float32),
                        "timestamp_ns": np.array([], dtype=np.int64),
                        "laser_id": np.array([], dtype=np.int8),
                    }
                )
            )
        return raw_value

    @property
    def allowed_modalities(self) -> list[Modality]:
        return [Modality.LIDAR_PC]


class EmptyHDMapDefaults(ModalityDefaults):
    """Substitute missing HD-map with an empty payload (all default lists)."""

    def _normalize(self, raw_value: Any, modality: Modality) -> Any:
        if modality is Modality.HD_MAP and raw_value is None:
            return HDMapData()
        return raw_value

    @property
    def allowed_modalities(self) -> list[Modality]:
        return [Modality.HD_MAP]


class EmptyDetectionsDefaults(ModalityDefaults):
    """Normalize the ``DETECTIONS_3D`` modality at load time.

    Two normalizations:

    1. ``None`` (modality missing on this frame) -> single zero-XYZ
       placeholder wrapped in ``FrameDetections3D``. The placeholder is
       needed because ``BatchedFrameDetections3D`` builds an internal
       ``BatchedTrajectory`` per frame and that container rejects an
       empty trajectory list. Downstream encoders can mask the
       placeholder via the agent id ``"__empty__"``.
    2. ``list[Detection3D]`` (the post-``FutureDetectionsAggregator``
       cached form) -> wrap into ``FrameDetections3D`` so the
       ``collate_modalities`` dispatch fires
       ``collate_frame_detections_fn`` and produces a
       ``BatchedFrameDetections3D``. Without this wrap the DataLoader
       would fall through to default ``list`` collate and crash trying
       to recurse into ``Detection3D`` Pydantic models.
    """

    PLACEHOLDER_AGENT_ID = "__empty__"

    def _normalize(self, raw_value: Any, modality: Modality) -> Any:
        if modality is not Modality.DETECTIONS_3D:
            return raw_value
        if raw_value is None:
            placeholder = Detection3D(
                unique_agent_id=self.PLACEHOLDER_AGENT_ID,
                detection_type=DetectionType.UNKNOWN,
                trajectory=Trajectory(
                    {
                        TC.TIMESTAMP: [0.0],
                        TC.X: [0.0],
                        TC.Y: [0.0],
                        TC.Z: [0.0],
                        TC.HEADING: [0.0],
                        TC.LENGTH: [0.0],
                        TC.WIDTH: [0.0],
                        TC.HEIGHT: [0.0],
                    }
                ),
            )
            return FrameDetections3D(detections=[placeholder])
        if isinstance(raw_value, list):
            # Cached perception form post-FutureDetectionsAggregator.
            # Wrap so collate_modalities dispatches by FrameDetections3D type.
            if not raw_value:
                # Empty post-aggregator list -> use the placeholder so
                # BatchedTrajectory does not reject an empty trajectory list.
                return self._normalize(None, modality)
            return FrameDetections3D(detections=raw_value)
        return raw_value

    @property
    def allowed_modalities(self) -> list[Modality]:
        return [Modality.DETECTIONS_3D]
