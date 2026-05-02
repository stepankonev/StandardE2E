from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from standard_e2e.enums import (
    CameraDirection,
    DetectionType,
    LidarComponent,
    MapElementType,
)
from standard_e2e.enums import TrajectoryComponent as TC

from .trajectory_data import BatchedTrajectory, Trajectory


class CameraData(BaseModel):
    """Camera sample containing image + calibration matrices.

    Validation rules:
    - intrinsics (K): shape (3,3) float32
    - extrinsics (T): shape (4,4) float32
    - distortion (D): optional, 1D float32
        * Accepts 3 -> expands to [k1,k2,0,0,k3] (Brown-Conrady)
        * Accepts 5 -> [k1,k2,p1,p2,k3] (Brown-Conrady)
        * Accepts 4 -> [k1,k2,k3,k4] (fisheye, kept as-is)
    - image: HxWxC uint8 (ndim==3)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    camera_direction: CameraDirection
    image: NDArray[np.uint8]
    intrinsics: NDArray[np.float32]  # K
    extrinsics: NDArray[np.float32]  # T
    distortion: Optional[NDArray[np.float32]] = None  # D
    # Optional explicit (H, W) tuple; inferred from image if omitted.
    size: Optional[tuple[int, int]] = None
    is_fisheye: bool = False

    # --- Coercion validators (before) ---
    @field_validator("intrinsics", mode="before")
    @classmethod
    def _coerce_intrinsics(cls, v):
        return np.asarray(v, dtype=np.float32)

    @field_validator("extrinsics", mode="before")
    @classmethod
    def _coerce_extrinsics(cls, v):
        return np.asarray(v, dtype=np.float32)

    @field_validator("distortion", mode="before")
    @classmethod
    def _coerce_distortion(cls, v):
        if v is None:
            return None
        base: NDArray[np.float32] = np.asarray(v, dtype=np.float32).reshape(-1)
        size = int(base.size)
        if size == 3:
            # Expand to Brown–Conrady 5-term with zeros for tangential terms
            return np.array([base[0], base[1], 0.0, 0.0, base[2]], dtype=np.float32)
        if size in (4, 5):
            return base  # already fisheye 4-term or BC 5-term
        raise ValueError("distortion must have 3, 4, or 5 elements")

    @field_validator("image", mode="before")
    @classmethod
    def _coerce_image(cls, v):
        return np.asarray(v)

    # --- Validation (after coercion) ---
    @field_validator("intrinsics")
    @classmethod
    def _validate_intrinsics(cls, v):
        if v.shape != (3, 3):
            raise ValueError(f"intrinsics must have shape (3,3); got {v.shape}")
        return v

    @field_validator("extrinsics")
    @classmethod
    def _validate_extrinsics(cls, v):
        if v.shape != (4, 4):
            raise ValueError(f"extrinsics must have shape (4,4); got {v.shape}")
        return v

    @field_validator("distortion")
    @classmethod
    def _validate_distortion(cls, v):
        if v is None:
            return v
        if v.ndim != 1:
            raise ValueError("distortion must be a 1D vector")
        if v.size not in (4, 5):
            raise ValueError(
                "distortion must have 4 (fisheye) or 5 (Brown–Conrady) elements"
            )
        return v

    @field_validator("image")
    @classmethod
    def _validate_image(cls, v):
        if v.ndim != 3:
            raise ValueError(
                f"image must be HxWxC (3 dims); got ndim={v.ndim}, shape={v.shape}"
            )
        if v.dtype != np.uint8:
            raise ValueError(f"image dtype must be uint8; got {v.dtype}")
        return v

    @model_validator(mode="after")
    def _infer_and_validate_dims(self):
        h, w, _ = self.image.shape
        if self.size is None:
            self.size = (int(h), int(w))
        else:
            if (
                not isinstance(self.size, tuple)
                or len(self.size) != 2
                or not all(isinstance(x, int) for x in self.size)
            ):
                raise ValueError("size must be a tuple (H, W) of ints")
            if self.size != (h, w):
                raise ValueError(
                    f"Provided size={self.size} does not match image size={(h, w)}"
                )
        return self

    # --- Convenience aliases ---
    @property
    def K(self) -> np.ndarray:  # intrinsics
        return self.intrinsics

    @property
    def T(self) -> np.ndarray:  # extrinsics
        return self.extrinsics

    @property
    def D(self) -> Optional[np.ndarray]:  # distortion vector
        return self.distortion

    # --- Dimension convenience ---
    @property
    def H(self) -> int:
        return int(self.size[0])  # type: ignore[index]

    @property
    def W(self) -> int:
        return int(self.size[1])  # type: ignore[index]

    # Backward convenience aliases (height/width) kept for transitional
    # compatibility in case earlier code on the branch referenced them.
    @property
    def height(self) -> int:
        return self.H

    @property
    def width(self) -> int:
        return self.W

    @property
    def shape(self) -> tuple[int, int, int]:
        h, w, c = self.image.shape
        return int(h), int(w), int(c)


class MapElement(BaseModel):
    """A single HD map element (polyline, polygon, or point) in vehicle frame.

    - ``points``: ``(N, 3)`` float32 array. ``N == 1`` for points (e.g.
      ``STOP_SIGN``); ``N >= 2`` for polylines / polygons.
    - ``is_closed``: True for polygons (last point connects to first); False
      for open polylines and points.
    - ``successor_ids`` / ``predecessor_ids``: lane-graph connectivity
      (empty for non-lane elements). Unused by the BEV rasterizer; kept on
      the schema for future vector-output adapters.
    - ``attrs``: dataset-specific per-element metadata (e.g. ``speed_limit``,
      ``mark_type``, ``lane_type``, ``is_intersection``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    id: str
    type: MapElementType
    points: NDArray[np.float32]
    is_closed: bool = False
    successor_ids: list[str] = Field(default_factory=list)
    predecessor_ids: list[str] = Field(default_factory=list)
    attrs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("points", mode="before")
    @classmethod
    def _coerce_points(cls, v):
        return np.asarray(v, dtype=np.float32)

    @field_validator("points")
    @classmethod
    def _validate_points(cls, v):
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"points must be (N, 3); got shape {v.shape}")
        if v.shape[0] == 0:
            raise ValueError("points must have at least one row")
        return v


class HDMap(BaseModel):
    """HD map snapshot in the vehicle frame at a frame's timestamp.

    Used as ``StandardFrameData.hd_map`` (in-memory during preprocessing
    only — not persisted to ``.npz``). Adapters such as ``HDMapBEVAdapter``
    consume it and emit modality-specific representations.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    elements: list[MapElement]


class LidarData(BaseModel):
    """Lidar point cloud container used in ``StandardFrameData``.

    - points: pandas DataFrame with mandatory columns matching ``LidarComponent``
        values (currently ``x``, ``y``, ``z``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    points: pd.DataFrame

    @field_validator("points")
    @classmethod
    def _validate_points(cls, v: pd.DataFrame):
        if not isinstance(v, pd.DataFrame):
            raise ValueError(f"points must be a pandas DataFrame; got {type(v)}")
        for component in LidarComponent:
            if component.value not in v.columns:
                raise ValueError(f"points must contain column '{component.value}'")
        return v


class LidarPointCloud:
    """Single lidar point cloud after the adapter (numpy-backed).

    - ``points``: ``(N, K)`` ``np.float32`` array.
    - ``components``: list of ``LidarComponent`` of length ``K``, in column order.
    """

    def __init__(
        self,
        points: np.ndarray,
        components: Sequence[LidarComponent],
    ) -> None:
        if not isinstance(points, np.ndarray):
            raise TypeError(f"points must be a numpy array, got {type(points)}")
        if points.ndim != 2:
            raise ValueError(f"points must be 2D (N, K); got shape {points.shape}")
        components_list = list(components)
        if not all(isinstance(c, LidarComponent) for c in components_list):
            raise TypeError("components must all be LidarComponent members")
        if len(set(components_list)) != len(components_list):
            raise ValueError("components must be unique")
        if points.shape[1] != len(components_list):
            raise ValueError(
                f"points has {points.shape[1]} columns but components has "
                f"{len(components_list)} entries"
            )
        self._points = points.astype(np.float32, copy=False)
        self._components = components_list

    @property
    def points(self) -> np.ndarray:
        return self._points

    @property
    def components(self) -> List[LidarComponent]:
        return list(self._components)

    @property
    def num_points(self) -> int:
        return int(self._points.shape[0])

    def get(
        self,
        components: Union[LidarComponent, Sequence[LidarComponent]],
    ) -> np.ndarray:
        """Return the requested component columns as a ``(N, K_req)`` array."""
        if isinstance(components, LidarComponent):
            requested = [components]
        else:
            requested = list(components)
            if not all(isinstance(c, LidarComponent) for c in requested):
                raise TypeError("components must all be LidarComponent members")
        missing = [c for c in requested if c not in self._components]
        if missing:
            available = ", ".join(c.name for c in self._components)
            needed = ", ".join(c.name for c in missing)
            raise KeyError(f"Missing component(s): {needed}. Available: [{available}]")
        idx = [self._components.index(c) for c in requested]
        return self._points[:, idx]

    def __len__(self) -> int:
        return self.num_points

    def __repr__(self) -> str:
        comps = ",".join(c.name for c in self._components)
        return f"LidarPointCloud(N={self.num_points}, components=[{comps}])"


class BatchedLidarPointCloud:
    """Batched lidar point clouds in concat-with-batch-index format.

    Single concatenated tensor of shape ``(sum_N, K)`` with a parallel
    ``batch_idx`` tensor of shape ``(sum_N,)`` mapping each point back to its
    sample.

    All inputs must share the same ``components`` list (validated).
    """

    def __init__(
        self,
        point_clouds: Sequence[LidarPointCloud],
        device: Optional[torch.device] = None,
    ) -> None:
        if not point_clouds:
            raise ValueError(
                "BatchedLidarPointCloud requires a non-empty list of LidarPointCloud."
            )
        if not all(isinstance(pc, LidarPointCloud) for pc in point_clouds):
            raise TypeError("all entries must be LidarPointCloud instances")
        first_components = point_clouds[0].components
        for i, pc in enumerate(point_clouds):
            if pc.components != first_components:
                raise ValueError(
                    f"sample {i} has components {[c.name for c in pc.components]}, "
                    f"expected {[c.name for c in first_components]}"
                )
        self._device = device or torch.device("cpu")
        self._components = first_components
        self._batch_size = len(point_clouds)
        sizes = [pc.num_points for pc in point_clouds]
        if sum(sizes) == 0:
            self._points = torch.zeros(
                (0, len(self._components)),
                dtype=torch.float32,
                device=self._device,
            )
            self._batch_idx = torch.zeros((0,), dtype=torch.int64, device=self._device)
        else:
            self._points = torch.from_numpy(
                np.concatenate([pc.points for pc in point_clouds], axis=0)
            ).to(device=self._device)
            self._batch_idx = torch.cat(
                [
                    torch.full((n,), i, dtype=torch.int64, device=self._device)
                    for i, n in enumerate(sizes)
                ]
            )

    @property
    def points(self) -> torch.Tensor:
        return self._points

    @property
    def batch_idx(self) -> torch.Tensor:
        return self._batch_idx

    @property
    def components(self) -> List[LidarComponent]:
        return list(self._components)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    def get(
        self,
        components: Union[LidarComponent, Sequence[LidarComponent]],
    ) -> torch.Tensor:
        """Return the requested component columns as a ``(sum_N, K_req)`` tensor."""
        if isinstance(components, LidarComponent):
            requested = [components]
        else:
            requested = list(components)
            if not all(isinstance(c, LidarComponent) for c in requested):
                raise TypeError("components must all be LidarComponent members")
        missing = [c for c in requested if c not in self._components]
        if missing:
            available = ", ".join(c.name for c in self._components)
            needed = ", ".join(c.name for c in missing)
            raise KeyError(f"Missing component(s): {needed}. Available: [{available}]")
        idx = [self._components.index(c) for c in requested]
        return self._points[:, idx]

    def to(self, device: torch.device) -> "BatchedLidarPointCloud":
        """Move points and batch index to ``device`` (in-place)."""
        if device == self._device:
            return self
        self._points = self._points.to(device=device, non_blocking=True)
        self._batch_idx = self._batch_idx.to(device=device, non_blocking=True)
        self._device = device
        return self

    def cuda(self, device: Optional[int] = None) -> "BatchedLidarPointCloud":
        dev = torch.device(f"cuda:{device}" if device is not None else "cuda")
        return self.to(dev)

    def __repr__(self) -> str:
        comps = ",".join(c.name for c in self._components)
        return (
            f"BatchedLidarPointCloud(batch_size={self._batch_size}, "
            f"sum_N={int(self._points.shape[0])}, components=[{comps}], "
            f"device={self._device})"
        )


class Detection3D(BaseModel):
    """Holds 3D detections"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    unique_agent_id: str
    detection_type: DetectionType
    trajectory: Trajectory


class FrameDetections3D(BaseModel):
    """Holds 3D detections for a single frame"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    detections: list[Detection3D]


class BatchedFrameDetections3D:
    """Holds 3D detections for multiple frames"""

    _trajectory_components = [
        TC.TIMESTAMP,
        TC.X,
        TC.Y,
        TC.Z,
        TC.HEADING,
        TC.LENGTH,
        TC.WIDTH,
        TC.HEIGHT,
    ]

    def __init__(self, frames_detections: list[FrameDetections3D]):
        """Batch detections across frames and expose trajectory tensors.

        Args:
            frames_detections: Sequence of per-frame detection containers.
        """
        if not isinstance(frames_detections, list) or not all(
            isinstance(fd, FrameDetections3D) for fd in frames_detections
        ):
            raise TypeError("frames_detections must be a sequence of FrameDetections3D")
        self._batched_detections = frames_detections
        self._batched_trajectories = []
        for frame_detections in frames_detections:
            frame_detections_trajectories = [
                detection.trajectory for detection in frame_detections.detections
            ]
            self._batched_trajectories.append(
                BatchedTrajectory(frame_detections_trajectories)
            )
        self._batched_trajectories_tensors = [
            td.get(self._trajectory_components) for td in self._batched_trajectories
        ]
        self._detection_types = []
        self._unique_agent_ids = []

        for frame_detections in frames_detections:
            self._detection_types.append(
                [detection.detection_type for detection in frame_detections.detections]
            )
            self._unique_agent_ids.append(
                [detection.unique_agent_id for detection in frame_detections.detections]
            )

    @property
    def trajectory_components(self) -> list[TC]:
        """Trajectory components preserved in the batched tensors."""
        return self._trajectory_components

    @property
    def detection_types(self):
        """Detection types for the first frame (assumed consistent across batch)."""
        return self._detection_types[0]

    @property
    def unique_agent_ids(self):
        """Unique agent identifiers for the first frame."""
        return self._unique_agent_ids[0]

    @property
    def trajectories(self):
        """Tensor view of trajectories for the first frame in the batch."""
        return self._batched_trajectories_tensors[0]
