from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from standard_e2e.enums import (
    CameraDirection,
    DetectionType,
    LidarComponent,
    MapElementType,
)

from .trajectory_data import Trajectory


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

    - ``points``: ``(N, 2)`` or ``(N, 3)`` float32 array. ``N == 1`` is
      allowed for point-like elements (e.g. ``STOP_SIGN``); ``N >= 2`` for
      polylines / polygons. Datasets that ship only XY (e.g. AV2, where
      vector Z is advisory and may be NaN outside the ground-height ROI)
      emit ``(N, 2)``; datasets with reliable 3D maps (Waymo) emit
      ``(N, 3)``. The BEV rasterizer slices ``[:, :2]`` and works for both.
    - ``is_closed``: True for polygons (last point connects to first); False
      for open polylines and points.
    - ``successor_ids`` / ``predecessor_ids``: lane-graph connectivity
      (empty for non-lane elements). Unused by the BEV rasterizer; kept on
      the schema for future vector-output adapters.
    - ``left_neighbor_id`` / ``right_neighbor_id``: lateral lane neighbours
      (None for non-lane elements). Single ID per side; if a source dataset
      ships a list, take the first.
    - ``attrs``: dataset-specific per-element metadata. Standardised keys
      documented in ``docs/map_element_translation.md`` (e.g.
      ``lane_type``, ``is_intersection``, ``speed_limit_mph``,
      ``paint_color``, ``paint_pattern``, ``paint_subtype_raw``,
      ``road_edge_subtype``, ``tl_state_per_ts``, ``controlled_lane_id``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    id: str
    type: MapElementType
    points: NDArray[np.float32]
    is_closed: bool = False
    successor_ids: list[str] = Field(default_factory=list)
    predecessor_ids: list[str] = Field(default_factory=list)
    left_neighbor_id: Optional[str] = None
    right_neighbor_id: Optional[str] = None
    attrs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("points", mode="before")
    @classmethod
    def _coerce_points(cls, v):
        return np.asarray(v, dtype=np.float32)

    @field_validator("points")
    @classmethod
    def _validate_points(cls, v):
        if v.ndim != 2 or v.shape[1] not in (2, 3):
            raise ValueError(f"points must be (N, 2) or (N, 3); got shape {v.shape}")
        if v.shape[0] == 0:
            raise ValueError("points must have at least one row")
        if not np.isfinite(v).all():
            raise ValueError("points must be finite (no NaN/inf)")
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
