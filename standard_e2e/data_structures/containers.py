from typing import Optional

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from standard_e2e.enums import (
    CameraDirection,
    DetectionType,
    LaneMarkType,
    LaneType,
    RoadEdgeType,
)
from standard_e2e.enums import TrajectoryComponent as TC

from .trajectory_data import BatchedTrajectory, Trajectory


class CameraData(BaseModel):
    """Camera sample containing image + calibration matrices.

    Coord-frame invariant:
    - extrinsics is T_ego_camera: a point p_cam in the camera frame maps to
      p_ego = extrinsics @ p_cam in the ego/vehicle frame at the frame's
      timestamp. Both Waymo (CameraCalibration.extrinsic) and AV2
      (ego_SE3_cam) use this direction; sources that publish the inverse
      must invert at parse time.

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


class BatchedCameraData:
    """Per-direction stacked camera tensors for a training batch.

    Internal shape ``dict[CameraDirection, Tensor]`` (NOT a single
    ``[B, N_cams, ...]`` stack) so:

    * Cameras present in only some frames stay missing keys.
    * Datasets with heterogeneous camera sets across directions (e.g. AV2
      stereo) batch without padding tricks.

    Per direction:
      * ``images``: ``Tensor[B, H, W, C]`` ``uint8`` (matches ``CameraData.image``).
      * ``intrinsics``: ``Tensor[B, 3, 3]`` ``float32``.
      * ``extrinsics``: ``Tensor[B, 4, 4]`` ``float32``.
    """

    def __init__(self, frames: list[dict[CameraDirection, CameraData]]):
        if not isinstance(frames, list):
            raise TypeError(
                "frames must be a list of dict[CameraDirection, CameraData]"
            )
        if not frames:
            raise ValueError("BatchedCameraData requires a non-empty list of frames")
        for frame in frames:
            if not isinstance(frame, dict):
                raise TypeError("each frame must be dict[CameraDirection, CameraData]")
            for k, v in frame.items():
                if not isinstance(k, CameraDirection):
                    raise TypeError("camera dict keys must be CameraDirection")
                if not isinstance(v, CameraData):
                    raise TypeError("camera dict values must be CameraData")
        self._batch_size = len(frames)
        # Only stack directions present in every frame; missing-in-any drops.
        common = set(frames[0].keys())
        for frame in frames[1:]:
            common &= set(frame.keys())
        self._directions: list[CameraDirection] = sorted(common, key=lambda d: d.value)
        self._images: dict[CameraDirection, torch.Tensor] = {}
        self._intrinsics: dict[CameraDirection, torch.Tensor] = {}
        self._extrinsics: dict[CameraDirection, torch.Tensor] = {}
        for direction in self._directions:
            images = np.stack([frame[direction].image for frame in frames], axis=0)
            intrinsics = np.stack(
                [frame[direction].intrinsics for frame in frames], axis=0
            )
            extrinsics = np.stack(
                [frame[direction].extrinsics for frame in frames], axis=0
            )
            self._images[direction] = torch.from_numpy(images)
            self._intrinsics[direction] = torch.from_numpy(intrinsics)
            self._extrinsics[direction] = torch.from_numpy(extrinsics)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def directions(self) -> list[CameraDirection]:
        return list(self._directions)

    @property
    def images(self) -> dict[CameraDirection, torch.Tensor]:
        return self._images

    @property
    def intrinsics(self) -> dict[CameraDirection, torch.Tensor]:
        return self._intrinsics

    @property
    def extrinsics(self) -> dict[CameraDirection, torch.Tensor]:
        return self._extrinsics

    def to(self, device: torch.device) -> "BatchedCameraData":
        for direction in self._directions:
            self._images[direction] = self._images[direction].to(
                device=device, non_blocking=True
            )
            self._intrinsics[direction] = self._intrinsics[direction].to(
                device=device, non_blocking=True
            )
            self._extrinsics[direction] = self._extrinsics[direction].to(
                device=device, non_blocking=True
            )
        return self


class LidarData(BaseModel):
    """Lidar point cloud container.

    Coord-frame invariant: points are in the ego/vehicle frame at the
    current frame's timestamp. Source-native frames (Waymo range-image
    sensor frame, AV2 lidar sensor frame) are lifted to ego at convert
    time via the corresponding laser/lidar extrinsic; consumers of
    Modality.LIDAR can rely on this without checking.

    - points: pandas DataFrame with mandatory columns x,y,z
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    points: pd.DataFrame

    @field_validator("points")
    @classmethod
    def _validate_points(cls, v: pd.DataFrame):
        if not isinstance(v, pd.DataFrame):
            raise ValueError(f"points must be a pandas DataFrame; got {type(v)}")
        for col in ("x", "y", "z"):
            if col not in v.columns:
                raise ValueError(f"points must contain column '{col}'")
        return v


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


class Lane(BaseModel):
    """One lane (per ADR 0006).

    Coord-frame is determined by the enclosing container (HDMapData = ego,
    RawSegmentHDMap = world).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    centerline: NDArray[np.float32]  # (N, 3) polyline
    left_boundary_id: str | None = None
    right_boundary_id: str | None = None
    predecessors: list[str] = []
    successors: list[str] = []
    is_intersection: bool = False
    lane_type: LaneType = LaneType.UNKNOWN

    @field_validator("centerline", mode="before")
    @classmethod
    def _coerce_centerline(cls, v):
        return np.asarray(v, dtype=np.float32)


class LaneBoundary(BaseModel):
    """A single lane-boundary polyline.

    ``source_boundary_id`` keeps the upstream identifier so consumers can
    re-deduplicate boundaries shared between adjacent lanes if they want
    to (Waymo's per-lane slicing makes the same physical boundary appear
    multiple times).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    polyline: NDArray[np.float32]  # (N, 3)
    boundary_type: LaneMarkType = LaneMarkType.UNKNOWN
    source_boundary_id: str | None = None

    @field_validator("polyline", mode="before")
    @classmethod
    def _coerce_polyline(cls, v):
        return np.asarray(v, dtype=np.float32)


class RoadEdge(BaseModel):
    """A road-edge polyline. Waymo-only at the source layer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    polyline: NDArray[np.float32]
    road_edge_type: RoadEdgeType = RoadEdgeType.UNKNOWN

    @field_validator("polyline", mode="before")
    @classmethod
    def _coerce_polyline(cls, v):
        return np.asarray(v, dtype=np.float32)


class Crosswalk(BaseModel):
    """Crosswalk polygon. AV2 stitches its two parallel edges into one
    polygon at parse time (per ADR 0006); Waymo provides one polygon
    natively."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    polygon: NDArray[np.float32]  # (M, 3)

    @field_validator("polygon", mode="before")
    @classmethod
    def _coerce_polygon(cls, v):
        return np.asarray(v, dtype=np.float32)


class StopSign(BaseModel):
    """A single stop sign. Waymo-only."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    position: NDArray[np.float32]  # (3,)
    lane_ids: list[str] = []

    @field_validator("position", mode="before")
    @classmethod
    def _coerce_position(cls, v):
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        if arr.size != 3:
            raise ValueError(f"position must have 3 components; got {arr.shape}")
        return arr


class SpeedBump(BaseModel):
    """Speed-bump polygon. Waymo-only."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    polygon: NDArray[np.float32]

    @field_validator("polygon", mode="before")
    @classmethod
    def _coerce_polygon(cls, v):
        return np.asarray(v, dtype=np.float32)


class DrivableArea(BaseModel):
    """Drivable-area polygon. AV2-only."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    polygon: NDArray[np.float32]

    @field_validator("polygon", mode="before")
    @classmethod
    def _coerce_polygon(cls, v):
        return np.asarray(v, dtype=np.float32)


class Driveway(BaseModel):
    """Driveway polygon. Waymo-only."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    polygon: NDArray[np.float32]

    @field_validator("polygon", mode="before")
    @classmethod
    def _coerce_polygon(cls, v):
        return np.asarray(v, dtype=np.float32)


class _HDMapFields(BaseModel):
    """Shared field set for HDMapData (ego) and RawSegmentHDMap (world).

    Defining the fields once keeps the two coord-frame variants in
    lockstep without runtime branching. Each variant subclasses this
    mixin and adds nothing — the *types are still distinct* so isinstance
    checks discriminate, which is the whole point of splitting them
    (ADR 0006 / 0007).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    lanes: list[Lane] = []
    lane_boundaries: list[LaneBoundary] = []
    road_edges: list[RoadEdge] = []
    crosswalks: list[Crosswalk] = []
    stop_signs: list[StopSign] = []
    speed_bumps: list[SpeedBump] = []
    drivable_areas: list[DrivableArea] = []
    driveways: list[Driveway] = []


class HDMapData(_HDMapFields):
    """Per-frame HD-map payload, **always in ego frame at the current
    frame's timestamp** (ADR 0006).

    This is what the cache stores under ``Modality.HD_MAP`` and what
    consumers receive. The world-frame intermediate is the distinct type
    ``RawSegmentHDMap``.
    """


class RawSegmentHDMap(_HDMapFields):
    """Segment-wide HD-map payload in **world frame**.

    Runtime-only Pydantic — never persisted (ADR 0007). Lives only inside
    a per-source aggregator's ``_process_segment`` call as the input to
    ``crop_hd_map_ego_relative`` (the single function that bridges world
    -> ego).
    """


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
