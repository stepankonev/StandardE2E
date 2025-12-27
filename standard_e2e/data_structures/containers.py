from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from standard_e2e.enums import CameraDirection, DetectionType
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


class LidarData(BaseModel):
    """Lidar point cloud container.

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
