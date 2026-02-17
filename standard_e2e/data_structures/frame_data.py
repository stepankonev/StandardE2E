import logging
import os
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from torch.utils.data._utils.collate import collate as _torch_collate
from torch.utils.data._utils.collate import default_collate_fn_map

from standard_e2e.constants import INDEX_FILE_NAME
from standard_e2e.data_structures.containers import (
    BatchedFrameDetections3D,
    CameraData,
    FrameDetections3D,
    LidarData,
)
from standard_e2e.data_structures.trajectory_data import BatchedTrajectory, Trajectory
from standard_e2e.dataset_utils.modality_defaults import ModalityDefaults
from standard_e2e.enums import (
    CameraDirection,
    Intent,
    Modality,
)
from standard_e2e.enums import TrajectoryComponent as TC


class StandardFrameData(BaseModel):
    """
    Represents a single frame data in intermediate standardized format:
    Raw frame data -> StandardFrameData -> TransformedFrameData.

    Attributes:
        timestamp (float): Timestamp of the frame in seconds.
        frame_id (int): Unique identifier of the frame within a sequence.
        segment_id (str): Unique identifier of the segment this frame belongs to.
        dataset_name (str): Name of the dataset this frame belongs to.
        split (str): Dataset split (e.g., "train", "val", "test").
        global_position (Optional[Trajectory]):
            Pose for the ego entity at this frame in global coordinates.
        intent (Optional[Intent]):
            Predicted or annotated intent associated with the frame.
        cameras (dict[CameraDirection, CameraData]):
            Camera data keyed by camera direction.
        lidar (Optional[LidarData]): LiDAR data for the frame, if available.
        future_states (Optional[Trajectory]):
            Future trajectory states relative to this frame.
        past_states (Optional[Trajectory]):
            Past trajectory states leading up to this frame.
        hd_map (Any): High-definition map data associated with the frame. WIP
        frame_detections_3d (Optional[FrameDetections3D]):
            3D detections present in the frame.
        aux_data (Optional[Dict[str, Any]]): Additional auxiliary data.
        extra_index_data (Optional[Dict[str, Any]]): Extra indexing or lookup data.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    timestamp: float
    frame_id: int
    segment_id: str
    dataset_name: str
    split: str
    global_position: Optional[Trajectory] = None
    intent: Optional[Intent] = None
    cameras: dict[CameraDirection, CameraData] = Field(default_factory=dict)
    lidar: Optional[LidarData] = None
    future_states: Optional[Trajectory] = None
    past_states: Optional[Trajectory] = None
    hd_map: Any = None
    frame_detections_3d: Optional[FrameDetections3D] = None
    aux_data: Optional[Dict[str, Any]] = None
    extra_index_data: Optional[Dict[str, Any]] = None

    @field_validator("cameras")
    @classmethod
    def _validate_cameras(cls, v):
        # Ensure keys are CameraDirection
        if not isinstance(v, dict):
            raise TypeError("cameras must be a dict")
        for k, val in v.items():
            if not isinstance(k, CameraDirection):
                raise TypeError("cameras keys must be CameraDirection")
            if not isinstance(val, CameraData):
                raise TypeError("cameras values must be CameraData")
        return v


class FrameIndexData(BaseModel):
    """Frame index metadata (for parquet serialization)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    dataset_name: str
    segment_id: str
    frame_id: int | str
    timestamp: float
    split: str
    filename: str
    extra_index_data: dict | None = None

    def to_index_dict(self) -> dict:
        """Convert index metadata to a flat dictionary used for Parquet writes.

        Extra fields stored in ``extra_index_data`` are prefixed with ``extra_`` to
        keep backward compatibility with legacy index files.
        """

        d = self.model_dump()
        # Flatten extra_index_data to prefixed keys as before
        extra = d.pop("extra_index_data", None)
        if extra:
            for k, v in extra.items():
                d[f"extra_{k}"] = v
        return d

    @classmethod
    def save_index_data(
        cls, index_data: list["FrameIndexData"], output_path: str
    ) -> pd.DataFrame:
        """Persist a list of index entries to a sorted Parquet file.

        Args:
            index_data: Sequence of frame index records to serialize.
            output_path: Directory where the Parquet file will be written.

        Returns:
            pd.DataFrame: The sorted index dataframe that was written to disk.
        """

        df = pd.DataFrame([d.to_index_dict() for d in index_data])
        df.sort_values(by=["segment_id", "frame_id"], inplace=True)
        df.to_parquet(os.path.join(output_path, INDEX_FILE_NAME), index=False)
        logging.info(
            "Index data saved to %s", os.path.join(output_path, INDEX_FILE_NAME)
        )
        return df


def _to_device_recursive(x: Any, device: torch.device) -> Any:
    """Recursively move tensors/BatchedTrajectory (and nested containers) to device."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, non_blocking=True)
    if isinstance(x, BatchedTrajectory):
        x.to(device)  # in-place; returns self
        return x
    if isinstance(x, dict):
        return {k: _to_device_recursive(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_device_recursive(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_device_recursive(v, device) for v in x)
    return x


class TransformedFrameData(BaseModel):
    """Represents a single frame data with associated metadata, transformed by Adapters.
    A finalized, training-ready structure, loaded by Dataset:
    Raw frame data -> StandardFrameData -> TransformedFrameData.

    Attributes:
        dataset_name (str): Name of the dataset containing this frame.
        segment_id (str): Identifier of the sequence/segment this frame belongs to.
        frame_id (int): Unique identifier of the frame within the segment.
        timestamp (float): Timestamp of the frame in seconds.
        split (str): Dataset split (e.g., train/val/test) for this frame.
        global_position (Trajectory | None): World-frame position data; defaults
            to a zeroed position if not provided.
        filename (str | None): Auto-generated file name of the frame npz; computed
            as ``{dataset_name}/{split}/{segment_id}_{frame_id}.npz`` when missing.
        aux_data (dict[str, Any] | None): Optional auxiliary metadata.
        extra_index_data (dict[str, Any] | None): Optional extra indexing metadata.
        timestamp_diff (float | None): Optional time delta to adjacent frames.
        modality_defaults (dict[Modality, ModalityDefaults] | None): Optional
            default handlers used to normalize modality data.

    Private Attributes:
        _modality_data (dict[Modality, Any]): Raw modality-specific payloads,
            stored privately for compatibility with legacy code.
        Use get_modality_data() to access.

    Methods:
        set_modality_data(modality, data): Store payload for a given modality.
        get_modality_data(modality, set_default=True): Retrieve payload for a
            modality, optionally normalizing via its default handler.
        get_present_modality_keys(): List the modalities currently stored.
        to_npz(path): Serialize the frame (including modality data) to a compressed
            ``.npz`` file.
        from_npz(path, required_modalities=None): Loads a frame from ``.npz``,
            optionally ensuring required modalities exist (inserting None) and
            applying defaults.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    dataset_name: str
    segment_id: str
    frame_id: int
    timestamp: float
    split: str
    global_position: Optional[Trajectory] = None
    filename: str | None = None  # auto-filled
    # Private so Pydantic doesn't complain about leading underscore but we can
    # still expose same attribute name for existing code. We'll handle manual
    # (de)serialization.
    _modality_data: Dict[Modality, Any] = PrivateAttr(default_factory=dict)
    aux_data: Optional[Dict[str, Any]] = None
    extra_index_data: Optional[Dict[str, Any]] = None
    timestamp_diff: Optional[float] = None
    modality_defaults: dict[Modality, ModalityDefaults] | None = None

    def __init__(self, **data: Any):
        # Allow construction with _modality_data while keeping it private.
        modality_data = data.pop("_modality_data", None)
        super().__init__(**data)
        if modality_data is not None:
            # Assign after validation (no validation enforced on dict contents here)
            self._modality_data.update(modality_data)

    @model_validator(mode="after")
    def _set_filename(self):
        if not self.filename:
            self.filename = os.path.join(
                self.dataset_name, self.split, f"{self.segment_id}_{self.frame_id}.npz"
            )
        return self

    @model_validator(mode="after")
    def _set_default_global_position(self):
        if not self.global_position:
            self.global_position = Trajectory(
                {
                    TC.TIMESTAMP: [self.timestamp],
                    TC.X: [0],
                    TC.Y: [0],
                    TC.Z: [0],
                    TC.HEADING: [0],
                    TC.IS_VALID: [0],
                }
            )

        return self

    def set_modality_data(self, modality: Modality, data: Any):
        """Store data associated with the given modality.

        Args:
            modality (Modality): The modality identifier used as the key.
            data (Any): The data object to associate with the modality.
        """
        self._modality_data[modality] = data

    def get_modality_data(self, modality: Modality, set_default: bool = True) -> Any:
        """
        Retrieve modality-specific data from the stored modality map.

        Args:
            modality (Modality): The modality key used to locate the associated data.
            set_default (bool, optional): If True, normalizes the raw modality data
                using the configured default handler for the given modality when
                available. If False, returns the raw modality value without
                normalization. Defaults to True.

        Returns:
            Any: The raw or normalized modality data, or None if the modality key
            is not present.
        """
        modality_raw_value = self._modality_data.get(modality)
        modality_default_handler = (
            self.modality_defaults.get(modality) if self.modality_defaults else None
        )
        if not set_default:
            return modality_raw_value
        return (
            modality_default_handler.normalize(modality_raw_value, modality)
            if modality_default_handler
            else modality_raw_value
        )

    def get_present_modality_keys(self) -> List[Modality]:
        """Return the list of modality keys currently present in the frame data.

        Returns:
            List[Modality]: The modalities for which data is stored.
        """
        return list(self._modality_data.keys())

    def to_npz(self, path: str):
        """Serialize the frame data to a compressed NPZ file.

        Args:
            path: Destination file path for the NPZ archive.
        """
        payload = self.model_dump()
        # Manually inject private attr for backward compatibility
        payload["_modality_data"] = self._modality_data
        # Avoid persisting modality_defaults object reference (non-serializable)
        payload.pop("modality_defaults", None)
        payload.pop("extra_index_data", None)
        np.savez_compressed(path, **payload)

    @classmethod
    def from_npz(
        cls, path: str, required_modalities: list[Modality] | None = None
    ) -> "TransformedFrameData":
        """Load the frame data from a .npz file."""
        data = np.load(path, allow_pickle=True)
        instance = cls(
            dataset_name=data["dataset_name"].item(),
            segment_id=data["segment_id"].item(),
            frame_id=int(data["frame_id"].item()),
            timestamp=float(data["timestamp"].item()),
            split=data["split"].item(),
            global_position=data["global_position"].item(),
            aux_data=data.get("aux_data", np.array({})).item(),
            extra_index_data=data.get("extra_index_data", np.array({})).item(),
        )
        if "_modality_data" in data:
            instance._modality_data = data["_modality_data"].item()
        if required_modalities is None:
            return instance
        required_modalities = [Modality(m) for m in required_modalities]
        for required_modality in required_modalities:
            if required_modality not in instance._modality_data:
                instance.set_modality_data(required_modality, None)
        # remove unwanted modalities
        for modality in list(instance._modality_data.keys()):
            if modality not in required_modalities:
                del instance._modality_data[modality]
        return instance


def collate_trajectory_fn(
    batch,
    *,
    # pylint: disable=unused-argument
    collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None,
):
    """Collate a batch of ``Trajectory`` instances into a ``BatchedTrajectory``."""

    return BatchedTrajectory(batch)


def collate_frame_detections_fn(
    batch,
    *,
    # pylint: disable=unused-argument
    collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None,
):
    """Collate a batch of ``FrameDetections3D`` into ``BatchedFrameDetections3D``."""

    return BatchedFrameDetections3D(batch)


def collate_modalities(
    batch: List[Any], *, device: Optional[torch.device] = None
) -> Any:
    """
    Exactly like torch's default_collate, except leaves of type `Trajectory`
    are turned into a `BatchedTrajectory`. Everything else is native behavior.
    """
    device = device or torch.device("cpu")
    # Collate fn map must accept broader key types required by torch's internal typing
    extended_map: dict[Union[type, tuple[type, ...]], Callable[..., Any]] = {
        Trajectory: collate_trajectory_fn,
        FrameDetections3D: collate_frame_detections_fn,
    }
    # default_collate_fn_map has type compatibility, update after copy
    extended_map.update(default_collate_fn_map)
    return _torch_collate(batch, collate_fn_map=extended_map)


class TransformedFrameDataBatch:
    """
    Data structure to hold a batch of frame data (PyTorch-friendly).
    """

    dataset_name: list[str]
    segment_id: list[str]
    frame_id: list[int]
    timestamp: torch.Tensor
    split: list[str]
    filename: list[str]
    _modality_data: Dict[Modality, Any]
    aux_data: Optional[Dict[str, Any]] = None
    timestamp_diff: Optional[torch.Tensor] = None

    def __init__(
        self,
        frames: list[TransformedFrameData],
        *,
        device: Optional[torch.device] = None,
    ):
        """Create a batched view of multiple ``TransformedFrameData`` instances.

        Args:
            frames: Non-empty list of frames to batch.
            device: Optional device to place tensor-like fields on during
                initialization. Defaults to CPU when not provided.
        """

        if not frames:
            raise ValueError(
                "TransformedFrameDataBatch requires a non-empty \
                    list of TransformedFrameData."
            )
        device = device or torch.device("cpu")

        self.dataset_name = [frame.dataset_name for frame in frames]
        self.segment_id = [frame.segment_id for frame in frames]
        self.frame_id = [frame.frame_id for frame in frames]
        self.timestamp = torch.tensor(
            [frame.timestamp for frame in frames], dtype=torch.float32, device=device
        )
        self.split = [frame.split for frame in frames]
        self.filename = []
        for frame in frames:
            if frame.filename is None:
                raise ValueError("All frames must have a filename before batching.")
            self.filename.append(frame.filename)

        # Union of modality keys across the batch (keeps missing as None)
        all_modalities = sorted(
            set().union(*[set(f._modality_data.keys()) for f in frames]),
            key=lambda m: m.name,
        )
        # Collate each modality across frames using PyTorch's
        # collate + Trajectory override
        self._modality_data = {
            modality: collate_modalities(
                [f.get_modality_data(modality) for f in frames],
                device=device,
            )
            for modality in all_modalities
        }

        # Aux data: if dict-like across frames,
        # you may also want to collate it with the same helper.
        # For now: copy the first if present.
        self.aux_data = (
            frames[0].aux_data if frames and frames[0].aux_data is not None else None
        )
        # timestamp_diff: keep None if any missing; otherwise stack to tensor
        if any(f.timestamp_diff is None for f in frames):
            self.timestamp_diff = None
        else:
            self.timestamp_diff = torch.tensor(
                [f.timestamp_diff for f in frames], dtype=torch.float32, device=device
            )

    def get_modality_data(self, modality: Modality) -> Any:
        """Get data for a specific modality."""
        return self._modality_data.get(modality)

    def cuda(self, device: Optional[int] = None):
        """Move batched tensors and nested modality payloads to a CUDA device."""

        dev = torch.device(f"cuda:{device}" if device is not None else "cuda")
        return self.to(dev)

    def to(self, device: torch.device):
        """Move all tensor-like fields to ``device`` (non-blocking where possible)."""

        self.timestamp = self.timestamp.to(device=device, non_blocking=True)
        if self.timestamp_diff is not None:
            self.timestamp_diff = self.timestamp_diff.to(
                device=device, non_blocking=True
            )
        self._modality_data = {
            k: _to_device_recursive(v, device) for k, v in self._modality_data.items()
        }
        return self
