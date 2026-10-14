"""Batching of processed frames into PyTorch tensors.

:meth:`UnifiedE2EDataset.collate_fn
<standard_e2e.training.UnifiedE2EDataset.collate_fn>` turns a list of
``TransformedFrameData`` into a :class:`TransformedFrameDataBatch`: trajectories,
lidar point clouds, cameras and 3D detections become the batched containers
below (a frame missing one of them batches as its empty element), a modality
that is ``None`` in every frame stays ``None``, and every other leaf goes
through PyTorch's default collation.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data._utils.collate import collate as _torch_collate
from torch.utils.data._utils.collate import default_collate_fn_map

from standard_e2e.data_structures.containers import (
    CameraData,
    FrameDetections3D,
    LidarPointCloud,
)
from standard_e2e.data_structures.frame_data import TransformedFrameData
from standard_e2e.data_structures.trajectory_data import (
    Array1DNP,
    Trajectory,
    _validate_components_arg,
)
from standard_e2e.enums import CameraDirection, DetectionType, LidarComponent, Modality
from standard_e2e.enums import TrajectoryComponent
from standard_e2e.enums import TrajectoryComponent as TC

# ======================= BatchedTrajectory (Torch) =======================


class BatchedTrajectory:
    """
    Data container for a batch of trajectories.

    - __init__ accepts a NON-EMPTY list of Trajectory objects.
    - Empty trajectories inside the list are allowed and zero-filled.
    - strict=True: all **non-empty** trajectories must have the same \
        component set (IS_VALID always included).
      strict=False: union of components; per-sample missing components are zero-filled.
    - Stores per-component tensors float32 (batch_size, sequence_length) on one device.
    - Exposes:
        - get(..., strict=...) -> (batch_size, sequence_length, num_components)
        - scores: (batch_size,) float32 (NaN where missing)
        - is_empty_mask: (batch_size,) bool
        - .to(device) / .cuda()
    """

    def _replace_none_with_empty_trajs(
        self, trajectories: Sequence[Trajectory]
    ) -> Sequence[Trajectory]:
        """Replace None trajectories with empty ones."""
        return [t if t is not None else Trajectory() for t in trajectories]

    def __init__(
        self,
        trajectories: Sequence[Trajectory],
        device: Optional[torch.device] = None,
        side: str = "right",
        strict: bool = False,
    ) -> None:
        if not trajectories or len(trajectories) == 0:
            raise ValueError(
                "BatchedTrajectory requires a non-empty list of trajectories."
            )
        self._side = side
        self._strict = strict
        trajectories = self._replace_none_with_empty_trajs(trajectories)
        self._original_trajectories = trajectories
        self._device = device or torch.device("cpu")
        self._data: Dict[TrajectoryComponent, torch.Tensor] = {}

        # Meta: scores and emptiness
        self._scores = torch.tensor(
            [
                np.float32(t.score) if t.score is not None else np.float32("nan")
                for t in trajectories
            ],
            dtype=torch.float32,
            device=self._device,
        )
        self._is_empty_mask = torch.tensor(
            [t.isEmpty for t in trajectories], dtype=torch.bool, device=self._device
        )

        # sequence lengths
        sequence_lengths = [t.length for t in trajectories]
        max_length = max(sequence_lengths)

        # determine component set to use
        component_sets = [set(t.components()) for t in trajectories]
        non_empty_indices = [i for i, t in enumerate(trajectories) if not t.isEmpty]

        if strict:
            if non_empty_indices:
                expected_components = set(component_sets[non_empty_indices[0]]).union(
                    {TrajectoryComponent.IS_VALID}
                )
                for idx in non_empty_indices:
                    if (
                        component_sets[idx].union({TrajectoryComponent.IS_VALID})
                        != expected_components
                    ):
                        got = component_sets[idx].union({TrajectoryComponent.IS_VALID})
                        expected_components_list = [
                            c.name
                            for c in sorted(expected_components, key=lambda c: c.name)
                        ]
                        raise ValueError(
                            (
                                "Strict mode: component mismatch at sample "
                                f"{idx}. Expected: {expected_components_list}, got:"
                                f"{[c.name for c in sorted(got, key=lambda c: c.name)]}"
                            )
                        )
            else:
                expected_components = {TrajectoryComponent.IS_VALID}
            components_to_use = sorted(expected_components, key=lambda c: c.name)
        else:
            union_components = (
                set().union(*component_sets).union({TrajectoryComponent.IS_VALID})
            )
            components_to_use = sorted(union_components, key=lambda c: c.name)
        for trajectory in trajectories:
            trajectory.pad(max_length, side=side)
        for component in components_to_use:
            values = []
            for trajectory in trajectories:
                values.append(trajectory.get(component, strict=False))
            # stack all values for this component
            stacked = torch.tensor(
                np.stack(values, axis=0), dtype=torch.float32, device=self._device
            )
            self._data[component] = stacked

    # ---- basics ----
    @property
    def device(self) -> torch.device:
        """Return the device on which the trajectory data is stored."""
        return self._device

    @property
    def batch_size(self) -> int:
        """Return the number of trajectories in the batch."""
        return int(self._scores.shape[0])

    @property
    def length(self) -> int:
        """Return the length of the trajectories in the batch (automatically padded)."""
        return 0 if not self._data else int(next(iter(self._data.values())).shape[1])

    @property
    def scores(self) -> torch.Tensor:
        """Return the scores associated with each trajectory in the batch."""
        return self._scores

    @property
    def is_empty_mask(self) -> torch.Tensor:
        """Return a mask indicating which trajectories in the batch are empty."""
        return self._is_empty_mask

    def components(self) -> List[TrajectoryComponent]:
        """List the trajectory components currently stored."""
        return list(self._data.keys())

    def has(self, component: TrajectoryComponent) -> bool:
        """Check if a trajectory component is present."""
        return component in self._data

    # ---- get ----
    def get(
        self,
        components: Union[TrajectoryComponent, Sequence[TrajectoryComponent]],
        *,
        strict: bool = True,
    ) -> torch.Tensor:
        """Retrieve stacked component tensors of shape ``(B, T, K)``.

        Args:
            components: Single component or sequence thereof.
            strict: If True, missing components raise ``KeyError``; otherwise zeros
                are returned for missing entries.
        """
        components_list = _validate_components_arg(components)

        batch_size = self.batch_size
        sequence_length = self.length

        columns: List[torch.Tensor] = []
        missing_components: List[TrajectoryComponent] = []

        for component in components_list:
            if component in self._data:
                columns.append(
                    self._data[component]
                )  # (batch_size, sequence_length, 1)
            else:
                if strict:
                    missing_components.append(component)
                else:
                    columns.append(
                        torch.zeros(
                            (batch_size, sequence_length, 1),
                            dtype=torch.float32,
                            device=self.device,
                        )
                    )

        if missing_components:
            available = ", ".join(c.name for c in self._data)
            needed = ", ".join(c.name for c in missing_components)
            raise KeyError(f"Missing component(s): {needed}. Available: [{available}]")

        if not columns:
            return torch.zeros(
                (batch_size, sequence_length, 0),
                dtype=torch.float32,
                device=self.device,
            )
        return torch.cat(columns, dim=-1)

    # # ---- device moves ----
    # @classmethod
    # def _from_internal(
    #     cls,
    #     data: Dict[TrajectoryComponent, torch.Tensor],
    #     scores: torch.Tensor,
    #     empty_mask: torch.Tensor,
    #     device: torch.device,
    # ) -> "BatchedTrajectory":
    #     obj = cls.__new__(cls)  # bypass __init__
    #     obj._data = data
    #     obj._scores = scores
    #     obj._is_empty_mask = empty_mask
    #     obj._device = device
    #     return obj

    def to(self, device: Optional[torch.device] = None) -> "BatchedTrajectory":
        if device is None or device == self.device:
            return self
        for k, v in self._data.items():
            self._data[k] = v.to(device=device, non_blocking=True)
        self._scores = self._scores.to(device=device, non_blocking=True)
        self._is_empty_mask = self._is_empty_mask.to(device=device, non_blocking=True)
        self._device = device
        return self

    def cuda(self, device: Optional[int] = None) -> "BatchedTrajectory":
        dev = torch.device(f"cuda:{device}" if device is not None else "cuda")
        return self.to(dev)

    # ---- batch-level pad/trim ----
    def trim(self, target_length: int, *, side: str = "right") -> "BatchedTrajectory":
        """Trim all sequences in the batch to ``target_length`` in-place."""
        current_length = self.length
        if target_length >= current_length:
            return self
        cut = current_length - target_length
        self._data = {
            k: (v[:, cut:] if side == "left" else v[:, :-cut])
            for k, v in self._data.items()
        }
        return self

    def pad(self, target_length: int, *, side: str = "right") -> "BatchedTrajectory":
        """Pad all sequences in the batch to ``target_length`` with zeros."""
        current_length = self.length
        if target_length <= current_length:
            return self
        pad = target_length - current_length
        # padded: Dict[TrajectoryComponent, torch.Tensor] = {}
        for component, tensor in self._data.items():
            zeros = torch.zeros(
                (tensor.shape[0], pad, tensor.shape[2]),
                dtype=torch.float32,
                device=self.device,
            )
            self._data[component] = (
                torch.cat([zeros, tensor], dim=1)
                if side == "left"
                else torch.cat([tensor, zeros], dim=1)
            )
        return self

    def pad_or_trim(
        self, target_length: int, *, side: str = "right"
    ) -> "BatchedTrajectory":
        """Pad or trim batch to ``target_length`` depending on current length."""
        current_length = self.length
        return (
            self.pad(target_length, side=side)
            if target_length > current_length
            else self.trim(target_length, side=side)
        )

    # ---- resample ----
    def resample(self, time_lattice: Array1DNP) -> "BatchedTrajectory":
        """Return a new BatchedTrajectory resampled onto a new time lattice.

        This constructs (optionally cloned) copies of the original per-sample
        Trajectory objects, calls their ``Trajectory.resample`` method, and
        returns a freshly built BatchedTrajectory. The current object is not
        modified.

        Args:
            time_lattice: 1D array-like of new timestamps (passed to each
                underlying ``Trajectory.resample`` call).

        Returns:
            A new ``BatchedTrajectory`` instance on the same device.
        """
        new_trajs: List[Trajectory] = []
        for traj in self._original_trajectories:
            new_trajs.append(traj.resample(time_lattice))
        return BatchedTrajectory(
            new_trajs, device=self.device, side=self._side, strict=self._strict
        )

    def __repr__(self) -> str:
        shapes = ", ".join(f"{k.name}:{tuple(v.shape)}" for k, v in self._data.items())
        return (
            f"BatchedTrajectory(batch_size={self.batch_size}, \
                sequence_length={self.length}, device={self.device}, "
            f"components=[{shapes}], scores_shape={tuple(self._scores.shape)}, "
            f"empty_mask={self._is_empty_mask.tolist()})"
        )


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


class BatchedCameraData:
    """Batched ``CameraData`` of one camera direction.

    Each ``CameraData`` field gains a leading batch dimension ``B``. A sample
    without this camera (``None``) is zero-filled and flagged in
    ``is_empty_mask``.

    - ``image``: ``(B, H, W, C)`` uint8. Images are zero-padded at the bottom and
      right to the largest ``H`` and ``W`` in the batch, which keeps the
      intrinsics valid; ``size`` holds each sample's unpadded ``(H, W)``.
    - ``intrinsics``: ``(B, 3, 3)`` and ``extrinsics``: ``(B, 4, 4)`` float32.
    - ``distortion``: ``(B, D)`` float32, or ``None`` when no sample has one;
      samples without one are zero-filled. All distortions must have the same
      length (a 4-term fisheye and a 5-term Brown–Conrady model do not mix).
    - ``is_fisheye`` and ``is_empty_mask``: ``(B,)`` bool.
    """

    def __init__(
        self,
        cameras: Sequence[Optional[CameraData]],
        device: Optional[torch.device] = None,
    ) -> None:
        present = [camera for camera in cameras if camera is not None]
        if not present:
            raise ValueError("BatchedCameraData requires at least one CameraData.")
        if not all(isinstance(camera, CameraData) for camera in present):
            raise TypeError("all entries must be CameraData instances or None")
        channels = {camera.shape[2] for camera in present}
        if len(channels) != 1:
            raise ValueError(f"images have different channel counts: {channels}")
        distortion_sizes = {
            camera.distortion.size
            for camera in present
            if camera.distortion is not None
        }
        if len(distortion_sizes) > 1:
            raise ValueError(f"distortions have different lengths: {distortion_sizes}")
        self._device = device or torch.device("cpu")
        batch_size = len(cameras)
        height = max(camera.shape[0] for camera in present)
        width = max(camera.shape[1] for camera in present)
        image = np.zeros((batch_size, height, width, channels.pop()), dtype=np.uint8)
        size = np.zeros((batch_size, 2), dtype=np.int64)
        intrinsics = np.zeros((batch_size, 3, 3), dtype=np.float32)
        extrinsics = np.zeros((batch_size, 4, 4), dtype=np.float32)
        distortion = (
            np.zeros((batch_size, distortion_sizes.pop()), dtype=np.float32)
            if distortion_sizes
            else None
        )
        is_fisheye = np.zeros(batch_size, dtype=bool)
        for i, camera in enumerate(cameras):
            if camera is None:
                continue
            h, w, _ = camera.shape
            image[i, :h, :w] = camera.image
            size[i] = (h, w)
            intrinsics[i] = camera.intrinsics
            extrinsics[i] = camera.extrinsics
            if distortion is not None and camera.distortion is not None:
                distortion[i] = camera.distortion
            is_fisheye[i] = camera.is_fisheye
        self._image = torch.from_numpy(image).to(device=self._device)
        self._size = torch.from_numpy(size).to(device=self._device)
        self._intrinsics = torch.from_numpy(intrinsics).to(device=self._device)
        self._extrinsics = torch.from_numpy(extrinsics).to(device=self._device)
        self._distortion = (
            None
            if distortion is None
            else torch.from_numpy(distortion).to(device=self._device)
        )
        self._is_fisheye = torch.from_numpy(is_fisheye).to(device=self._device)
        self._is_empty_mask = torch.tensor(
            [camera is None for camera in cameras],
            dtype=torch.bool,
            device=self._device,
        )

    @property
    def image(self) -> torch.Tensor:
        return self._image

    @property
    def size(self) -> torch.Tensor:
        return self._size

    @property
    def intrinsics(self) -> torch.Tensor:
        return self._intrinsics

    @property
    def extrinsics(self) -> torch.Tensor:
        return self._extrinsics

    @property
    def distortion(self) -> Optional[torch.Tensor]:
        return self._distortion

    @property
    def is_fisheye(self) -> torch.Tensor:
        return self._is_fisheye

    @property
    def is_empty_mask(self) -> torch.Tensor:
        return self._is_empty_mask

    @property
    def batch_size(self) -> int:
        return int(self._image.shape[0])

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device) -> "BatchedCameraData":
        """Move every tensor to ``device`` (in-place)."""
        if device == self._device:
            return self
        self._image = self._image.to(device=device, non_blocking=True)
        self._size = self._size.to(device=device, non_blocking=True)
        self._intrinsics = self._intrinsics.to(device=device, non_blocking=True)
        self._extrinsics = self._extrinsics.to(device=device, non_blocking=True)
        if self._distortion is not None:
            self._distortion = self._distortion.to(device=device, non_blocking=True)
        self._is_fisheye = self._is_fisheye.to(device=device, non_blocking=True)
        self._is_empty_mask = self._is_empty_mask.to(device=device, non_blocking=True)
        self._device = device
        return self

    def cuda(self, device: Optional[int] = None) -> "BatchedCameraData":
        dev = torch.device(f"cuda:{device}" if device is not None else "cuda")
        return self.to(dev)

    def __repr__(self) -> str:
        return (
            f"BatchedCameraData(batch_size={self.batch_size}, "
            f"image_shape={tuple(self._image.shape)}, device={self._device})"
        )


class BatchedFrameDetections3D:
    """3D detections of a batch of frames in concat-with-batch-index format.

    Every detection of the batch is one row, as the points of a
    :class:`BatchedLidarPointCloud` are: ``batch_idx`` maps each row to its
    frame, and ``unique_agent_ids`` and ``detection_types`` are flat lists in
    row order. A frame without detections has no rows.

    ``trajectories`` is a ``(sum_N, T, K)`` tensor over ``trajectory_components``
    built from one :class:`BatchedTrajectory` of all detection trajectories:
    ``T`` is the longest trajectory in the batch, shorter ones are zero-padded
    on the right and ``is_valid`` is ``False`` on the padded steps. Components a
    detection does not carry are zero, so per-frame snapshots (all 8
    components) and the X/Y/HEADING-only tracks of ``FutureDetectionsAggregator``
    batch alike.
    """

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

    def __init__(
        self,
        frames_detections: list[FrameDetections3D],
        device: Optional[torch.device] = None,
    ):
        """Batch the detections of ``frames_detections``, one entry per frame."""
        if not isinstance(frames_detections, list) or not all(
            isinstance(fd, FrameDetections3D) for fd in frames_detections
        ):
            raise TypeError("frames_detections must be a sequence of FrameDetections3D")
        self._device = device or torch.device("cpu")
        self._batch_size = len(frames_detections)
        detections = [
            detection
            for frame_detections in frames_detections
            for detection in frame_detections.detections
        ]
        self._batch_idx = torch.tensor(
            [
                i
                for i, frame_detections in enumerate(frames_detections)
                for _ in frame_detections.detections
            ],
            dtype=torch.int64,
            device=self._device,
        )
        self._unique_agent_ids = [d.unique_agent_id for d in detections]
        self._detection_types = [d.detection_type for d in detections]
        num_components = len(self._trajectory_components)
        if detections:
            batched = BatchedTrajectory(
                [d.trajectory for d in detections], device=self._device
            )
            self._trajectories = batched.get(self._trajectory_components, strict=False)
            self._is_valid = batched.get(TC.IS_VALID)[..., 0].bool()
        else:
            self._trajectories = torch.zeros(
                (0, 0, num_components), dtype=torch.float32, device=self._device
            )
            self._is_valid = torch.zeros((0, 0), dtype=torch.bool, device=self._device)

    @property
    def trajectory_components(self) -> list[TC]:
        """Trajectory components, in the order of the last ``trajectories`` axis."""
        return self._trajectory_components

    @property
    def trajectories(self) -> torch.Tensor:
        """``(sum_N, T, K)`` float32 detection trajectories."""
        return self._trajectories

    @property
    def is_valid(self) -> torch.Tensor:
        """``(sum_N, T)`` bool; ``False`` on the zero-padded steps."""
        return self._is_valid

    @property
    def batch_idx(self) -> torch.Tensor:
        """``(sum_N,)`` int64 index of each detection's frame in the batch."""
        return self._batch_idx

    @property
    def unique_agent_ids(self) -> list[str]:
        """Agent identifier of each detection, in row order."""
        return self._unique_agent_ids

    @property
    def detection_types(self) -> list[DetectionType]:
        """Type of each detection, in row order."""
        return self._detection_types

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device) -> "BatchedFrameDetections3D":
        """Move every tensor to ``device`` (in-place)."""
        if device == self._device:
            return self
        self._trajectories = self._trajectories.to(device=device, non_blocking=True)
        self._is_valid = self._is_valid.to(device=device, non_blocking=True)
        self._batch_idx = self._batch_idx.to(device=device, non_blocking=True)
        self._device = device
        return self

    def cuda(self, device: Optional[int] = None) -> "BatchedFrameDetections3D":
        dev = torch.device(f"cuda:{device}" if device is not None else "cuda")
        return self.to(dev)

    def __repr__(self) -> str:
        return (
            f"BatchedFrameDetections3D(batch_size={self._batch_size}, "
            f"sum_N={int(self._trajectories.shape[0])}, "
            f"length={int(self._trajectories.shape[1])}, device={self._device})"
        )


def _to_device_recursive(x: Any, device: torch.device) -> Any:
    """Recursively move tensors/BatchedTrajectory (and nested containers) to device."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, non_blocking=True)
    if isinstance(x, BatchedTrajectory):
        x.to(device)  # in-place; returns self
        return x
    if isinstance(x, BatchedLidarPointCloud):
        x.to(device)  # in-place; returns self
        return x
    if isinstance(x, BatchedCameraData):
        x.to(device)  # in-place; returns self
        return x
    if isinstance(x, BatchedFrameDetections3D):
        x.to(device)  # in-place; returns self
        return x
    if isinstance(x, dict):
        return {k: _to_device_recursive(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_device_recursive(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_device_recursive(v, device) for v in x)
    return x


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


def collate_lidar_fn(
    batch,
    *,
    # pylint: disable=unused-argument
    collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None,
):
    """Collate a batch of ``LidarPointCloud`` into ``BatchedLidarPointCloud``."""

    return BatchedLidarPointCloud(batch)


def _collate_camera_dicts(
    batch: Sequence[Optional[Dict[CameraDirection, CameraData]]],
) -> Dict[CameraDirection, BatchedCameraData]:
    """Collate per-frame camera dicts into ``{direction: BatchedCameraData}``.

    The directions are the union over the batch, in ``CameraDirection`` order;
    a frame without a direction (or without cameras, ``None``) is an empty
    sample of that direction.
    """
    frames: List[Dict[CameraDirection, CameraData]] = []
    for cameras in batch:
        if cameras is not None and not isinstance(cameras, dict):
            raise TypeError(
                "frames hold cameras in different forms: per-camera dicts "
                f"(cameras_identity_adapter) and {type(cameras).__name__} "
                "(pano_adapter); process the datasets with the same camera adapter"
            )
        frames.append(cameras or {})
    return {
        direction: BatchedCameraData([cameras.get(direction) for cameras in frames])
        for direction in CameraDirection
        if any(direction in cameras for cameras in frames)
    }


def collate_modalities(
    batch: List[Any], *, device: Optional[torch.device] = None
) -> Any:
    """Collate one modality across a batch of frames.

    Like torch's default collate, with StandardE2E's containers batched into
    the classes above: ``Trajectory`` to :class:`BatchedTrajectory`,
    ``FrameDetections3D`` to :class:`BatchedFrameDetections3D`,
    ``LidarPointCloud`` to :class:`BatchedLidarPointCloud` and
    ``{CameraDirection: CameraData}`` dicts to
    ``{CameraDirection: BatchedCameraData}``.

    A frame that lacks the modality (``None``) batches as the container's
    empty element: an empty trajectory, no detections, no points or no
    cameras, each flagged by the container. A modality that is ``None`` in
    every frame collates to ``None``. Arrays, numbers and lists have no empty
    element, so ``None`` among them raises; a ``ModalityDefaults`` handler
    declares what fills a missing modality of that kind.
    """
    device = device or torch.device("cpu")
    present = [sample for sample in batch if sample is not None]
    if batch and not present:
        return None
    if any(
        isinstance(sample, dict)
        and any(isinstance(value, CameraData) for value in sample.values())
        for sample in present
    ):
        return _collate_camera_dicts(batch)
    # torch dispatches on the type of the first sample, which may be None;
    # dispatch on the first present one and fill the missing ones.
    first = present[0] if present else None
    if isinstance(first, Trajectory):
        return BatchedTrajectory(batch)
    if isinstance(first, FrameDetections3D):
        return BatchedFrameDetections3D(
            [
                sample if sample is not None else FrameDetections3D(detections=[])
                for sample in batch
            ]
        )
    if isinstance(first, LidarPointCloud):
        empty = LidarPointCloud(
            np.zeros((0, len(first.components)), dtype=np.float32), first.components
        )
        return BatchedLidarPointCloud(
            [sample if sample is not None else empty for sample in batch]
        )
    if len(present) != len(batch):
        raise ValueError(
            f"{len(batch) - len(present)} of {len(batch)} frames have None for a "
            f"{type(first).__name__} modality, which has no empty element; set a "
            "ModalityDefaults handler to fill it"
        )
    # Collate fn map must accept broader key types required by torch's internal typing
    extended_map: dict[Union[type, tuple[type, ...]], Callable[..., Any]] = {
        Trajectory: collate_trajectory_fn,
        FrameDetections3D: collate_frame_detections_fn,
        LidarPointCloud: collate_lidar_fn,
    }
    # default_collate_fn_map has type compatibility, update after copy
    extended_map.update(default_collate_fn_map)
    return _torch_collate(batch, collate_fn_map=extended_map)


def _describe_payload(value: Any) -> str:
    """One-line description of a modality payload, for batching error reports."""
    if value is None:
        return "None"
    if isinstance(value, np.ndarray):
        return f"ndarray {value.dtype} {tuple(value.shape)}"
    if isinstance(value, dict) and any(
        isinstance(camera, CameraData) for camera in value.values()
    ):
        directions = ", ".join(str(getattr(k, "value", k)) for k in value)
        shapes = sorted({camera.shape for camera in value.values()})
        return f"cameras {{{directions}}} with images {shapes}"
    if isinstance(value, Trajectory):
        components = ", ".join(c.name for c in value.components())
        return f"Trajectory(N={value.length}, components=[{components}])"
    if isinstance(value, FrameDetections3D):
        return f"FrameDetections3D({len(value.detections)} detections)"
    if isinstance(value, (list, tuple)):
        inner = _describe_payload(value[0]) if value else "nothing"
        return f"{type(value).__name__}[{len(value)}] of {inner}"
    if isinstance(value, dict):
        return f"dict with keys {sorted(str(k) for k in value)}"
    return f"{type(value).__name__} {value!r}"[:80]


def _batching_report(
    modality: Modality, frames: Sequence[TransformedFrameData], values: Sequence[Any]
) -> str:
    """Which frames held what for ``modality``: the frames grouped by payload,
    each group with the files that reproduce it."""
    groups: Dict[str, List[str]] = {}
    for frame, value in zip(frames, values):
        groups.setdefault(_describe_payload(value), []).append(str(frame.filename))
    lines = [f"while batching modality {modality.name} of {len(frames)} frames:"]
    for description, filenames in groups.items():
        more = f", +{len(filenames) - 5} more" if len(filenames) > 5 else ""
        lines.append(
            f"  {len(filenames)} with {description}: {', '.join(filenames[:5])}{more}"
        )
    return "\n".join(lines)


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
            raise ValueError("TransformedFrameDataBatch requires a non-empty \
                    list of TransformedFrameData.")
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
        self._modality_data = {}
        for modality in all_modalities:
            values = [f.get_modality_data(modality) for f in frames]
            try:
                self._modality_data[modality] = collate_modalities(
                    values, device=device
                )
            except Exception as exc:
                # Batches are sampled at random, so name the frames: they
                # reproduce the failure without the DataLoader.
                exc.add_note(_batching_report(modality, frames, values))
                raise

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
