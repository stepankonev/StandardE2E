from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch

from standard_e2e.enums import TrajectoryComponent
from standard_e2e.utils import _assert_strictly_increasing

# ======================= Helpers (module-level) =======================

Array1DNP = Union[
    np.ndarray, List[float], List[int], Tuple[float, ...], Tuple[int, ...]
]


def _to_numpy_1d_float32(values: Array1DNP) -> np.ndarray:
    """Convert ndarray/list/tuple to 1D np.float32. Accepts (N,) or (N,1)."""
    if isinstance(values, np.ndarray):
        array = values
    elif isinstance(values, (list, tuple)):
        array = np.asarray(values)
    else:
        raise TypeError(f"Expected NumPy array or list/tuple, got {type(values)!r}")

    if array.ndim == 1:
        pass
    elif array.ndim == 2 and array.shape[1] == 1:
        array = array.reshape(-1)
    else:
        raise ValueError(f"Expected shape (N,) or (N,1); got {tuple(array.shape)}")

    return array.astype(np.float32, copy=False)


def _pad_numpy_1d(array: np.ndarray, target_length: int, side: str) -> np.ndarray:
    """Zero-pad a 1D array to target_length on 'left' or 'right'."""
    length = array.shape[0]
    if length >= target_length:
        return array
    pad = target_length - length
    zeros = np.zeros((pad,), dtype=np.float32)
    if side == "left":
        return np.concatenate([zeros, array])
    if side == "right":
        return np.concatenate([array, zeros])
    raise ValueError("side must be 'left' or 'right'")


def _validate_components_arg(
    components: Union[TrajectoryComponent, Sequence[TrajectoryComponent]],
) -> List[TrajectoryComponent]:
    """Normalize and validate a component or a list of components."""
    if isinstance(components, TrajectoryComponent):
        components_list = [components]
    elif isinstance(components, Sequence) and components:
        # check this is a sequence of TrajectoryComponent
        components_list = [c for c in components]  # type: ignore[misc]
        if not all(isinstance(c, TrajectoryComponent) for c in components_list):
            valid = ", ".join(m.name for m in TrajectoryComponent)
            raise TypeError(
                f"All elements must be TrajectoryComponent. Valid: [{valid}]"
            )
    else:
        raise TypeError(
            "components must be a TrajectoryComponent or a non-empty sequence of them"
        )

    return components_list


# ======================= Trajectory (NumPy) =======================


class Trajectory:
    """
    Single trajectory data container (numpy) with helpers for processing.

    - Components: 1D np.float32 arrays of equal length N.
    - Accepts shapes (N,) or (N,1) from ndarray/list/tuple; stored as (N,).
    - Auto IS_VALID: first non-IS_VALID set -> is_valid = 1;
        user-provided IS_VALID overrides.
    - score: optional scalar (float).
    - get(one or many, strict=True) -> (N, K) np.float32.
    - pad/trim/pad_or_trim with zeros; padded rows marked invalid via is_valid zeros.
    """

    def __init__(
        self,
        data: Dict[TrajectoryComponent, Array1DNP] | None = None,
        score: Optional[float] = None,
        time_lattice: Optional[Array1DNP] = None,
    ) -> None:
        self._data: Dict[TrajectoryComponent, np.ndarray] = {}
        self.score: Optional[float] = None if score is None else float(score)
        if data:
            self.set_many(data)
        if time_lattice is not None:
            self.resample(time_lattice, inplace=True)

    # ----- basics -----
    @property
    def length(self) -> int:
        """Return the number of timesteps stored in this trajectory.
        If no data has been added yet, this returns 0.
        Returns:
            int: Number of timesteps (rows) in the trajectory.
        """
        if not self._data:
            return 0
        first_key = next(iter(self._data))
        return int(self._data[first_key].shape[0])

    @property
    def isEmpty(self) -> bool:
        """Check if the trajectory is empty or has no valid data."""
        if self.length == 0:
            return True
        if self.get(TrajectoryComponent.IS_VALID, strict=False).sum() == 0:
            return True
        return False

    def components(self) -> List[TrajectoryComponent]:
        """List the trajectory components currently stored."""
        return list(self._data.keys())

    def has(self, component: TrajectoryComponent) -> bool:
        """Check if a trajectory component is present."""
        return component in self._data

    # ----- set/get -----
    def set(self, component: TrajectoryComponent, values: Array1DNP) -> "Trajectory":
        """Add or replace a trajectory component, enforcing length consistency."""
        array = _to_numpy_1d_float32(values)
        if component is TrajectoryComponent.TIMESTAMP:
            _assert_strictly_increasing(array)
        if self.length and array.shape[0] != self.length:
            raise ValueError(
                f"Length mismatch for {component.name}: \
                    {array.shape[0]} != {self.length}"
            )
        self._data[component] = array

        # Auto-create is_valid=1 for first real component
        if (
            component is not TrajectoryComponent.IS_VALID
            and TrajectoryComponent.IS_VALID not in self._data
        ):
            self._data[TrajectoryComponent.IS_VALID] = np.ones(
                array.shape[0], dtype=np.float32
            )
        return self

    def set_many(
        self, mapping: Mapping[TrajectoryComponent, Array1DNP]
    ) -> "Trajectory":
        """Set multiple components in one call, respecting IS_VALID ordering."""
        for component, values in mapping.items():
            if component is not TrajectoryComponent.IS_VALID:
                self.set(component, values)
        if TrajectoryComponent.IS_VALID in mapping:
            self.set(
                TrajectoryComponent.IS_VALID, mapping[TrajectoryComponent.IS_VALID]
            )
        return self

    def get(
        self,
        components: Union[TrajectoryComponent, Sequence[TrajectoryComponent]],
        *,
        strict: bool = True,
    ) -> np.ndarray:
        """Fetch one or more components as a stacked ``(N, K)`` array."""
        components_list = _validate_components_arg(components)
        length = self.length

        columns: List[np.ndarray] = []
        missing_components: List[TrajectoryComponent] = []

        for component in components_list:
            if component in self._data:
                columns.append(self._data[component].reshape(length, 1))
            else:
                if strict:
                    missing_components.append(component)
                else:
                    columns.append(np.zeros((length, 1), dtype=np.float32))

        if missing_components:
            available = ", ".join(c.name for c in self._data)
            needed = ", ".join(c.name for c in missing_components)
            raise KeyError(f"Missing component(s): {needed}. Available: [{available}]")

        if not columns:
            return np.zeros((self.length, 0), dtype=np.float32)
        out = np.concatenate(columns, axis=1).astype(np.float32, copy=False)
        return cast(np.ndarray, out)

    # ----- pad/trim -----
    def pad(self, target_length: int, *, side: str = "right") -> "Trajectory":
        """Pad all components with zeros to ``target_length`` (left/right)."""
        if target_length <= self.length:
            return self
        if TrajectoryComponent.IS_VALID not in self._data:
            self._data[TrajectoryComponent.IS_VALID] = np.zeros(
                self.length, dtype=np.float32
            )
        for component, array in list(self._data.items()):
            self._data[component] = _pad_numpy_1d(array, target_length, side)
        return self

    def trim(self, target_length: int, *, side: str = "right") -> "Trajectory":
        """Trim components to ``target_length`` from the specified side."""
        current_length = self.length
        if target_length >= current_length:
            return self
        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")
        cut = current_length - target_length
        for component, array in list(self._data.items()):
            self._data[component] = array[cut:] if side == "left" else array[:-cut]
        return self

    def pad_or_trim(self, target_length: int, *, side: str = "right") -> "Trajectory":
        """Pad or trim to ``target_length`` depending on current length."""
        return (
            self.pad(target_length, side=side)
            if target_length > self.length
            else self.trim(target_length, side=side)
        )

    def resample(self, time_lattice: Array1DNP, inplace: bool = False) -> "Trajectory":
        """
        Resample trajectory components onto a new time lattice using linear
        interpolation.

        Args:
            time_lattice: New timestamps to interpolate onto (1D array-like).

        Returns:
            self: Modified trajectory with resampled data.

        Raises:
            ValueError: If TIMESTAMP component is missing or time_lattice is invalid.
        """
        if time_lattice is None:
            raise ValueError("time_lattice cannot be None")

        # Convert time_lattice to numpy array
        new_times = _to_numpy_1d_float32(time_lattice)
        _assert_strictly_increasing(new_times)

        if len(new_times) == 0:
            raise ValueError("time_lattice cannot be empty")
        if self.isEmpty:
            # For empty trajectory, create new data with all components
            # set to 0 and IS_VALID to False
            new_data = {}
            for component in self.components():
                new_data[component] = np.zeros(len(new_times), dtype=np.float32)
            new_data[TrajectoryComponent.TIMESTAMP] = new_times.copy()
            new_data[TrajectoryComponent.IS_VALID] = np.zeros(
                len(new_times), dtype=np.float32
            )
            self._data = new_data
            return self

        # Check if we have timestamp data
        if TrajectoryComponent.TIMESTAMP not in self._data:
            raise ValueError("Cannot resample trajectory without TIMESTAMP component")

        original_times = self._data[TrajectoryComponent.TIMESTAMP]
        # Validate stored timestamps each resample call (in case legacy object)
        _assert_strictly_increasing(original_times)

        # Handle edge cases - if original trajectory has only one point
        if len(original_times) == 1:
            new_data = {}
            original_time = original_times[0]

            for component, values in self._data.items():
                if component == TrajectoryComponent.TIMESTAMP:
                    new_data[component] = new_times.copy()
                elif component == TrajectoryComponent.IS_VALID:
                    # Set IS_VALID to True only for exact timestamp matches
                    new_data[component] = np.zeros(len(new_times), dtype=np.float32)
                    # Find exact matches with the original timestamp
                    exact_matches = np.isclose(
                        new_times, original_time, rtol=1e-6, atol=1e-6
                    )
                    new_data[component][exact_matches] = 1.0
                else:
                    # Set to zero for extrapolated points, original value for
                    # exact matches
                    new_data[component] = np.zeros(len(new_times), dtype=np.float32)
                    exact_matches = np.isclose(
                        new_times, original_time, rtol=1e-6, atol=1e-6
                    )
                    new_data[component][exact_matches] = values[0]
            self._data = new_data
            return self

        # Determine time boundaries for validity
        min_time = np.min(original_times)
        max_time = np.max(original_times)

        # Create new data dictionary
        new_data = {}

        # Always add timestamp
        new_data[TrajectoryComponent.TIMESTAMP] = new_times.copy()

        # Handle IS_VALID component and determine valid time ranges preserving gaps
        new_is_valid = np.zeros(len(new_times), dtype=np.float32)
        interpolation_bounds = (min_time, max_time)  # default
        valid_intervals: List[Tuple[float, float]] = []

        if TrajectoryComponent.IS_VALID in self._data:
            original_is_valid = self._data[TrajectoryComponent.IS_VALID]
            valid_mask = original_is_valid == 1.0
            if np.any(valid_mask):
                # Build contiguous intervals (in index space) of valid points
                start_idx: Optional[int] = None
                for idx, is_v in enumerate(valid_mask):
                    if is_v and start_idx is None:
                        start_idx = idx
                    elif not is_v and start_idx is not None:
                        # end previous interval
                        end_idx = idx - 1
                        valid_intervals.append(
                            (
                                float(original_times[start_idx]),
                                float(original_times[end_idx]),
                            )
                        )
                        start_idx = None
                if start_idx is not None:
                    valid_intervals.append(
                        (
                            float(original_times[start_idx]),
                            float(original_times[len(valid_mask) - 1]),
                        )
                    )

                # Mark new_times valid if they lie inside any valid interval
                for i, t in enumerate(new_times):
                    for a, b in valid_intervals:
                        if a <= t <= b:
                            new_is_valid[i] = 1.0
                            break
                # Interpolation bounds: overall min/max of valid points for
                # outside zeroing
                all_valid_times = original_times[valid_mask]
                interpolation_bounds = (
                    float(np.min(all_valid_times)),
                    float(np.max(all_valid_times)),
                )
            else:
                # All invalid: leave new_is_valid zeros, keep default bounds
                pass
        else:
            # No IS_VALID provided: treat full original span as one valid interval
            new_is_valid = ((new_times >= min_time) & (new_times <= max_time)).astype(
                np.float32
            )
            valid_intervals.append((float(min_time), float(max_time)))
            interpolation_bounds = (float(min_time), float(max_time))

        new_data[TrajectoryComponent.IS_VALID] = new_is_valid

        # Mask for overall interpolation bounds (outermost valid extent)
        in_bounds = (new_times >= interpolation_bounds[0]) & (
            new_times <= interpolation_bounds[1]
        )

        # Interpolate all other components
        for component, values in self._data.items():
            if component in [
                TrajectoryComponent.TIMESTAMP,
                TrajectoryComponent.IS_VALID,
            ]:
                continue  # Already handled

            # Determine which original data points to use for interpolation
            if TrajectoryComponent.IS_VALID in self._data:
                original_is_valid = self._data[TrajectoryComponent.IS_VALID]
                valid_mask = original_is_valid == 1.0
                if np.any(valid_mask):
                    # Use only valid timestamps and values for interpolation
                    valid_original_times = original_times[valid_mask]
                    valid_original_values = values[valid_mask]
                    # Linear interpolation using only valid points
                    interpolated_values = np.interp(
                        new_times, valid_original_times, valid_original_values
                    )
                else:
                    # No valid original points, fill with zeros
                    interpolated_values = np.zeros_like(new_times)
            else:
                # Use all original data for interpolation
                interpolated_values = np.interp(new_times, original_times, values)

            # Zero out values outside interpolation bounds OR not marked valid
            valid_new_mask = new_is_valid.astype(bool)
            interpolated_values[~in_bounds | ~valid_new_mask] = 0.0

            new_data[component] = interpolated_values.astype(np.float32)

        if inplace:
            self._data = new_data
            return self
        return Trajectory(
            data=cast(Dict[TrajectoryComponent, Array1DNP], new_data),
            score=self.score,
        )

    def __len__(self) -> int:
        return self.length

    def __repr__(self) -> str:
        shapes = ", ".join(f"{k.name}:{tuple(v.shape)}" for k, v in self._data.items())
        return f"Trajectory(N={self.length}, isEmpty={self.isEmpty}, \
            score={self.score}, comps=[{shapes}])"


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
        self._original_trajectories = trajectories
        self._side = side
        self._strict = strict
        trajectories = self._replace_none_with_empty_trajs(trajectories)
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
