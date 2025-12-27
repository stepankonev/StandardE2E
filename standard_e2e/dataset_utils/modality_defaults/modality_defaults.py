from abc import ABC, abstractmethod
from typing import Any, final

from standard_e2e.data_structures.trajectory_data import Trajectory
from standard_e2e.enums import Intent, Modality


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
