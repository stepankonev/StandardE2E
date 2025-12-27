from abc import ABC, abstractmethod
from typing import Any, final

from standard_e2e.data_structures import StandardFrameData
from standard_e2e.enums import Modality


class AbstractAdapter(ABC):
    """Base class for converting ``StandardFrameData`` into modality payloads."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the adapter. Must be implemented by subclasses."""

    @final
    def transform(self, standard_frame_data: StandardFrameData) -> dict[Modality, Any]:
        """Validate input frame and dispatch to subclass implementation."""
        if standard_frame_data is None:
            raise ValueError("standard_frame_data cannot be None")
        if not isinstance(standard_frame_data, StandardFrameData):
            raise TypeError(
                "standard_frame_data must be an instance of StandardFrameData, "
                f"got {type(standard_frame_data)}"
            )
        return self._transform(standard_frame_data)

    @abstractmethod
    def _transform(self, standard_frame_data: StandardFrameData) -> dict[Modality, Any]:
        """
        Transform the StandardFrameData into a format suitable for the specific adapter.
        """
