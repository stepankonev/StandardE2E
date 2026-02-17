from abc import ABC, abstractmethod

import pandas as pd


class FrameSelector(ABC):
    """Base class for selecting frame indices relative to a reference frame."""

    _ALLOWED_LOCATIONS = {"labels", "features"}

    def __init__(self, location: str, **kwargs):
        """Initialize selector.

        Args:
            location: Either "labels" or "features" indicating which side of
                the pipeline the selector is used for; affects validation rules.
            **kwargs: May include ``index_data`` used to precompute mappings.
        """
        self._index_data = kwargs.get("index_data")
        if location not in self._ALLOWED_LOCATIONS:
            raise ValueError(
                f"Invalid location '{location}'. Allowed locations are: \
                    {self._ALLOWED_LOCATIONS}"
            )
        self._location = location
        self._validate_params()

    def _validate_params(self):
        """Hook for subclasses to validate constructor parameters."""

    def set_index_data(self, index_data: pd.DataFrame):
        """
        Set the index data for the frame selector.
        """
        self._index_data = index_data

    @property
    def index_data(self) -> pd.DataFrame | None:
        """Return the index data (DataFrame) or None if not set."""
        return self._index_data

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the selector."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def select_frame(
        self, current_frame_iloc: int, index_data: pd.DataFrame | None = None
    ) -> int:
        """Select a frame given the current frame location within a segment."""
        raise NotImplementedError("Subclasses must implement this method.")
