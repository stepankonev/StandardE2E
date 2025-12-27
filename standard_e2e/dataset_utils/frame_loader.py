import os

import pandas as pd

from standard_e2e.data_structures.frame_data import TransformedFrameData
from standard_e2e.dataset_utils.modality_defaults import (
    ModalityDefaults,
    _check_modality_defaults_dict,
)
from standard_e2e.dataset_utils.selector import (
    FrameSelector,
    create_frame_selector_from_config,
)
from standard_e2e.enums import Modality


class FrameLoader:
    """Loads individual frames from processed data with selection logic."""

    def __init__(
        self,
        frame_name: str,
        required_modalities: list[Modality],
        frame_selector: FrameSelector,
        processed_data_path: str | None = None,
    ):
        """Configure a frame loader.

        Args:
            frame_name: Key used to identify the loaded frame in dataset outputs.
            required_modalities: Modalities that must be present in the returned
                frame (missing ones are inserted as ``None`` by ``from_npz``).
            frame_selector: Strategy used to pick which frame to fetch relative to
                the current index (e.g., current/closest timestamp).
            processed_data_path: Root path to the processed ``.npz`` frames; can be
                set later via ``set_processed_data_path``.
        """
        self._processed_data_path = processed_data_path
        self._frame_name = frame_name
        self._required_modalities = required_modalities
        self._frame_selector = frame_selector

    def set_processed_data_path(self, processed_data_path: str):
        """Set the processed data path."""
        self._processed_data_path = processed_data_path

    @property
    def processed_data_path(self) -> str | None:
        """Get the processed data path."""
        return self._processed_data_path

    def set_index_data(self, index_data: pd.DataFrame):
        """Set the index data for the frame selector."""
        self._frame_selector.set_index_data(index_data)

    def load_frame(
        self,
        current_frame_idx: int,
        index_data: pd.DataFrame,
        modality_defaults: dict[Modality, ModalityDefaults] | None = None,
    ) -> dict[str, TransformedFrameData]:
        """Load and materialize a frame selected relative to ``current_frame_idx``.

        Uses the configured ``FrameSelector`` to map the requested index to the
        required frame, loads it from disk, attaches modality defaults, and
        computes ``timestamp_diff`` vs. the current frame.

        Returns a mapping of ``frame_name -> TransformedFrameData`` for easy
        merging across multiple loaders.
        """
        _check_modality_defaults_dict(modality_defaults)
        if self._processed_data_path is None:
            raise ValueError("Processed data path is not set.")
        current_frame_timestamp = index_data.iloc[current_frame_idx]["timestamp"]
        required_frame_idx = self._frame_selector.select_frame(
            current_frame_idx, index_data
        )
        required_frame_timestamp = index_data.iloc[required_frame_idx]["timestamp"]
        frame_data_path = index_data.iloc[required_frame_idx]["filename"]
        frame_data = TransformedFrameData.from_npz(
            path=os.path.join(self._processed_data_path, frame_data_path),
            required_modalities=self._required_modalities,
        )
        frame_data.modality_defaults = modality_defaults
        frame_data.timestamp_diff = required_frame_timestamp - current_frame_timestamp
        return {self._frame_name: frame_data}

    def __str__(self):
        return f"FrameLoader(name={self._frame_name}, \
              required_modalities={self._required_modalities})"


def create_frame_loaders_from_config(
    processed_data_path: str,
    frames_description: list[dict],
    location: str,
    index_data: pd.DataFrame | None = None,
) -> list[FrameLoader]:
    """Instantiate ``FrameLoader`` objects from a configuration list.

    Args:
        processed_data_path: Root directory containing processed frame files.
        frames_description: List of dictionaries describing loaders; each requires
            ``frame_name``, ``modalities``, and ``selector`` sub-config.
        location: Selector location ("features" or "labels") forwarded to
            ``create_frame_selector_from_config``.
        index_data: Optional index dataframe used to precompute selector mappings.

    Returns:
        list[FrameLoader]: Concrete loaders ready to retrieve frames.
    """
    frame_loaders = []
    for frame_description in frames_description:
        frame_selector = create_frame_selector_from_config(
            frame_description["selector"], location, index_data
        )
        frame_loader = FrameLoader(
            processed_data_path=processed_data_path,
            frame_name=frame_description["frame_name"],
            required_modalities=frame_description["modalities"],
            frame_selector=frame_selector,
        )
        frame_loaders.append(frame_loader)
    return frame_loaders
