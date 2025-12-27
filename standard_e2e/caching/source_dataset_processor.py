import logging
import os
from abc import ABC, abstractmethod
from typing import Any, final

from standard_e2e.caching.adapters import AbstractAdapter
from standard_e2e.caching.segment_context import SegmentContextAggregator
from standard_e2e.data_structures import (
    FrameIndexData,
    StandardFrameData,
    TransformedFrameData,
)
from standard_e2e.indexing import IndexDataGenerator
from standard_e2e.utils import _check_list_of_objects_or_none


class SourceDatasetProcessor(ABC):
    """Abstract base class for processing source datasets."""

    def __init__(
        self,
        common_output_path: str,
        split: str,
        index_data_generator: IndexDataGenerator | None = None,
        adapters: list[AbstractAdapter] | None = None,
        context_aggregators: list[SegmentContextAggregator] | None = None,
    ):
        _check_list_of_objects_or_none(adapters, AbstractAdapter)
        _check_list_of_objects_or_none(context_aggregators, SegmentContextAggregator)
        if not isinstance(index_data_generator, (IndexDataGenerator, type(None))):
            raise TypeError(
                "index_data_generator must be an instance of IndexDataGenerator"
                f"or None, got {type(index_data_generator)}"
            )
        self._split = split
        self._common_output_path = common_output_path
        self._specific_output_path = self._prepare_output_directory()
        self._inner_path = os.path.relpath(
            self._specific_output_path, common_output_path
        )
        self._adapters = self._get_default_adapters() if adapters is None else adapters
        self._context_aggregators = (
            self._get_default_context_aggregators()
            if context_aggregators is None
            else context_aggregators
        )
        self._index_data_generator = (
            index_data_generator if index_data_generator else IndexDataGenerator()
        )
        if self._split not in self.allowed_splits:
            raise ValueError(
                f"Invalid split: {self._split}. Must be one of {self.allowed_splits}."
            )
        logging.info("Initialized %s processor", self.dataset_name)
        logging.info("Using adapters: %s", [a.name for a in self._adapters])
        logging.info("Specific output path: %s", self._specific_output_path)

    def _get_default_adapters(self) -> list[AbstractAdapter]:
        raise NotImplementedError("Subclasses must implement this method.")

    def _get_default_context_aggregators(self) -> list[SegmentContextAggregator]:
        return []

    @final
    def _prepare_output_directory(self) -> str:
        """Prepare the output directory for specific processed data."""
        specific_output_path = os.path.join(
            self._common_output_path, self.dataset_name, self.split
        )
        if not os.path.exists(specific_output_path):
            os.makedirs(specific_output_path)
            logging.info("Created output directory: %s", specific_output_path)
        else:
            logging.warning("Output directory already exists: %s", specific_output_path)
        return specific_output_path

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Return the name of the dataset."""
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def allowed_splits(self) -> list[str]:
        """Return the list of allowed splits for the dataset."""
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def context_aggregators(self):
        return self._context_aggregators

    @final
    def process_frame(
        self, raw_frame_data: Any
    ) -> tuple[TransformedFrameData, FrameIndexData]:
        standard_frame_data = self._prepare_standardized_frame_data(raw_frame_data)
        if not isinstance(standard_frame_data, StandardFrameData):
            raise TypeError(
                "_prepare_standardized_frame_data must return StandardFrameData, "
                f"got {type(standard_frame_data)}"
            )
        transformed_modalities = {}
        for adapter in self._adapters:
            transformed_modalities.update(adapter.transform(standard_frame_data))
        transformed_frame_data = TransformedFrameData(
            dataset_name=standard_frame_data.dataset_name,
            segment_id=standard_frame_data.segment_id,
            frame_id=standard_frame_data.frame_id,
            timestamp=standard_frame_data.timestamp,
            split=standard_frame_data.split,
            global_position=standard_frame_data.global_position,
            aux_data=standard_frame_data.aux_data,
            extra_index_data=standard_frame_data.extra_index_data,
            _modality_data=transformed_modalities,
        )
        frame_index_data = self._index_data_generator.generate_index_data(
            transformed_frame_data
        )
        return transformed_frame_data, frame_index_data

    @abstractmethod
    def _prepare_standardized_frame_data(
        self, raw_frame_data: Any
    ) -> StandardFrameData:
        """Process a single frame of data."""
        # Implement the logic to process a single frame
        raise NotImplementedError("Subclasses must implement this method.")

    @final
    def process_frame_and_save_data(self, raw_frame_data: Any) -> FrameIndexData:
        """
        Process a single frame of raw data, save the processed frame data to disk,
        and return the corresponding FrameIndexData.
        """
        frame_data: TransformedFrameData
        frame_index_data: FrameIndexData
        frame_data, frame_index_data = self.process_frame(raw_frame_data)
        filename = frame_data.filename
        if filename is None:
            raise ValueError("Frame data must have a filename before saving.")
        frame_data.to_npz(os.path.join(self._common_output_path, filename))
        return frame_index_data

    @property
    def split(self) -> str:
        """Return the dataset split."""
        return self._split

    @property
    def output_path(self) -> str:
        """Return the output path for the processed dataset."""
        return self._common_output_path

    @property
    def inner_path(self) -> str:
        """Return the inner path relative to the common output path."""
        return self._inner_path

    @property
    def specific_output_path(self) -> str:
        """Return the specific output path for the dataset."""
        return self._specific_output_path
