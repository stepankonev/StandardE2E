"""
Module for converting source datasets to target formats with optional parallel
processing and TFRecord support.
"""

import argparse
import logging
import multiprocessing
import os
from abc import ABC, abstractmethod
from typing import Optional, Union, cast, final

import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from standard_e2e.caching.segment_context import SegmentContextAggregator
from standard_e2e.caching.source_dataset_processor import SourceDatasetProcessor
from standard_e2e.data_structures import FrameIndexData


class SourceDatasetConverter(ABC):
    """Base class orchestrating conversion from raw datasets to processed frames."""

    def __init__(
        self,
        source_processor: SourceDatasetProcessor,
        input_path: str,
        split: str,
        num_workers: int = 0,
        do_parallel_processing: bool = True,
        arguments: Optional[Union[argparse.Namespace, dict]] = None,
    ):
        self._source_processor = source_processor
        self._input_path = input_path
        self._split = split
        self._num_workers = num_workers
        if self._num_workers == 0:
            self._num_workers = multiprocessing.cpu_count()
        self._do_parallel_processing = do_parallel_processing
        # Normalize arguments into argparse.Namespace
        if arguments is None:
            self._args = argparse.Namespace()
        elif isinstance(arguments, dict):
            self._args = argparse.Namespace(**arguments)
        else:
            self._args = arguments
        self._source_dataset_iterator = self._get_source_dataset_iterator()

    @property
    def dataset_name(self) -> str:
        """Return the name of the dataset."""
        return self._source_processor.dataset_name

    @classmethod
    def get_arg_parser(cls):
        """Return an argument parser for the converter."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--input_path",
            type=str,
            required=True,
            help="Path to the input directory containing the source dataset.",
        )
        parser.add_argument(
            "--output_path",
            type=str,
            required=True,
            help="Path to the output directory where the converted data will be saved.",
        )
        parser.add_argument(
            "--split", type=str, required=True, help="Split of the dataset to process."
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=multiprocessing.cpu_count(),
            help="Number of worker processes to use for data processing.",
        )
        parser.add_argument(
            "--do_parallel_processing",
            action="store_true",
            help="Whether to use parallel processing for data conversion.",
        )
        parser.add_argument(
            "--config_file",
            type=str,
            required=True,
            help="Path to the configuration file.",
        )
        return parser

    @abstractmethod
    def _get_source_dataset_iterator(self):
        """Return an iterator over the source dataset."""
        raise NotImplementedError("Subclasses must implement this method.")

    @final
    def _run_context_aggregators(self, index_df: pd.DataFrame) -> None:
        logging.info("Running context aggregators...")
        for context_aggregator in self._source_processor.context_aggregators:
            logging.info(
                "Processing with context aggregator: %s",
                context_aggregator.__class__.__name__,
            )
            if not isinstance(context_aggregator, SegmentContextAggregator):
                raise TypeError(
                    "context_aggregator must be an instance of SegmentContextAggregator"
                    f", got {type(context_aggregator)}"
                )
            context_aggregator.process(index_df)

    @final
    def _convert_frames(self) -> pd.DataFrame:
        """Convert the source dataset to the target format."""
        logging.info("Processing input path: %s", self._input_path)
        logging.info("Output path: %s", self._source_processor.output_path)
        logging.info("Processing split: %s", self._split)
        if self._do_parallel_processing:
            logging.info(
                "Using parallel processing with %d workers for dataset conversion.",
                self._num_workers,
            )
            with multiprocessing.Pool(self._num_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap(
                            self._source_processor.process_frame_and_save_data,
                            self._source_dataset_iterator,
                        ),
                        desc=f"Processing \
                            {self._source_processor.dataset_name} dataset",
                    )
                )
        else:
            logging.info(
                "Processing dataset without parallelization for %s.",
                self._source_processor.dataset_name,
            )
            results = []
            for raw_frame_data in tqdm(
                self._source_dataset_iterator,
                desc=f"Processing {self._source_processor.dataset_name} dataset",
            ):
                results.append(
                    self._source_processor.process_frame_and_save_data(raw_frame_data)
                )
        logging.info(
            "Conversion completed for %s dataset.", self._source_processor.dataset_name
        )
        logging.info("Converted %d frames from the source dataset.", len(results))
        logging.info("Data saved to %s", self._source_processor.specific_output_path)
        index_df = FrameIndexData.save_index_data(
            results, self._source_processor.specific_output_path
        )
        return index_df

    @final
    def convert(self) -> None:
        """Convert all frames then run any configured context aggregators."""
        index_df = self._convert_frames()
        self._run_context_aggregators(index_df)


class TFRecSourceDatasetConverter(SourceDatasetConverter, ABC):
    """
    TFRecSourceDatasetConverter is an abstract base class for converting source
    datasets into TensorFlow Record (TFRecord) format.

    This class extends SourceDatasetConverter and provides additional functionality
    for handling sharded TFRecord datasets.
    It defines command-line arguments for sharding, enforces implementation of file
    retrieval logic, and provides an iterator
    over the source dataset with optional sharding support.

    Methods
    -------
    get_arg_parser(cls):
        Returns an argument parser with additional arguments for
        sharding (n_shards, shard_id).

    _get_processing_files(self):
        Abstract method that must be implemented by subclasses to return
        a list of files to process.

    _source_dataset_iterator(self):
        Returns a tf.data.TFRecordDataset iterator over the files to process,
        applying sharding if specified.

    Attributes
    ----------
    _args : argparse.Namespace
        Parsed command-line arguments, expected to include n_shards and shard_id.
    _input_path : str
        Path to the input data directory.
    _split : str
        Name of the dataset split (e.g., 'train', 'val', 'test').
    """

    def __init__(self, *args, **kwargs):
        # Provide default sharding args if not supplied
        if kwargs.get("arguments") is None:
            kwargs["arguments"] = {
                "n_shards": 1,
                "shard_id": 0,
            }
        super().__init__(*args, **kwargs)

    @classmethod
    def get_arg_parser(cls):
        """Return an argument parser for the converter."""
        parser = super().get_arg_parser()
        parser.add_argument(
            "--n_shards",
            type=int,
            default=1,
            help="Number of shards to split the output data into.",
        )
        parser.add_argument(
            "--shard_id", type=int, default=0, help="ID of the shard to process."
        )
        return parser

    @abstractmethod
    def _get_processing_files(self):
        """Return a list of files to process."""
        raise NotImplementedError("Subclasses must implement this method.")

    def _get_source_dataset_iterator(self):
        processing_files = self._get_processing_files()
        processing_files = tf.io.matching_files(processing_files)
        if os.environ.get("STANDARD_E2E_DEBUG", "false").lower() == "true":
            logging.info(
                "STANDARD_E2E_DEBUG is set to true, processing only the first file."
            )
            processing_files = processing_files[:1]
        if len(processing_files) == 0:
            raise FileNotFoundError(
                f"No files found in {self._input_path} with pattern \
                    {self._split}*.tfrecord*"
            )
        logging.info(
            "Found %d files to process for split '%s' of dataset '%s' before sharding.",
            len(processing_files),
            self._split,
            self._source_processor.dataset_name,
        )
        # Use generic Dataset variable for typing compatibility
        # (mypy mismatch with subclass)
        dataset = cast(
            tf.data.Dataset,
            tf.data.TFRecordDataset(
                processing_files,
                compression_type="",
            ),
        )
        if getattr(self._args, "n_shards", 1) > 1:
            logging.info(
                "Sharding dataset into %d shards, processing shard %d.",
                self._args.n_shards,
                self._args.shard_id,
            )
            dataset = dataset.shard(self._args.n_shards, self._args.shard_id)
        else:
            logging.info("Processing without sharding.")
        return dataset
