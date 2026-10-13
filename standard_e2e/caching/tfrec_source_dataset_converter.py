"""Base converter for datasets stored as TFRecord files (the Waymo datasets).

Kept apart from :mod:`standard_e2e.caching.source_dataset_converter` because it
needs TensorFlow, which only the environments of TFRecord-backed datasets
install (``datasets/waymo_*/``).
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import cast

import tensorflow as tf

from standard_e2e.caching.source_dataset_converter import SourceDatasetConverter


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
