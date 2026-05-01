"""Tests for ``SourceDatasetProcessor.extra_processor_kwargs`` classmethod."""

from __future__ import annotations

import argparse

import pytest


def test_base_returns_empty_dict():
    """Base implementation returns ``{}`` so most processors stay unaffected."""
    from standard_e2e.caching.source_dataset_processor import SourceDatasetProcessor

    args = argparse.Namespace(
        input_path="/in", output_path="/out", split="training"
    )
    assert SourceDatasetProcessor.extra_processor_kwargs(args) == {}


def test_waymo_e2e_inherits_empty_dict():
    """WaymoE2E processor takes no extra CLI kwargs (no override)."""
    pytest.importorskip("tensorflow")
    from standard_e2e.caching.src_datasets.waymo_e2e import (
        WaymoE2EDatasetProcessor,
    )

    args = argparse.Namespace(
        input_path="/in", output_path="/out", split="training"
    )
    assert WaymoE2EDatasetProcessor.extra_processor_kwargs(args) == {}


def test_waymo_perception_returns_source_data_path():
    """Waymo Perception threads through the per-split source tfrecord directory."""
    pytest.importorskip("tensorflow")
    from standard_e2e.caching.src_datasets.waymo_perception import (
        WaymoPerceptionDatasetProcessor,
    )

    args = argparse.Namespace(
        input_path="/data/waymo",
        output_path="/cache",
        split="validation",
    )
    out = WaymoPerceptionDatasetProcessor.extra_processor_kwargs(args)
    assert out == {"source_data_path": "/data/waymo/validation"}
