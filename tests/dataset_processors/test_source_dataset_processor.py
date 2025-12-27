"""Tests for the generic SourceDatasetProcessor base class.

We create lightweight concrete subclasses + dummy adapters so we can exercise:
 - constructor validation (adapters param, index_data_generator type, split)
 - default adapters usage
 - process_frame aggregation of multiple adapters
 - error when _prepare_standardized_frame_data returns wrong type
 - process_frame_and_save_data happy path + error path (missing filename)
"""

# pylint: disable=redefined-outer-name

from __future__ import annotations

from pathlib import Path

import pytest

from standard_e2e.caching.adapters import AbstractAdapter
from standard_e2e.caching.source_dataset_processor import SourceDatasetProcessor
from standard_e2e.data_structures import StandardFrameData
from standard_e2e.enums import Intent, Modality

# --- Dummy adapters for testing ---


class DummyAdapterSpeed(AbstractAdapter):
    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return "DummyAdapterSpeed"

    def _transform(self, standard_frame_data: StandardFrameData):  # pragma: no cover
        return {Modality.SPEED: 123.0}


class DummyAdapterIntent(AbstractAdapter):
    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return "DummyAdapterIntent"

    def _transform(self, standard_frame_data: StandardFrameData):  # pragma: no cover
        return {Modality.INTENT: standard_frame_data.intent}


# --- Test concrete processors ---


class SampleProcessor(SourceDatasetProcessor):
    def _get_default_adapters(self):
        return [DummyAdapterSpeed()]

    @property
    def dataset_name(self) -> str:  # pragma: no cover - simple property
        return "test_dataset"

    @property
    def allowed_splits(self):  # pragma: no cover - simple property
        return ["train", "val"]

    def _prepare_standardized_frame_data(self, raw_frame_data):
        # Minimal valid StandardFrameData
        return StandardFrameData(
            dataset_name=self.dataset_name,
            segment_id="segmentA",
            frame_id=1,
            timestamp=0.5,
            split=self.split,
            intent=Intent.GO_LEFT,
        )


class BadReturnProcessor(SampleProcessor):
    def _prepare_standardized_frame_data(
        self, raw_frame_data
    ):  # type: ignore[override]
        return object()  # Wrong type to trigger TypeError in process_frame


# --- Fixtures ---


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "out"


# --- Constructor validation tests ---


def test_constructor_invalid_adapters_type(output_dir: Path):
    with pytest.raises(TypeError):
        SampleProcessor(str(output_dir), "train", adapters=42)  # type: ignore[arg-type]


def test_constructor_invalid_adapters_element(output_dir: Path):
    with pytest.raises(TypeError):
        SampleProcessor(
            str(output_dir), "train", adapters=[123]
        )  # type: ignore[list-item]


def test_constructor_invalid_index_generator_type(output_dir: Path):
    with pytest.raises(TypeError):
        SampleProcessor(
            str(output_dir), "train", index_data_generator=object()
        )  # type: ignore[arg-type]


def test_constructor_invalid_split(output_dir: Path):
    with pytest.raises(ValueError):
        SampleProcessor(str(output_dir), "bad_split")


def test_constructor_defaults(output_dir: Path):
    p = SampleProcessor(str(output_dir), "train")
    # Default adapter applied
    # noqa: SLF001 - testing internal state pylint: disable=protected-access
    assert len(p._adapters) == 1
    assert isinstance(p._adapters[0], DummyAdapterSpeed)  # noqa: SLF001


# --- process_frame tests ---


def test_process_frame_aggregates_modalities(output_dir: Path):
    p = SampleProcessor(
        str(output_dir),
        "train",
        adapters=[DummyAdapterSpeed(), DummyAdapterIntent()],
    )
    transformed, index = p.process_frame(raw_frame_data=None)
    assert transformed.get_modality_data(Modality.SPEED) == 123.0
    assert transformed.get_modality_data(Modality.INTENT) == Intent.GO_LEFT
    # Index data coherency
    assert index.dataset_name == transformed.dataset_name
    assert index.frame_id == transformed.frame_id


def test_process_frame_wrong_return_type(output_dir: Path):
    p = BadReturnProcessor(str(output_dir), "train")
    with pytest.raises(TypeError):
        p.process_frame(None)


# --- process_frame_and_save_data tests ---


def test_process_frame_and_save_data_saves_file(output_dir: Path):
    p = SampleProcessor(str(output_dir), "train")
    frame_index = p.process_frame_and_save_data(raw_frame_data=None)
    # File should exist relative to common output path
    saved_path = Path(output_dir) / frame_index.filename
    assert saved_path.exists()
