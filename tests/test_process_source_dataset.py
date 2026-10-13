"""Tests for the preprocessing CLI's dataset registry (process_source_dataset)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from standard_e2e.caching import process_source_dataset as cli
from standard_e2e.caching.src_datasets.kitscenes_longtail import (
    KITScenesLongTailDatasetConverter,
    KITScenesLongTailDatasetProcessor,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_get_dataset_arg_name_splits_off_the_dataset():
    name, rest = cli.get_dataset_arg_name(["vod", "--split=val", "--num_workers=2"])
    assert name == "vod"
    assert rest == ["--split=val", "--num_workers=2"]


def test_every_registered_dataset_has_a_package_and_an_environment():
    for name in cli.DATASETS:
        package = _REPO_ROOT / "standard_e2e" / "caching" / "src_datasets" / name
        assert (package / "__init__.py").is_file(), name
        assert (_REPO_ROOT / "datasets" / name / "pyproject.toml").is_file(), name
        assert (_REPO_ROOT / "datasets" / name / "README.md").is_file(), name


def test_load_dataset_classes_returns_the_registered_classes():
    converter, processor = cli.load_dataset_classes("kitscenes_longtail")
    assert converter is KITScenesLongTailDatasetConverter
    assert processor is KITScenesLongTailDatasetProcessor


def test_load_dataset_classes_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown dataset name"):
        cli.load_dataset_classes("not_a_dataset")


def test_missing_sdk_names_the_dataset_environment(monkeypatch):
    # Hide nuscenes-devkit (absent from the core environment anyway) and drop any
    # cached import of the nuScenes package so it is imported afresh.
    monkeypatch.setitem(sys.modules, "nuscenes", None)
    for module in list(sys.modules):
        if module.startswith("standard_e2e.caching.src_datasets.nuscenes"):
            monkeypatch.delitem(sys.modules, module)
    with pytest.raises(ModuleNotFoundError) as info:
        cli.load_dataset_classes("nuscenes")
    message = str(info.value)
    assert info.value.name.split(".")[0] == "nuscenes"
    assert f"needs {info.value.name!r}" in message
    assert "uvx stepix run datasets/nuscenes/README.md" in message
    assert "uv run --project datasets/nuscenes" in message


def test_missing_standard_e2e_module_is_not_reworded(monkeypatch):
    # A broken StandardE2E import is a bug, not a missing SDK: re-raised as is.
    monkeypatch.setitem(sys.modules, "standard_e2e.caching.src_datasets.vod", None)
    with pytest.raises(ModuleNotFoundError) as info:
        cli.load_dataset_classes("vod")
    assert info.value.name == "standard_e2e.caching.src_datasets.vod"
    assert "stepix" not in str(info.value)


def test_main_resolves_the_dataset_before_parsing_its_arguments(capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["kitscenes_longtail", "--help"])
    assert info.value.code == 0
    assert "--input_path" in capsys.readouterr().out
