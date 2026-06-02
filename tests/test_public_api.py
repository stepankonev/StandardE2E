"""The convenience re-exports on ``standard_e2e`` resolve to the canonical
submodule objects (aliases, not copies) and stay in sync with ``__all__``."""

import importlib

import standard_e2e


def test_top_level_reexports_are_the_canonical_objects():
    from standard_e2e.data_structures import TransformedFrameData
    from standard_e2e.enums import Modality, TrajectoryComponent

    # the exact line we want user code / notebooks to be able to write
    assert standard_e2e.TransformedFrameData is TransformedFrameData
    assert standard_e2e.Modality is Modality
    assert standard_e2e.TrajectoryComponent is TrajectoryComponent


def test_all_names_are_importable_and_exported():
    for name in standard_e2e.__all__:
        assert hasattr(standard_e2e, name), f"{name} in __all__ but not importable"
    # spot-check a couple resolve to their documented homes
    ds = importlib.import_module("standard_e2e.data_structures")
    en = importlib.import_module("standard_e2e.enums")
    assert standard_e2e.LidarData is ds.LidarData
    assert standard_e2e.CameraDirection is en.CameraDirection
