import os
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from standard_e2e.data_structures import Trajectory, TransformedFrameData
from standard_e2e.dataset_utils.modality_defaults import (
    IntentDefaults,
    PreferredTrajectoryDefaults,
)
from standard_e2e.enums import Intent, Modality


def make_frame(**overrides) -> TransformedFrameData:
    kwargs = dict(
        dataset_name="ds",
        segment_id="segA",
        frame_id=7,
        timestamp=123.45,
        split="train",
    )
    kwargs.update(overrides)
    return TransformedFrameData(**kwargs)


def _aux_equals(a, b) -> bool:
    """Helper to compare aux_data that may be raw dict or 0-d np.object_ array."""
    if isinstance(a, np.ndarray) and a.dtype == object and a.shape == ():
        a = a.item()
    if isinstance(b, np.ndarray) and b.dtype == object and b.shape == ():
        b = b.item()
    return a == b


def test_filename_generation():
    f = make_frame(
        dataset_name="dataset", segment_id="segment", frame_id=42, split="val"
    )
    expected = os.path.join("dataset", "val", "segment_42.npz")
    assert f.filename == expected


def test_set_and_get_modality_data_without_defaults():
    f = make_frame()
    f.set_modality_data(Modality.SPEED, 12.34)
    assert f.get_modality_data(Modality.SPEED) == 12.34
    # Missing modality without defaults returns None
    assert f.get_modality_data(Modality.INTENT) is None


def test_get_modality_data_with_defaults_preferred_trajectory():
    f = make_frame(
        modality_defaults={
            Modality.PREFERENCE_TRAJECTORY: PreferredTrajectoryDefaults(n=2)
        }
    )
    # Not setting the modality should yield default list of Trajectory
    val = f.get_modality_data(Modality.PREFERENCE_TRAJECTORY)
    assert isinstance(val, list)
    assert len(val) == 2
    assert all(isinstance(t, Trajectory) for t in val)


def test_get_modality_data_set_default_flag_controls_defaults():
    f = make_frame(
        modality_defaults={
            Modality.PREFERENCE_TRAJECTORY: PreferredTrajectoryDefaults(n=1)
        }
    )
    # No data set for preference trajectory
    raw = f.get_modality_data(Modality.PREFERENCE_TRAJECTORY, set_default=False)
    assert raw is None
    # With default, we get a list of length 1
    val = f.get_modality_data(Modality.PREFERENCE_TRAJECTORY, set_default=True)
    assert isinstance(val, list) and len(val) == 1 and isinstance(val[0], Trajectory)


def test_get_present_modality_keys():
    f = make_frame()
    f.set_modality_data(Modality.SPEED, 1.0)
    f.set_modality_data(Modality.INTENT, Intent.GO_LEFT)
    present = set(f.get_present_modality_keys())
    assert {Modality.SPEED, Modality.INTENT}.issubset(present)


def test_npz_roundtrip_basic(tmp_path: Path):
    f = make_frame()
    # Use Python-native types to ensure pickle-based save/load works
    f.set_modality_data(Modality.SPEED, [1.0, 2.0, 3.0])
    f.set_modality_data(Modality.INTENT, Intent.GO_RIGHT)
    f.aux_data = {"info": 99}

    out = tmp_path / "frame.npz"
    f.to_npz(str(out))

    loaded = TransformedFrameData.from_npz(str(out))
    assert loaded.dataset_name == f.dataset_name
    assert loaded.segment_id == f.segment_id
    assert loaded.frame_id == f.frame_id
    assert loaded.timestamp == f.timestamp
    assert loaded.split == f.split
    # filename is recomputed from metadata
    assert loaded.filename == f.filename
    # _modality_data should persist
    assert loaded.get_modality_data(Modality.SPEED) == [1.0, 2.0, 3.0]
    # aux_data may be returned as a 0-d object array; normalize before compare
    assert _aux_equals(loaded.aux_data, f.aux_data)
    assert loaded.get_modality_data(Modality.INTENT) == Intent.GO_RIGHT


def test_from_npz_required_modalities_non_strict_fills_and_trims(tmp_path: Path):
    # Save with SPEED and INTENT, require only SPEED
    f = make_frame()
    f.set_modality_data(Modality.SPEED, 123)
    f.set_modality_data(Modality.INTENT, Intent.GO_LEFT)
    out = tmp_path / "frame_req2.npz"
    f.to_npz(str(out))

    loaded = TransformedFrameData.from_npz(
        str(out), required_modalities=[Modality.SPEED]
    )

    # Required present and unchanged
    assert loaded.get_modality_data(Modality.SPEED) == 123
    # Unwanted removed
    assert Modality.INTENT not in set(loaded.get_present_modality_keys())


def test_from_npz_required_modalities_non_strict_adds_missing_as_none(tmp_path: Path):
    # Save with no modalities, then require two
    f = make_frame()
    out = tmp_path / "frame_req3.npz"
    f.to_npz(str(out))

    loaded = TransformedFrameData.from_npz(
        str(out), required_modalities=[Modality.SPEED, Modality.INTENT]
    )
    # Both required modalities should be present and set to None
    assert set(loaded.get_present_modality_keys()) == {Modality.SPEED, Modality.INTENT}
    assert loaded.get_modality_data(Modality.SPEED, set_default=False) is None
    assert loaded.get_modality_data(Modality.INTENT, set_default=False) is None


# --- modality_defaults validation (constructor & assignment) ---


def _base_kwargs():
    return dict(
        dataset_name="ds",
        segment_id="segX",
        frame_id=99,
        timestamp=0.0,
        split="train",
    )


def test_tfd_constructor_rejects_non_dict_modality_defaults():
    import pytest

    with pytest.raises(ValidationError):
        TransformedFrameData(
            **_base_kwargs(),
            modality_defaults=123,  # type: ignore[arg-type]
        )


def test_tfd_constructor_rejects_bad_key():
    import pytest

    with pytest.raises(ValidationError):
        TransformedFrameData(
            **_base_kwargs(),
            modality_defaults={"bad": IntentDefaults()},  # type: ignore[arg-type]
        )


def test_tfd_constructor_rejects_bad_value():
    import pytest

    with pytest.raises(ValidationError):
        TransformedFrameData(
            **_base_kwargs(),
            modality_defaults={Modality.INTENT: object()},  # type: ignore[arg-type]
        )


def test_tfd_constructor_accepts_valid_modality_defaults():
    f = TransformedFrameData(
        **_base_kwargs(),
        modality_defaults={Modality.INTENT: IntentDefaults()},
    )
    assert Modality.INTENT in f.modality_defaults


def test_tfd_assignment_rejects_bad_dict():
    import pytest

    f = TransformedFrameData(**_base_kwargs())
    with pytest.raises(ValidationError):
        f.modality_defaults = 5  # type: ignore[assignment]


def test_tfd_assignment_rejects_bad_key_or_value():
    import pytest

    f = TransformedFrameData(**_base_kwargs())
    with pytest.raises(ValidationError):
        f.modality_defaults = {"bad": IntentDefaults()}  # type: ignore[assignment]
    with pytest.raises(ValidationError):
        f.modality_defaults = {Modality.SPEED: object()}  # type: ignore[assignment]


def test_tfd_assignment_accepts_valid():
    f = TransformedFrameData(**_base_kwargs())
    f.modality_defaults = {Modality.INTENT: IntentDefaults()}
    assert Modality.INTENT in f.modality_defaults
