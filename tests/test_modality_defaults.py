import pytest

from standard_e2e.data_structures import Trajectory
from standard_e2e.dataset_utils.modality_defaults import (
    IntentDefaults,
    ModalityDefaults,
    PreferredTrajectoryDefaults,
    _check_modality_defaults_dict,
)
from standard_e2e.enums import Intent, Modality


class DummyDefaults(ModalityDefaults):
    """Simple concrete defaults for testing allowed modalities plumbing."""

    def __init__(self, allowed):
        self._allowed = allowed

    def _normalize(self, raw_value, modality):  # pragma: no cover - trivial
        return raw_value

    @property
    def allowed_modalities(self):
        return self._allowed


def test_check_modality_defaults_accepts_none():
    # Should not raise
    _check_modality_defaults_dict(None)


def test_check_modality_defaults_rejects_non_dict():
    with pytest.raises(TypeError):
        _check_modality_defaults_dict(123)  # type: ignore[arg-type]


def test_check_modality_defaults_rejects_non_modality_key():
    with pytest.raises(TypeError):
        _check_modality_defaults_dict(
            {"not_mod": DummyDefaults([Modality.SPEED])}  # type: ignore[arg-type]
        )


def test_check_modality_defaults_rejects_non_defaults_value():
    with pytest.raises(TypeError):
        _check_modality_defaults_dict(
            {Modality.SPEED: object()}  # type: ignore[arg-type]
        )


def test_check_modality_defaults_passes_with_valid_entries():
    _check_modality_defaults_dict({Modality.SPEED: DummyDefaults([Modality.SPEED])})


def test_preferred_traj_defaults_invalid_n():
    with pytest.raises(ValueError):
        PreferredTrajectoryDefaults(n=0)


def test_preferred_traj_defaults_returns_list_when_missing():
    handler = PreferredTrajectoryDefaults(n=3)
    val = handler.normalize(None, Modality.PREFERENCE_TRAJECTORY)
    assert isinstance(val, list) and len(val) == 3
    assert all(isinstance(t, Trajectory) for t in val)


def test_preferred_traj_defaults_returns_list_when_empty_iterable():
    handler = PreferredTrajectoryDefaults(n=2)
    val = handler.normalize([], Modality.PREFERENCE_TRAJECTORY)
    assert isinstance(val, list) and len(val) == 2


def test_preferred_traj_defaults_preserves_existing():
    handler = PreferredTrajectoryDefaults(n=5)
    existing = [Trajectory()]
    val = handler.normalize(existing, Modality.PREFERENCE_TRAJECTORY)
    assert val is existing


def test_preferred_traj_defaults_wrong_modality():
    handler = PreferredTrajectoryDefaults()
    with pytest.raises(ValueError):  # modality not allowed
        handler.normalize(None, Modality.SPEED)


def test_preferred_traj_defaults_non_modality_argument():
    handler = PreferredTrajectoryDefaults()
    with pytest.raises(ValueError):  # not a Modality enum
        handler.normalize(None, "intent")  # type: ignore[arg-type]


def test_intent_defaults_returns_unknown_when_none():
    handler = IntentDefaults()
    assert handler.normalize(None, Modality.INTENT) == Intent.UNKNOWN


def test_intent_defaults_pass_through_existing():
    handler = IntentDefaults()
    assert handler.normalize(Intent.GO_LEFT, Modality.INTENT) == Intent.GO_LEFT


def test_intent_defaults_wrong_modality():
    handler = IntentDefaults()
    with pytest.raises(ValueError):
        handler.normalize(None, Modality.SPEED)
