"""Default modality settings (moved from standard_e2e.modality_defaults).

Public re-export location for modality defaults now under dataset namespace.
"""

from standard_e2e.enums import Modality

from .modality_defaults import (
    IntentDefaults,
    ModalityDefaults,
    PreferredTrajectoryDefaults,
)


def _check_modality_defaults_dict(
    modality_defaults: dict[Modality, ModalityDefaults] | None,
) -> None:
    """Validate a modality->defaults mapping used by frame loaders/datasets.

    Raises ``TypeError`` when keys are not ``Modality`` or values are not
    ``ModalityDefaults`` instances. ``None`` is accepted to indicate the absence
    of defaults.
    """

    if modality_defaults is None:
        return
    if not isinstance(modality_defaults, dict):
        raise TypeError(
            f"modality_defaults must be a dictionary, got {type(modality_defaults)}"
        )
    for modality, default_handler in modality_defaults.items():
        if not isinstance(modality, Modality):
            raise TypeError(
                f"modality_defaults keys must be Modality, got {type(modality)}"
            )
        if not isinstance(default_handler, ModalityDefaults):
            raise TypeError(
                "modality_defaults values must be ModalityDefaults, "
                f"got {type(default_handler)}"
            )


__all__ = [
    "IntentDefaults",
    "ModalityDefaults",
    "PreferredTrajectoryDefaults",
    "_check_modality_defaults_dict",
]
