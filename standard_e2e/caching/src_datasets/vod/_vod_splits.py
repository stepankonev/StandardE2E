"""Vendored View-of-Delft scene table (official scene -> frame-range -> split).

VoD's detection release numbers every keyframe globally (``00000`` .. ``09930``)
and the public documentation groups those indices into 24 recording *scenes*
(``delft_1`` .. ``delft_27``; scenes 5/15/17 are absent from the public release)
each carrying an official train / val / test assignment. StandardE2E needs the
scene grouping to build per-segment past/future ego trajectories -- a segment
must never span two recordings -- and needs the split to select frames; both come
from this single table.

The table is transcribed verbatim from the VoD devkit's
``docs/SENSORS_AND_DATA.md`` (Apache-2.0); no devkit code is imported. The docs
label the validation scenes ``Valid`` -- normalised to ``"val"`` here.

Directory layout consequence: train and val frames share ``lidar/training/``
(labelled), while test-scene frames live under ``lidar/testing/`` with sensor
data only (the test labels are withheld by the benchmark).
"""

from __future__ import annotations

from dataclasses import dataclass

ALLOWED_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class VodScene:
    """One VoD recording scene: a contiguous global frame range + its split."""

    name: str
    start_frame: int  # inclusive
    end_frame: int  # inclusive
    split: str  # "train" | "val" | "test"

    @property
    def subdir(self) -> str:
        """Top-level sensor folder holding this scene's frames.

        Train and val frames are both stored under ``training/``; only test
        frames live under ``testing/`` (labels withheld there).
        """
        return "testing" if self.split == "test" else "training"

    def contains(self, frame_index: int) -> bool:
        return self.start_frame <= frame_index <= self.end_frame


# (name, start_frame, end_frame, split) -- verbatim from docs/SENSORS_AND_DATA.md.
_SCENES_RAW: list[tuple[str, int, int, str]] = [
    ("delft_1", 0, 543, "val"),
    ("delft_2", 544, 1311, "train"),
    ("delft_3", 1312, 1802, "train"),
    ("delft_4", 1803, 2199, "train"),
    ("delft_6", 2200, 2531, "train"),
    ("delft_7", 2532, 2797, "test"),
    ("delft_8", 2798, 3276, "test"),
    ("delft_9", 3277, 3574, "train"),
    ("delft_10", 3575, 3609, "val"),
    ("delft_11", 3610, 4047, "train"),
    ("delft_12", 4049, 4386, "train"),
    ("delft_13", 4387, 4651, "train"),
    ("delft_14", 4652, 5085, "val"),
    ("delft_16", 6334, 6570, "test"),
    ("delft_18", 6571, 6758, "test"),
    ("delft_19", 6759, 7542, "train"),
    ("delft_20", 7543, 7899, "test"),
    ("delft_21", 7900, 8197, "test"),
    ("delft_22", 8198, 8480, "val"),
    ("delft_23", 8481, 8748, "train"),
    ("delft_24", 8749, 9095, "train"),
    ("delft_25", 9096, 9517, "test"),
    ("delft_26", 9518, 9775, "train"),
    ("delft_27", 9776, 9930, "train"),
]

VOD_SCENES: list[VodScene] = [VodScene(*row) for row in _SCENES_RAW]


def scenes_for_split(split: str) -> list[VodScene]:
    """All scenes assigned to ``split`` (scene-order). Raises on unknown split."""
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Invalid split {split!r}; must be one of {ALLOWED_SPLITS}.")
    return [scene for scene in VOD_SCENES if scene.split == split]


def scene_for_frame(frame_index: int) -> VodScene | None:
    """The scene whose range contains ``frame_index`` (``None`` if in a gap).

    Used to cross-check official ImageSets frame lists against the vendored
    grouping; gaps correspond to frames dropped from the public release.
    """
    for scene in VOD_SCENES:
        if scene.contains(frame_index):
            return scene
    return None
