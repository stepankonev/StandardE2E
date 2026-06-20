"""KITScenes Multimodal split handling.

KITScenes (KIT / MRT, arXiv 2606.02956) ships ~1000 recording scenes, each a
directory named by a scene UUID, grouped into the official **geo-disjoint**
splits the devkit publishes under
``kitscenes/split/generated_splits/default_geo_split_v1_0/`` (Apache-2.0):
``train`` (534), ``validation`` (117), ``test`` (206), ``test-e2e`` (127) and
``overlap_train_val`` (23).

The Hugging Face release organises the tarballs **by split**
(``data/<split>/<scene_uuid>.tar``), so once a split is downloaded and
extracted the split is encoded in the directory tree -- we select scenes by the
on-disk split folder rather than vendoring ~1000 UUIDs. :mod:`._kitscenes_io`
resolves scene directories for the requested split (and falls back to "every
scene present" for a flat layout such as the single-scene sample, where ``split``
becomes a passthrough output label). This module only defines the user-facing
split names and how each maps to the official split-folder name(s).
"""

from __future__ import annotations

# User-facing splits accepted on the CLI / in ``allowed_splits``. ``all`` is an
# escape hatch that selects every scene found under ``--input_path`` regardless
# of any split folder (useful for the flat single-scene sample).
ALLOWED_SPLITS: tuple[str, ...] = (
    "train",
    "val",
    "test",
    "test_e2e",
    "overlap_train_val",
    "all",
)

# User-facing split -> the split-folder name(s) it may appear under on disk.
# Both the underscored form and the devkit's hyphenated / spelled-out form are
# accepted so either the HF layout or a hand-arranged tree resolves. ``all`` has
# no folder constraint (handled specially in :mod:`._kitscenes_io`).
SPLIT_DIR_ALIASES: dict[str, tuple[str, ...]] = {
    "train": ("train",),
    "val": ("val", "validation"),
    "test": ("test",),
    "test_e2e": ("test_e2e", "test-e2e"),
    "overlap_train_val": ("overlap_train_val",),
    "all": (),
}


def split_dir_names(split: str) -> tuple[str, ...]:
    """Return the on-disk split-folder name(s) a user-facing ``split`` maps to.

    Raises ``ValueError`` for an unknown split so a typo fails fast rather than
    silently selecting nothing.
    """
    if split not in SPLIT_DIR_ALIASES:
        raise ValueError(
            f"Invalid split {split!r}; must be one of {sorted(ALLOWED_SPLITS)}."
        )
    return SPLIT_DIR_ALIASES[split]
