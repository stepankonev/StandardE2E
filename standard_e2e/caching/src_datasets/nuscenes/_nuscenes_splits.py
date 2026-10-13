"""nuScenes split labels StandardE2E accepts, with their scenes and metadata version.

The official scene lists come from nuscenes-devkit
(:func:`nuscenes.utils.splits.create_splits_scenes`); this module only adds the
metadata version each split lives in.
"""

from __future__ import annotations

from nuscenes.utils.splits import create_splits_scenes

# Each split lives in exactly one downloaded metadata version. These are the
# only split labels the converter accepts.
VERSION_FOR_SPLIT: dict[str, str] = {
    "train": "v1.0-trainval",
    "val": "v1.0-trainval",
    "test": "v1.0-test",
    "mini_train": "v1.0-mini",
    "mini_val": "v1.0-mini",
}


def scenes_for_split(split: str) -> set[str]:
    """Scene names belonging to ``split`` (raises ``KeyError`` if unknown)."""
    if split not in VERSION_FOR_SPLIT:
        raise KeyError(split)
    return set(create_splits_scenes()[split])
