"""Regression: dict-collate must not misclassify generic dicts as camera batches.

``_is_cameras_batch`` early-exits on the first key — but if that early-exit
ever regresses to "any dict is a cameras batch", a plain ``aux_data``-style
dict would silently route through ``collate_cameras_fn`` and crash on the
``BatchedCameraData`` validator. This test pins the per-key recurse path
for non-camera dicts.
"""

from __future__ import annotations

import torch

from standard_e2e.data_structures import BatchedCameraData
from standard_e2e.data_structures.frame_data import collate_modalities


def test_non_camera_dict_batch_recurses_per_key():
    """A batch of generic dicts is collated per-key, not as cameras."""
    batch = [
        {"a": 1, "b": 2.0},
        {"a": 3, "b": 4.0},
    ]
    out = collate_modalities(batch)
    assert isinstance(out, dict)
    assert not isinstance(out, BatchedCameraData)
    assert isinstance(out["a"], torch.Tensor)
    assert out["a"].tolist() == [1, 3]
    assert isinstance(out["b"], torch.Tensor)
    assert out["b"].tolist() == [2.0, 4.0]


def test_dict_with_string_keys_not_camera_batch():
    """String-keyed dicts must not match the cameras-batch heuristic."""
    batch = [
        {"intent": 1, "speed": 5.5},
        {"intent": 2, "speed": 6.5},
    ]
    out = collate_modalities(batch)
    assert isinstance(out, dict)
    assert not isinstance(out, BatchedCameraData)
