# flake8: noqa: E501
"""Tests for the KITScenes LongTail processor.

* **Unit tests** (always run, no real data): the fixed-rig calibration (axis
  convention + shapes), the ``driving_instruction`` -> ``Intent`` folding, the
  trajectory ``-100`` withheld sentinel, the inline-JPEG per-timestep frame
  decode, the ego-relative history re-expression, per-timestep ``FrameRef``
  enumeration on a synthetic parquet, and the full
  ``_prepare_standardized_frame_data`` unroll on a synthetic scenario
  (per-frame cameras + ego-relative past; future / counterfactual preference /
  prediction-frame target attached to the t=0 frame only).
* **Real-frame checks** (skipped unless ``KITSCENES_LONGTAIL_ROOT`` points at a
  downloaded LongTail dataset root): the first ``train`` scenario unrolls into
  one frame per timestep; the t=0 frame yields six ring cameras, a full past, a
  5 s expert future, and present counterfactual preference trajectories.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from standard_e2e.caching.src_datasets.kitscenes_longtail import (
    _kitscenes_longtail_io as io,
)
from standard_e2e.caching.src_datasets.kitscenes_longtail._kitscenes_longtail_calib import (
    CAMERA_TO_DIRECTION,
    build_calibrations,
)
from standard_e2e.caching.src_datasets.kitscenes_longtail.kitscenes_longtail_dataset_processor import (
    KITScenesLongTailDatasetProcessor,
    ego_relative_history,
    intent_from_instruction,
)
from standard_e2e.constants import PREFERENCE_TRAJECTORIES_KEY
from standard_e2e.enums import CameraDirection, Intent, StandardFrameDataField
from standard_e2e.enums import TrajectoryComponent as TC
from standard_e2e.utils import decode_image_bytes

_LONGTAIL_ROOT = os.environ.get("KITSCENES_LONGTAIL_ROOT", "")
_CAMERA_KEYS = list(CAMERA_TO_DIRECTION)


# --------------------------------------------------------------------------- #
# Calibration unit tests (no data required)
# --------------------------------------------------------------------------- #
def test_build_calibrations_covers_six_ring_cameras():
    calibs = build_calibrations()
    assert set(calibs) == set(CAMERA_TO_DIRECTION.values())
    assert len(calibs) == 6
    for calib in calibs.values():
        assert calib.intrinsics.shape == (3, 3) and calib.intrinsics.dtype == np.float32
        assert calib.extrinsics.shape == (4, 4)
        # A rigid camera->ego transform: orthonormal rotation, det +1.
        rot = calib.extrinsics[:3, :3]
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-5)
        assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-5)


def test_calibration_optical_axes_point_outward():
    # extrinsics is T_ego_from_camera; the camera optical axis (cam +z) in ego is
    # its third column. Front looks +x (forward), rear looks -x (backward), the
    # left cameras have +y (left) and the right cameras -y components.
    calibs = build_calibrations()

    def axis(direction: CameraDirection) -> np.ndarray:
        return calibs[direction].extrinsics[:3, :3] @ np.array([0.0, 0.0, 1.0])

    assert axis(CameraDirection.FRONT)[0] > 0.99
    assert axis(CameraDirection.REAR)[0] < -0.99
    assert axis(CameraDirection.FRONT_LEFT)[1] > 0.3
    assert axis(CameraDirection.FRONT_RIGHT)[1] < -0.3
    assert axis(CameraDirection.REAR_LEFT)[1] > 0.3
    assert axis(CameraDirection.REAR_RIGHT)[1] < -0.3


# --------------------------------------------------------------------------- #
# Intent folding (no data required)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "instruction,expected",
    [
        ("turn left", Intent.GO_LEFT),
        ("use left lane", Intent.GO_LEFT),
        ("turn right", Intent.GO_RIGHT),
        ("use right lane", Intent.GO_RIGHT),
        ("overtake car driving on the right", Intent.GO_RIGHT),
        ("drive straight on", Intent.GO_STRAIGHT),
        ("u-turn", Intent.UNKNOWN),
    ],
)
def test_intent_from_instruction(instruction, expected):
    assert intent_from_instruction(instruction) is expected


# --------------------------------------------------------------------------- #
# Ego-relative history re-expression (no data required)
# --------------------------------------------------------------------------- #
def test_ego_relative_history_last_frame_is_identity():
    # The last frame is the native t=0 ego frame: history kept as-is.
    past = np.array([[-3.0, 0.0], [-2.0, 0.0], [-1.0, 0.0], [-0.1, 0.05]])
    traj = ego_relative_history(past, frame_index=3)
    xy = traj.get([TC.X, TC.Y])
    assert traj.length == 4
    np.testing.assert_allclose(xy, past, atol=1e-6)
    np.testing.assert_allclose(traj.get(TC.TIMESTAMP).reshape(-1)[-1], 0.0)


def test_ego_relative_history_intermediate_is_ego_centric():
    # Straight +x motion: at an intermediate frame the ego is at the origin and
    # the history trails behind along -x (y ~ 0).
    past = np.array([[-3.0, 0.0], [-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    traj = ego_relative_history(past, frame_index=2)
    xy = traj.get([TC.X, TC.Y])
    assert traj.length == 3
    np.testing.assert_allclose(xy[-1], [0.0, 0.0], atol=1e-6)  # ego at origin
    assert xy[0, 0] < -1.0 and abs(xy[0, 1]) < 1e-6  # history behind, on-axis
    np.testing.assert_allclose(traj.get(TC.TIMESTAMP).reshape(-1), [-0.4, -0.2, 0.0])


def test_ego_relative_history_rotates_into_heading():
    # A left turn (last leg heads +y): the ego faces +y, so a point that was
    # straight behind in world maps to the ego's rear (-x), not its side.
    past = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])  # moving +y
    xy = ego_relative_history(past, frame_index=1).get([TC.X, TC.Y])
    np.testing.assert_allclose(xy[-1], [0.0, 0.0], atol=1e-6)  # ego at origin
    np.testing.assert_allclose(xy[0], [-1.0, 0.0], atol=1e-6)  # prior step is behind


# --------------------------------------------------------------------------- #
# IO unit tests (synthetic data)
# --------------------------------------------------------------------------- #
def test_trajectory_xy_valid_and_withheld_sentinel():
    valid = {"k": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}
    arr, ok = io.trajectory_xy(valid, "k")
    assert ok and arr.shape == (3, 2)
    np.testing.assert_allclose(arr[0], [1.0, 2.0])
    # The (1, 2) [[-100, -100]] sentinel is "withheld / not applicable".
    withheld = {"k": [[-100.0, -100.0]]}
    arr2, ok2 = io.trajectory_xy(withheld, "k")
    assert not ok2 and arr2.shape == (1, 2)


def _jpeg_cell(n_frames: int, colors: list[tuple[int, int, int]]):
    """A camera cell: a list of ``{"bytes", "path"}`` JPEG frames."""
    cell = []
    for i in range(n_frames):
        bgr = np.full((4, 6, 3), colors[i % len(colors)], dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", bgr)
        assert ok
        cell.append({"bytes": buf.tobytes(), "path": f"{i:010d}.jpg"})
    return cell


def test_decode_frame_indexes_and_clamps():
    # First BGR blue (255,0,0)->RGB(0,0,255); last BGR red (0,0,255)->RGB(255,0,0).
    cell = _jpeg_cell(2, [(255, 0, 0), (0, 0, 255)])
    first = io.decode_frame(cell, 0)
    assert first is not None and first.shape == (4, 6, 3) and first.dtype == np.uint8
    assert first[0, 0, 2] > 200 and first[0, 0, 0] < 60  # RGB blue
    last = io.decode_frame(cell, 1)
    assert last[0, 0, 0] > 200 and last[0, 0, 2] < 60  # RGB red
    # Out-of-range index clamps to the last available frame (ragged sequences).
    np.testing.assert_array_equal(io.decode_frame(cell, 9), last)
    assert io.decode_frame([], 0) is None


def test_decode_image_bytes_roundtrip_rgb():
    bgr = np.zeros((3, 3, 3), dtype=np.uint8)
    bgr[..., 2] = 255  # BGR red
    ok, buf = cv2.imencode(".png", bgr)  # PNG: lossless for an exact check
    assert ok
    rgb = decode_image_bytes(buf.tobytes())
    assert rgb.shape == (3, 3, 3) and rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb[0, 0], [255, 0, 0])  # RGB red


# --------------------------------------------------------------------------- #
# Synthetic shard: schema-faithful parquet for IO + processor coverage
# --------------------------------------------------------------------------- #
_TRAJ_KEYS = [io.PAST_KEY, io.EXPERT_KEY, *io.COUNTERFACTUAL_KEYS]
_N_FRAMES = 4  # synthetic scenario length (camera frames == past points)


def _xy(n: int, x0: float) -> list[list[float]]:
    return [[x0 + 0.2 * i, 0.0] for i in range(n)]


def _write_shard(path, scenarios: list[dict]) -> None:
    """Write a LongTail-schema parquet (one row-group per scenario)."""
    image_struct = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    cam_type = pa.list_(image_struct)
    traj_type = pa.struct([(k, pa.list_(pa.list_(pa.float64()))) for k in _TRAJ_KEYS])
    schema = pa.schema(
        [("scenario_id", pa.string())]
        + [(f"{io.CAMERA_COLUMN_PREFIX}{k}", cam_type) for k in _CAMERA_KEYS]
        + [
            ("driving_instruction", pa.string()),
            ("scenario_type", pa.string()),
            ("trajectory", traj_type),
        ]
    )
    columns: dict[str, pa.Array] = {
        "scenario_id": pa.array([s["scenario_id"] for s in scenarios], pa.string()),
        "driving_instruction": pa.array(
            [s["driving_instruction"] for s in scenarios], pa.string()
        ),
        "scenario_type": pa.array([s["scenario_type"] for s in scenarios], pa.string()),
        "trajectory": pa.array([s["trajectory"] for s in scenarios], traj_type),
    }
    for key in _CAMERA_KEYS:
        columns[f"{io.CAMERA_COLUMN_PREFIX}{key}"] = pa.array(
            [s["cells"][key] for s in scenarios], cam_type
        )
    table = pa.table(columns, schema=schema)
    pq.write_table(table, path, row_group_size=1)


def _scenario(scenario_id: str, instruction: str, with_counterfactuals: bool) -> dict:
    sentinel = [[-100.0, -100.0]]
    trajectory = {
        io.PAST_KEY: _xy(_N_FRAMES, -4.0),  # past length == camera frame count
        io.EXPERT_KEY: _xy(25, 0.2),
        "wrong_speed": _xy(25, 0.1) if with_counterfactuals else sentinel,
        "neglect_instruction": sentinel,
        "off_road": _xy(25, 0.15) if with_counterfactuals else sentinel,
        "crash": sentinel,
    }
    cells = {k: _jpeg_cell(_N_FRAMES, [(0, 0, 255)]) for k in _CAMERA_KEYS}
    return {
        "scenario_id": scenario_id,
        "driving_instruction": instruction,
        "scenario_type": "4 intersection",
        "trajectory": trajectory,
        "cells": cells,
    }


def test_iter_frame_refs_one_per_timestep(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_shard(
        data_dir / "train-00000-of-00001.parquet",
        [
            _scenario("000", "turn left", True),
            _scenario("001", "turn right", True),
        ],
    )
    # Two scenarios x _N_FRAMES timesteps each, scenario- then frame-major.
    refs = list(io.iter_frame_refs(str(tmp_path), "train"))
    assert len(refs) == 2 * _N_FRAMES
    assert [(r.row_group, r.frame_index) for r in refs[:_N_FRAMES]] == [
        (0, i) for i in range(_N_FRAMES)
    ]
    # Pointing at data/ resolves the same way as the root holding data/.
    assert list(io.iter_frame_refs(str(data_dir), "train")) == refs


def test_iter_frame_refs_missing_split_raises(tmp_path):
    (tmp_path / "data").mkdir()
    with pytest.raises(FileNotFoundError):
        list(io.iter_frame_refs(str(tmp_path), "test"))


# --------------------------------------------------------------------------- #
# Processor unroll on a synthetic scenario
# --------------------------------------------------------------------------- #
def _processor(tmp_path, scenario: dict, adapters=None):
    shard = tmp_path / "data" / "train-00000-of-00001.parquet"
    shard.parent.mkdir(exist_ok=True)
    _write_shard(shard, [scenario])
    proc = KITScenesLongTailDatasetProcessor(
        common_output_path=str(tmp_path / "out"),
        split="train",
        adapters=adapters,
        context_aggregators=[],
    )
    return proc, str(shard)


def test_prediction_frame_full_mapping(tmp_path):
    proc, shard = _processor(
        tmp_path, _scenario("000400", "turn left", with_counterfactuals=True)
    )
    fd = proc._prepare_standardized_frame_data(io.FrameRef(shard, 0, _N_FRAMES - 1))

    assert fd.segment_id == "000400" and fd.frame_id == _N_FRAMES - 1
    assert fd.timestamp == (_N_FRAMES - 1) * 0.2
    assert fd.extra_index_data["is_prediction_frame"] is True
    # Six ring cameras, each pinhole-calibrated.
    assert set(fd.cameras) == set(CAMERA_TO_DIRECTION.values())
    cam = fd.cameras[CameraDirection.FRONT]
    assert cam.image.ndim == 3 and cam.intrinsics.shape == (3, 3)
    assert cam.extrinsics.shape == (4, 4)
    # Full past at t=0; expert future (25 pts); counterfactual preference set.
    assert fd.past_states.length == _N_FRAMES and fd.future_states.length == 25
    pref = fd.aux_data[PREFERENCE_TRAJECTORIES_KEY]
    assert [t.length for t in pref] == [25, 25]
    assert fd.aux_data["preference_trajectory_labels"] == ["wrong_speed", "off_road"]
    assert fd.intent is Intent.GO_LEFT
    assert fd.extra_index_data["driving_instruction"] == "turn left"
    assert fd.extra_index_data[f"has_{PREFERENCE_TRAJECTORIES_KEY}"] is True


def test_observation_frame_has_no_target(tmp_path):
    proc, shard = _processor(
        tmp_path, _scenario("000400", "turn left", with_counterfactuals=True)
    )
    fd = proc._prepare_standardized_frame_data(io.FrameRef(shard, 0, 1))

    assert fd.frame_id == 1 and fd.extra_index_data["is_prediction_frame"] is False
    assert set(fd.cameras) == set(CAMERA_TO_DIRECTION.values())
    # Observation frame: ego-relative history (ego at origin), no prediction target.
    assert fd.past_states.length == 2
    np.testing.assert_allclose(
        fd.past_states.get([TC.X, TC.Y])[-1], [0.0, 0.0], atol=1e-6
    )
    assert fd.future_states is None
    assert fd.aux_data[PREFERENCE_TRAJECTORIES_KEY] is None
    assert fd.extra_index_data[f"has_{PREFERENCE_TRAJECTORIES_KEY}"] is False
    assert fd.intent is Intent.GO_LEFT  # scenario command on every frame


def test_prediction_frame_without_counterfactuals(tmp_path):
    proc, shard = _processor(
        tmp_path, _scenario("000401", "drive straight on", with_counterfactuals=False)
    )
    fd = proc._prepare_standardized_frame_data(io.FrameRef(shard, 0, _N_FRAMES - 1))
    assert fd.future_states.length == 25  # expert still present
    assert fd.aux_data[PREFERENCE_TRAJECTORIES_KEY] is None
    assert fd.aux_data["preference_trajectory_labels"] == []
    assert fd.intent is Intent.GO_STRAIGHT


def test_prepare_frame_skips_cameras_when_not_consumed(tmp_path):
    from standard_e2e.caching.adapters import FutureStatesIdentityAdapter

    proc, shard = _processor(
        tmp_path,
        _scenario("000402", "turn right", with_counterfactuals=True),
        adapters=[FutureStatesIdentityAdapter()],
    )
    assert not proc.needs_attr(StandardFrameDataField.CAMERAS)
    fd = proc._prepare_standardized_frame_data(io.FrameRef(shard, 0, _N_FRAMES - 1))
    assert fd.cameras == {}
    assert fd.future_states.length == 25


# --------------------------------------------------------------------------- #
# Real-frame checks (skipped without data)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_refs_and_proc(tmp_path_factory):
    if not _LONGTAIL_ROOT or not os.path.isdir(_LONGTAIL_ROOT):
        pytest.skip(
            "no LongTail data (set KITSCENES_LONGTAIL_ROOT to the dataset root)"
        )
    refs = list(io.iter_frame_refs(_LONGTAIL_ROOT, "train"))
    if not refs:
        pytest.skip("no train shards under KITSCENES_LONGTAIL_ROOT")
    out = tmp_path_factory.mktemp("longtail")
    proc = KITScenesLongTailDatasetProcessor(
        common_output_path=str(out), split="train", context_aggregators=[]
    )
    return refs, proc


def test_real_scenario_unrolls_into_frames(real_refs_and_proc):
    refs, _proc = real_refs_and_proc
    # The first scenario contributes several consecutive frames (a 4 s window).
    first_key = (refs[0].shard_path, refs[0].row_group)
    n_first = sum(1 for r in refs if (r.shard_path, r.row_group) == first_key)
    assert n_first > 1
    assert [r.frame_index for r in refs[:n_first]] == list(range(n_first))


def test_real_prediction_frame_has_cameras_and_targets(real_refs_and_proc):
    refs, proc = real_refs_and_proc
    first_key = (refs[0].shard_path, refs[0].row_group)
    last_of_first = max(
        (r for r in refs if (r.shard_path, r.row_group) == first_key),
        key=lambda r: r.frame_index,
    )
    fd = proc._prepare_standardized_frame_data(last_of_first)
    assert set(fd.cameras) == set(CAMERA_TO_DIRECTION.values())
    for cam in fd.cameras.values():
        assert cam.image.ndim == 3 and cam.image.shape[2] == 3
        assert cam.intrinsics.shape == (3, 3) and np.isfinite(cam.extrinsics).all()
    assert fd.past_states.length == fd.frame_id + 1 and fd.future_states.length == 25
    pref = fd.aux_data[PREFERENCE_TRAJECTORIES_KEY]
    assert pref is not None and len(pref) >= 1
