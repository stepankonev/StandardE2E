# flake8: noqa: E501
"""Tests for the nuScenes processor.

* **Unit tests** (always run): the ``category_name`` -> ``DetectionType``
  taxonomy (incl. the two-wheeler split and the unknown fallback), the
  ``CAM_*`` -> ``CameraDirection`` mapping, the global -> ego box transform
  (translation, +90 deg rotation, ``size`` -> length/width/height reorder), the
  lidar ``float32 x 5`` -> xyz decode, the vendored split lists, and -- on a
  synthetic JSON tree -- the reverse-index reconstruction
  (``sample['data']``/``anns``, keyframe filtering, channel + category joins)
  plus a full hermetic processor build with tiny on-disk image / lidar files.
* **Real-frame checks** (skipped unless ``NUSCENES_DATAROOT`` points at an
  extracted dataroot): a built frame exposes the 6 surround cameras under their
  canonical ``CameraDirection`` with pinhole intrinsics and 4x4 extrinsics,
  ego-frame LiDAR, ego-frame detections, and a rigid ego pose.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from standard_e2e.caching.src_datasets.nuscenes import _nuscenes_arcline as arc
from standard_e2e.caching.src_datasets.nuscenes import _nuscenes_io as io
from standard_e2e.caching.src_datasets.nuscenes._nuscenes_map import NuscMap
from standard_e2e.caching.src_datasets.nuscenes._nuscenes_splits import (
    SPLITS,
    VERSION_FOR_SPLIT,
    scenes_for_split,
)
from standard_e2e.caching.src_datasets.nuscenes.nuscenes_dataset_processor import (
    _CHANNEL_TO_DIRECTION,
    NuscenesDatasetProcessor,
    box_to_ego_xyzhwl,
    detection_type_for_category,
)
from standard_e2e.enums import CameraDirection, DetectionType, MapElementType
from standard_e2e.enums import TrajectoryComponent as TC
from standard_e2e.utils import quat_wxyz_to_rotmat, se3

# Real-frame checks read nuScenes only from this env var; they skip when unset.
_NUSCENES_DATAROOT = os.environ.get("NUSCENES_DATAROOT", "")
_NUSCENES_VERSION = os.environ.get("NUSCENES_VERSION", "v1.0-mini")


# --------------------------------------------------------------------------- #
# Taxonomy / mapping unit tests (no data required)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "category_name,expected",
    [
        ("vehicle.car", DetectionType.VEHICLE),
        ("vehicle.truck", DetectionType.VEHICLE),
        ("vehicle.bus.rigid", DetectionType.VEHICLE),
        ("vehicle.trailer", DetectionType.VEHICLE),
        ("vehicle.construction", DetectionType.VEHICLE),
        ("vehicle.emergency.police", DetectionType.VEHICLE),
        ("vehicle.bicycle", DetectionType.BICYCLE),
        ("vehicle.motorcycle", DetectionType.BICYCLE),
        ("human.pedestrian.adult", DetectionType.PEDESTRIAN),
        ("human.pedestrian.child", DetectionType.PEDESTRIAN),
        ("animal", DetectionType.PEDESTRIAN),
        ("movable_object.barrier", DetectionType.UNKNOWN),
        ("movable_object.trafficcone", DetectionType.UNKNOWN),
        ("movable_object.debris", DetectionType.UNKNOWN),
        ("static_object.bicycle_rack", DetectionType.UNKNOWN),
    ],
)
def test_detection_type_for_category(category_name, expected):
    assert detection_type_for_category(category_name) is expected


def test_detection_type_unknown_fallback():
    # A label outside the (frozen) taxonomy never crashes -> UNKNOWN.
    assert detection_type_for_category("future.new_class") is DetectionType.UNKNOWN
    # Two-wheeler check precedes the generic vehicle prefix.
    assert detection_type_for_category("vehicle.bicycle") is DetectionType.BICYCLE


def test_channel_to_direction_is_exact_six():
    assert set(_CHANNEL_TO_DIRECTION) == {
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    }
    # 6 distinct canonical directions, and CAM_BACK maps to REAR (not a new slot).
    assert len(set(_CHANNEL_TO_DIRECTION.values())) == 6
    assert _CHANNEL_TO_DIRECTION["CAM_BACK"] is CameraDirection.REAR
    assert _CHANNEL_TO_DIRECTION["CAM_BACK_LEFT"] is CameraDirection.REAR_LEFT


# --------------------------------------------------------------------------- #
# Geometry unit tests (no data required)
# --------------------------------------------------------------------------- #
def _pose(translation, quat_wxyz=(1.0, 0.0, 0.0, 0.0)) -> np.ndarray:
    return se3(quat_wxyz_to_rotmat(quat_wxyz), np.asarray(translation, float))


def _box(center, wlh=(2.0, 4.0, 1.5), quat_wxyz=(1.0, 0.0, 0.0, 0.0)) -> io.NuscBox:
    return io.NuscBox(
        category_name="vehicle.car",
        instance_token="inst",
        center_global=np.asarray(center, dtype=np.float64),
        wlh=np.asarray(wlh, dtype=np.float64),
        rotation_wxyz=np.asarray(quat_wxyz, dtype=np.float64),
    )


def test_box_to_ego_identity_reorders_size():
    # Identity ego pose: the global box passes through; size (w,l,h) -> l,w,h.
    t_ego_from_global = np.linalg.inv(_pose([0.0, 0.0, 0.0]))
    x, y, z, heading, length, width, height = box_to_ego_xyzhwl(
        _box([10.0, 0.0, 0.0], wlh=(2.0, 4.0, 1.5)), t_ego_from_global
    )
    np.testing.assert_allclose([x, y, z, heading], [10.0, 0.0, 0.0, 0.0], atol=1e-9)
    assert (length, width, height) == (4.0, 2.0, 1.5)


def test_box_to_ego_translation_only():
    # Ego sits at global (10, 0, 0): a box there lands at the ego origin.
    t_ego_from_global = np.linalg.inv(_pose([10.0, 0.0, 0.0]))
    x, y, z, *_ = box_to_ego_xyzhwl(_box([10.0, 0.0, 0.0]), t_ego_from_global)
    np.testing.assert_allclose([x, y, z], [0.0, 0.0, 0.0], atol=1e-9)


def test_box_to_ego_rotation_maps_axes_and_heading():
    # Ego yawed +90 deg about z (wxyz = (cos45, 0, 0, sin45)).
    s = np.sqrt(0.5)
    t_ego_from_global = np.linalg.inv(_pose([0.0, 0.0, 0.0], (s, 0.0, 0.0, s)))
    # A global +x box appears on the ego -y axis...
    x, y, z, heading, *_ = box_to_ego_xyzhwl(_box([10.0, 0.0, 0.0]), t_ego_from_global)
    np.testing.assert_allclose([x, y], [0.0, -10.0], atol=1e-9)
    # ...and a globally-axis-aligned box has ego heading -90 deg.
    np.testing.assert_allclose(heading, -np.pi / 2, atol=1e-9)


# --------------------------------------------------------------------------- #
# IO unit tests (synthetic files)
# --------------------------------------------------------------------------- #
def test_read_lidar_xyz_keeps_first_three_columns(tmp_path):
    points = np.arange(2 * 5, dtype=np.float32).reshape(2, 5)
    path = tmp_path / "a.pcd.bin"
    points.tofile(path)
    xyz = io.read_lidar_xyz(str(path))
    assert xyz.shape == (2, 3) and xyz.dtype == np.float32
    np.testing.assert_allclose(xyz, points[:, :3])


def test_read_lidar_xyz_empty_and_bad_size(tmp_path):
    empty = tmp_path / "empty.pcd.bin"
    empty.write_bytes(b"")
    assert io.read_lidar_xyz(str(empty)).shape == (0, 3)
    bad = tmp_path / "bad.pcd.bin"
    np.arange(7, dtype=np.float32).tofile(bad)  # 7 not divisible by 5
    with pytest.raises(ValueError):
        io.read_lidar_xyz(str(bad))


# --------------------------------------------------------------------------- #
# Split unit tests (no data required)
# --------------------------------------------------------------------------- #
def test_vendored_splits_counts_and_versions():
    assert {k: len(v) for k, v in SPLITS.items()} == {
        "train": 700,
        "val": 150,
        "test": 150,
        "mini_train": 8,
        "mini_val": 2,
    }
    assert scenes_for_split("mini_val") == {"scene-0103", "scene-0916"}
    assert VERSION_FOR_SPLIT["mini_train"] == "v1.0-mini"
    assert VERSION_FOR_SPLIT["train"] == "v1.0-trainval"
    assert VERSION_FOR_SPLIT["test"] == "v1.0-test"
    # train and val are disjoint within the trainval version.
    assert not (set(SPLITS["train"]) & set(SPLITS["val"]))


# --------------------------------------------------------------------------- #
# Synthetic-tree tests for the reverse-index reconstruction + processor build
# --------------------------------------------------------------------------- #
def _write_tree(root: Path, *, version: str = "v1.0-mini") -> Path:
    """Write a minimal but schema-faithful nuScenes metadata tree.

    One scene, two samples (linked), CAM_FRONT + LIDAR_TOP keyframes plus a
    non-keyframe CAM sweep (which must be excluded from ``sample['data']``), and
    two annotations on the first sample (a pedestrian and a car).
    """
    meta = root / version
    meta.mkdir(parents=True)
    intr = [[1000.0, 0.0, 800.0], [0.0, 1000.0, 450.0], [0.0, 0.0, 1.0]]
    tables = {
        "sensor": [
            {"token": "sen_cam", "channel": "CAM_FRONT", "modality": "camera"},
            {"token": "sen_lid", "channel": "LIDAR_TOP", "modality": "lidar"},
        ],
        "category": [
            {"token": "cat_ped", "name": "human.pedestrian.adult", "description": ""},
            {"token": "cat_car", "name": "vehicle.car", "description": ""},
        ],
        "instance": [
            {"token": "inst_1", "category_token": "cat_ped"},
            {"token": "inst_2", "category_token": "cat_car"},
        ],
        "calibrated_sensor": [
            {
                "token": "cs_cam",
                "sensor_token": "sen_cam",
                "translation": [1.0, 0.0, 1.5],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "camera_intrinsic": intr,
            },
            {
                "token": "cs_lid",
                "sensor_token": "sen_lid",
                "translation": [0.9, 0.0, 1.8],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "camera_intrinsic": [],
            },
        ],
        "ego_pose": [
            {
                "token": "ego_0",
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "translation": [100.0, 200.0, 0.0],
                "timestamp": 0,
            },
            {
                "token": "ego_1",
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "translation": [101.0, 200.0, 0.0],
                "timestamp": 500000,
            },
        ],
        "log": [
            {
                "token": "log_0",
                "logfile": "n0",
                "vehicle": "v",
                "date_captured": "2018",
                "location": "singapore-onenorth",
            }
        ],
        "scene": [
            {
                "token": "scene_0",
                "log_token": "log_0",
                "nbr_samples": 2,
                "first_sample_token": "samp_0",
                "last_sample_token": "samp_1",
                "name": "scene-0103",
                "description": "",
            }
        ],
        "sample": [
            {
                "token": "samp_0",
                "timestamp": 0,
                "prev": "",
                "next": "samp_1",
                "scene_token": "scene_0",
            },
            {
                "token": "samp_1",
                "timestamp": 500000,
                "prev": "samp_0",
                "next": "",
                "scene_token": "scene_0",
            },
        ],
        "sample_data": [
            {
                "token": "sd_cam_0",
                "sample_token": "samp_0",
                "ego_pose_token": "ego_0",
                "calibrated_sensor_token": "cs_cam",
                "timestamp": 0,
                "fileformat": "jpg",
                "is_key_frame": True,
                "height": 4,
                "width": 4,
                "filename": "samples/CAM_FRONT/a.jpg",
                "prev": "",
                "next": "",
            },
            {
                "token": "sd_lid_0",
                "sample_token": "samp_0",
                "ego_pose_token": "ego_0",
                "calibrated_sensor_token": "cs_lid",
                "timestamp": 0,
                "fileformat": "bin",
                "is_key_frame": True,
                "height": 0,
                "width": 0,
                "filename": "samples/LIDAR_TOP/a.pcd.bin",
                "prev": "",
                "next": "",
            },
            {
                "token": "sd_cam_0_sweep",
                "sample_token": "samp_0",
                "ego_pose_token": "ego_0",
                "calibrated_sensor_token": "cs_cam",
                "timestamp": 100,
                "fileformat": "jpg",
                "is_key_frame": False,
                "height": 4,
                "width": 4,
                "filename": "sweeps/CAM_FRONT/x.jpg",
                "prev": "",
                "next": "",
            },
            {
                "token": "sd_cam_1",
                "sample_token": "samp_1",
                "ego_pose_token": "ego_1",
                "calibrated_sensor_token": "cs_cam",
                "timestamp": 500000,
                "fileformat": "jpg",
                "is_key_frame": True,
                "height": 4,
                "width": 4,
                "filename": "samples/CAM_FRONT/b.jpg",
                "prev": "",
                "next": "",
            },
            {
                "token": "sd_lid_1",
                "sample_token": "samp_1",
                "ego_pose_token": "ego_1",
                "calibrated_sensor_token": "cs_lid",
                "timestamp": 500000,
                "fileformat": "bin",
                "is_key_frame": True,
                "height": 0,
                "width": 0,
                "filename": "samples/LIDAR_TOP/b.pcd.bin",
                "prev": "",
                "next": "",
            },
        ],
        "sample_annotation": [
            {
                "token": "ann_0",
                "sample_token": "samp_0",
                "instance_token": "inst_1",
                "visibility_token": "1",
                "attribute_tokens": [],
                "translation": [110.0, 200.0, 0.0],
                "size": [0.6, 0.8, 1.7],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "prev": "",
                "next": "",
                "num_lidar_pts": 5,
                "num_radar_pts": 0,
            },
            {
                "token": "ann_1",
                "sample_token": "samp_0",
                "instance_token": "inst_2",
                "visibility_token": "4",
                "attribute_tokens": [],
                "translation": [120.0, 205.0, 0.0],
                "size": [1.9, 4.5, 1.6],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "prev": "",
                "next": "",
                "num_lidar_pts": 30,
                "num_radar_pts": 4,
            },
        ],
    }
    for name, rows in tables.items():
        (meta / f"{name}.json").write_text(json.dumps(rows))
    return root


def test_resolve_frame_rebuilds_reverse_indices(tmp_path):
    root = _write_tree(tmp_path)
    tables = io.NuscTables(str(root), "v1.0-mini")
    # Sample order follows the ``next`` chain.
    assert tables.scene_sample_tokens(tables.scenes[0]) == ["samp_0", "samp_1"]

    frame = tables.resolve_frame("samp_0")
    assert frame.scene_name == "scene-0103"
    assert frame.frame_id == 0
    # The non-keyframe CAM sweep is excluded: exactly one camera (CAM_FRONT).
    assert [c.channel for c in frame.cameras] == ["CAM_FRONT"]
    assert frame.cameras[0].intrinsics.shape == (3, 3)
    assert frame.cameras[0].extrinsics.shape == (4, 4)
    assert frame.cameras[0].image_path.endswith("samples/CAM_FRONT/a.jpg")
    # LIDAR_TOP resolved with ego-frame extrinsics.
    assert frame.lidar_path is not None and frame.lidar_path.endswith("a.pcd.bin")
    assert frame.lidar_extrinsics.shape == (4, 4)
    # category_name resolved via instance -> category (not on the annotation).
    assert {b.category_name for b in frame.detections} == {
        "human.pedestrian.adult",
        "vehicle.car",
    }
    # Ego pose is the LIDAR_TOP ego_pose (translation passthrough).
    np.testing.assert_allclose(frame.pose_global_from_ego[:3, 3], [100.0, 200.0, 0.0])
    # The second sample carries no annotations.
    assert tables.resolve_frame("samp_1").detections == ()


def test_scene_has_sensor_data_detects_partial_download(tmp_path):
    # The converter uses this to skip scenes whose sensor blob isn't on disk yet
    # (the trainval *_blobs.tgz arrive one at a time), so a partial download
    # converts cleanly instead of crashing on a missing file.
    root = _write_tree(tmp_path)
    tables = io.NuscTables(str(root), "v1.0-mini")
    scene = tables.scenes[0]
    # Metadata present but no sensor files materialised yet -> not downloaded.
    assert tables.scene_has_sensor_data(scene) is False
    # Materialise the first sample's LIDAR_TOP file -> downloaded.
    (root / "samples/LIDAR_TOP").mkdir(parents=True)
    (root / "samples/LIDAR_TOP/a.pcd.bin").write_bytes(b"\x00" * 20)
    assert tables.scene_has_sensor_data(scene) is True


def test_processor_builds_frame_hermetically(tmp_path):
    root = _write_tree(tmp_path)
    # Materialise the tiny sensor files the processor reads.
    (root / "samples/CAM_FRONT").mkdir(parents=True)
    (root / "samples/LIDAR_TOP").mkdir(parents=True)
    cv2.imwrite(
        str(root / "samples/CAM_FRONT/a.jpg"),
        np.zeros((4, 4, 3), dtype=np.uint8),
    )
    # 6 lidar points: float32 x 5 columns (x, y, z, intensity, ring).
    lidar = np.array([[c, c + 1, c + 2, 0.5, 0] for c in range(6)], dtype=np.float32)
    lidar.tofile(root / "samples/LIDAR_TOP/a.pcd.bin")

    tables = io.NuscTables(str(root), "v1.0-mini")
    frame = tables.resolve_frame("samp_0")

    out = tmp_path / "out"
    proc = NuscenesDatasetProcessor(
        common_output_path=str(out), split="mini_val", context_aggregators=[]
    )
    fd = proc._prepare_standardized_frame_data(frame)

    assert fd.segment_id == "scene-0103"
    assert set(fd.cameras) == {CameraDirection.FRONT}
    assert fd.cameras[CameraDirection.FRONT].image.shape == (4, 4, 3)
    # Lidar moved into the ego frame (cs_lid translation (0.9, 0, 1.8) added).
    xyz = fd.lidar.points[["x", "y", "z"]].to_numpy()
    assert xyz.shape == (6, 3)
    np.testing.assert_allclose(xyz[0], [0.9, 1.0, 3.8], atol=1e-5)
    # Two ego-frame detections with the expected coarse types.
    dets = fd.frame_detections_3d.detections
    assert {d.detection_type for d in dets} == {
        DetectionType.PEDESTRIAN,
        DetectionType.VEHICLE,
    }
    # Pedestrian at global (110, 200, 0); ego at (100, 200, 0) -> ego x = 10.
    ped = next(d for d in dets if d.detection_type is DetectionType.PEDESTRIAN)
    np.testing.assert_allclose(
        float(np.asarray(ped.trajectory.get(TC.X)).reshape(-1)[0]), 10.0, atol=1e-6
    )


# --------------------------------------------------------------------------- #
# Real-frame checks (skipped without data)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def built():
    if not _NUSCENES_DATAROOT or not Path(_NUSCENES_DATAROOT).exists():
        pytest.skip("no nuScenes data available (set NUSCENES_DATAROOT)")
    try:
        tables = io.NuscTables(_NUSCENES_DATAROOT, _NUSCENES_VERSION)
    except FileNotFoundError:
        pytest.skip(f"no {_NUSCENES_VERSION} metadata under NUSCENES_DATAROOT")
    scene = tables.scenes[0]
    frame = tables.resolve_frame(tables.scene_sample_tokens(scene)[0])
    out = Path(_NUSCENES_DATAROOT).parent / "_nusc_test_out"
    proc = NuscenesDatasetProcessor(
        common_output_path=str(out), split="mini_val", context_aggregators=[]
    )
    return frame, proc._prepare_standardized_frame_data(frame)


def test_real_cameras_are_canonical_with_calibration(built):
    _frame, fd = built
    canonical = set(_CHANNEL_TO_DIRECTION.values())
    assert len(fd.cameras) >= 1
    for direction, cam in fd.cameras.items():
        assert direction in canonical
        assert cam.camera_direction is direction
        assert cam.image.ndim == 3 and cam.image.dtype == np.uint8
        assert cam.image.shape[2] == 3
        assert cam.intrinsics.shape == (3, 3)
        assert cam.extrinsics.shape == (4, 4)
        assert np.isfinite(cam.extrinsics).all()


def test_real_lidar_is_ego_frame(built):
    _frame, fd = built
    assert fd.lidar is not None
    xyz = fd.lidar.points[["x", "y", "z"]].to_numpy()
    assert xyz.dtype == np.float32 and len(xyz) > 0
    assert np.isfinite(xyz).all()
    # LIDAR_TOP returns reach tens of metres in front of the ego.
    assert xyz[:, 0].max() > 20.0


def test_real_detections_are_ego_frame(built):
    _frame, fd = built
    detections = fd.frame_detections_3d.detections
    assert len(detections) > 0
    for det in detections:
        assert isinstance(det.detection_type, DetectionType)
        tr = det.trajectory
        for comp in (TC.X, TC.Y, TC.Z, TC.LENGTH, TC.WIDTH, TC.HEIGHT, TC.HEADING):
            value = float(np.asarray(tr.get(comp)).reshape(-1)[0])
            assert np.isfinite(value)
        for comp in (TC.LENGTH, TC.WIDTH, TC.HEIGHT):
            assert float(np.asarray(tr.get(comp)).reshape(-1)[0]) >= 0.0


def test_real_ego_pose_is_rigid(built):
    _frame, fd = built
    t = fd.aux_data["pose_matrix"]
    assert t.shape == (4, 4) and np.isfinite(t).all()
    np.testing.assert_allclose(t[:3, :3] @ t[:3, :3].T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(t[:3, :3]), 1.0, atol=1e-6)


def test_real_hd_map_present(built):
    _frame, fd = built
    if fd.hd_map is None:
        pytest.skip("map-expansion pack not present in NUSCENES_DATAROOT/maps/")
    types = {e.type for e in fd.hd_map.elements}
    assert MapElementType.LANE_CENTER in types
    assert MapElementType.DRIVABLE_AREA in types
    for element in fd.hd_map.elements:
        assert element.points.ndim == 2 and element.points.shape[1] == 2
        assert np.isfinite(element.points).all()


# --------------------------------------------------------------------------- #
# HD-map unit tests (vendored arcline + synthetic vector map; no data required)
# --------------------------------------------------------------------------- #
def test_arcline_discretize_straight_segment():
    # One 10 m straight middle segment ('S'), zero-length end arcs.
    path = {
        "start_pose": [0.0, 0.0, 0.0],
        "shape": "RSR",
        "radius": 1000.0,
        "segment_length": [0.0, 10.0, 0.0],
    }
    pts = np.array(arc.discretize_lane([path], resolution_meters=1.0))
    assert pts.shape[0] == 11
    np.testing.assert_allclose(pts[0], [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(pts[-1], [10.0, 0.0, 0.0], atol=1e-6)
    assert np.all(np.abs(pts[:, 1]) < 1e-6)  # collinear along x


def _write_map_json(path: Path) -> None:
    """A minimal but schema-faithful map-expansion location: one straight lane,
    a lane divider, a square polygon reused for crossing / drivable / intersection."""
    data = {
        "version": "1.3",
        "node": [
            {"token": "n0", "x": 0.0, "y": 0.0},
            {"token": "n1", "x": 10.0, "y": 0.0},
            {"token": "p0", "x": 0.0, "y": 0.0},
            {"token": "p1", "x": 4.0, "y": 0.0},
            {"token": "p2", "x": 4.0, "y": 4.0},
            {"token": "p3", "x": 0.0, "y": 4.0},
        ],
        "line": [{"token": "line0", "node_tokens": ["n0", "n1"]}],
        "polygon": [
            {
                "token": "poly0",
                "exterior_node_tokens": ["p0", "p1", "p2", "p3"],
                "holes": [],
            }
        ],
        "lane": [{"token": "lane0", "polygon_token": "poly0", "lane_type": "CAR"}],
        "lane_connector": [],
        "arcline_path_3": {
            "lane0": [
                {
                    "start_pose": [0.0, 0.0, 0.0],
                    "shape": "RSR",
                    "radius": 1000.0,
                    "segment_length": [0.0, 10.0, 0.0],
                }
            ]
        },
        "connectivity": {"lane0": {"incoming": [], "outgoing": ["laneX"]}},
        "lane_divider": [
            {"token": "ld0", "line_token": "line0", "lane_divider_segments": []}
        ],
        "road_divider": [],
        "ped_crossing": [
            {"token": "pc0", "polygon_token": "poly0", "road_segment_token": ""}
        ],
        "walkway": [],
        "stop_line": [],
        "drivable_area": [{"token": "da0", "polygon_tokens": ["poly0"]}],
        "road_segment": [
            {
                "token": "rs0",
                "polygon_token": "poly0",
                "is_intersection": True,
                "drivable_area_token": "",
            }
        ],
        "carpark_area": [],
        "traffic_light": [],
    }
    path.write_text(json.dumps(data))


def test_nuscmap_parse_and_build(tmp_path):
    p = tmp_path / "test-loc.json"
    _write_map_json(p)
    nusc_map = NuscMap.from_json(str(p))
    # lane_center + lane_boundary + crosswalk + drivable_area + intersection.
    assert nusc_map.num_elements == 5

    # Identity ego pose -> ego frame == global frame.
    hd = nusc_map.build_hd_map(np.eye(4), radius_m=100.0)
    assert {e.type for e in hd.elements} == {
        MapElementType.LANE_CENTER,
        MapElementType.LANE_BOUNDARY,
        MapElementType.CROSSWALK,
        MapElementType.DRIVABLE_AREA,
        MapElementType.INTERSECTION,
    }
    lane = next(e for e in hd.elements if e.type is MapElementType.LANE_CENTER)
    assert lane.points.dtype == np.float32 and not lane.is_closed
    np.testing.assert_allclose(lane.points[0], [0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(lane.points[-1], [10.0, 0.0], atol=1e-4)
    assert lane.successor_ids == ["laneX"]  # from connectivity
    assert lane.attrs["lane_type"] == "vehicle"  # CAR -> vehicle
    assert next(
        e for e in hd.elements if e.type is MapElementType.DRIVABLE_AREA
    ).is_closed


def test_nuscmap_roi_excludes_far_elements(tmp_path):
    p = tmp_path / "test-loc.json"
    _write_map_json(p)
    nusc_map = NuscMap.from_json(str(p))
    # Ego 1 km away: nothing within a 50 m ROI.
    far = np.eye(4)
    far[0, 3] = 1000.0
    assert nusc_map.build_hd_map(far, radius_m=50.0).elements == []
