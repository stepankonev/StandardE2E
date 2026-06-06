# flake8: noqa: E501
"""Tests for the TruckDrive processor.

* **Unit tests** (always run): the class-id -> ``DetectionType`` taxonomy (incl.
  ego-vehicle / DontCare exclusion and the prefix fallback), the transform-tree
  BFS direction convention, ``T_world_from_ego`` assembly, the Aeva
  ``float64 x 11`` -> xyz decode, sync-stem parsing, camera-name resolution, and
  scene discovery / trajectory parsing on a synthetic layout.
* **Real-frame checks** (skipped unless ``TRUCKDRIVE_ROOT`` points at extracted
  scenes): a built frame exposes every rig camera under its own
  ``CameraDirection`` with pinhole intrinsics and 4x4 extrinsics, ego-frame
  LiDAR with the expected long range, ego-frame detections with the ego vehicle
  excluded, and a rigid ego pose.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from standard_e2e.caching.src_datasets.truckdrive import _truckdrive_io as io
from standard_e2e.caching.src_datasets.truckdrive._truckdrive_geometry import (
    build_tf_graph,
    find_transform,
    pose_world_from_ego,
)
from standard_e2e.caching.src_datasets.truckdrive.truckdrive_dataset_processor import (
    _CAMERA_NAME_TO_DIRECTION,
    TruckDriveDatasetProcessor,
    _camera_direction_for,
    detection_type_for_class_id,
)
from standard_e2e.enums import CameraDirection, DetectionType
from standard_e2e.enums import TrajectoryComponent as TC

# Real-frame checks read TruckDrive only from this env var; they skip when unset.
_TRUCKDRIVE_ROOT = os.environ.get("TRUCKDRIVE_ROOT", "")


# --------------------------------------------------------------------------- #
# Taxonomy unit tests (no data required)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "class_id,expected",
    [
        ("Vehicle", DetectionType.VEHICLE),
        ("Vehicle-Passenger", DetectionType.VEHICLE),
        ("Vehicle-SemiTruck-Cab", DetectionType.VEHICLE),
        ("Vehicle-SemiTruck-Trailer", DetectionType.VEHICLE),
        ("Vehicle-Emergency", DetectionType.VEHICLE),
        ("VRUvehicle-Bicycle", DetectionType.BICYCLE),
        ("Vehicle-Motorcycle", DetectionType.BICYCLE),
        ("Person", DetectionType.PEDESTRIAN),
        ("Animal", DetectionType.PEDESTRIAN),
        ("RoadObstruction-Barrel", DetectionType.UNKNOWN),
        ("RoadDebris-Tire", DetectionType.UNKNOWN),
        ("TrafficSign", DetectionType.SIGN),
        ("TrafficSignal-GreenSolid", DetectionType.SIGN),
        ("WalkSignal", DetectionType.SIGN),
    ],
)
def test_detection_type_for_known_class_ids(class_id, expected):
    assert detection_type_for_class_id(class_id) is expected


@pytest.mark.parametrize(
    "class_id",
    [
        "Vehicle-EgoVehicle-Cab",
        "Vehicle-EgoVehicle-Trailer",
        "DelineatorGroupDontCare",
        "OutOfLidarRangeVehicleGroup",
        "ParkingLotVehicleGroup",
    ],
)
def test_detection_type_excludes_ego_and_dontcare(class_id):
    assert detection_type_for_class_id(class_id) is None


def test_detection_type_prefix_fallback_for_unknown_labels():
    # Labels not in the vendored Table-2 map fall back to prefix rules.
    assert (
        detection_type_for_class_id("TrafficSign-BrandNewVariant") is DetectionType.SIGN
    )
    assert detection_type_for_class_id("Vehicle-SomeNewClass") is DetectionType.VEHICLE
    assert detection_type_for_class_id("Person-NewRole") is DetectionType.PEDESTRIAN
    assert detection_type_for_class_id("SomethingEgoVehicleLike") is None
    assert detection_type_for_class_id("CompletelyUnknown") is DetectionType.UNKNOWN


# --------------------------------------------------------------------------- #
# Geometry unit tests (no data required)
# --------------------------------------------------------------------------- #
def _leaf(parent: str, child: str, t=(0.0, 0.0, 0.0), q=(0.0, 0.0, 0.0, 1.0)):
    """One TF-tree entry (rotation quaternion is (x, y, z, w))."""
    return {
        "header": {"frame_id": parent},
        "child_frame_id": child,
        "transform": {
            "translation": {"x": t[0], "y": t[1], "z": t[2]},
            "rotation": {"x": q[0], "y": q[1], "z": q[2], "w": q[3]},
        },
    }


def test_find_transform_chains_translations_and_inverts():
    # vehicle -> cab (+2 x) -> velodyne (+1 x): velodyne sits +3 x of vehicle.
    tree = {
        "a": _leaf("vehicle", "cab", t=(2.0, 0.0, 0.0)),
        "b": _leaf("cab", "velodyne", t=(1.0, 0.0, 0.0)),
    }
    graph = build_tf_graph(tree)
    # T_vehicle_from_velodyne maps a velodyne-frame point into vehicle coords.
    t_vehicle_from_velodyne = find_transform(graph, "velodyne", "vehicle")
    np.testing.assert_allclose(
        t_vehicle_from_velodyne[:3, 3], [3.0, 0.0, 0.0], atol=1e-9
    )
    # The reverse is its inverse.
    t_velodyne_from_vehicle = find_transform(graph, "vehicle", "velodyne")
    np.testing.assert_allclose(
        t_velodyne_from_vehicle[:3, 3], [-3.0, 0.0, 0.0], atol=1e-9
    )
    np.testing.assert_allclose(
        t_vehicle_from_velodyne @ t_velodyne_from_vehicle, np.eye(4), atol=1e-9
    )


def test_find_transform_applies_rotation():
    # cab -> sensor rotated +90 deg about z (quat xyzw = (0,0,sin45,cos45)).
    s = np.sqrt(0.5)
    tree = {"a": _leaf("cab", "sensor", t=(0.0, 0.0, 0.0), q=(0.0, 0.0, s, s))}
    graph = build_tf_graph(tree)
    t_cab_from_sensor = find_transform(graph, "sensor", "cab")
    # A sensor-frame +x point lands on cab +y.
    pt = t_cab_from_sensor @ np.array([1.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(pt[:3], [0.0, 1.0, 0.0], atol=1e-9)


def test_find_transform_unknown_frame_raises():
    graph = build_tf_graph({"a": _leaf("vehicle", "cab")})
    with pytest.raises(KeyError):
        find_transform(graph, "nope", "vehicle")
    with pytest.raises(KeyError):
        find_transform(graph, "vehicle", "nope")


def test_pose_world_from_ego_identity_and_rotation():
    # Identity quaternion -> rotation I, translation passthrough.
    t = pose_world_from_ego(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    np.testing.assert_allclose(t[:3, 3], [1.0, 2.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(t[:3, :3], np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(t[:3, :3]), 1.0)
    # +90 deg about z maps ego +x -> world +y.
    s = np.sqrt(0.5)
    t = pose_world_from_ego(np.zeros(3), np.array([0.0, 0.0, s, s]))
    np.testing.assert_allclose(t[:3, :3] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-9)


# --------------------------------------------------------------------------- #
# IO unit tests (synthetic files)
# --------------------------------------------------------------------------- #
def test_read_aeva_xyz_keeps_first_three_columns(tmp_path):
    points = np.arange(2 * 11, dtype=np.float64).reshape(2, 11)
    path = tmp_path / "0000_0.bin"
    points.tofile(path)
    xyz = io.read_aeva_xyz(str(path))
    assert xyz.shape == (2, 3) and xyz.dtype == np.float32
    np.testing.assert_allclose(xyz, points[:, :3].astype(np.float32))


def test_read_aeva_xyz_empty_and_bad_size(tmp_path):
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert io.read_aeva_xyz(str(empty)).shape == (0, 3)
    bad = tmp_path / "bad.bin"
    np.arange(5, dtype=np.float64).tofile(bad)  # 5 not divisible by 11
    with pytest.raises(ValueError):
        io.read_aeva_xyz(str(bad))


def test_parse_sync_stem():
    assert io.parse_sync_stem("0063_3172004958.jpg") == (63, 3172004958)
    assert io.parse_sync_stem("0000_0.bin") == (0, 0)
    assert io.parse_sync_stem("gt_trajectory.txt") is None
    assert io.parse_sync_stem("not_a_sync_name.json") is None


def test_camera_direction_resolution():
    # Canonical surround members are reused for the cameras that match a facing.
    assert _camera_direction_for("forward_center_medium") is CameraDirection.FRONT
    assert _camera_direction_for("forward_left_wide") is CameraDirection.FRONT_LEFT
    assert (
        _camera_direction_for("sideward_right_front_wide") is CameraDirection.SIDE_RIGHT
    )
    assert (
        _camera_direction_for("rearward_left_bottom_medium")
        is CameraDirection.REAR_LEFT
    )
    # Only the genuinely-extra views get dedicated members.
    assert (
        _camera_direction_for("forward_left_narrow")
        is CameraDirection.FRONT_LEFT_NARROW
    )
    assert (
        _camera_direction_for("sideward_left_back_wide")
        is CameraDirection.SIDE_LEFT_BACK
    )
    # The 11 public-rig cameras map to 11 distinct directions.
    assert len(set(_CAMERA_NAME_TO_DIRECTION.values())) == 11
    # A 15-rig camera not mapped in the public set resolves to None.
    assert _camera_direction_for("forward_left_medium") is None


def _make_scene(root: Path, sync_rows: list[tuple[int, float]]) -> io.SceneRef:
    scene_dir = root / "scene_28_1"
    (scene_dir / "poses").mkdir(parents=True)
    lines = ["SYNC_KEY, TIMESTAMP, X, Y, Z, R_X, R_Y, R_Z, R_W"]
    for i, (sync_id, ts) in enumerate(sync_rows):
        lines.append(f"{sync_id:04d} {ts} {float(i)} 0.0 0.0 0.0 0.0 0.0 1.0")
    (scene_dir / "poses" / "gt_trajectory.txt").write_text("\n".join(lines) + "\n")
    return io.SceneRef(str(scene_dir), scene_dir.name)


def test_discover_scenes_finds_and_orders(tmp_path):
    _make_scene(tmp_path, [(0, 0.0)])
    scenes = io.discover_scenes(str(tmp_path))
    assert [s.scene_id for s in scenes] == ["scene_28_1"]


def test_discover_scenes_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.discover_scenes(str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError):
        io.discover_scenes(str(tmp_path))  # exists but holds no scenes


def test_read_trajectory_and_sync_ids(tmp_path):
    ref = _make_scene(tmp_path, [(0, 0.0), (2, 0.2), (4, 0.4)])
    sync_ids, timestamps_s, poses_raw = io.read_trajectory(ref)
    assert sync_ids == [0, 2, 4]
    np.testing.assert_allclose(timestamps_s, [0.0, 0.2, 0.4])
    assert poses_raw.shape == (3, 7)
    np.testing.assert_allclose(
        poses_raw[:, 0], [0.0, 1.0, 2.0]
    )  # X column == row index
    assert io.read_pose_sync_ids(ref) == [0, 2, 4]


# --------------------------------------------------------------------------- #
# Real-frame checks (skipped without data)
# --------------------------------------------------------------------------- #
def _first_scene():
    if not _TRUCKDRIVE_ROOT or not Path(_TRUCKDRIVE_ROOT).exists():
        return None
    try:
        return io.discover_scenes(_TRUCKDRIVE_ROOT)[0]
    except (FileNotFoundError, IndexError):
        return None


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    ref = _first_scene()
    if ref is None:
        pytest.skip("no TruckDrive data available (set TRUCKDRIVE_ROOT)")
    # A frame with all modalities: the first sync_id that has a box file and is
    # also present in the pose table.
    box_syncs = set(io.box_sync_paths(ref))
    pose_syncs = set(io.read_pose_sync_ids(ref))
    candidates = sorted(box_syncs & pose_syncs)
    if not candidates:
        pytest.skip("no fully-annotated frame in the first scene")
    sync_id = candidates[0]
    out = tmp_path_factory.mktemp("truckdrive")
    proc = TruckDriveDatasetProcessor(
        common_output_path=str(out), split="all", context_aggregators=[]
    )
    fd = proc._prepare_standardized_frame_data((ref, sync_id))
    return ref, sync_id, fd


def test_cameras_are_rig_members_with_calibration(built):
    _ref, _sync_id, fd = built
    assert len(fd.cameras) >= 1
    rig_directions = set(_CAMERA_NAME_TO_DIRECTION.values())
    for direction, cam in fd.cameras.items():
        assert direction in rig_directions  # a mapped rig camera
        assert cam.camera_direction is direction
        assert cam.image.ndim == 3 and cam.image.dtype == np.uint8
        assert cam.image.shape[2] == 3
        assert cam.intrinsics.shape == (3, 3)
        assert cam.extrinsics.shape == (4, 4)
        assert np.isfinite(cam.extrinsics).all()


def test_lidar_is_ego_frame_long_range(built):
    _ref, _sync_id, fd = built
    assert fd.lidar is not None
    pts = fd.lidar.points
    assert {"x", "y", "z"}.issubset(pts.columns)
    xyz = pts[["x", "y", "z"]].to_numpy()
    assert xyz.dtype == np.float32 and len(xyz) > 0
    assert np.isfinite(xyz).all()
    # FMCW long-range rig: forward returns reach well beyond short-range datasets.
    assert xyz[:, 0].max() > 100.0


def test_detections_are_ego_frame_and_exclude_ego_vehicle(built):
    _ref, _sync_id, fd = built
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
    # The ego truck's own cab/trailer boxes must not appear as detections.
    assert not any("EgoVehicle" in d.unique_agent_id for d in detections)


def test_ego_pose_is_rigid(built):
    _ref, _sync_id, fd = built
    t = fd.aux_data["pose_matrix"]
    assert t.shape == (4, 4) and np.isfinite(t).all()
    np.testing.assert_allclose(t[:3, :3] @ t[:3, :3].T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(t[:3, :3]), 1.0, atol=1e-6)
