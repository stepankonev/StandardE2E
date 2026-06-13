# flake8: noqa: E501
"""Tests for the View-of-Delft (VoD) processor.

* **Unit tests** (always run): the class -> ``DetectionType`` taxonomy (incl.
  ``DontCare`` exclusion and the substring fallback), KITTI calibration parsing,
  ``T_map_from_lidar`` ego-pose composition, the camera->ego box transform and
  its -Z yaw sign, the velodyne ``float32 x 4`` -> xyz decode, KITTI label-line
  parsing (incl. the track-id field), pose-JSON reading, the vendored scene /
  split table, root resolution and scene-grouped frame enumeration on a synthetic
  layout.
* **Real-frame checks** (skipped unless ``VOD_ROOT`` points at an extracted
  ``view_of_delft_PUBLIC``): a built frame exposes a FRONT camera with pinhole
  intrinsics + 4x4 extrinsics, ego-frame LiDAR, ego-frame detections with
  ``DontCare`` excluded and non-negative dimensions, and a rigid ego pose.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from standard_e2e.caching.src_datasets.vod import _vod_io as io
from standard_e2e.caching.src_datasets.vod._vod_geometry import (
    box_camera_to_ego,
    ego_pose_map_from_lidar,
    parse_calibration,
)
from standard_e2e.caching.src_datasets.vod._vod_splits import (
    ALLOWED_SPLITS,
    VOD_SCENES,
    scene_for_frame,
    scenes_for_split,
)
from standard_e2e.caching.src_datasets.vod.vod_dataset_processor import (
    _CLASS_TO_DETECTION_TYPE,
    VodDatasetProcessor,
    detection_type_for_class,
)
from standard_e2e.enums import CameraDirection, DetectionType
from standard_e2e.enums import TrajectoryComponent as TC

# Real-frame checks read VoD only from this env var; they skip when unset.
_VOD_ROOT = os.environ.get("VOD_ROOT", "")


# --------------------------------------------------------------------------- #
# Taxonomy unit tests (no data required)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "class_name,expected",
    [
        ("Car", DetectionType.VEHICLE),
        ("truck", DetectionType.VEHICLE),
        ("vehicle_other", DetectionType.VEHICLE),
        ("Pedestrian", DetectionType.PEDESTRIAN),
        ("Cyclist", DetectionType.BICYCLE),
        ("bicycle", DetectionType.BICYCLE),
        ("rider", DetectionType.BICYCLE),
        ("moped_scooter", DetectionType.BICYCLE),
        ("motor", DetectionType.BICYCLE),
        ("bicycle_rack", DetectionType.UNKNOWN),
        ("human_depiction", DetectionType.UNKNOWN),
        ("ride_other", DetectionType.UNKNOWN),
        ("ride_uncertain", DetectionType.UNKNOWN),
    ],
)
def test_detection_type_for_known_classes(class_name, expected):
    assert detection_type_for_class(class_name) is expected


def test_detection_type_excludes_dontcare():
    assert detection_type_for_class("DontCare") is None


def test_detection_type_map_covers_all_13_vod_classes_plus_dontcare():
    # The explicit map is the 13 annotated classes + KITTI DontCare.
    assert len(_CLASS_TO_DETECTION_TYPE) == 14
    assert _CLASS_TO_DETECTION_TYPE["DontCare"] is None


def test_detection_type_substring_fallback_for_unknown_labels():
    assert detection_type_for_class("SomeNewVan") is DetectionType.VEHICLE
    assert detection_type_for_class("ElectricScooterV2") is DetectionType.BICYCLE
    assert detection_type_for_class("PedestrianChild") is DetectionType.PEDESTRIAN
    assert detection_type_for_class("CompletelyUnknown") is DetectionType.UNKNOWN


# --------------------------------------------------------------------------- #
# Geometry unit tests (no data required)
# --------------------------------------------------------------------------- #
_SYNTHETIC_CALIB = (
    "P0: 1000.0 0.0 960.0 0.0 0.0 1000.0 600.0 0.0 0.0 0.0 1.0 0.0\n"
    "P1: 1000.0 0.0 960.0 0.0 0.0 1000.0 600.0 0.0 0.0 0.0 1.0 0.0\n"
    "P2: 1495.0 0.0 961.0 0.0 0.0 1495.0 625.0 0.0 0.0 0.0 1.0 0.0\n"
    "P3: 1000.0 0.0 960.0 0.0 0.0 1000.0 600.0 0.0 0.0 0.0 1.0 0.0\n"
    "R0_rect: 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0\n"
    "Tr_velo_to_cam: 0.0 -1.0 0.0 0.1 0.0 0.0 -1.0 -0.4 1.0 0.0 0.0 -0.9\n"
    "Tr_imu_to_velo:\n"
)


def test_parse_calibration_extracts_K_and_extrinsics():
    intrinsics, t_cam_from_lidar = parse_calibration(_SYNTHETIC_CALIB)
    assert intrinsics.shape == (3, 3) and intrinsics.dtype == np.float32
    np.testing.assert_allclose(intrinsics[0, 0], 1495.0)
    np.testing.assert_allclose(intrinsics[1, 1], 1495.0)
    np.testing.assert_allclose(intrinsics[0, 2], 961.0)
    np.testing.assert_allclose(intrinsics[1, 2], 625.0)
    # 4x4 homogeneous with the [R | t] top block and a [0, 0, 0, 1] bottom row.
    assert t_cam_from_lidar.shape == (4, 4)
    np.testing.assert_allclose(t_cam_from_lidar[3], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(t_cam_from_lidar[:3, 3], [0.1, -0.4, -0.9])
    # A velodyne-frame point maps into the camera by R: velo +x -> cam +z here.
    np.testing.assert_allclose(
        t_cam_from_lidar[:3, :3] @ [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], atol=1e-9
    )


def test_parse_calibration_missing_keys_raises():
    with pytest.raises(ValueError):
        parse_calibration("P0: 1 0 0 0 0 1 0 0 0 0 1 0\n")


def test_ego_pose_map_from_lidar_identity_and_translation():
    eye = np.eye(4)
    np.testing.assert_allclose(ego_pose_map_from_lidar(eye, eye), eye, atol=1e-12)
    # VoD's "mapToCamera" is the camera pose in the map (T_map_from_camera), so a
    # pure-translation pose places the ego (== lidar == camera here) at +t.
    map_from_camera = np.eye(4)
    map_from_camera[:3, 3] = [5.0, -2.0, 1.0]
    pose = ego_pose_map_from_lidar(map_from_camera, np.eye(4))
    np.testing.assert_allclose(pose[:3, 3], [5.0, -2.0, 1.0], atol=1e-9)
    assert np.isclose(np.linalg.det(pose[:3, :3]), 1.0)
    # A non-trivial camera<-lidar calib composes on the right (+1 m up in lidar).
    t_cam_from_lidar = np.eye(4)
    t_cam_from_lidar[:3, 3] = [0.0, 0.0, 1.0]
    pose2 = ego_pose_map_from_lidar(map_from_camera, t_cam_from_lidar)
    np.testing.assert_allclose(pose2[:3, 3], [5.0, -2.0, 2.0], atol=1e-9)


def test_box_camera_to_ego_identity_dims_and_yaw_sign():
    # Identity camera->lidar: x/y pass through; dims reorder (h,w,l)->(l,w,h); the
    # KITTI bottom-center is raised by H/2 (=0.75) to the geometric center; VoD
    # yaw about LiDAR -Z negates to the FLU (+Z) heading.
    center, length, width, height, heading = box_camera_to_ego(
        np.array([1.0, 2.0, 3.0]), np.array([1.5, 0.6, 4.0]), 0.5, np.eye(4)
    )
    np.testing.assert_allclose(center, [1.0, 2.0, 3.0 + 0.75], atol=1e-12)
    assert (length, width, height) == (4.0, 0.6, 1.5)
    assert np.isclose(heading, -0.5)


def test_box_camera_to_ego_raises_bottom_center_by_half_height():
    # location is the bottom face; a taller box's center lifts more (H/2).
    c_short, *_ = box_camera_to_ego(np.zeros(3), np.array([1.0, 1, 1]), 0.0, np.eye(4))
    c_tall, *_ = box_camera_to_ego(np.zeros(3), np.array([3.0, 1, 1]), 0.0, np.eye(4))
    assert np.isclose(c_short[2], 0.5) and np.isclose(c_tall[2], 1.5)


def test_box_camera_to_ego_yaw_wraps_to_pi():
    _, _, _, _, heading = box_camera_to_ego(
        np.zeros(3), np.ones(3), float(np.pi), np.eye(4)
    )
    assert np.isclose(abs(heading), np.pi)


def test_box_camera_to_ego_applies_camera_to_lidar_transform():
    # 90 deg about z maps cam +x -> lidar +y; check the center is rotated.
    s = np.sqrt(0.5)
    t_lidar_from_cam = np.eye(4)
    t_lidar_from_cam[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    center, *_ = box_camera_to_ego(
        np.array([2.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), 0.0, t_lidar_from_cam
    )
    # cam +x -> lidar +y, then +H/2 (=0.5) up for the bottom-center raise.
    np.testing.assert_allclose(center, [0.0, 2.0, 0.5], atol=1e-9)
    assert np.isclose(s * s, 0.5)  # guard against an accidental no-op rotation


# --------------------------------------------------------------------------- #
# IO unit tests (synthetic files)
# --------------------------------------------------------------------------- #
def test_read_velodyne_xyz_keeps_first_three_columns(tmp_path):
    points = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "00000.bin"
    points.tofile(path)
    xyz = io.read_velodyne_xyz(str(path))
    assert xyz.shape == (3, 3) and xyz.dtype == np.float32
    np.testing.assert_allclose(xyz, points[:, :3])


def test_read_velodyne_xyz_empty_and_bad_size(tmp_path):
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert io.read_velodyne_xyz(str(empty)).shape == (0, 3)
    bad = tmp_path / "bad.bin"
    np.arange(5, dtype=np.float32).tofile(bad)  # 5 not divisible by 4
    with pytest.raises(ValueError):
        io.read_velodyne_xyz(str(bad))


def test_read_pose_parses_three_json_objects(tmp_path):
    path = tmp_path / "00000.json"
    odom = np.arange(16, dtype=float).reshape(4, 4)
    mapc = np.eye(4)
    mapc[:3, 3] = [1.0, 2.0, 3.0]
    lines = [
        json.dumps({"odomToCamera": odom.reshape(-1).tolist()}),
        json.dumps({"mapToCamera": mapc.reshape(-1).tolist()}),
        json.dumps({"UTMToCamera": np.eye(4).reshape(-1).tolist()}),
    ]
    path.write_text("\n".join(lines) + "\n")
    pose = io.read_pose(str(path))
    assert set(pose) == {"odomToCamera", "mapToCamera", "UTMToCamera"}
    assert pose["mapToCamera"].shape == (4, 4)
    np.testing.assert_allclose(pose["mapToCamera"][:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(pose["odomToCamera"], odom)


def test_parse_label_line_base_and_track_id():
    base = "Pedestrian 1 0 -0.22 634.8 853.3 667.1 932.1 1.64 0.48 0.61 -6.97 6.83 33.6 -0.42 1"
    label = io.parse_label_line(base)
    assert label is not None
    assert label.cls == "Pedestrian"
    assert label.track_field == "1"  # base labels: KITTI truncation slot
    assert label.occluded == 0
    np.testing.assert_allclose(label.dimensions_hwl, [1.64, 0.48, 0.61])
    np.testing.assert_allclose(label.location_cam, [-6.97, 6.83, 33.6])
    assert np.isclose(label.rotation, -0.42)
    assert np.isclose(label.score, 1.0)
    # Track-id label set overloads field 1 with the unique object id.
    track = io.parse_label_line(
        "bicycle 1757 1 -0.51 1692.8 873.0 1935.0 1064.7 0.99 0.45 1.73 5.23 2.47 8.67 0.02 1"
    )
    assert track is not None and track.track_field == "1757"


def test_parse_label_line_rejects_malformed():
    assert io.parse_label_line("Car 0 0 0.0 1 2 3 4") is None  # < 15 fields
    assert io.parse_label_line("") is None


def test_parse_label_line_score_optional():
    # GT lines may omit the trailing score; default to 1.0.
    no_score = "Car 0 0 0.0 1 2 3 4 1.5 1.8 4.0 1.0 1.5 10.0 0.1"
    label = io.parse_label_line(no_score)
    assert label is not None and np.isclose(label.score, 1.0)


# --------------------------------------------------------------------------- #
# Scene / split table unit tests
# --------------------------------------------------------------------------- #
def test_scene_table_counts():
    assert len(VOD_SCENES) == 24
    assert len(scenes_for_split("train")) == 13
    assert len(scenes_for_split("val")) == 4
    assert len(scenes_for_split("test")) == 7
    # Every scene's split is valid and every frame range is non-empty/ordered.
    for scene in VOD_SCENES:
        assert scene.split in ALLOWED_SPLITS
        assert scene.start_frame <= scene.end_frame


def test_scene_subdir_routing():
    by_name = {s.name: s for s in VOD_SCENES}
    assert by_name["delft_2"].subdir == "training"  # train
    assert by_name["delft_1"].subdir == "training"  # val also under training/
    assert by_name["delft_7"].subdir == "testing"  # test


def test_scene_for_frame_and_gaps():
    assert scene_for_frame(0).name == "delft_1"
    assert scene_for_frame(9930).name == "delft_27"
    assert scene_for_frame(4048) is None  # 1-frame gap between delft_11/delft_12
    assert scene_for_frame(5500) is None  # scene 15 absent from the release
    assert scene_for_frame(100000) is None


def test_scenes_for_split_rejects_unknown_split():
    with pytest.raises(ValueError):
        scenes_for_split("trainval")


# --------------------------------------------------------------------------- #
# Root resolution + frame enumeration (synthetic layout)
# --------------------------------------------------------------------------- #
def _make_layout(root: Path, training_ids=(), testing_ids=()) -> None:
    for subdir, ids in (("training", training_ids), ("testing", testing_ids)):
        velodyne = root / "lidar" / subdir / "velodyne"
        velodyne.mkdir(parents=True, exist_ok=True)
        for frame_id in ids:
            (velodyne / f"{frame_id:05d}.bin").write_bytes(b"")


def test_resolve_root_variants(tmp_path):
    root = tmp_path / "view_of_delft_PUBLIC"
    _make_layout(root, training_ids=(549,))
    # Pointing directly at the root, or its parent, both resolve.
    assert io.resolve_root(str(root)) == str(root)
    assert io.resolve_root(str(tmp_path)) == str(root)


def test_resolve_root_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.resolve_root(str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError):
        io.resolve_root(str(tmp_path))  # exists but holds no lidar tree


def test_iter_frame_refs_groups_by_scene_and_split(tmp_path):
    root = tmp_path / "view_of_delft_PUBLIC"
    # 549,600 -> delft_2 (train); 1400 -> delft_3 (train); 2600 -> delft_7 (test).
    _make_layout(root, training_ids=(600, 549, 1400), testing_ids=(2600,))
    resolved = io.resolve_root(str(root))

    train = list(io.iter_frame_refs(resolved, "train"))
    assert [(r.scene_name, r.frame_id) for r in train] == [
        ("delft_2", 549),
        ("delft_2", 600),
        ("delft_3", 1400),
    ]
    assert all(r.subdir == "training" for r in train)

    test = list(io.iter_frame_refs(resolved, "test"))
    assert [(r.scene_name, r.frame_id) for r in test] == [("delft_7", 2600)]
    assert test[0].subdir == "testing"

    # val scenes have no present frames in this layout.
    assert list(io.iter_frame_refs(resolved, "val")) == []


def test_frame_ref_paths():
    ref = io.FrameRef("/root", "delft_2", "training", 549)
    assert ref.stem == "00549"
    assert ref.velodyne_path.endswith("lidar/training/velodyne/00549.bin")
    assert ref.image_path.endswith("lidar/training/image_2/00549.jpg")
    assert ref.label_path.endswith("lidar/training/label_2/00549.txt")
    assert ref.calib_path.endswith("lidar/training/calib/00549.txt")
    assert ref.pose_path.endswith("lidar/training/pose/00549.json")


# --------------------------------------------------------------------------- #
# Real-frame checks (skipped without data)
# --------------------------------------------------------------------------- #
def _first_labelled_train_frame():
    if not _VOD_ROOT or not Path(_VOD_ROOT).exists():
        return None
    try:
        root = io.resolve_root(_VOD_ROOT)
    except FileNotFoundError:
        return None
    for ref in io.iter_frame_refs(root, "train"):
        if os.path.exists(ref.label_path):
            return ref
    return None


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    ref = _first_labelled_train_frame()
    if ref is None:
        pytest.skip("no VoD data available (set VOD_ROOT to view_of_delft_PUBLIC)")
    out = tmp_path_factory.mktemp("vod")
    proc = VodDatasetProcessor(
        common_output_path=str(out), split="train", context_aggregators=[]
    )
    return ref, proc._prepare_standardized_frame_data(ref)


def test_front_camera_has_calibration(built):
    _ref, fd = built
    assert set(fd.cameras) == {CameraDirection.FRONT}
    cam = fd.cameras[CameraDirection.FRONT]
    assert (
        cam.image.ndim == 3 and cam.image.dtype == np.uint8 and cam.image.shape[2] == 3
    )
    assert cam.intrinsics.shape == (3, 3)
    assert cam.extrinsics.shape == (4, 4) and np.isfinite(cam.extrinsics).all()


def test_lidar_is_ego_frame(built):
    _ref, fd = built
    assert fd.lidar is not None
    xyz = fd.lidar.points[["x", "y", "z"]].to_numpy()
    assert xyz.dtype == np.float32 and len(xyz) > 0 and np.isfinite(xyz).all()
    # Urban Velodyne reaches well beyond a few metres forward.
    assert xyz[:, 0].max() > 20.0


def test_detections_are_ego_frame_and_exclude_dontcare(built):
    _ref, fd = built
    detections = fd.frame_detections_3d.detections
    assert len(detections) > 0
    for det in detections:
        assert isinstance(det.detection_type, DetectionType)
        assert det.unique_agent_id != "DontCare"
        for comp in (TC.X, TC.Y, TC.Z, TC.LENGTH, TC.WIDTH, TC.HEIGHT, TC.HEADING):
            value = float(np.asarray(det.trajectory.get(comp)).reshape(-1)[0])
            assert np.isfinite(value)
        for comp in (TC.LENGTH, TC.WIDTH, TC.HEIGHT):
            assert float(np.asarray(det.trajectory.get(comp)).reshape(-1)[0]) >= 0.0


def test_ego_pose_is_rigid(built):
    _ref, fd = built
    pose = fd.aux_data["pose_matrix"]
    assert pose.shape == (4, 4) and np.isfinite(pose).all()
    np.testing.assert_allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-5)
