# flake8: noqa: E501
"""Tests for the KITScenes Multimodal processor.

* **Unit tests** (always run): calibration parsing (camera ``K`` + extrinsics,
  the LiDAR extrinsic, the missing-``lidar_top`` error), ego-pose construction
  from a TUM row (translation, quaternion rotation, heading), the ring-camera ->
  ``CameraDirection`` map, the discretized-LiDAR parquet decode (de-discretize +
  invalid-return sentinel drop), ``poses.txt`` parsing, scene-directory
  resolution for flat and by-split layouts, the Lanelet2 -> ``HDMap`` translation
  and centerline / orientation / resample geometry, and a full hermetic processor
  build on a synthetic on-disk scene.
* **Real-frame checks** (skipped unless ``KITSCENES_ROOT`` points at a directory
  of extracted scenes): a built frame exposes the six ring cameras with pinhole
  intrinsics + 4x4 extrinsics, ego-frame LiDAR, an empty detection set, a rigid
  ego pose, and an ego-frame HD map carrying lane centerlines.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from standard_e2e.caching.src_datasets.kitscenes_multimodal import _kitscenes_io as io
from standard_e2e.caching.src_datasets.kitscenes_multimodal._kitscenes_geometry import (
    RING_CAMERA_TO_DIRECTION,
    parse_calibration,
    pose_from_tum,
)
from standard_e2e.caching.src_datasets.kitscenes_multimodal._kitscenes_map import (
    KITScenesMap,
    _lanelet_centerline,
    _orient_to,
    _resample_polyline,
)
from standard_e2e.caching.src_datasets.kitscenes_multimodal._kitscenes_splits import (
    ALLOWED_SPLITS,
    split_dir_names,
)
from standard_e2e.caching.src_datasets.kitscenes_multimodal.kitscenes_multimodal_dataset_processor import (
    KITScenesMultimodalDatasetProcessor,
)
from standard_e2e.enums import CameraDirection, MapElementType
from standard_e2e.enums import TrajectoryComponent as TC
from standard_e2e.utils import matrix_to_xyz_heading

# Real-frame checks read KITScenes only from this env var; they skip when unset.
_KITSCENES_ROOT = os.environ.get("KITSCENES_ROOT", "")


# --------------------------------------------------------------------------- #
# Calibration / pose geometry (no data required)
# --------------------------------------------------------------------------- #
def _identity_4x4_list() -> list[list[float]]:
    return np.eye(4).tolist()


def _synthetic_calib() -> dict:
    front = np.eye(4)
    front[:3, 3] = [0.2, 0.0, -0.18]
    return {
        "reference": "base_frame",
        "lidar_top": {"T_to_reference": _identity_4x4_list()},
        "camera_ring_front": {
            "T_to_reference": front.tolist(),
            "camera_model": "pinhole",
            "intrinsics": {
                "focal_length": 1841.0,
                "principal_point_u": 1740.0,
                "principal_point_v": 1139.0,
            },
            "resolution": {"width": 3504, "height": 2272},
        },
    }


def test_parse_calibration_extracts_K_and_extrinsics():
    cameras, lidar_top_extrinsic = parse_calibration(_synthetic_calib())
    assert set(cameras) == {CameraDirection.FRONT}
    cam = cameras[CameraDirection.FRONT]
    assert cam.intrinsics.shape == (3, 3) and cam.intrinsics.dtype == np.float32
    np.testing.assert_allclose(cam.intrinsics[0, 0], 1841.0)
    np.testing.assert_allclose(cam.intrinsics[1, 1], 1841.0)
    np.testing.assert_allclose(cam.intrinsics[0, 2], 1740.0)
    np.testing.assert_allclose(cam.intrinsics[1, 2], 1139.0)
    assert cam.extrinsics.shape == (4, 4)
    np.testing.assert_allclose(cam.extrinsics[:3, 3], [0.2, 0.0, -0.18])
    # base_frame == ego, so lidar_top's extrinsic is the identity (ego origin).
    np.testing.assert_allclose(lidar_top_extrinsic, np.eye(4))


def test_parse_calibration_missing_lidar_top_raises():
    calib = _synthetic_calib()
    del calib["lidar_top"]
    with pytest.raises(KeyError):
        parse_calibration(calib)


def test_pose_from_tum_translation_and_identity():
    # qx qy qz qw = 0 0 0 1 -> identity rotation; translation passes through.
    pose = pose_from_tum(np.array([0.05, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]))
    assert pose.shape == (4, 4)
    np.testing.assert_allclose(pose[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(pose[:3, 3], [1.0, 2.0, 3.0])
    x, y, z, heading = matrix_to_xyz_heading(pose)
    assert (x, y, z) == (1.0, 2.0, 3.0) and np.isclose(heading, 0.0)


def test_pose_from_tum_rotation_about_z():
    # 90 deg about +z (scalar-last xyzw): maps ego +x -> map +y, heading = +pi/2.
    s = np.sqrt(0.5)
    pose = pose_from_tum(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, s, s]))
    np.testing.assert_allclose(
        pose[:3, :3] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-9
    )
    _, _, _, heading = matrix_to_xyz_heading(pose)
    assert np.isclose(heading, np.pi / 2)
    # Rigid: R R^T = I, det = 1.
    np.testing.assert_allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(pose[:3, :3]), 1.0)


def test_pose_from_tum_rejects_bad_row():
    with pytest.raises(ValueError):
        pose_from_tum(np.zeros(7))


def test_ring_camera_map_is_six_distinct_canonical_directions():
    assert len(RING_CAMERA_TO_DIRECTION) == 6
    assert len(set(RING_CAMERA_TO_DIRECTION.values())) == 6
    assert all(name.startswith("camera_ring_") for name in RING_CAMERA_TO_DIRECTION)
    # All map onto canonical surround members (not the TruckDrive extras).
    canonical = {
        CameraDirection.FRONT,
        CameraDirection.FRONT_LEFT,
        CameraDirection.FRONT_RIGHT,
        CameraDirection.REAR,
        CameraDirection.REAR_LEFT,
        CameraDirection.REAR_RIGHT,
    }
    assert set(RING_CAMERA_TO_DIRECTION.values()) == canonical


# --------------------------------------------------------------------------- #
# LiDAR / pose IO (synthetic files)
# --------------------------------------------------------------------------- #
def _write_lidar_parquet(path, xyz_int, reflectivity, resolution=0.005):
    table = pa.table(
        {
            "x": pa.array(np.asarray(xyz_int)[:, 0], pa.int32()),
            "y": pa.array(np.asarray(xyz_int)[:, 1], pa.int32()),
            "z": pa.array(np.asarray(xyz_int)[:, 2], pa.int32()),
            "reflectivity": pa.array(np.asarray(reflectivity), pa.float32()),
        }
    )
    table = table.replace_schema_metadata(
        {"discretization_resolution": f"{resolution:f}"}
    )
    pq.write_table(table, path)


def test_read_lidar_top_xyz_de_discretizes_and_drops_sentinels(tmp_path):
    path = tmp_path / "0000000000.parquet"
    # row0: sentinel (refl == -1 AND xyz == 0) -> dropped.
    # row1: kept -> int * 0.005.
    # row2: refl == -1 but xyz != 0 -> NOT a sentinel -> kept.
    _write_lidar_parquet(
        path,
        xyz_int=[[0, 0, 0], [200, 100, -400], [400, 0, 0]],
        reflectivity=[-1.0, 0.5, -1.0],
    )
    xyz = io.read_lidar_top_xyz(str(path))
    assert xyz.shape == (2, 3) and xyz.dtype == np.float32
    np.testing.assert_allclose(xyz, [[1.0, 0.5, -2.0], [2.0, 0.0, 0.0]], atol=1e-6)


def test_read_lidar_top_xyz_empty(tmp_path):
    path = tmp_path / "empty.parquet"
    _write_lidar_parquet(path, xyz_int=np.zeros((0, 3), dtype=int), reflectivity=[])
    assert io.read_lidar_top_xyz(str(path)).shape == (0, 3)


def test_read_poses_parses_eight_columns_and_rejects_bad(tmp_path):
    good = tmp_path / "poses.txt"
    good.write_text(
        "0.05 -1.0 2.0 3.0 0.0 0.0 0.0 1.0\n0.15 -1.1 2.1 3.0 0.0 0.0 0.1 0.99\n"
    )
    poses = io.read_poses(str(good))
    assert poses.shape == (2, 8)
    np.testing.assert_allclose(poses[0, 1:4], [-1.0, 2.0, 3.0])

    bad = tmp_path / "bad.txt"
    bad.write_text("0.05 1.0 2.0 3.0\n")  # 4 columns
    with pytest.raises(ValueError):
        io.read_poses(str(bad))


# --------------------------------------------------------------------------- #
# Split table + scene-directory resolution
# --------------------------------------------------------------------------- #
def test_split_dir_aliases_and_validation():
    assert set(ALLOWED_SPLITS) == {
        "train",
        "val",
        "test",
        "test_e2e",
        "overlap_train_val",
        "all",
    }
    assert split_dir_names("val") == ("val", "validation")
    assert split_dir_names("test_e2e") == ("test_e2e", "test-e2e")
    with pytest.raises(ValueError):
        split_dir_names("trainval")


def _make_scene(scene_dir: Path) -> None:
    (scene_dir / "calibration").mkdir(parents=True, exist_ok=True)
    (scene_dir / "calibration" / "calib.json").write_text("{}")


def test_resolve_scene_dirs_flat_layout_is_passthrough(tmp_path):
    # Flat: scene UUID directly under root, no split folder. Any split returns it.
    _make_scene(tmp_path / "uuid-a")
    for split in ("train", "val", "test", "all"):
        resolved = io.resolve_scene_dirs(str(tmp_path), split)
        assert [d.name for d in resolved] == ["uuid-a"]


def test_resolve_scene_dirs_by_split_layout_selects_split(tmp_path):
    _make_scene(tmp_path / "train" / "uuid-train")
    _make_scene(tmp_path / "validation" / "uuid-val")
    assert [d.name for d in io.resolve_scene_dirs(str(tmp_path), "train")] == [
        "uuid-train"
    ]
    # "val" resolves via the "validation" alias folder.
    assert [d.name for d in io.resolve_scene_dirs(str(tmp_path), "val")] == ["uuid-val"]
    # "all" ignores the split folders and returns both.
    assert sorted(d.name for d in io.resolve_scene_dirs(str(tmp_path), "all")) == [
        "uuid-train",
        "uuid-val",
    ]


def test_resolve_scene_dirs_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.resolve_scene_dirs(str(tmp_path / "nope"), "val")
    with pytest.raises(FileNotFoundError):
        io.resolve_scene_dirs(str(tmp_path), "val")  # exists but no scene dirs


def test_scene_ref_paths():
    ref = io.SceneRef("uuid", "/root/uuid")
    assert ref.calib_path.endswith("uuid/calibration/calib.json")
    assert ref.poses_path.endswith("uuid/poses.txt")
    assert ref.map_osm_path.endswith("uuid/maps/map.osm")
    assert ref.camera_path("camera_ring_front", 7).endswith(
        "uuid/camera_ring_front/0000000007.jpg"
    )
    assert ref.lidar_top_path(7).endswith("uuid/lidar_top/0000000007.parquet")


# --------------------------------------------------------------------------- #
# Map geometry helpers
# --------------------------------------------------------------------------- #
def test_resample_polyline_count_and_endpoints():
    out = _resample_polyline(np.array([[0.0, 0.0], [10.0, 0.0]]), 3)
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out, [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])


def test_orient_to_flips_reversed_bound():
    left = np.array([[0.0, 0.0], [10.0, 0.0]])
    right_reversed = np.array([[10.0, 2.0], [0.0, 2.0]])
    oriented = _orient_to(left, right_reversed)
    np.testing.assert_allclose(oriented, [[0.0, 2.0], [10.0, 2.0]])


def test_lanelet_centerline_is_midpoint_of_bounds():
    left = np.array([[0.0, 0.0], [10.0, 0.0]])
    right = np.array([[0.0, 2.0], [10.0, 2.0]])
    center = _lanelet_centerline(left, right)
    np.testing.assert_allclose(center, [[0.0, 1.0], [10.0, 1.0]], atol=1e-9)


# --------------------------------------------------------------------------- #
# Lanelet2 OSM -> HDMap (synthetic map)
# --------------------------------------------------------------------------- #
_MAP_ORIGIN = {"latitude": 50.10, "longitude": 8.68}


def _write_map(map_dir: Path) -> None:
    """A tiny Lanelet2 map: one road lanelet (two ~north-offset bounds running
    east) plus a standalone stop line, anchored at ``_MAP_ORIGIN``."""
    map_dir.mkdir(parents=True, exist_ok=True)
    (map_dir / "origin.json").write_text(json.dumps(_MAP_ORIGIN))

    lat0, lon0 = _MAP_ORIGIN["latitude"], _MAP_ORIGIN["longitude"]
    dlat = 9.0e-5  # ~10 m north
    dlon = 1.4e-4  # ~10 m east
    # left bound (north), right bound (south), each a 2-node line running east.
    nodes = {
        1: (lat0 + dlat, lon0),
        2: (lat0 + dlat, lon0 + dlon),
        3: (lat0 - dlat, lon0),
        4: (lat0 - dlat, lon0 + dlon),
        5: (lat0, lon0 + dlon),
        6: (lat0 + dlat, lon0 + dlon),
    }
    node_xml = "".join(
        f'<node id="{nid}" lat="{lat:.10f}" lon="{lon:.10f}"><tag k="ele" v="100.0"/></node>'
        for nid, (lat, lon) in nodes.items()
    )
    ways_xml = (
        '<way id="10"><nd ref="1"/><nd ref="2"/>'
        '<tag k="type" v="line_thin"/><tag k="subtype" v="solid"/></way>'
        '<way id="11"><nd ref="3"/><nd ref="4"/>'
        '<tag k="type" v="line_thin"/><tag k="subtype" v="dashed"/></way>'
        '<way id="12"><nd ref="5"/><nd ref="6"/><tag k="type" v="stop_line"/></way>'
    )
    relations_xml = (
        '<relation id="20"><member type="way" role="left" ref="10"/>'
        '<member type="way" role="right" ref="11"/>'
        '<tag k="type" v="lanelet"/><tag k="subtype" v="road"/></relation>'
    )
    (map_dir / "map.osm").write_text(
        f'<?xml version="1.0" encoding="UTF-8"?><osm version="0.6">'
        f"{node_xml}{ways_xml}{relations_xml}</osm>"
    )


def test_kitscenes_map_parse_and_build(tmp_path):
    _write_map(tmp_path / "maps")
    kit_map = KITScenesMap.from_files(
        str(tmp_path / "maps" / "map.osm"), str(tmp_path / "maps" / "origin.json")
    )
    # 1 lane center (lanelet) + 2 lane boundaries (line_thin) + 1 stop line.
    assert kit_map.num_elements == 4
    hd = kit_map.build_hd_map(np.eye(4), radius_m=100.0)
    types = {e.type for e in hd.elements}
    assert MapElementType.LANE_CENTER in types
    assert MapElementType.LANE_BOUNDARY in types
    assert MapElementType.STOP_LINE in types

    lane = next(e for e in hd.elements if e.type is MapElementType.LANE_CENTER)
    assert lane.points.shape[1] == 2 and lane.points.dtype == np.float32
    assert lane.attrs.get("lane_type") == "vehicle"
    # Centerline runs between the two bounds: its |y| is smaller than a bound's.
    boundary = next(e for e in hd.elements if e.type is MapElementType.LANE_BOUNDARY)
    assert np.abs(lane.points[:, 1]).max() < np.abs(boundary.points[:, 1]).max()


def test_kitscenes_map_roi_excludes_far_elements(tmp_path):
    _write_map(tmp_path / "maps")
    kit_map = KITScenesMap.from_files(
        str(tmp_path / "maps" / "map.osm"), str(tmp_path / "maps" / "origin.json")
    )
    far = np.eye(4)
    far[:2, 3] = [5000.0, 5000.0]
    assert kit_map.build_hd_map(far, radius_m=50.0).elements == []


# --------------------------------------------------------------------------- #
# Hermetic processor build (synthetic on-disk scene)
# --------------------------------------------------------------------------- #
def _write_scene(scene_dir: Path) -> None:
    """A minimal two-frame scene: calib, poses, timestamps, one ring camera and
    one top LiDAR per frame, plus the synthetic Lanelet2 map."""
    (scene_dir / "calibration").mkdir(parents=True, exist_ok=True)
    (scene_dir / "calibration" / "calib.json").write_text(
        json.dumps(_synthetic_calib())
    )
    # Ego at the map origin (0, 0, 0) so the map ROI catches the synthetic lane.
    (scene_dir / "poses.txt").write_text(
        "0.05 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n0.15 1.0 0.0 0.0 0.0 0.0 0.0 1.0\n"
    )
    (scene_dir / "timestamp.reference.txt").write_text("0.05\n0.15\n")

    cam_dir = scene_dir / "camera_ring_front"
    cam_dir.mkdir(parents=True, exist_ok=True)
    lidar_dir = scene_dir / "lidar_top"
    lidar_dir.mkdir(parents=True, exist_ok=True)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    for frame_index in (0, 1):
        cv2.imwrite(str(cam_dir / f"{frame_index:010d}.jpg"), image)
        _write_lidar_parquet(
            lidar_dir / f"{frame_index:010d}.parquet",
            xyz_int=[[2000, 0, 0], [0, 0, 0]],
            reflectivity=[0.5, -1.0],  # 2nd point is a sentinel -> dropped
        )
    _write_map(scene_dir / "maps")


def test_processor_builds_frame_hermetically(tmp_path):
    scene_dir = tmp_path / "scene-uuid"
    _write_scene(scene_dir)
    out = tmp_path / "out"

    proc = KITScenesMultimodalDatasetProcessor(
        common_output_path=str(out), split="val", context_aggregators=[]
    )
    ref = io.SceneRef(scene_id="scene-uuid", scene_dir=str(scene_dir))
    assert io.scene_frame_count(str(scene_dir)) == 2

    fd = proc._prepare_standardized_frame_data((ref, 0))
    assert fd.segment_id == "scene-uuid" and fd.frame_id == 0
    assert np.isclose(fd.timestamp, 0.05)

    # Cameras: only the ring camera present in the calib.
    assert set(fd.cameras) == {CameraDirection.FRONT}
    cam = fd.cameras[CameraDirection.FRONT]
    assert cam.image.shape == (4, 4, 3) and cam.image.dtype == np.uint8

    # LiDAR: de-discretized, sentinel dropped, identity extrinsic -> sensor == ego.
    assert fd.lidar is not None
    xyz = fd.lidar.points[["x", "y", "z"]].to_numpy()
    assert xyz.shape == (1, 3)
    np.testing.assert_allclose(xyz[0], [10.0, 0.0, 0.0], atol=1e-6)

    # Ego pose at the origin; KITScenes ships no boxes.
    assert fd.frame_detections_3d.detections == []
    assert fd.aux_data["pose_matrix"].shape == (4, 4)
    np.testing.assert_allclose(fd.aux_data["pose_matrix"][:3, 3], [0.0, 0.0, 0.0])

    # HD map present (default adapters include HDMapBEVAdapter) with a lane center.
    assert fd.hd_map is not None
    assert MapElementType.LANE_CENTER in {e.type for e in fd.hd_map.elements}


# --------------------------------------------------------------------------- #
# Real-frame checks (skipped without data)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not _KITSCENES_ROOT or not Path(_KITSCENES_ROOT).exists():
        pytest.skip("no KITScenes data available (set KITSCENES_ROOT)")
    try:
        scene_dirs = io.resolve_scene_dirs(_KITSCENES_ROOT, "all")
    except FileNotFoundError:
        pytest.skip("no KITScenes scenes under KITSCENES_ROOT")
    ref = io.SceneRef(scene_id=scene_dirs[0].name, scene_dir=str(scene_dirs[0]))
    out = tmp_path_factory.mktemp("kitscenes")
    proc = KITScenesMultimodalDatasetProcessor(
        common_output_path=str(out), split="all", context_aggregators=[]
    )
    return ref, proc._prepare_standardized_frame_data((ref, 0))


def test_real_ring_cameras_have_calibration(built):
    _ref, fd = built
    assert len(fd.cameras) >= 1
    for direction, cam in fd.cameras.items():
        assert direction in set(RING_CAMERA_TO_DIRECTION.values())
        assert cam.image.ndim == 3 and cam.image.dtype == np.uint8
        assert cam.intrinsics.shape == (3, 3)
        assert cam.extrinsics.shape == (4, 4) and np.isfinite(cam.extrinsics).all()


def test_real_lidar_is_ego_frame(built):
    _ref, fd = built
    assert fd.lidar is not None
    xyz = fd.lidar.points[["x", "y", "z"]].to_numpy()
    assert xyz.dtype == np.float32 and len(xyz) > 0 and np.isfinite(xyz).all()
    assert xyz[:, 0].max() > 20.0  # a top sweep reaches well beyond a few metres


def test_real_frame_has_no_detections(built):
    _ref, fd = built
    assert fd.frame_detections_3d.detections == []


def test_real_ego_pose_is_rigid(built):
    _ref, fd = built
    pose = fd.aux_data["pose_matrix"]
    assert pose.shape == (4, 4) and np.isfinite(pose).all()
    np.testing.assert_allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-5)


def test_real_hd_map_has_lane_centers(built):
    _ref, fd = built
    assert fd.hd_map is not None and len(fd.hd_map.elements) > 0
    types = {e.type for e in fd.hd_map.elements}
    assert MapElementType.LANE_CENTER in types
    for element in fd.hd_map.elements:
        assert element.points.shape[1] == 2 and np.isfinite(element.points).all()
