# flake8: noqa: E501
"""Tests for the MUHOS processor.

* **Units** (always run in the MUHOS environment): set-id cleanup, stamp parsing, the
  image-to-sweep matching, motion compensation, and the rig geometry (rigid transforms;
  the ego frame puts the road normal on +z and the travel direction on +x).
* **Synthetic set end-to-end** (hermetic): a 150 s drive written to ``tmp_path`` -- CAN
  speed/yaw rate and a PCAN-GPS log (1 Hz fixes repeated every 100 ms, the position 0.2 s
  late, speed and course 1 s late, plus its gyro), both stamped by a bus clock that runs
  0.6 s off the sensors' and drifts; the lidar IMU on the sensor clock; LAZ sweeps with
  missing returns and car-body returns; 25 Hz PNGs -- goes through the GPS reader, the
  bus-clock alignment, the fused trajectory (from CAN, and from GPS + gyro as for the set
  with an empty CAN log), the converter and the processor.
* **Real-data checks** (skipped unless ``MUHOS_ROOT`` points at an extracted set): the
  fused trajectory stays within a few metres of the GPS, and the built frame's lidar puts
  the road at z = 0.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

# laspy (+ lazrs) and pyproj are installed in the MUHOS environment only (datasets/muhos/).
laspy = pytest.importorskip("laspy")
pytest.importorskip("pyproj")

from standard_e2e.caching.src_datasets.muhos import _muhos_calibration as calib
from standard_e2e.caching.src_datasets.muhos import _muhos_io as io
from standard_e2e.caching.src_datasets.muhos._muhos_ego import (
    GPS_POSITION_LAG_S,
    PCAN_GYRO_DELAY_S,
    PCAN_GYRO_SCALE,
    build_ego_trajectory,
    estimate_bus_clock,
    read_gps_fixes,
)
from standard_e2e.caching.src_datasets.muhos.muhos_dataset_converter import (
    FrameRef,
    MuhosDatasetConverter,
    match_images_to_sweeps,
)
from standard_e2e.caching.src_datasets.muhos.muhos_dataset_processor import (
    MuhosDatasetProcessor,
    motion_compensate,
    world_from_ego,
)
from standard_e2e.enums import CameraDirection
from standard_e2e.enums import TrajectoryComponent as TC

_MUHOS_ROOT = os.environ.get("MUHOS_ROOT", "")
T0 = 1_791_368_400.0  # 2026-10-07, like the release
COURSE_SPEED_LAG_S = 1.0  # the GPS speed and course lag more than its position
CAN_SPEED_FACTOR = 0.985  # the wheel speed reads 1.5 % low
BUS_LAG_S, BUS_DRIFT = -0.6, -200e-6  # bus-log time = sensor time + lag (+ drift)
ANCHOR_LAT, ANCHOR_LON = 65.0, 25.5
DURATION_S = 150.0
TURNS = (
    (15.0, 1),
    (40.0, -1),
    (70.0, 1),
    (100.0, -1),
    (130.0, 1),
)  # (start s, direction), 10 s each


# --- units -----------------------------------------------------------------------


def test_set_id_from_folder_strips_the_bucket_affixes():
    assert io.set_id_from_folder("imu_highway_processed") == "highway"
    assert (
        io.set_id_from_folder("imu_alakylantie_80kmh_processed") == "alakylantie_80kmh"
    )
    assert io.set_id_from_folder("my_set") == "my_set"


def test_list_stamped_files_sorts_by_the_stamp_not_the_name(tmp_path):
    for name in (
        "image_10_900000000.png",
        "image_9_5.png",
        "image_10_000000001.png",
        "notes.txt",
    ):
        (tmp_path / name).write_bytes(b"")
    stamps, paths = io.list_stamped_files(str(tmp_path))
    assert list(stamps) == [9_000_000_005, 10_000_000_001, 10_900_000_000]
    assert [Path(p).name for p in paths] == [
        "image_9_5.png",
        "image_10_000000001.png",
        "image_10_900000000.png",
    ]
    assert io.list_stamped_files(str(tmp_path / "missing"))[1] == []


def test_match_images_to_sweeps_takes_the_image_closest_to_the_sweep_middle():
    sweeps = np.array([0, 100_000_000, 200_000_000, 300_000_000])
    images = np.array([10_000_000, 45_000_000, 90_000_000, 150_000_000, 305_000_000])
    # sweep 0 -> 45 ms (closest to 50 ms); sweep 1 -> 150 ms; sweep 2 -> none in [200, 300); sweep 3 -> 305 ms
    assert list(match_images_to_sweeps(sweeps, images)) == [1, 3, -1, 4]
    assert list(match_images_to_sweeps(sweeps, np.zeros(0, dtype=np.int64))) == [
        -1,
        -1,
        -1,
        -1,
    ]


def _straight_ref(speed: float, t_frame: float) -> FrameRef:
    t = np.linspace(t_frame - 0.15, t_frame + 0.15, 31)
    return FrameRef(
        set_id="s",
        frame_index=0,
        lidar_path="",
        lidar_stamp_ns=0,
        image_path="",
        image_stamp_ns=0,
        timestamp=t_frame,
        pose=(speed * t_frame, 0.0, 0.0, speed),
        trajectory_slice=np.stack(
            [t, speed * t, np.zeros_like(t), np.zeros_like(t)], axis=1
        ),
        anchor_lat=ANCHOR_LAT,
        anchor_lon=ANCHOR_LON,
    )


def test_motion_compensation_moves_points_to_the_frame_time():
    # A wall 20 m ahead, seen while driving forward at 20 m/s: 50 ms before the frame time
    # the ego was 1 m further back, so the wall was measured 1 m further away.
    ref = _straight_ref(20.0, 100.0)
    measured = np.array([[21.0, 0.0, 1.0], [20.0, 2.0, 1.0], [19.0, -1.0, 0.5]])
    times = np.array([99.95, 100.0, 100.05])
    out = motion_compensate(measured, times, ref)
    np.testing.assert_allclose(
        out, [[20.0, 0.0, 1.0], [20.0, 2.0, 1.0], [20.0, -1.0, 0.5]], atol=1e-9
    )


def test_world_from_ego_is_a_planar_pose():
    pose = world_from_ego(3.0, -2.0, np.pi / 2)
    np.testing.assert_allclose(
        pose @ [1.0, 0.0, 0.0, 1.0], [3.0, -1.0, 0.0, 1.0], atol=1e-12
    )


@pytest.mark.parametrize(
    "transform",
    [
        calib.T_EGO_FROM_LIDAR,
        calib.T_LIDAR_FROM_LEFT,
        calib.T_AUX_FROM_LEFT,
        calib.T_EGO_FROM_AUX,
        calib.T_EGO_FROM_RIGHT,
    ],
)
def test_rig_transforms_are_rigid(transform):
    rotation = transform[:3, :3]
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    np.testing.assert_allclose(transform[3], [0, 0, 0, 1])


def test_ego_frame_levels_the_lidar():
    rotation = calib.T_EGO_FROM_LIDAR[:3, :3]
    normal = rotation @ (
        calib.GROUND_NORMAL_LIDAR / np.linalg.norm(calib.GROUND_NORMAL_LIDAR)
    )
    np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1e-9)
    travel = rotation @ calib.TRAVEL_DIRECTION_LIDAR
    assert travel[0] > 0.99 and abs(travel[1]) < 1e-9
    # The lidar itself sits LIDAR_HEIGHT_M above the ego origin (the road below it).
    np.testing.assert_allclose(
        calib.T_EGO_FROM_LIDAR[:3, 3], [0.0, 0.0, calib.LIDAR_HEIGHT_M]
    )


def test_colour_camera_looks_forward_and_down_from_the_roof():
    t_ego_from_aux = calib.T_EGO_FROM_AUX
    optical_axis = t_ego_from_aux[:3, :3] @ [0.0, 0.0, 1.0]
    assert optical_axis[0] > 0.9  # forward
    assert -0.4 < optical_axis[2] < -0.1  # pitched down
    assert 1.3 < t_ego_from_aux[2, 3] < 2.0  # roof height, below the lidar
    assert calib.AUX_DISTORTION.shape == (4,)


# --- synthetic set -----------------------------------------------------------------


def _truth(t):
    """Speed oscillates; straight stretches between 10 s turns (TURNS), sensor-clock time."""
    tau = np.asarray(t) - T0
    speed = 12.0 + 4.0 * np.sin(2 * np.pi * tau / 17.0)
    turn = lambda start: np.where(
        (tau >= start) & (tau < start + 10.0),
        np.sin(np.pi * (tau - start) / 10.0) ** 2,
        0.0,
    )
    yaw_rate = np.radians(9.0) * sum(d * turn(start) for start, d in TURNS)
    return speed, yaw_rate


def _bus(t):
    """Bus-log stamp of an event at sensor time ``t``."""
    return t + BUS_LAG_S + BUS_DRIFT * (t - T0)


def _integrate_truth(times):
    grid = np.arange(T0, T0 + DURATION_S + 1.0, 0.001)
    speed, yaw_rate = _truth(grid)
    heading = (
        np.radians(30.0) + np.cumsum(yaw_rate) * 0.001
    )  # counter-clockwise from east
    x = np.cumsum(speed * np.cos(heading)) * 0.001
    y = np.cumsum(speed * np.sin(heading)) * 0.001
    return (
        np.interp(times, grid, x),
        np.interp(times, grid, y),
        np.interp(times, grid, heading),
        np.interp(times, grid, speed),
    )


def _write_can_and_gps(can_dir: Path) -> None:
    from pyproj import Transformer

    rows_can = []
    for t in np.arange(T0, T0 + DURATION_S, 1 / 38.0):
        speed, _ = _truth(t)
        rows_can.append(
            {
                "timestamp": _bus(t),
                "frame_id": "0xB4",
                "message": "SPEED",
                "signal": "SPEED",
                "value": speed * CAN_SPEED_FACTOR * 3.6,
            }
        )
    for t in np.arange(T0, T0 + DURATION_S, 0.1):
        _, yaw_rate = _truth(t)
        rows_can.append(
            {
                "timestamp": _bus(t),
                "frame_id": "0x24",
                "message": "KINEMATICS",
                "signal": "YAW_RATE",
                "value": np.degrees(yaw_rate),
            }
        )
    to_wgs84 = Transformer.from_crs(
        f"+proj=aeqd +lat_0={ANCHOR_LAT} +lon_0={ANCHOR_LON} +datum=WGS84 +units=m",
        "EPSG:4326",
        always_xy=True,
    )
    rows_gps = []
    fix = None
    for k, cycle in enumerate(np.arange(T0 + 0.07, T0 + DURATION_S, 0.1)):
        if (
            k % 10 == 0
        ):  # a new 1 Hz fix: the position GPS_POSITION_LAG_S old, speed and course older
            x, y, _, _ = _integrate_truth(np.array([cycle - GPS_POSITION_LAG_S]))
            _, _, heading, speed = _integrate_truth(
                np.array([cycle - COURSE_SPEED_LAG_S])
            )
            lon, lat = to_wgs84.transform(x[0], y[0])
            utc = int(cycle)
            fix = dict(
                lat=lat,
                lon=lon,
                speed=speed[0],
                course=(90.0 - np.degrees(heading[0])) % 360.0,
                utc=utc,
            )
        assert fix is not None
        sec = time.gmtime(fix["utc"])
        for msg, sig, val, dt in [
            ("GPS_CourseSpeed", "GPS_Course", fix["course"], 0.0),
            ("GPS_CourseSpeed", "GPS_Speed", fix["speed"] * 3.6, 0.0),
            (
                "GPS_PositionLongitude",
                "GPS_Longitude_Degree",
                float(int(fix["lon"])),
                0.0002,
            ),
            (
                "GPS_PositionLongitude",
                "GPS_Longitude_Minutes",
                (fix["lon"] - int(fix["lon"])) * 60.0,
                0.0002,
            ),
            ("GPS_PositionLongitude", "GPS_IndicatorEW", float(ord("E")), 0.0002),
            (
                "GPS_PositionLatitude",
                "GPS_Latitude_Degree",
                float(int(fix["lat"])),
                0.0004,
            ),
            (
                "GPS_PositionLatitude",
                "GPS_Latitude_Minutes",
                (fix["lat"] - int(fix["lat"])) * 60.0,
                0.0004,
            ),
            ("GPS_PositionLatitude", "GPS_IndicatorNS", float(ord("N")), 0.0004),
            ("GPS_DateTime", "UTC_Second", float(sec.tm_sec), 0.001),
        ]:
            rows_gps.append(
                {
                    "timestamp": _bus(cycle + dt),
                    "frame_id": "0x62x",
                    "message": msg,
                    "signal": sig,
                    "value": val,
                }
            )
    for t in np.arange(
        T0 + 0.02, T0 + DURATION_S, 0.05
    ):  # the PCAN-GPS gyro reads low and late
        _, yaw_rate = _truth(t)
        rows_gps.append(
            {
                "timestamp": _bus(t) + PCAN_GYRO_DELAY_S,
                "frame_id": "0x631",
                "message": "L3GD20_Rotation_B",
                "signal": "Rotation_Z",
                "value": np.degrees(yaw_rate) / PCAN_GYRO_SCALE,
            }
        )
    (can_dir / "toyota_CAN_test.json").write_text(json.dumps(rows_can))
    (can_dir / "toyota_GPS_test.json").write_text(json.dumps(rows_gps))


def _write_sweep(path: Path, stamp: float) -> None:
    """A sweep (lidar frame) of a flat road, a missing return and points on the car body."""
    rng = np.random.default_rng(int(stamp * 10) % 1000)
    ego = np.c_[
        rng.uniform(5, 30, 200), rng.uniform(-5, 5, 200), np.zeros(200)
    ]  # road, z = 0 in ego
    body = np.array([[1.5, 0.0, 1.05], [-1.0, 0.3, 1.7]])  # hood and roof
    pts_ego = np.vstack([ego, body])
    lidar_from_ego = np.linalg.inv(calib.T_EGO_FROM_LIDAR)
    pts = (np.c_[pts_ego, np.ones(len(pts_ego))] @ lidar_from_ego.T)[:, :3]
    pts = np.vstack([pts, np.zeros((1, 3))])  # a missing return
    las = laspy.create(point_format=3, file_version="1.2")
    las.header.scales = np.array([0.001, 0.001, 0.001])
    las.header.offsets = np.array([0.0, 0.0, 0.0])
    las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
    las.intensity = np.full(len(pts), 7, dtype=np.uint16)
    las.gps_time = stamp + np.linspace(0.0, 0.1, len(pts)) - io.GPS_UNIX_OFFSET_S
    las.write(str(path))


def _write_lidar_imu(path: Path) -> None:
    """The lidar's IMU on the sensor clock, 100 Hz, as two concatenated JSON arrays."""
    rows = []
    for t in np.arange(T0, T0 + DURATION_S, 0.01):
        _, yaw_rate = _truth(t)
        rows.append(
            {
                "timestamp_ns": int(round(t * 1e9)),
                "angular_velocity": {"x": 0.001, "y": -0.002, "z": float(yaw_rate)},
            }
        )
    half = len(rows) // 2
    path.write_text(json.dumps(rows[half:]) + "\n" + json.dumps(rows[:half]))


@pytest.fixture(scope="module")
def synthetic_set(tmp_path_factory):
    root = tmp_path_factory.mktemp("muhos")
    set_dir = root / "imu_testdrive_processed"
    for sub in ("lidar_points", "aux_image_color_anonymized", "can", "imu"):
        (set_dir / sub).mkdir(parents=True)
    _write_can_and_gps(set_dir / "can")
    _write_lidar_imu(set_dir / "imu" / "lidar_imu.json")
    for k in range(5):
        stamp = T0 + 30.0 + 0.1 * k
        sec, nsec = int(stamp), int(round((stamp - int(stamp)) * 1e9))
        _write_sweep(set_dir / "lidar_points" / f"pointcloud_{sec}_{nsec}.laz", stamp)
    for k in range(10):
        if k == 6:
            continue  # sweep 2 ([30.2, 30.3)) gets no image -> skipped
        stamp = (
            T0 + 30.01 + calib.CAMERA_DELAY_S + 0.04 * k
        )  # exposed CAMERA_DELAY_S before the stamp
        sec, nsec = int(stamp), int(round((stamp - int(stamp)) * 1e9))
        image = np.full((54, 96, 3), 40 + 10 * k, dtype=np.uint8)
        cv2.imwrite(
            str(set_dir / "aux_image_color_anonymized" / f"image_{sec}_{nsec}.png"),
            image,
        )
    return root


def test_read_gps_fixes_one_row_per_1hz_fix(synthetic_set):
    rows = io.read_json(
        str(synthetic_set / "imu_testdrive_processed" / "can" / "toyota_GPS_test.json")
    )
    fixes = read_gps_fixes(rows)
    assert 148 <= len(fixes.t) <= 150
    np.testing.assert_allclose(np.diff(fixes.t), 1.0, atol=0.01)
    assert np.all(np.abs(fixes.lat - ANCHOR_LAT) < 0.03) and np.all(
        np.abs(fixes.lon - ANCHOR_LON) < 0.06
    )  # a 1.8 km drive


def test_read_imu_merges_and_sorts_the_concatenated_arrays(synthetic_set):
    t, w = io.read_imu(
        str(synthetic_set / "imu_testdrive_processed" / "imu" / "lidar_imu.json")
    )
    assert (
        len(t) == int(DURATION_S * 100)
        and np.all(np.diff(t) > 0)
        and w.shape == (len(t), 3)
    )


def test_bus_clock_is_recovered_from_the_lidar_imu(synthetic_set):
    ref = io.discover_sets(str(synthetic_set))[0]
    can = io.read_json(ref.can_json)
    yaw = io.read_signal(can, "KINEMATICS", "YAW_RATE")
    t_imu, w_imu = io.read_imu(ref.imu_json)
    clock = estimate_bus_clock(yaw[:, 0], np.radians(yaw[:, 1]), t_imu, w_imu[:, 2])
    t_sensor = np.array([T0 + 20.0, T0 + 75.0, T0 + 130.0])
    np.testing.assert_allclose(clock.to_sensor(_bus(t_sensor)), t_sensor, atol=0.005)
    assert clock.n_windows >= 3


def _trajectory_error(ego):
    """Max position (m) and heading (rad) error against the truth (sensor clock) over 10-140 s
    (the truth starts at the origin, the fused frame at the first fix: compare displacements).
    """
    times = np.arange(T0 + 10.0, T0 + 140.0, 0.5)
    x, y, heading, speed = ego.at(times)
    tx, ty, th, ts = _integrate_truth(times)
    dx, dy = (x - x[0]) - (tx - tx[0]), (y - y[0]) - (ty - ty[0])
    return (
        np.max(np.hypot(dx, dy)),
        np.max(np.abs(np.angle(np.exp(1j * (heading - th))))),
        speed / ts,
    )


def test_fused_trajectory_follows_the_truth(synthetic_set):
    ref = io.discover_sets(str(synthetic_set))[0]
    ego = build_ego_trajectory(
        io.read_json(ref.can_json),
        io.read_json(ref.gps_json),
        io.read_imu(ref.imu_json),
    )
    assert ego.source == "can" and abs(ego.clock.lag0 - BUS_LAG_S) < 0.05
    assert abs(ego.speed_scale - 1.0 / CAN_SPEED_FACTOR) < 0.003
    position_error, heading_error, speed_ratio = _trajectory_error(ego)
    assert position_error < 0.2 and heading_error < np.radians(0.2)
    np.testing.assert_allclose(speed_ratio, 1.0, atol=0.004)
    assert np.median(ego.gps_residual_m) < 0.3


def test_trajectory_without_can_uses_gps_and_the_pcan_gyro(synthetic_set):
    ref = io.discover_sets(str(synthetic_set))[0]
    ego = build_ego_trajectory(
        [], io.read_json(ref.gps_json), io.read_imu(ref.imu_json)
    )  # like alakylantie_80kmh's empty CAN log
    assert ego.source == "gps+pcan_gyro"
    position_error, heading_error, speed_ratio = _trajectory_error(ego)
    assert position_error < 0.75 and heading_error < np.radians(0.5)
    assert np.median(np.abs(speed_ratio - 1.0)) < 0.02


def test_converter_pairs_sweeps_with_images(synthetic_set):
    converter = MuhosDatasetConverter(
        source_processor=MuhosDatasetProcessor(
            common_output_path=str(synthetic_set / "out"), split="all"
        ),
        input_path=str(synthetic_set),
        split="all",
        do_parallel_processing=False,
    )
    refs = list(converter._get_source_dataset_iterator())
    assert [r.frame_index for r in refs] == [0, 1, 2, 3]
    assert all(r.set_id == "testdrive" for r in refs)
    for r in refs:
        # the frame time is the image's capture time (its stamp minus the camera delay), inside its sweep
        capture_ns = r.image_stamp_ns - int(round(calib.CAMERA_DELAY_S * 1e9))
        assert r.lidar_stamp_ns <= capture_ns < r.lidar_stamp_ns + 100_000_000
        assert r.timestamp == capture_ns / 1e9
        assert r.trajectory_slice.shape[1] == 4


def test_processor_builds_fisheye_front_camera_and_clean_lidar(synthetic_set):
    processor = MuhosDatasetProcessor(
        common_output_path=str(synthetic_set / "out"), split="all"
    )
    converter = MuhosDatasetConverter(
        source_processor=processor,
        input_path=str(synthetic_set),
        split="all",
        do_parallel_processing=False,
    )
    ref = next(iter(converter._get_source_dataset_iterator()))
    frame = processor._prepare_standardized_frame_data(ref)
    camera = frame.cameras[CameraDirection.FRONT]
    assert (
        camera.is_fisheye
        and camera.distortion is not None
        and camera.distortion.shape == (4,)
    )
    np.testing.assert_allclose(camera.intrinsics, calib.AUX_INTRINSICS)
    np.testing.assert_allclose(camera.extrinsics, calib.T_EGO_FROM_AUX, atol=1e-6)
    points = frame.lidar.points[["x", "y", "z"]].to_numpy()
    assert (
        len(points) == 200
    )  # the missing return and the two car-body returns are gone
    np.testing.assert_allclose(points[:, 2], 0.0, atol=0.01)  # the road at z = 0
    assert (
        frame.segment_id == "testdrive" and frame.frame_detections_3d.detections == []
    )
    gp = frame.global_position
    assert gp.get([TC.SPEED])[0, 0] > 5.0
    assert frame.aux_data["pose_matrix"].shape == (4, 4)
    lat, lon = frame.extra_index_data["latitude"], frame.extra_index_data["longitude"]
    assert abs(lat - ANCHOR_LAT) < 0.03 and abs(lon - ANCHOR_LON) < 0.06


# --- real data ---------------------------------------------------------------------


@pytest.mark.skipif(not _MUHOS_ROOT, reason="set MUHOS_ROOT to an extracted MUHOS set")
def test_real_trajectory_follows_the_gps():
    ref = io.discover_sets(_MUHOS_ROOT)[0]
    ego = build_ego_trajectory(
        io.read_json(ref.can_json),
        io.read_json(ref.gps_json),
        io.read_imu(ref.imu_json),
    )
    assert ego.source == "can" and ego.clock.n_windows >= 2
    assert 1.0 < ego.speed_scale < 1.03  # the wheel speed reads ~1.4 % low on every set
    assert np.median(ego.gps_residual_m) < 2.0


@pytest.mark.skipif(not _MUHOS_ROOT, reason="set MUHOS_ROOT to an extracted MUHOS set")
def test_real_frame_road_is_at_ego_z_zero(tmp_path):
    processor = MuhosDatasetProcessor(common_output_path=str(tmp_path), split="all")
    converter = MuhosDatasetConverter(
        source_processor=processor,
        input_path=_MUHOS_ROOT,
        split="all",
        do_parallel_processing=False,
    )
    refs = list(converter._get_source_dataset_iterator())
    heights = []
    for ref in [refs[k] for k in np.linspace(0, len(refs) - 1, 5).astype(int)]:
        points = (
            processor._prepare_standardized_frame_data(ref)
            .lidar.points[["x", "y", "z"]]
            .to_numpy()
        )
        near = points[
            (points[:, 0] > 4) & (points[:, 0] < 8) & (np.abs(points[:, 1]) < 2)
        ]
        heights.append(
            np.percentile(near[:, 2], 25)
        )  # the road under whatever stands on it
    assert abs(np.median(heights)) < 0.1
