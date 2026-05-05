"""Tests for ``Av2SensorDatasetProcessor`` map handling.

Two layers:

* **Construction defaults** — adapters / context aggregators wired correctly.
  Always runs.

* **HD-map round-trip against AV2's own API** — load a real AV2 log via
  ``ArgoverseStaticMap``, run our processor's ``_build_hd_map``, then assert
  the output is consistent with AV2's idiom (XY canonical, Z advisory). Also
  verifies the same code path on a log that *does* contain NaN Z values in
  its source JSON, to confirm the slice-then-transform pipeline keeps NaN
  out of the validator. Skipped automatically when the local AV2 sensor
  dataset isn't mounted.

Why these particular logs?

* ``00a6ffc1-...`` (PIT) — first sample log, every waypoint XYZ is finite.
* ``01bb304d-...`` (WDC) — surveyed by us earlier and confirmed to ship
  literal ``NaN`` Z values in the JSON (10/205 lane boundaries, 6/7
  drivable-area boundaries). This is the ground truth for "AV2 may store
  NaN Z; vector XY is still authoritative".
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from av2.map.map_api import ArgoverseStaticMap
from scipy.spatial.transform import Rotation

from standard_e2e.caching.adapters import (
    Detections3DIdentityAdapter,
    HDMapBEVAdapter,
    LidarAdapter,
    PanoImageAdapter,
)
from standard_e2e.caching.src_datasets.av2_sensor.av2_sensor_dataset_processor import (
    Av2SensorDatasetProcessor,
)
from standard_e2e.enums import MapElementType

AV2_ROOT = Path("/mnt/bigdisk/datasets/argoverse2/sensor/train")
GOOD_LOG = "00a6ffc1-6ce9-3bc3-a060-6006e9893a1a"  # PIT, all finite
BAD_LOG = "01bb304d-7bd8-35f8-bbef-7086b688e35e"  # WDC, has NaN Z


# --- construction defaults ---------------------------------------------------


def test_av2_sensor_defaults(tmp_path: Path):
    proc = Av2SensorDatasetProcessor(str(tmp_path), split="train")
    assert proc.dataset_name == "av2_sensor"
    assert set(("train", "val", "test")).issubset(set(proc.allowed_splits))
    adapters = getattr(proc, "_adapters")
    assert len(adapters) == 4
    assert isinstance(adapters[0], PanoImageAdapter)
    assert isinstance(adapters[1], LidarAdapter)
    assert isinstance(adapters[2], HDMapBEVAdapter)
    assert isinstance(adapters[3], Detections3DIdentityAdapter)


# --- HD-map round-trip vs the AV2 API ----------------------------------------


def _ego_pose_at_first_frame(log_path: Path) -> np.ndarray:
    """Read the first ego pose from ``city_SE3_egovehicle.feather`` as a 4x4."""
    df = pd.read_feather(log_path / "city_SE3_egovehicle.feather")
    row = df.iloc[0]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Rotation.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix()
    T[:3, 3] = (row.tx_m, row.ty_m, row.tz_m)
    return T


def _build_processor_with_map(log_path: Path) -> Av2SensorDatasetProcessor:
    """Construct a processor without running ``__init__`` and inject a real map.

    The full ``__init__`` requires a writable output directory and triggers
    side effects we don't need for HD-map unit tests.
    """
    proc = Av2SensorDatasetProcessor.__new__(Av2SensorDatasetProcessor)
    proc._map = ArgoverseStaticMap.from_map_dir(log_path / "map", build_raster=False)
    return proc


@pytest.fixture
def good_log_processor() -> Av2SensorDatasetProcessor:
    if not (AV2_ROOT / GOOD_LOG / "map").exists():
        pytest.skip("AV2 good log not available")
    return _build_processor_with_map(AV2_ROOT / GOOD_LOG)


@pytest.fixture
def bad_log_processor() -> Av2SensorDatasetProcessor:
    if not (AV2_ROOT / BAD_LOG / "map").exists():
        pytest.skip("AV2 bad log not available")
    return _build_processor_with_map(AV2_ROOT / BAD_LOG)


@pytest.fixture
def good_log_pose() -> np.ndarray:
    if not (AV2_ROOT / GOOD_LOG).exists():
        pytest.skip("AV2 good log not available")
    return _ego_pose_at_first_frame(AV2_ROOT / GOOD_LOG)


@pytest.fixture
def bad_log_pose() -> np.ndarray:
    if not (AV2_ROOT / BAD_LOG).exists():
        pytest.skip("AV2 bad log not available")
    return _ego_pose_at_first_frame(AV2_ROOT / BAD_LOG)


def test_clean_log_emits_finite_2d_elements(good_log_processor, good_log_pose):
    """Sanity: the clean log's HD map round-trips with no warnings, all (N,2) finite."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hd_map = good_log_processor._build_hd_map(good_log_pose)
    assert hd_map is not None
    assert len(hd_map.elements) > 0
    for e in hd_map.elements:
        assert e.points.ndim == 2 and e.points.shape[1] == 2
        assert np.isfinite(e.points).all()


def test_dirty_log_does_not_propagate_nan(bad_log_processor, bad_log_pose):
    """The bad WDC log has literal NaN Z in its JSON; we must not surface it.

    Setting ``warnings.simplefilter('error')`` turns any silent NaN-cast
    warning into a test failure -- this is the defensive check that keeps
    us from regressing back into the original ``invalid value encountered
    in cast`` situation.
    """
    # First sanity: confirm the source actually has the NaN we're testing
    # the pipeline against, otherwise the test is vacuous.
    nan_in_source = any(
        not np.isfinite(da.xyz).all()
        for da in bad_log_processor._map.vector_drivable_areas.values()
    )
    assert nan_in_source, "expected NaN in source data; test premise broken"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hd_map = bad_log_processor._build_hd_map(bad_log_pose)

    assert hd_map is not None
    for e in hd_map.elements:
        assert e.points.ndim == 2 and e.points.shape[1] == 2
        assert np.isfinite(e.points).all(), f"non-finite point survived in {e.id}"


@pytest.mark.parametrize(
    "proc_fixture,pose_fixture",
    [
        ("good_log_processor", "good_log_pose"),
        ("bad_log_processor", "bad_log_pose"),
    ],
)
def test_element_counts_per_type(proc_fixture, pose_fixture, request):
    """Per the unified-format spec (docs/map_element_translation.md §5.2):

    - exactly one ``LANE_CENTER`` per lane segment;
    - one ``LANE_BOUNDARY`` per lane edge whose ``mark_type != NONE``
      (NONE / UNKNOWN suppress emission);
    - one ``INTERSECTION`` per lane segment with ``is_intersection=True``;
    - one ``DRIVABLE_AREA`` per ``vector_drivable_areas`` entry;
    - one ``CROSSWALK`` per ``vector_pedestrian_crossings`` entry.
    """
    proc = request.getfixturevalue(proc_fixture)
    pose = request.getfixturevalue(pose_fixture)
    avm = proc._map

    expected_lane_center = len(avm.vector_lane_segments)
    expected_intersection = sum(
        1 for ls in avm.vector_lane_segments.values() if ls.is_intersection
    )
    expected_lane_boundary = sum(
        sum(
            1
            for mt in (ls.left_mark_type, ls.right_mark_type)
            if str(mt) not in ("LaneMarkType.NONE", "LaneMarkType.UNKNOWN")
        )
        for ls in avm.vector_lane_segments.values()
    )
    expected_drivable_area = len(avm.vector_drivable_areas)
    expected_crosswalk = len(avm.vector_pedestrian_crossings)

    hd_map = proc._build_hd_map(pose)
    by_type = {t: 0 for t in MapElementType}
    for e in hd_map.elements:
        by_type[e.type] += 1

    assert by_type[MapElementType.LANE_CENTER] == expected_lane_center
    assert by_type[MapElementType.LANE_BOUNDARY] == expected_lane_boundary
    assert by_type[MapElementType.INTERSECTION] == expected_intersection
    assert by_type[MapElementType.DRIVABLE_AREA] == expected_drivable_area
    assert by_type[MapElementType.CROSSWALK] == expected_crosswalk
    # Channels we don't emit from AV2 Sensor:
    assert by_type[MapElementType.ROAD_EDGE] == 0
    assert by_type[MapElementType.STOP_SIGN] == 0
    assert by_type[MapElementType.SPEED_BUMP] == 0
    assert by_type[MapElementType.DRIVEWAY] == 0
    assert by_type[MapElementType.WALKWAY] == 0
    assert by_type[MapElementType.TRAFFIC_LIGHT] == 0


def test_lane_center_carries_graph_and_attrs(good_log_processor, good_log_pose):
    """LANE_CENTER elements carry lane_type, is_intersection, predecessors, successors."""
    hd_map = good_log_processor._build_hd_map(good_log_pose)
    centers = [e for e in hd_map.elements if e.type == MapElementType.LANE_CENTER]
    assert len(centers) > 0
    n_with_lane_type = sum(1 for e in centers if "lane_type" in e.attrs)
    n_with_is_intersection = sum(1 for e in centers if "is_intersection" in e.attrs)
    n_with_successors = sum(1 for e in centers if e.successor_ids)
    assert n_with_lane_type == len(centers)
    assert n_with_is_intersection == len(centers)
    assert n_with_successors > 0  # at least some lanes have successors


def test_lane_boundary_carries_paint_attrs(good_log_processor, good_log_pose):
    """LANE_BOUNDARY elements carry paint_color + paint_pattern + paint_subtype_raw."""
    hd_map = good_log_processor._build_hd_map(good_log_pose)
    boundaries = [e for e in hd_map.elements if e.type == MapElementType.LANE_BOUNDARY]
    assert len(boundaries) > 0
    for e in boundaries:
        assert "paint_color" in e.attrs
        assert "paint_pattern" in e.attrs
        assert "paint_subtype_raw" in e.attrs
        # paint_color values stay in the normalised vocabulary
        assert e.attrs["paint_color"] in ("white", "yellow", "blue")
        assert e.attrs["paint_pattern"] in (
            "solid",
            "dashed",
            "double_solid",
            "double_dashed",
            "solid_dashed",
        )


def test_2d_affine_matches_full_3d_transform_when_z_known(
    good_log_processor, good_log_pose
):
    """Confirm our 2D city→ego affine is the XY-restriction of the full SE(3).

    We pick a real lane boundary with finite Z, transform it two ways:

    * ``ego_3d_xy``: full SE(3) on (x, y, z, 1), then take XY of the result.
      This is the AV2 API path (``transform_point_cloud`` does the same).
    * ``ego_2d_xy``: our processor's path -- ``R[:2,:2] @ xy + t[:2]``.

    They should agree to within a few centimetres for the small pitch/roll
    of a typical ego pose. This bounds the "Z-drop drift" we discussed: the
    only difference between the two outputs is ``R[0,2]*z`` and
    ``R[1,2]*z`` which vanish for yaw-only rotations.
    """
    avm = good_log_processor._map
    T_city_from_ego = good_log_pose
    T_ego_from_city = np.linalg.inv(T_city_from_ego).astype(np.float32)

    ls = next(iter(avm.vector_lane_segments.values()))
    xyz_city = np.asarray(ls.left_lane_boundary.xyz, dtype=np.float64)
    assert np.isfinite(xyz_city).all()

    # Full 3D transform (the AV2 API convention)
    homog = np.column_stack([xyz_city, np.ones(len(xyz_city), dtype=np.float64)])
    ego_3d_xy = (homog @ T_ego_from_city.astype(np.float64).T)[:, :2]

    # Our 2D affine
    R_xy = T_ego_from_city[:2, :2].astype(np.float64)
    t_xy = T_ego_from_city[:2, 3].astype(np.float64)
    ego_2d_xy = xyz_city[:, :2] @ R_xy.T + t_xy

    # Drift bound: |R[i,2]| * |z| for i in {0, 1}. With AV2 ego z ~14m and
    # near-yaw-only rotations, this should stay well under ~1 m. We assert
    # 0.5 m as a comfortable upper bound.
    np.testing.assert_allclose(ego_2d_xy, ego_3d_xy, atol=0.5)


def test_2d_affine_exact_match_for_z_zero(good_log_processor, good_log_pose):
    """When source Z = 0, the 2D affine matches the 3D transform exactly.

    This is the algebraic identity that justifies dropping Z: the only
    Z-dependent term in transformed XY is ``R[i,2] * z``, so setting z=0
    eliminates it.
    """
    avm = good_log_processor._map
    T_ego_from_city = np.linalg.inv(good_log_pose).astype(np.float64)

    ls = next(iter(avm.vector_lane_segments.values()))
    xyz_city = np.asarray(ls.left_lane_boundary.xyz, dtype=np.float64)
    xyz_city[:, 2] = 0.0  # force z = 0

    homog = np.column_stack([xyz_city, np.ones(len(xyz_city), dtype=np.float64)])
    ego_3d_xy = (homog @ T_ego_from_city.T)[:, :2]

    R_xy = T_ego_from_city[:2, :2]
    t_xy = T_ego_from_city[:2, 3]
    ego_2d_xy = xyz_city[:, :2] @ R_xy.T + t_xy

    np.testing.assert_allclose(ego_2d_xy, ego_3d_xy, atol=1e-9)
