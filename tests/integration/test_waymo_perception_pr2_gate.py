"""PR 2 gate: Waymo Perception integration smoke + visual rendering.

Runs the first N frames of one real Waymo Perception segment through
``WaymoPerceptionDatasetProcessor`` (with HD-map source path wired),
runs the segment-context aggregators, and asserts shape +
cross-modality invariants. Renders 5 modalities x 3 frames PNGs into
``tests/visual_inspection/pr2/``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from standard_e2e.data_structures import (
    Detection3D,
    HDMapData,
    LidarData,
    TransformedFrameData,
)
from standard_e2e.enums import CameraDirection, Modality
from tests.integration._render_assertions import assert_png_has_real_content

WAYMO_PERCEPTION_VAL = Path(
    "/mnt/bigdisk/datasets/waymo/waymo_open_dataset_v_1_4_3"
    "/individual_files/validation"
)
SEGMENT_TFRECORD = (
    WAYMO_PERCEPTION_VAL
    / "segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RENDER_SCRIPT = REPO_ROOT / "scripts" / "render_frame.py"
VISUAL_OUT = REPO_ROOT / "tests" / "visual_inspection" / "pr2"

NUM_FRAMES = 10
RENDER_FRAMES = 3
RENDER_MODALITIES = ["cameras", "lidar_pc", "hd_map", "detections_3d", "combined"]
EXPECTED_CAMERAS = {
    CameraDirection.FRONT,
    CameraDirection.FRONT_LEFT,
    CameraDirection.FRONT_RIGHT,
    CameraDirection.SIDE_LEFT,
    CameraDirection.SIDE_RIGHT,
}

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def render_module():
    spec = importlib.util.spec_from_file_location("_render_frame", RENDER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load render script from {RENDER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_render_frame"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def processed_segment(tmp_path_factory) -> tuple[Path, list[Path]]:
    if not SEGMENT_TFRECORD.exists():
        pytest.skip(f"Waymo Perception segment not found at {SEGMENT_TFRECORD}")

    import tensorflow as tf

    from standard_e2e.caching.src_datasets.waymo_perception import (
        WaymoPerceptionDatasetProcessor,
    )

    cache_dir = tmp_path_factory.mktemp("waymo_perception_pr2")
    processor = WaymoPerceptionDatasetProcessor(
        common_output_path=str(cache_dir),
        split="validation",
        source_data_path=str(WAYMO_PERCEPTION_VAL),
    )

    written: list[Path] = []
    index_records: list[dict] = []
    raw_iter = tf.data.TFRecordDataset(str(SEGMENT_TFRECORD), compression_type="")
    for raw in raw_iter.take(NUM_FRAMES):
        frame_data, frame_index_data = processor.process_frame(raw)
        target = Path(cache_dir) / frame_data.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        frame_data.to_npz(str(target))
        written.append(target)
        index_records.append(
            {
                "segment_id": frame_data.segment_id,
                "timestamp": frame_data.timestamp,
                "filename": frame_data.filename,
            }
        )

    assert len(written) == NUM_FRAMES

    index_df = (
        pd.DataFrame(index_records).sort_values(by="timestamp").reset_index(drop=True)
    )
    for aggregator in processor.context_aggregators:
        aggregator.process(index_df)

    return cache_dir, written


def test_all_four_modalities_populated(processed_segment):
    _cache_dir, frame_paths = processed_segment
    for path in frame_paths:
        frame = TransformedFrameData.from_npz(str(path))
        present = set(frame.get_present_modality_keys())
        for required in (
            Modality.CAMERAS,
            Modality.LIDAR_PC,
            Modality.DETECTIONS_3D,
            Modality.HD_MAP,
        ):
            assert required in present, f"frame {path.name} missing {required}"


def test_hd_map_payload_is_ego_type_not_world_type(processed_segment):
    """Regression guard: catches accidental persistence of world-frame
    RawSegmentHDMap."""
    from standard_e2e.data_structures import RawSegmentHDMap

    _cache_dir, frame_paths = processed_segment
    for path in frame_paths:
        frame = TransformedFrameData.from_npz(str(path))
        payload = frame.get_modality_data(Modality.HD_MAP)
        assert isinstance(payload, HDMapData)
        assert not isinstance(payload, RawSegmentHDMap)


def test_no_segment_map_npz_files_written_by_default(processed_segment):
    """No ``{segment_id}__map.npz`` artifact lands by default — only
    per-frame ego HDMapData inside the standard
    ``{segment_id}_{frame_id}.npz``."""
    cache_dir, _ = processed_segment
    map_artifacts = list(Path(cache_dir).rglob("*__map.npz"))
    assert map_artifacts == []


def test_frame_loader_has_no_hd_map_specific_branch():
    """Regression guard against re-introducing a load-time HD-map
    code path. ``TransformedFrameData.from_npz`` should remain agnostic
    to the modality."""
    from standard_e2e.data_structures import frame_data as fd_module

    src = Path(fd_module.__file__).read_text()
    forbidden = ("hd_map", "HD_MAP", "HDMap")
    from_npz_idx = src.index("def from_npz")
    next_def_idx = src.find("\ndef ", from_npz_idx + 1)
    body = src[from_npz_idx:next_def_idx]
    for token in forbidden:
        assert token not in body, (
            f"FrameLoader path mentions {token!r}; HD-map-specific code "
            f"is forbidden in TransformedFrameData.from_npz."
        )


def test_cross_modality_coord_frame_consistency(processed_segment):
    """Detection box centers should sit roughly inside the lidar's
    99th-percentile ego radius. If lidar leaks into world frame, the
    detections will not — and vice versa.
    """
    _cache_dir, frame_paths = processed_segment
    for path in frame_paths:
        frame = TransformedFrameData.from_npz(str(path))

        lidar: LidarData = frame.get_modality_data(Modality.LIDAR_PC)
        assert lidar is not None and len(lidar.points) > 0
        lidar_pts = lidar.points[["x", "y", "z"]].to_numpy()
        lidar_radius = np.linalg.norm(lidar_pts, axis=1)
        # Per the breakdown: 99th-percentile radius < 150 m on Waymo TOP.
        assert (
            np.percentile(lidar_radius, 99) < 150.0
        ), "lidar 99th-pct radius > 150 m; coord-frame leak (likely world)"

        detections = frame.get_modality_data(Modality.DETECTIONS_3D)
        # After FutureDetectionsAggregator, DETECTIONS_3D is list[Detection3D];
        # current-frame box centers are at the 0th element of each trajectory.
        if isinstance(detections, list):
            for det in detections:
                assert isinstance(det, Detection3D)
                # Aggregated trajectory may be empty if the agent only
                # appears in future frames; skip those.
                if det.trajectory.length == 0:
                    continue
                # Aggregated box centers are in ego-relative space and
                # should not be at kilometer scale.
                from standard_e2e.enums import TrajectoryComponent as TC

                xy = det.trajectory.get([TC.X, TC.Y])
                assert np.all(
                    np.abs(xy) < 200.0
                ), f"detection center > 200 m from ego on {path.name}: {xy}"


def test_aggregated_detections_carry_all_eight_components(processed_segment):
    """After ``FutureDetectionsAggregator`` runs, each aggregated
    ``Detection3D`` must carry the full 8-component shape that source
    processors produce — TIMESTAMP, X, Y, Z, HEADING, LENGTH, WIDTH,
    HEIGHT — not the {X, Y, HEADING}-only subset. Prevents regression
    of the renderer falling back to ``"x"`` markers because pose-
    invariant size components were silently dropped.
    """
    from standard_e2e.enums import TrajectoryComponent as TC

    full_shape = (
        TC.TIMESTAMP,
        TC.X,
        TC.Y,
        TC.Z,
        TC.HEADING,
        TC.LENGTH,
        TC.WIDTH,
        TC.HEIGHT,
    )
    _cache_dir, frame_paths = processed_segment
    asserted_at_least_one = False
    for path in frame_paths:
        frame = TransformedFrameData.from_npz(str(path))
        detections = frame.get_modality_data(Modality.DETECTIONS_3D)
        if not isinstance(detections, list):
            continue
        for det in detections:
            assert isinstance(det, Detection3D)
            if det.trajectory.length == 0:
                continue
            # All 8 components must be retrievable without KeyError.
            full = det.trajectory.get(list(full_shape))
            assert full.shape[1] == 8
            asserted_at_least_one = True
    assert asserted_at_least_one, (
        "no non-empty aggregated detection was found across the segment; "
        "test is no longer exercising the aggregator output"
    )


def test_render_visual_gate_pngs(processed_segment, render_module):
    _cache_dir, frame_paths = processed_segment
    VISUAL_OUT.mkdir(parents=True, exist_ok=True)
    selected = frame_paths[:RENDER_FRAMES]
    produced: list[Path] = []
    for i, npz_path in enumerate(selected):
        for modality in RENDER_MODALITIES:
            out = VISUAL_OUT / f"frame{i}_{modality}.png"
            render_module.render_frame(
                npz_path=str(npz_path), modality=modality, out_path=str(out)
            )
            assert out.exists()
            assert_png_has_real_content(out)
            produced.append(out)
    assert len(produced) == RENDER_FRAMES * len(RENDER_MODALITIES)


def test_hd_map_content_is_finite_and_in_extent(processed_segment):
    """Audit HD-map polyline / polygon content.

    The existing PR2 gate checks the *type* of the HD-map payload
    (``HDMapData`` rather than ``RawSegmentHDMap``) and the *absence*
    of a ``__map.npz`` artifact, but it does not look at the numbers.
    Three classes of bug pass that gate today:

    * A NaN / Inf vertex slipping into a lane centerline (e.g. via a
      degenerate world->ego transform when ego z is undefined).
    * A masking bug in the crop step that emits ``lane_boundaries`` /
      ``road_edges`` with points outside the ±extent box - the renderer
      would then draw map elements far outside the visible plot.
    * A coord-frame leak that ships world-frame km-scale coordinates
      under the ego-frame ``HDMapData`` type.

    Lane centerlines in Waymo are kept whole when *any* point intersects
    the box (per ``crop_hd_map_ego_relative``), so they legitimately
    extend past ±extent (we observe lanes reaching ~390 m forward of
    ego in the validation segment used here). We therefore enforce two
    different bounds:

    * ``lane_boundaries`` / ``road_edges`` / ``stop_signs``: all points
      must sit inside ±extent within a small float-rounding slack -
      these collections *are* mask-filtered to the inside set.
    * ``lanes`` and polygons (crosswalks etc.): only require at least
      one in-extent vertex (the cropping criterion) plus a coarse
      sanity bound (no point > 1 km from ego) to catch full world-frame
      leaks.
    """
    # Mirrors ``WaymoPerceptionDatasetProcessor.DEFAULT_HD_MAP_CROP_EXTENT_M``
    # without forcing a top-level Waymo proto import (other integration tests
    # in this file follow the same lazy-import convention).
    extent = 75.0
    # Float slack: world->ego rotation can place a vertex at ±extent
    # plus rounding error from the rigid transform.
    slack = 1e-2
    # Coarse outer bound: any point > 1 km from ego in either axis is
    # implausible at the 75 m crop level, even for forward-stretching
    # lane centerlines.
    coarse_bound = 1000.0

    _cache_dir, frame_paths = processed_segment
    asserted_at_least_one_lane = False
    for path in frame_paths:
        frame = TransformedFrameData.from_npz(str(path))
        hd = frame.get_modality_data(Modality.HD_MAP)
        assert isinstance(hd, HDMapData)

        # 1) Mask-filtered collections: every point must be inside ±extent.
        for label, items, attr in (
            ("lane_boundaries", hd.lane_boundaries, "polyline"),
            ("road_edges", hd.road_edges, "polyline"),
        ):
            for idx, item in enumerate(items):
                pts = getattr(item, attr)
                assert pts.ndim == 2 and pts.shape[1] == 3, (
                    f"{path.name}: {label}[{idx}] {attr} has unexpected shape "
                    f"{pts.shape}"
                )
                assert np.isfinite(pts).all(), (
                    f"{path.name}: {label}[{idx}] {attr} contains NaN/Inf - "
                    "indicates a degenerate world->ego transform leaked into "
                    "the cropped HD map."
                )
                assert (np.abs(pts[:, 0]) <= extent + slack).all(), (
                    f"{path.name}: {label}[{idx}] x out of crop: max |x|="
                    f"{float(np.abs(pts[:, 0]).max()):.2f} m exceeds "
                    f"{extent} m + {slack} slack."
                )
                assert (np.abs(pts[:, 1]) <= extent + slack).all(), (
                    f"{path.name}: {label}[{idx}] y out of crop: max |y|="
                    f"{float(np.abs(pts[:, 1]).max()):.2f} m exceeds "
                    f"{extent} m + {slack} slack."
                )

        # 2) Stop signs: single position, must be inside the box.
        for idx, sign in enumerate(hd.stop_signs):
            pos = sign.position
            assert np.isfinite(pos).all(), (
                f"{path.name}: stop_signs[{idx}].position contains NaN/Inf"
            )
            assert abs(float(pos[0])) <= extent + slack, (
                f"{path.name}: stop_signs[{idx}] x={pos[0]} outside ±{extent} m"
            )
            assert abs(float(pos[1])) <= extent + slack, (
                f"{path.name}: stop_signs[{idx}] y={pos[1]} outside ±{extent} m"
            )

        # 3) Lanes + polygons: at least one in-extent point and no
        # km-scale leaks. Centerlines extend past the crop legitimately
        # so we don't lock them to ±extent.
        polygon_collections = (
            ("crosswalks", hd.crosswalks, "polygon"),
            ("speed_bumps", hd.speed_bumps, "polygon"),
            ("drivable_areas", hd.drivable_areas, "polygon"),
            ("driveways", hd.driveways, "polygon"),
        )
        for label, items, attr in polygon_collections:
            for idx, item in enumerate(items):
                pts = getattr(item, attr)
                assert np.isfinite(pts).all(), (
                    f"{path.name}: {label}[{idx}] {attr} contains NaN/Inf"
                )
                in_extent = (np.abs(pts[:, 0]) <= extent + slack) & (
                    np.abs(pts[:, 1]) <= extent + slack
                )
                assert in_extent.any(), (
                    f"{path.name}: {label}[{idx}] kept after crop but no point "
                    f"sits inside ±{extent} m - cropping criterion broken."
                )
                assert (np.abs(pts[:, :2]) <= coarse_bound).all(), (
                    f"{path.name}: {label}[{idx}] has point > {coarse_bound} m "
                    f"from ego (max |xy|={float(np.abs(pts[:, :2]).max()):.1f} m); "
                    "indicates world-frame coordinates leaked into HDMapData."
                )

        for idx, lane in enumerate(hd.lanes):
            pts = lane.centerline
            assert pts.ndim == 2 and pts.shape[1] == 3, (
                f"{path.name}: lane[{idx}].centerline has unexpected shape "
                f"{pts.shape}"
            )
            assert np.isfinite(pts).all(), (
                f"{path.name}: lane[{idx}].centerline contains NaN/Inf - "
                "indicates a degenerate world->ego transform."
            )
            in_extent = (np.abs(pts[:, 0]) <= extent + slack) & (
                np.abs(pts[:, 1]) <= extent + slack
            )
            assert in_extent.any(), (
                f"{path.name}: lane[{idx}] centerline has no point inside "
                f"±{extent} m; cropping criterion broken."
            )
            assert (np.abs(pts[:, :2]) <= coarse_bound).all(), (
                f"{path.name}: lane[{idx}] centerline has point > "
                f"{coarse_bound} m from ego "
                f"(max |xy|={float(np.abs(pts[:, :2]).max()):.1f} m); "
                "indicates world-frame coordinates leaked into HDMapData."
            )
            asserted_at_least_one_lane = True

    assert asserted_at_least_one_lane, (
        "no lanes were exercised across the segment; the test is no "
        "longer reaching the lane-content assertion path."
    )


def test_processed_frame_has_five_cameras(processed_segment):
    _cache_dir, frame_paths = processed_segment
    for path in frame_paths:
        frame = TransformedFrameData.from_npz(str(path))
        # Pano-stitched modality lives at Modality.CAMERAS for Waymo
        # Perception (per processor's PanoImageAdapter default), so the
        # individual 5 cameras are not exposed under that key. We assert
        # the pano produced something image-shaped, and the lidar /
        # detections / hd_map slots above are the multi-modality gate.
        pano = frame.get_modality_data(Modality.CAMERAS)
        assert pano is not None
        assert pano.ndim == 3 and pano.shape[2] == 3
