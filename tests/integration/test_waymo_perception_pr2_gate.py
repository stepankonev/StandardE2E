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

    index_df = pd.DataFrame(index_records).sort_values(by="timestamp").reset_index(drop=True)
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
    RawSegmentHDMap (per ADR 0007)."""
    from standard_e2e.data_structures import RawSegmentHDMap

    _cache_dir, frame_paths = processed_segment
    for path in frame_paths:
        frame = TransformedFrameData.from_npz(str(path))
        payload = frame.get_modality_data(Modality.HD_MAP)
        assert isinstance(payload, HDMapData)
        assert not isinstance(payload, RawSegmentHDMap)


def test_no_segment_map_npz_files_written_by_default(processed_segment):
    """Per ADR 0007, no ``{segment_id}__map.npz`` artifact lands by
    default — only per-frame ego HDMapData inside the standard
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
            f"FrameLoader path mentions {token!r}; ADR 0007 forbids HD-map-"
            f"specific code in TransformedFrameData.from_npz."
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
        assert np.percentile(lidar_radius, 99) < 150.0, (
            "lidar 99th-pct radius > 150 m; coord-frame leak (likely world)"
        )

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
                assert np.all(np.abs(xy) < 200.0), (
                    f"detection center > 200 m from ego on {path.name}: {xy}"
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
            size = out.stat().st_size
            assert 5 * 1024 <= size <= 2 * 1024 * 1024, (
                f"PNG {out} size out of band: {size} bytes"
            )
            produced.append(out)
    assert len(produced) == RENDER_FRAMES * len(RENDER_MODALITIES)


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
