"""Multi-segment scale + byte-determinism gate for Waymo Perception.

Runs the full Perception pipeline (adapters + every default aggregator
including the segment-scoped HD-map crop) over more than one segment and
verifies the per-frame npz outputs are byte-identical across two
independent passes over the same input.

Two failure modes this catches that the single-segment PR2 gate cannot:

* **Per-segment lifecycle leaks** — e.g. a segment-scoped object such as
  ``RawSegmentHDMap`` accidentally retained on the aggregator instance
  and consulted on the next segment. Single-segment
  tests cannot see this; two segments will diverge from a single-
  segment baseline if such retention exists.

* **Non-determinism leaking from dict ordering / unsorted dataframes.**
  Re-running the same input must produce the same bytes; if a sort
  is missing somewhere (groupby in an aggregator, ``set`` iteration
  in an adapter), one of the two passes ends up writing differently
  ordered modality arrays into the npz.

The two-pass byte equality assertion is morally a regression net for
*any future change* that introduces nondeterminism, not just the
hypothetical lifecycle leak.
"""

from __future__ import annotations

import filecmp
from pathlib import Path

import pandas as pd
import pytest

from standard_e2e.data_structures import HDMapData, TransformedFrameData
from standard_e2e.enums import Modality

WAYMO_PERCEPTION_VAL = Path(
    "/mnt/bigdisk/datasets/waymo/waymo_open_dataset_v_1_4_3"
    "/individual_files/validation"
)
NUM_SEGMENTS = 10
NUM_FRAMES_PER_SEGMENT = 3
DETERMINISM_FRAMES = 4

pytestmark = pytest.mark.integration


def _process_segments(
    cache_dir: Path,
    segment_paths: list[Path],
    num_frames: int,
) -> list[Path]:
    """Run the default Waymo Perception pipeline over multiple segments.

    Returns the list of written ``.npz`` paths in the order the
    aggregators ran (segment-major, frame-minor by timestamp).
    """
    import tensorflow as tf

    from standard_e2e.caching.src_datasets.waymo_perception import (
        WaymoPerceptionDatasetProcessor,
    )

    processor = WaymoPerceptionDatasetProcessor(
        common_output_path=str(cache_dir),
        split="validation",
        source_data_path=str(WAYMO_PERCEPTION_VAL),
    )
    written: list[Path] = []
    index_records: list[dict] = []
    for segment_path in segment_paths:
        raw_iter = tf.data.TFRecordDataset(str(segment_path), compression_type="")
        for raw in raw_iter.take(num_frames):
            frame_data, _ = processor.process_frame(raw)
            target = cache_dir / frame_data.filename
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

    index_df = (
        pd.DataFrame(index_records)
        .sort_values(by=["segment_id", "timestamp"])
        .reset_index(drop=True)
    )
    for aggregator in processor.context_aggregators:
        aggregator.process(index_df)
    return written


@pytest.fixture(scope="module")
def segment_paths() -> list[Path]:
    if not WAYMO_PERCEPTION_VAL.exists():
        pytest.skip(f"Waymo Perception validation dir missing: {WAYMO_PERCEPTION_VAL}")
    paths = sorted(WAYMO_PERCEPTION_VAL.glob("*.tfrecord"))[:NUM_SEGMENTS]
    if len(paths) < NUM_SEGMENTS:
        pytest.skip(
            f"Need >= {NUM_SEGMENTS} Waymo Perception segments, "
            f"got {len(paths)} under {WAYMO_PERCEPTION_VAL}"
        )
    return paths


@pytest.fixture(scope="module")
def multi_segment_run(
    tmp_path_factory, segment_paths: list[Path]
) -> tuple[Path, list[Path]]:
    cache_dir = tmp_path_factory.mktemp("perception_scale")
    written = _process_segments(cache_dir, segment_paths, NUM_FRAMES_PER_SEGMENT)
    return cache_dir, written


def test_multi_segment_per_segment_hd_map_is_ego_typed(
    multi_segment_run: tuple[Path, list[Path]],
):
    """Across all segments, every frame must have ``Modality.HD_MAP``
    typed as ``HDMapData`` (the ego-frame variant). A leaked
    segment-scoped ``RawSegmentHDMap`` from segment 0 would surface here
    if it accidentally got persisted to a later segment's frame."""
    _cache_dir, written = multi_segment_run
    assert (
        len(written) == NUM_SEGMENTS * NUM_FRAMES_PER_SEGMENT
    ), f"expected {NUM_SEGMENTS * NUM_FRAMES_PER_SEGMENT} npz files, got {len(written)}"

    for path in written:
        frame = TransformedFrameData.from_npz(str(path))
        present = frame.get_present_modality_keys()
        assert Modality.HD_MAP in present, f"frame {path.name} missing HD_MAP"
        payload = frame.get_modality_data(Modality.HD_MAP)
        assert type(payload) is HDMapData, (
            f"frame {path.name} HD_MAP type drift: {type(payload)} (expected "
            "exactly HDMapData; a RawSegmentHDMap subclass would mean a "
            "segment-scoped world-frame map leaked into a per-frame slot)."
        )


def test_multi_segment_unique_segment_ids_observed(
    multi_segment_run: tuple[Path, list[Path]],
):
    """Sanity: the scaled run actually exercised distinct segments. If
    the iteration logic regresses to processing the first tfrecord only,
    this fires before the more expensive determinism check."""
    _cache_dir, written = multi_segment_run
    seen_segment_ids = {
        TransformedFrameData.from_npz(str(p)).segment_id for p in written
    }
    assert len(seen_segment_ids) == NUM_SEGMENTS, (
        f"expected {NUM_SEGMENTS} distinct segments, got {len(seen_segment_ids)}: "
        f"{sorted(seen_segment_ids)}"
    )


def test_byte_identical_across_two_passes(tmp_path_factory, segment_paths: list[Path]):
    """Two passes over the same shard produce byte-identical npz files.

    Catches non-determinism from missing sorts (groupby + iteration
    order), set iteration, threading, or random seeds in any adapter
    or aggregator. We restrict to a single shard to keep this < 30 s -
    the per-segment lifecycle test above already covers multi-segment
    interactions; this test isolates the byte-equality assertion.
    """
    shard = segment_paths[0]
    pass_a = tmp_path_factory.mktemp("perception_det_a")
    pass_b = tmp_path_factory.mktemp("perception_det_b")
    written_a = _process_segments(pass_a, [shard], DETERMINISM_FRAMES)
    written_b = _process_segments(pass_b, [shard], DETERMINISM_FRAMES)
    assert len(written_a) == len(written_b) == DETERMINISM_FRAMES

    relative_a = sorted(p.relative_to(pass_a) for p in written_a)
    relative_b = sorted(p.relative_to(pass_b) for p in written_b)
    assert (
        relative_a == relative_b
    ), f"per-pass relative paths diverge: {relative_a} vs {relative_b}"

    diverged: list[str] = []
    for rel in relative_a:
        a_path = pass_a / rel
        b_path = pass_b / rel
        # ``filecmp.cmp(shallow=False)`` reads bytes, ignores stat metadata.
        if not filecmp.cmp(str(a_path), str(b_path), shallow=False):
            diverged.append(str(rel))

    assert not diverged, (
        f"{len(diverged)} npz file(s) diverged byte-for-byte between two passes "
        f"over the same shard - non-determinism in the cache pipeline. "
        f"Files: {diverged[:5]}{'...' if len(diverged) > 5 else ''}"
    )
