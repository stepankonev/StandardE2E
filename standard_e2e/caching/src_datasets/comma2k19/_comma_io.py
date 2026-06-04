"""On-disk access helpers for comma2k19 (extracted segments).

comma2k19 is distributed as ten ``Chunk_*.zip`` archives. **Extract them before
processing** (the same convention as WayveScenes101). After extraction each
~1-minute *segment* is a directory::

    <root>/.../<dongle_id>|<route_datetime>/<segment_number>/
        video.hevc                      # 20 Hz road camera, HEVC
        global_pose/{frame_positions, frame_orientations,
                     frame_velocities, frame_times, ...}
        processed_log/...               # CAN / IMU / GNSS (unused here)

This module discovers those segment directories and reads their per-segment
files. :class:`SegmentRef` is a small, picklable locator carried through the
multiprocessing pool as part of each per-frame task.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_POSE_FILES = (
    "frame_positions",
    "frame_orientations",
    "frame_velocities",
    "frame_times",
)
# Relative path that marks a segment directory (under the segment root).
_SEGMENT_MARKER = "global_pose/frame_positions"
_VIDEO_NAME = "video.hevc"


@dataclass(frozen=True)
class SegmentRef:
    """Locator for one extracted comma2k19 segment.

    Attributes:
        path: the segment directory (holds ``video.hevc`` and ``global_pose/``).
        segment_id: globally-unique, filename-safe id
            (``<dongle>_<route_datetime>_<segment_number>``).
        route: the source ``<dongle_id>|<route_datetime>`` directory name.
        segment: the segment number (e.g. ``"8"``).
    """

    path: str
    segment_id: str
    route: str
    segment: str

    @property
    def video_path(self) -> str:
        return os.path.join(self.path, _VIDEO_NAME)


def _segment_id(route: str, segment: str) -> str:
    # '|' is legal but awkward in filenames; flatten to '_'. The route
    # datetime and dongle id contain only hyphens/hex, so the result stays
    # unambiguous and globally unique.
    return f"{route.replace('|', '_')}_{segment}"


def discover_segments(input_path: str) -> list[SegmentRef]:
    """Find every extracted comma2k19 segment under ``input_path``.

    Raises ``FileNotFoundError`` (pointing at the required unzip step) when no
    extracted segment directories are found.
    """
    root = Path(input_path)
    if not root.exists():
        raise FileNotFoundError(f"comma2k19 input path not found: {root}")

    segments: list[SegmentRef] = []
    for marker in root.rglob(_SEGMENT_MARKER):
        seg_dir = marker.parent.parent
        if not (seg_dir / _VIDEO_NAME).is_file():
            continue
        route, segment = seg_dir.parts[-2], seg_dir.parts[-1]
        segments.append(
            SegmentRef(str(seg_dir), _segment_id(route, segment), route, segment)
        )
    if not segments:
        raise FileNotFoundError(
            f"No extracted comma2k19 segments under {root} (expected "
            f"<segment>/{_SEGMENT_MARKER}). Extract the distributed "
            f"Chunk_*.zip archives first."
        )
    return sorted(segments, key=lambda r: r.segment_id)


def read_frame_times(ref: SegmentRef) -> np.ndarray:
    """Read just ``global_pose/frame_times`` (cheap; used to count frames)."""
    return np.asarray(np.load(os.path.join(ref.path, "global_pose", "frame_times")))


def load_pose_arrays(ref: SegmentRef) -> dict[str, np.ndarray]:
    """Load the ``global_pose`` arrays needed to build per-frame poses."""
    gp = os.path.join(ref.path, "global_pose")
    arrays: dict[str, np.ndarray] = {}
    for name in _POSE_FILES:
        arrays[name] = np.load(os.path.join(gp, name))
    return arrays
