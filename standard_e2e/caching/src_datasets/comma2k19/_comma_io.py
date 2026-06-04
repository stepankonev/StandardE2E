"""On-disk access helpers for comma2k19.

comma2k19 ships as ten ``Chunk_*.zip`` archives. Each archive holds many
~1-minute *segments* laid out as::

    Chunk_<c>/<dongle_id>|<route_datetime>/<segment_number>/
        video.hevc                      # 20 Hz road camera, HEVC
        global_pose/{frame_positions, frame_orientations,
                     frame_velocities, frame_times, ...}
        processed_log/...               # CAN / IMU / GNSS (unused here)

This module discovers segments and reads their per-segment files, transparently
supporting **both** layouts:

* **extracted** -- a directory tree containing ``<seg>/global_pose/...`` dirs
  (preferred when present; matches the WayveScenes "unzip first" convention),
* **archived**  -- the distributed ``Chunk_*.zip`` files read in place, so the
  pipeline runs against the raw download with no manual extraction. The small
  ``global_pose`` arrays are read straight from the zip into memory; only
  ``video.hevc`` (which OpenCV must open by path) is materialised to a scratch
  file by the processor.

:class:`SegmentRef` is a small, picklable locator carried through the
multiprocessing pool as part of each per-frame task.
"""

from __future__ import annotations

import io
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    """Locator for one comma2k19 segment.

    Attributes:
        kind: ``"dir"`` for an extracted segment directory, ``"zip"`` for a
            segment still inside a ``Chunk_*.zip`` archive.
        container: the segment directory (``dir``) or the zip path (``zip``).
        prefix: the in-archive segment path (``zip`` only; ``""`` for ``dir``).
        segment_id: globally-unique, filename-safe id
            (``<dongle>_<route_datetime>_<segment_number>``).
        route: the source ``<dongle_id>|<route_datetime>`` directory name.
        segment: the segment number (e.g. ``"8"``).
    """

    kind: str
    container: str
    prefix: str
    segment_id: str
    route: str
    segment: str


def _segment_id(route: str, segment: str) -> str:
    # '|' is legal but awkward in filenames; flatten to '_'. The route
    # datetime and dongle id contain only hyphens/hex, so the result stays
    # unambiguous and globally unique.
    return f"{route.replace('|', '_')}_{segment}"


def _ref_from_segment_path(
    parts: list[str], kind: str, container: str, prefix: str
) -> SegmentRef:
    route, segment = parts[-2], parts[-1]
    return SegmentRef(
        kind=kind,
        container=container,
        prefix=prefix,
        segment_id=_segment_id(route, segment),
        route=route,
        segment=segment,
    )


def _discover_dir_segments(root: Path) -> list[SegmentRef]:
    segments: list[SegmentRef] = []
    for marker in root.rglob(_SEGMENT_MARKER):
        seg_dir = marker.parent.parent
        if not (seg_dir / _VIDEO_NAME).is_file():
            continue
        parts = list(seg_dir.parts)
        segments.append(_ref_from_segment_path(parts, "dir", str(seg_dir), ""))
    return sorted(segments, key=lambda r: r.segment_id)


def _discover_zip_segments(zip_paths: list[Path]) -> list[SegmentRef]:
    suffix = "/" + _SEGMENT_MARKER
    segments: list[SegmentRef] = []
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
        members = set(names)
        for name in names:
            if not name.endswith(suffix):
                continue
            prefix = name[: -len(suffix)]
            if f"{prefix}/{_VIDEO_NAME}" not in members:
                continue
            parts = prefix.split("/")
            segments.append(_ref_from_segment_path(parts, "zip", str(zip_path), prefix))
    return sorted(segments, key=lambda r: r.segment_id)


def discover_segments(input_path: str) -> list[SegmentRef]:
    """Find every comma2k19 segment under ``input_path``.

    Prefers extracted segment directories; falls back to reading
    ``Chunk_*.zip`` archives in place. Raises ``FileNotFoundError`` with a
    descriptive message when neither layout is found.
    """
    root = Path(input_path)
    if not root.exists():
        raise FileNotFoundError(f"comma2k19 input path not found: {root}")

    dir_segments = _discover_dir_segments(root)
    if dir_segments:
        return dir_segments

    zip_paths = sorted(set(root.glob("*.zip")) | set(root.glob("Chunk_*/*.zip")))
    zip_segments = _discover_zip_segments(zip_paths)
    if zip_segments:
        return zip_segments

    raise FileNotFoundError(
        f"No comma2k19 segments found under {root}. Expected either extracted "
        f"segment directories (<seg>/{_SEGMENT_MARKER}) or Chunk_*.zip archives."
    )


def _open_member(ref: SegmentRef, relpath: str, zip_handle: Optional[zipfile.ZipFile]):
    """Return raw bytes for ``<segment>/<relpath>`` from a dir or zip ref."""
    if ref.kind == "dir":
        with open(os.path.join(ref.container, relpath), "rb") as fh:
            return fh.read()
    zf = zip_handle if zip_handle is not None else zipfile.ZipFile(ref.container)
    try:
        return zf.read(f"{ref.prefix}/{relpath}")
    finally:
        if zip_handle is None:
            zf.close()


def read_frame_times(
    ref: SegmentRef, zip_handle: Optional[zipfile.ZipFile] = None
) -> np.ndarray:
    """Read just ``global_pose/frame_times`` (cheap; used to count frames)."""
    raw = _open_member(ref, "global_pose/frame_times", zip_handle)
    return np.asarray(np.load(io.BytesIO(raw)))


def load_pose_arrays(
    ref: SegmentRef, zip_handle: Optional[zipfile.ZipFile] = None
) -> dict[str, np.ndarray]:
    """Load the ``global_pose`` arrays needed to build per-frame poses."""
    arrays: dict[str, np.ndarray] = {}
    for name in _POSE_FILES:
        raw = _open_member(ref, f"global_pose/{name}", zip_handle)
        arrays[name] = np.load(io.BytesIO(raw))
    return arrays


def materialize_video(
    ref: SegmentRef,
    scratch_dir: str,
    zip_handle: Optional[zipfile.ZipFile] = None,
) -> tuple[str, bool]:
    """Return a filesystem path to the segment's ``video.hevc``.

    For extracted segments this is the file in place. For archived segments
    the video is copied out to ``scratch_dir`` (OpenCV cannot open a zip
    member by path) and the second tuple element is ``True`` to signal the
    caller owns (and should later delete) the scratch copy.

    Returns:
        ``(video_path, is_scratch_copy)``.
    """
    if ref.kind == "dir":
        return os.path.join(ref.container, _VIDEO_NAME), False

    os.makedirs(scratch_dir, exist_ok=True)
    dst = os.path.join(scratch_dir, f"{ref.segment_id}.hevc")
    zf = zip_handle if zip_handle is not None else zipfile.ZipFile(ref.container)
    try:
        with zf.open(f"{ref.prefix}/{_VIDEO_NAME}") as src, open(dst, "wb") as out:
            shutil.copyfileobj(src, out, length=1 << 20)
    finally:
        if zip_handle is None:
            zf.close()
    return dst, True
