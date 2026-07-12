"""On-disk access helpers for the NATIX Multi-Camera Driving Dataset.

The dataset (Hugging Face ``natix-network-org/natix-multi-camera-driving-dataset``;
the full release is pulled from a Cloudflare R2 bucket, the repository itself
carries a ``dataset-sample/``) needs **no extraction** -- it is downloaded as a
plain directory tree::

    <root>/<Country>/[<State>/]<trip-piece>/
        fixed_metadata.json                  # per-trip camera calibration
        trip_insight.json                    # per-trip metadata (IANA timezone)
        <HH-MM-SS>/                          # ~1-minute clip
            FRONT_FOLDER/FRONT_<date>_<time>.{mp4,csv,mcap}
            REAR_FOLDER/...                  # + LEFT/RIGHT (4-cam trips) or
            ...                              #   *_REPEATER / *_PILLAR (6-cam)

A *trip piece* (``<trip-id>_<n>`` with an optional ``_seqNN`` suffix) contains
continuous minutes and is the StandardE2E **segment**; it is discovered by its
``fixed_metadata.json`` marker. Camera folders are discovered by their
``*_FOLDER`` suffix (``mapping.txt`` just restates those names and is not
needed). Each camera's ``.csv`` carries one row per video frame -- an
ISO-8601 **local-time** timestamp (localized here via ``trip_insight.json``'s
IANA timezone into epoch seconds) plus, on rows where a GPS fix landed, the
fix itself; other rows hold the literal string ``na``. The per-camera
``.mcap`` duplicates the CSV content and is not read.

:func:`build_trip_frame_table` builds the per-trip frame table that defines
the StandardE2E frames: **one frame per front-camera GPS fix** (~1-10 Hz; the
frames between fixes have no pose and are not emitted), clips concatenated in
chronological order (clip order comes from the date+time in the front CSV
filename, so a trip crossing midnight still sorts correctly). Because per-clip
camera streams differ in frame rate and duration, every other camera is
matched to the front fix **by timestamp** -- the nearest row of that camera's
CSV -- and its matched ``frame_number`` (1-based, synced to its ``.mp4``) is
recorded per frame; ``-1`` marks a camera with no usable match (folder / file
missing for the clip, or nearest row further than ``max_camera_sync_s``).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_FIXED_METADATA_NAME = "fixed_metadata.json"
_TRIP_INSIGHT_NAME = "trip_insight.json"
_CAMERA_FOLDER_SUFFIX = "_FOLDER"
_CLIP_DIR_PATTERN = re.compile(r"^\d{2}-\d{2}-\d{2}$")
# FRONT_2025-11-14_08-49-42.csv -> the clip's wall-clock start.
_FILE_DATETIME_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})")

FRONT_CAMERA_KEY = "FRONT"

# The GPS columns consumed from a camera CSV (missing values are the literal
# string "na"); velocity components back-fill a missing speed.
_CSV_NA_VALUES = ["na"]


@dataclass(frozen=True)
class CameraClipFiles:
    """One camera's files inside one clip directory."""

    camera_key: str  # folder prefix, e.g. "FRONT", "LEFT_REPEATER"
    video_path: str
    csv_path: str


@dataclass(frozen=True)
class ClipRef:
    """One ~1-minute clip directory inside a trip piece."""

    path: str
    clip_name: str  # the HH-MM-SS directory name
    start: datetime  # naive local wall-clock, parsed from the front filename
    cameras: dict[str, CameraClipFiles]  # keyed by camera folder prefix


@dataclass(frozen=True)
class TripRef:
    """Locator for one trip piece (one StandardE2E segment).

    Attributes:
        path: the trip-piece directory (holds ``fixed_metadata.json``).
        trip_name: the directory name (``<trip-id>_<n>[_seqNN]``) -- globally
            unique (UUID-based) and filename-safe; used as the segment id.
    """

    path: str
    trip_name: str

    @property
    def fixed_metadata_path(self) -> str:
        return os.path.join(self.path, _FIXED_METADATA_NAME)

    @property
    def trip_insight_path(self) -> str:
        return os.path.join(self.path, _TRIP_INSIGHT_NAME)


def discover_trips(input_path: str) -> list[TripRef]:
    """Find every trip piece under ``input_path`` (any nesting depth).

    ``input_path`` may point at the dataset root, a country/state directory,
    or any parent of them (e.g. the Hugging Face snapshot root, whose trips
    live under ``dataset-sample/``). Raises ``FileNotFoundError`` when no
    ``fixed_metadata.json`` markers are found.
    """
    root = Path(input_path)
    if not root.exists():
        raise FileNotFoundError(f"NATIX input path not found: {root}")

    trips: list[TripRef] = []
    for marker in root.rglob(_FIXED_METADATA_NAME):
        trip_dir = marker.parent
        trips.append(TripRef(path=str(trip_dir), trip_name=trip_dir.name))
    if not trips:
        raise FileNotFoundError(
            f"No NATIX trips under {root} (expected directories containing "
            f"{_FIXED_METADATA_NAME}). Point --input_path at the downloaded "
            "dataset root (or its dataset-sample/ folder)."
        )
    return sorted(trips, key=lambda ref: ref.trip_name)


def _camera_files_for_clip(clip_dir: Path) -> dict[str, CameraClipFiles]:
    cameras: dict[str, CameraClipFiles] = {}
    for camera_dir in sorted(clip_dir.iterdir()):
        if not (
            camera_dir.is_dir() and camera_dir.name.endswith(_CAMERA_FOLDER_SUFFIX)
        ):
            continue
        camera_key = camera_dir.name.removesuffix(_CAMERA_FOLDER_SUFFIX)
        videos = sorted(camera_dir.glob(f"{camera_key}_*.mp4"))
        csvs = sorted(camera_dir.glob(f"{camera_key}_*.csv"))
        if not videos or not csvs:
            logging.warning(
                "NATIX: camera folder %s lacks its .mp4/.csv pair; skipping.",
                camera_dir,
            )
            continue
        if len(videos) > 1 or len(csvs) > 1:
            logging.warning(
                "NATIX: multiple .mp4/.csv files in %s; using the first.",
                camera_dir,
            )
        cameras[camera_key] = CameraClipFiles(
            camera_key=camera_key,
            video_path=str(videos[0]),
            csv_path=str(csvs[0]),
        )
    return cameras


def _clip_start_from_filename(front_csv_path: str) -> datetime | None:
    match = _FILE_DATETIME_PATTERN.search(os.path.basename(front_csv_path))
    if match is None:
        return None
    return datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y-%m-%d_%H-%M-%S")


def discover_clips(trip: TripRef) -> list[ClipRef]:
    """List a trip piece's clip directories in chronological order.

    Clips without a usable front camera (its ``.mp4`` + ``.csv``) are skipped
    with a warning: the front stream is the GPS-alignment anchor (the dataset
    aligns GPS to the front footage), so frames in such a clip cannot be
    anchored.
    """
    clips: list[ClipRef] = []
    for clip_dir in sorted(Path(trip.path).iterdir()):
        if not (clip_dir.is_dir() and _CLIP_DIR_PATTERN.match(clip_dir.name)):
            continue
        cameras = _camera_files_for_clip(clip_dir)
        front = cameras.get(FRONT_CAMERA_KEY)
        if front is None:
            logging.warning(
                "NATIX: clip %s has no usable front camera; skipping the clip.",
                clip_dir,
            )
            continue
        start = _clip_start_from_filename(front.csv_path)
        if start is None:
            logging.warning(
                "NATIX: cannot parse the clip start from %s; skipping the clip.",
                front.csv_path,
            )
            continue
        clips.append(
            ClipRef(
                path=str(clip_dir),
                clip_name=clip_dir.name,
                start=start,
                cameras=cameras,
            )
        )
    return sorted(clips, key=lambda clip: clip.start)


def read_fixed_metadata(trip: TripRef) -> dict:
    """Load the trip's ``fixed_metadata.json`` (camera calibration)."""
    with open(trip.fixed_metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(
            f"fixed_metadata must be a JSON object: {trip.fixed_metadata_path}"
        )
    return metadata


def read_trip_insight(trip: TripRef) -> dict:
    """Load the trip's ``trip_insight.json`` (``{}`` when unreadable)."""
    try:
        with open(trip.trip_insight_path, "r", encoding="utf-8") as handle:
            insight = json.load(handle)
    except (OSError, json.JSONDecodeError):
        logging.warning("NATIX: unreadable %s.", trip.trip_insight_path)
        return {}
    return insight if isinstance(insight, dict) else {}


def trip_timezone(insight: dict) -> str | None:
    """The trip's IANA timezone (None if absent -- timestamps then read as UTC).

    CSV timestamps are local wall-clock times with no offset; this is the
    timezone they must be localized with (verified against
    ``trip_insight.json``'s ``startEpochMs``).
    """
    timezone = insight.get("timezone")
    return str(timezone) if timezone else None


def _first_minute(insight: dict) -> dict:
    minutes = insight.get("minutes")
    if not isinstance(minutes, list) or not minutes:
        return {}
    first = minutes[0]
    if not isinstance(first, dict):
        return {}
    # Tolerate both the flat per-minute dict and the README's
    # {"YYYY-MM-DD_HH-MM": {...}} single-key wrapping.
    if len(first) == 1 and isinstance(next(iter(first.values())), dict):
        return next(iter(first.values()))
    return first


def trip_location(insight: dict) -> tuple[str, str]:
    """``(country, region)`` from the trip's first per-minute record.

    The directory layout (``Country/[State]/<trip>``) is not parsed for this:
    the depth varies by country and by download root, while the insight file
    is self-contained per trip. Empty strings when unavailable.
    """
    first = _first_minute(insight)
    return str(first.get("country") or ""), str(first.get("region") or "")


def read_camera_csv(csv_path: str) -> pd.DataFrame:
    """Read a per-camera GPS-metadata CSV (``na`` -> NaN), minimally cleaned:
    rows with an unparseable timestamp or frame number are dropped (with a
    warning), and rows are sorted by timestamp."""
    df = pd.read_csv(csv_path, na_values=_CSV_NA_VALUES)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["frame_number"] = pd.to_numeric(df["frame_number"], errors="coerce")
    bad_rows = df["timestamp"].isna() | df["frame_number"].isna()
    if bad_rows.any():
        logging.warning(
            "NATIX: dropping %d malformed row(s) from %s.",
            int(bad_rows.sum()),
            csv_path,
        )
        df = df[~bad_rows]
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def localize_timestamps(timestamps: pd.Series, timezone_name: str | None) -> np.ndarray:
    """Naive local-time stamps -> ``(N,)`` float64 epoch seconds.

    DST edge cases are resolved deterministically (ambiguous fall-back times
    read as standard time, nonexistent spring-forward times shifted forward);
    the strict-monotonicity guard in :func:`build_trip_frame_table` drops any
    frame a transition would fold backwards.
    """
    localized = timestamps.dt.tz_localize(
        timezone_name or "UTC", ambiguous=False, nonexistent="shift_forward"
    )
    # Anchor-subtraction keeps this correct whatever datetime64 resolution
    # pandas inferred for the parsed strings (ms for these CSVs, not ns).
    seconds = (localized - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)
    return np.asarray(seconds.to_numpy(), dtype=np.float64)


@dataclass
class TripFrameTable:
    """Per-trip table with one row per emitted frame (one front GPS fix).

    ``frame_numbers`` maps each requested camera key to the 1-based video
    frame matched to every emitted frame (``-1`` = no usable match).
    ``clip_indices`` indexes into the ``discover_clips`` list.
    """

    timestamps: np.ndarray  # (N,) float64, epoch seconds
    latitudes: np.ndarray  # (N,) float64, degrees
    longitudes: np.ndarray  # (N,) float64, degrees
    speeds_mps: np.ndarray  # (N,) float64, NaN when unavailable
    headings_deg: np.ndarray  # (N,) float64, NaN when unavailable
    clip_indices: np.ndarray  # (N,) int64
    frame_numbers: dict[str, np.ndarray]  # camera key -> (N,) int64

    @property
    def n_frames(self) -> int:
        return int(len(self.timestamps))


def _nearest_row_indices(reference: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Index of the nearest ``reference`` value for each query (both sorted)."""
    if len(reference) == 1:
        return np.zeros(len(queries), dtype=np.int64)
    right = np.searchsorted(reference, queries)
    right = np.clip(right, 1, len(reference) - 1)
    left = right - 1
    choose_right = np.abs(reference[right] - queries) < np.abs(
        reference[left] - queries
    )
    return np.where(choose_right, right, left)


def _effective_speeds(df: pd.DataFrame) -> np.ndarray:
    """``speed_mps``, back-filled from the north/east velocity when missing."""
    speeds = pd.to_numeric(df["speed_mps"], errors="coerce").to_numpy(np.float64)
    if {"velocity_north_mps", "velocity_east_mps"} <= set(df.columns):
        velocity_north = pd.to_numeric(
            df["velocity_north_mps"], errors="coerce"
        ).to_numpy(np.float64)
        velocity_east = pd.to_numeric(
            df["velocity_east_mps"], errors="coerce"
        ).to_numpy(np.float64)
        from_velocity = np.hypot(velocity_north, velocity_east)
        speeds = np.where(np.isfinite(speeds), speeds, from_velocity)
    return speeds


def build_trip_frame_table(
    clips: list[ClipRef],
    camera_keys: list[str],
    timezone_name: str | None,
    max_camera_sync_s: float = 1.0,
) -> TripFrameTable:
    """Build the frame table for one trip piece (see the module docstring).

    Args:
        clips: the trip's clips, chronological (from :func:`discover_clips`).
        camera_keys: camera folder prefixes to align (from the calibration).
        timezone_name: IANA timezone of the CSV wall-clock timestamps.
        max_camera_sync_s: nearest-row matches further than this from the
            front fix mark the camera as missing for that frame (``-1``).
            Clip streams differ by up to ~0.4 s in duration; 1 s tolerates
            that while rejecting gross misalignment.
    """
    columns: dict[str, list[np.ndarray]] = {key: [] for key in camera_keys}
    timestamps: list[np.ndarray] = []
    latitudes: list[np.ndarray] = []
    longitudes: list[np.ndarray] = []
    speeds: list[np.ndarray] = []
    headings: list[np.ndarray] = []
    clip_indices: list[np.ndarray] = []

    for clip_index, clip in enumerate(clips):
        front_df = read_camera_csv(clip.cameras[FRONT_CAMERA_KEY].csv_path)
        front_epoch = localize_timestamps(front_df["timestamp"], timezone_name)
        has_fix = (
            front_df["GPS_latitude_deg"].notna() & front_df["GPS_longitude_deg"].notna()
        ).to_numpy()
        if not has_fix.any():
            logging.warning(
                "NATIX: no GPS fixes in %s; skipping the clip.",
                clip.cameras[FRONT_CAMERA_KEY].csv_path,
            )
            continue
        anchor_epoch = front_epoch[has_fix]
        n_anchor = len(anchor_epoch)

        timestamps.append(anchor_epoch)
        latitudes.append(front_df["GPS_latitude_deg"].to_numpy(np.float64)[has_fix])
        longitudes.append(front_df["GPS_longitude_deg"].to_numpy(np.float64)[has_fix])
        speeds.append(_effective_speeds(front_df)[has_fix])
        headings.append(
            pd.to_numeric(front_df["heading_deg"], errors="coerce").to_numpy(
                np.float64
            )[has_fix]
        )
        clip_indices.append(np.full(n_anchor, clip_index, dtype=np.int64))

        for camera_key in camera_keys:
            if camera_key == FRONT_CAMERA_KEY:
                columns[camera_key].append(
                    front_df["frame_number"].to_numpy(np.int64)[has_fix]
                )
                continue
            files = clip.cameras.get(camera_key)
            if files is None:
                columns[camera_key].append(np.full(n_anchor, -1, dtype=np.int64))
                continue
            camera_df = read_camera_csv(files.csv_path)
            if camera_df.empty:
                columns[camera_key].append(np.full(n_anchor, -1, dtype=np.int64))
                continue
            camera_epoch = localize_timestamps(camera_df["timestamp"], timezone_name)
            nearest = _nearest_row_indices(camera_epoch, anchor_epoch)
            matched = camera_df["frame_number"].to_numpy(np.int64)[nearest]
            too_far = np.abs(camera_epoch[nearest] - anchor_epoch) > max_camera_sync_s
            columns[camera_key].append(np.where(too_far, -1, matched))

    if not timestamps:
        return TripFrameTable(
            timestamps=np.zeros(0, dtype=np.float64),
            latitudes=np.zeros(0, dtype=np.float64),
            longitudes=np.zeros(0, dtype=np.float64),
            speeds_mps=np.zeros(0, dtype=np.float64),
            headings_deg=np.zeros(0, dtype=np.float64),
            clip_indices=np.zeros(0, dtype=np.int64),
            frame_numbers={key: np.zeros(0, dtype=np.int64) for key in camera_keys},
        )

    table = TripFrameTable(
        timestamps=np.concatenate(timestamps),
        latitudes=np.concatenate(latitudes),
        longitudes=np.concatenate(longitudes),
        speeds_mps=np.concatenate(speeds),
        headings_deg=np.concatenate(headings),
        clip_indices=np.concatenate(clip_indices),
        frame_numbers={key: np.concatenate(chunks) for key, chunks in columns.items()},
    )

    # Strict-monotonicity guard: GPS glitches (or a DST fold) can step a
    # timestamp backwards across clip boundaries; such frames would corrupt
    # the future/past trajectory ordering, so they are dropped loudly.
    keep = np.ones(table.n_frames, dtype=bool)
    if table.n_frames > 1:
        running_max = np.maximum.accumulate(table.timestamps)
        keep[1:] = table.timestamps[1:] > running_max[:-1]
    if not keep.all():
        logging.warning(
            "NATIX: dropping %d frame(s) with non-increasing timestamps.",
            int((~keep).sum()),
        )
        table = TripFrameTable(
            timestamps=table.timestamps[keep],
            latitudes=table.latitudes[keep],
            longitudes=table.longitudes[keep],
            speeds_mps=table.speeds_mps[keep],
            headings_deg=table.headings_deg[keep],
            clip_indices=table.clip_indices[keep],
            frame_numbers={
                key: values[keep] for key, values in table.frame_numbers.items()
            },
        )
    return table
