"""Edge-case annotation helpers for the NATIX Edge Case Driving Dataset.

The release ships one ``edge-case.json`` next to the trip tree (``data/
edge-case.json`` in the Hugging Face layout) -- a JSON array with one entry
per **annotated clip**: a ``clip_path`` (``<Country>/[<State>/]<trip>/
<HH-MM-SS>/``, relative to the file's directory) and its ``events``, each a
``label`` (may be empty), an ``ai_analysis`` object (VLM-generated; five
fixed uppercase keys) and a ``[start_sec, end_sec]`` window in **video
seconds from the clip start**.

* :func:`find_edge_case_json` walks **up** from a trip directory to the
  nearest ancestor holding ``edge-case.json`` -- robust to where
  ``--input_path`` points (the download root, ``data/``, or deeper), exactly
  like the marker-based trip discovery.
* :func:`load_edge_case_events` keys the parsed events by
  ``(trip_name, clip_name)`` -- the last two ``clip_path`` components --
  instead of resolving paths, so annotation lookup is independent of the
  on-disk nesting (trip names are UUID-based and globally unique).
* :func:`clip_video_anchor` maps the window's video seconds onto the frame
  timeline: the epoch time of **video frame 1** for a clip, extrapolated from
  the front CSV's first row as ``epoch[0] - (frame_number[0] - 1) * period``
  (a clip's CSV can start past frame 1 -- observed ``frame_number=8`` -- so
  the first row is not necessarily the video start) with the per-frame period
  fitted from the CSV's own (frame_number, epoch) extremes.

An event then covers a frame iff ``start_sec <= frame_epoch - anchor <=
end_sec`` (inclusive; windows are integer seconds, far coarser than the
~0.03 s frame period, and the VLM windows themselves are approximate).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from standard_e2e.caching.src_datasets.natix_multicam import _natix_io as io

EDGE_CASE_JSON_NAME = "edge-case.json"

# Fallback per-frame period for a degenerate CSV (a single usable row); the
# footage is ~36 fps. Only used when the fit below has no baseline.
_FALLBACK_FRAME_PERIOD_S = 1.0 / 36.0


@dataclass(frozen=True)
class EdgeCaseEvent:
    """One annotated event within one clip (video-relative window, seconds)."""

    label: str
    start_sec: float
    end_sec: float
    ai_analysis: dict

    @property
    def summary(self) -> str:
        """The ``label``, falling back to the VLM ``EVENT CLASSIFICATION``
        (the label is empty on some entries; the classification never is)."""
        return self.label or str(self.ai_analysis.get("EVENT CLASSIFICATION") or "")

    def as_aux_dict(self) -> dict:
        """Plain-dict form stored in ``aux_data`` (npz-friendly)."""
        return {
            "label": self.label,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "ai_analysis": dict(self.ai_analysis),
        }


def find_edge_case_json(trip_path: str) -> str | None:
    """Nearest ancestor ``edge-case.json`` for a trip directory (None if none).

    The file sits at the trip tree's root (``data/`` in the released layout),
    which is always an ancestor of every trip directory, wherever
    ``--input_path`` pointed.
    """
    for ancestor in Path(trip_path).resolve().parents:
        candidate = ancestor / EDGE_CASE_JSON_NAME
        if candidate.is_file():
            return str(candidate)
    return None


def _clip_key_from_path(clip_path: str) -> tuple[str, str] | None:
    """``.../<trip>/<HH-MM-SS>/`` -> ``(trip_name, clip_name)`` (None if malformed)."""
    parts = [part for part in clip_path.replace("\\", "/").split("/") if part]
    if len(parts) < 2:
        return None
    return parts[-2], parts[-1]


def load_edge_case_events(json_path: str) -> dict[tuple[str, str], list[EdgeCaseEvent]]:
    """Parse ``edge-case.json`` into events keyed by ``(trip_name, clip_name)``.

    Malformed entries are skipped with a warning rather than failing the run:
    the annotations are auxiliary to the footage.
    """
    with open(json_path, "r", encoding="utf-8") as handle:
        entries = json.load(handle)
    if not isinstance(entries, list):
        logging.warning("NATIX edge-case: %s is not a JSON array.", json_path)
        return {}

    events_by_clip: dict[tuple[str, str], list[EdgeCaseEvent]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = _clip_key_from_path(str(entry.get("clip_path") or ""))
        if key is None:
            logging.warning(
                "NATIX edge-case: unusable clip_path %r in %s; skipping.",
                entry.get("clip_path"),
                json_path,
            )
            continue
        clip_events = events_by_clip.setdefault(key, [])
        for event in entry.get("events") or []:
            try:
                ai_analysis = event.get("ai_analysis") or {}
                clip_events.append(
                    EdgeCaseEvent(
                        label=str(event.get("label") or ""),
                        start_sec=float(event["start_sec"]),
                        end_sec=float(event["end_sec"]),
                        ai_analysis=(
                            dict(ai_analysis) if isinstance(ai_analysis, dict) else {}
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError):
                logging.warning(
                    "NATIX edge-case: malformed event for clip %r in %s; skipping.",
                    entry.get("clip_path"),
                    json_path,
                )
    return events_by_clip


def clip_video_anchor(front_csv_path: str, timezone_name: str | None) -> float:
    """Epoch seconds of **video frame 1** of a clip's front stream.

    Extrapolates from the first usable CSV row (``epoch - (frame_number - 1)
    * period``); the period comes from the CSV's (frame_number, epoch)
    extremes, so leading rows missing from the CSV (observed in the release)
    do not shift the anchor.
    """
    df = io.read_camera_csv(front_csv_path)
    if df.empty:
        raise ValueError(f"no usable rows in {front_csv_path}")
    epochs = io.localize_timestamps(df["timestamp"], timezone_name)
    frame_numbers = df["frame_number"].to_numpy(np.int64)
    if len(df) > 1 and frame_numbers[-1] > frame_numbers[0]:
        period = (epochs[-1] - epochs[0]) / float(frame_numbers[-1] - frame_numbers[0])
    else:
        period = _FALLBACK_FRAME_PERIOD_S
    return float(epochs[0] - (frame_numbers[0] - 1) * period)


def events_covering(
    events: list[EdgeCaseEvent], anchor_epoch: float, frame_epoch: float
) -> list[EdgeCaseEvent]:
    """The events whose window covers ``frame_epoch`` (inclusive bounds)."""
    video_sec = frame_epoch - anchor_epoch
    return [event for event in events if event.start_sec <= video_sec <= event.end_sec]
