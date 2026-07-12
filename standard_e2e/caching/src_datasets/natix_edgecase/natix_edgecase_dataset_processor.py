"""Source-dataset processor for the NATIX Edge Case Driving Dataset.

NATIX Edge Case (Hugging Face
``natix-network-org/natix-edge-case-driving-dataset``) is the **curated
edge-case sibling** of the NATIX Multi-Camera Driving Dataset -- rare,
challenging real-world scenarios (construction zones, adverse weather, road
deterioration, illegal maneuvers, obstructions) crowd-sourced from the same
decentralized Tesla-dashcam network. The first public release is 20 minutes /
86 mp4 clips across six US states (4- and 6-camera rigs) with **21
VLM-annotated events**.

The footage, GPS metadata, calibration and folder layout are **identical to
``natix_multicam``**, so this processor subclasses
:class:`NatixMulticamDatasetProcessor` -- one frame per front-camera GPS fix,
one segment per trip piece, cameras keyed by facing-verified
:class:`CameraDirection` -- and adds the release's one new ingredient, the
**edge-case annotations** (``edge-case.json``; see
:mod:`._natix_edgecase_events`):

* ``aux_data["edge_case_events"]``: the events whose ``[start_sec, end_sec]``
  window (video seconds from clip start) covers the frame's timestamp -- each
  a dict with the ``label``, the window and the full VLM ``ai_analysis``.
  Present on every frame (empty list outside events).
* ``extra_index_data``: on top of the multicam trip / country / region /
  camera_count -- ``edge_case_count`` (events covering the frame; filter
  ``> 0`` for in-event frames) and ``edge_case`` (their summaries, ``" | "``
  joined; the ``label``, falling back to the VLM ``EVENT CLASSIFICATION``
  when the label is empty).

A trip tree without ``edge-case.json`` processes fine with a warning and
empty annotations (the footage is self-contained); data-quality bounds are
the multicam ones (consumer GPS, nominal calibrations), plus the release's
own caveat that annotations are **VLM-generated best-effort** -- windows are
integer seconds and descriptions are model-written, so treat them as
guidance, not ground truth. One further observed GPS degradation (passed
through as-shipped, like the rest): a trip's fixes can **freeze** -- lat/lon
stuck with speed 0 for tens of seconds while the footage keeps moving
(observed for the first ~27 s of one release trip, overlapping its event
window) -- so the pose-derived past/future trajectories are degenerate
(zero-motion) over such stretches even though the cameras are fine.
"""

from __future__ import annotations

import logging
from typing import Any

from standard_e2e.caching.src_datasets.natix_edgecase import (
    _natix_edgecase_events as events_io,
)
from standard_e2e.caching.src_datasets.natix_multicam import _natix_io as io
from standard_e2e.caching.src_datasets.natix_multicam.natix_multicam_dataset_processor import (  # noqa: E501
    NatixMulticamDatasetProcessor,
)
from standard_e2e.data_structures import StandardFrameData


class NatixEdgeCaseDatasetProcessor(NatixMulticamDatasetProcessor):
    """Processor for the NATIX Edge Case Driving Dataset.

    Reuses the whole multicam pipeline (per-trip cache, forward-only video
    readers) and joins the edge-case annotations onto every emitted frame.
    ``edge-case.json`` is parsed once per worker (keyed by its resolved
    path); per-clip video anchors are computed lazily, only for annotated
    clips of the current trip.
    """

    DATASET_NAME = "natix_edgecase"

    _MISSING_JSON_SENTINEL = ""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Per-worker caches (lazily populated, keyed to survive trip changes).
        self._events_cache: dict[
            str, dict[tuple[str, str], list[events_io.EdgeCaseEvent]]
        ] = {}
        self._trip_events: dict[str, list[events_io.EdgeCaseEvent]] = {}
        self._clip_anchors: dict[str, float] = {}
        self._timezone_name: str | None = None
        self._warned_missing_json = False

    # --- per-trip cache -----------------------------------------------------

    def _events_for_trip_tree(
        self, ref: io.TripRef
    ) -> dict[tuple[str, str], list[events_io.EdgeCaseEvent]]:
        json_path = events_io.find_edge_case_json(ref.path)
        if json_path is None:
            if not self._warned_missing_json:
                logging.warning(
                    "NATIX edge-case: no %s found above %s; frames will "
                    "carry empty edge-case annotations.",
                    events_io.EDGE_CASE_JSON_NAME,
                    ref.path,
                )
                self._warned_missing_json = True
            json_path = self._MISSING_JSON_SENTINEL
        if json_path not in self._events_cache:
            self._events_cache[json_path] = (
                events_io.load_edge_case_events(json_path) if json_path else {}
            )
        return self._events_cache[json_path]

    def _refresh_trip_cache(self, ref: io.TripRef) -> None:
        if self._cached_trip_name == ref.trip_name:
            return
        super()._refresh_trip_cache(ref)
        self._timezone_name = io.trip_timezone(io.read_trip_insight(ref))
        all_events = self._events_for_trip_tree(ref)
        self._trip_events = {
            clip.clip_name: all_events[(ref.trip_name, clip.clip_name)]
            for clip in self._clips
            if (ref.trip_name, clip.clip_name) in all_events
        }
        self._clip_anchors = {}

    # --- per-frame annotation join -------------------------------------------

    def _clip_anchor(self, clip: io.ClipRef) -> float:
        anchor = self._clip_anchors.get(clip.clip_name)
        if anchor is None:
            anchor = events_io.clip_video_anchor(
                clip.cameras[io.FRONT_CAMERA_KEY].csv_path, self._timezone_name
            )
            self._clip_anchors[clip.clip_name] = anchor
        return anchor

    def _events_covering_frame(self, frame_index: int) -> list[events_io.EdgeCaseEvent]:
        assert self._table is not None
        clip = self._clips[int(self._table.clip_indices[frame_index])]
        events = self._trip_events.get(clip.clip_name)
        if not events:
            return []
        return events_io.events_covering(
            events,
            self._clip_anchor(clip),
            float(self._table.timestamps[frame_index]),
        )

    # --- main entry point ---------------------------------------------------

    def _prepare_standardized_frame_data(
        self, raw_frame_data: Any
    ) -> StandardFrameData:
        frame = super()._prepare_standardized_frame_data(raw_frame_data)
        _ref, frame_index = raw_frame_data
        events = self._events_covering_frame(frame_index)
        assert frame.aux_data is not None and frame.extra_index_data is not None
        frame.aux_data["edge_case_events"] = [event.as_aux_dict() for event in events]
        frame.extra_index_data["edge_case_count"] = len(events)
        frame.extra_index_data["edge_case"] = " | ".join(
            event.summary for event in events
        )
        return frame
