"""Concrete ``WaymoHDMapEgoCropAggregator`` (per ADR 0007).

Implements the abstract ``_parse_world_segment_map(segment_id)`` hook by
opening the segment's tfrecord under ``source_data_path``, taking the
first record (Waymo Perception populates ``map_features`` only on the
first frame of each segment), and calling
``parse_waymo_map_features`` to build the world-frame
``RawSegmentHDMap``. The base class handles the per-frame ego crop and
persistence; the world-frame map exists only as a local in
``_process_segment``.

The Waymo segment id (``frame.context.name``) is the tfrecord filename
without the ``.tfrecord`` extension. ``source_data_path`` should point
at the directory holding the segment tfrecords (e.g. the
``validation/`` subdirectory of the Waymo Perception v1.4.x release).
"""

from __future__ import annotations

from pathlib import Path

from standard_e2e.caching.segment_context.hd_map_ego_crop import (
    HDMapEgoCropAggregator,
)
from standard_e2e.caching.src_datasets.waymo_perception.hd_map_parser import (
    parse_waymo_map_features,
)
from standard_e2e.data_structures import RawSegmentHDMap


class WaymoHDMapEgoCropAggregator(HDMapEgoCropAggregator):
    """Per-segment Waymo HD-map parse + per-frame ego crop."""

    def _resolve_tfrecord_path(self, segment_id: str) -> Path:
        """Find the tfrecord for ``segment_id`` under ``source_data_path``.

        ``segment_id`` is ``frame.context.name`` from the Waymo proto.
        Two layouts exist on disk:

        * v1.0.0 / E2E-style: filename equals ``segment_id`` exactly.
        * v1.4.x Perception: filename is
          ``segment-{segment_id}_with_camera_labels.tfrecord`` (the
          proto ``context.name`` strips the ``segment-`` prefix and the
          ``_with_camera_labels`` suffix that the filename keeps).

        Resolution: try the direct ``{segment_id}.tfrecord`` first, then
        fall back to a glob that matches both layouts. If a glob pattern
        matches more than one file (e.g., layered installs of multiple
        Waymo versions under the same root), raise rather than silently
        loading the wrong segment's map.
        """
        root = Path(self._source_data_path)
        direct = root / f"{segment_id}.tfrecord"
        if direct.exists():
            return direct
        for pattern in (
            f"**/{segment_id}.tfrecord",
            f"**/segment-{segment_id}*.tfrecord",
            f"**/*{segment_id}*.tfrecord",
        ):
            matches = list(root.glob(pattern))
            if not matches:
                continue
            if len(matches) > 1:
                listing = ", ".join(str(m) for m in sorted(matches))
                raise FileNotFoundError(
                    f"Ambiguous tfrecord for segment '{segment_id}' under "
                    f"{self._source_data_path}: glob '{pattern}' matched "
                    f"{len(matches)} files: [{listing}]. Disambiguate the "
                    "data directory before running."
                )
            return matches[0]
        raise FileNotFoundError(
            f"No tfrecord for segment '{segment_id}' under {self._source_data_path}"
        )

    def _parse_world_segment_map(self, segment_id: str) -> RawSegmentHDMap:
        # TF is required for tfrecord IO; imported lazily so the rest
        # of the cache pipeline does not pull TF when only the lidar /
        # detection paths run.
        import tensorflow as tf  # noqa: PLC0415

        # pylint: disable=no-name-in-module
        from standard_e2e.third_party.waymo_open_dataset.dataset_pb2 import (
            Frame as WaymoFrame,
        )

        tfrecord_path = self._resolve_tfrecord_path(segment_id)
        dataset = tf.data.TFRecordDataset([str(tfrecord_path)], compression_type="")
        for raw_record in dataset:
            frame = WaymoFrame()
            frame.ParseFromString(raw_record.numpy())
            return parse_waymo_map_features(frame)
        raise ValueError(
            f"Segment tfrecord {tfrecord_path} (segment_id={segment_id!r}) "
            "is empty; cannot parse HD map."
        )


__all__ = ["WaymoHDMapEgoCropAggregator"]
