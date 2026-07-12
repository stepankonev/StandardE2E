"""Source-dataset converter for the NATIX Edge Case Driving Dataset.

The release is a plain directory tree in the exact ``natix_multicam`` layout
(trips discovered by their ``fixed_metadata.json`` marker at any nesting
depth, plus the release-level ``edge-case.json`` the **processor** joins in),
so the whole iteration scheme is inherited from
:class:`NatixMulticamDatasetConverter`: one ``(TripRef, frame_index)`` task
per front-camera GPS fix, trip-major / frame-ascending for forward-only mp4
decode, ``--frame_stride`` subsampling, first-trip-only under
``STANDARD_E2E_DEBUG=true``.
"""

from __future__ import annotations

from standard_e2e.caching.src_datasets.natix_multicam.natix_multicam_dataset_converter import (  # noqa: E501
    NatixMulticamDatasetConverter,
)


class NatixEdgeCaseDatasetConverter(NatixMulticamDatasetConverter):
    """Iterates NATIX edge-case trip pieces frame-by-frame."""
