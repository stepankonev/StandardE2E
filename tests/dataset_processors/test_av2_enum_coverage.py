"""AV2 enum-coverage spike — read-only mapping table.

Tabulates how Waymo Perception's source enums and AV2 Sensor's source
enums collapse into the canonical ``LaneType`` / ``LaneMarkType`` /
``RoadEdgeType`` taxonomy. Driven by ``standard_e2e/enums.py`` and the
existing Waymo mappings in
``standard_e2e/caching/src_datasets/waymo_perception/hd_map_parser.py``.

This test is **skipped** until AV2 source fixtures land. It exists now
to surface enum-coverage gaps before AV2 implementation begins: if a
row below has no canonical match, the canonical enum either needs
extension or AV2's value should map to ``UNKNOWN`` — that decision
should be made up-front, not under PR3 deadline pressure.

Mapping table:

| Canonical                  | Waymo LaneCenter / RoadLine / RoadEdge      | AV2 LaneType / LaneMarkType                 |
|----------------------------|---------------------------------------------|---------------------------------------------|
| LaneType.UNKNOWN           | TYPE_UNDEFINED                              | UNKNOWN                                     |
| LaneType.VEHICLE           | TYPE_FREEWAY, TYPE_SURFACE_STREET           | VEHICLE                                     |
| LaneType.BIKE              | TYPE_BIKE_LANE                              | BIKE                                        |
| LaneType.BUS               | (no Waymo source value -> UNKNOWN today)    | BUS                                         |
| LaneMarkType.UNKNOWN       | TYPE_UNKNOWN                                | UNKNOWN                                     |
| LaneMarkType.NONE          | (no Waymo source value)                     | NONE                                        |
| LaneMarkType.SOLID_WHITE   | TYPE_SOLID_SINGLE_WHITE                     | SOLID_WHITE, SOLID_BLUE *(see note)*        |
| LaneMarkType.SOLID_YELLOW  | TYPE_SOLID_SINGLE_YELLOW                    | SOLID_YELLOW                                |
| LaneMarkType.DASHED_WHITE  | TYPE_BROKEN_SINGLE_WHITE                    | DASHED_WHITE                                |
| LaneMarkType.DASHED_YELLOW | TYPE_BROKEN_SINGLE_YELLOW,                  | DASHED_YELLOW                               |
|                            | TYPE_BROKEN_DOUBLE_YELLOW                   |                                             |
| LaneMarkType.DOUBLE_SOLID_WHITE  | TYPE_SOLID_DOUBLE_WHITE              | DOUBLE_SOLID_WHITE                          |
| LaneMarkType.DOUBLE_SOLID_YELLOW | TYPE_SOLID_DOUBLE_YELLOW             | DOUBLE_SOLID_YELLOW                         |
| LaneMarkType.PASSING_DOUBLE_DASH | TYPE_PASSING_DOUBLE_YELLOW           | DOUBLE_DASH_WHITE, DOUBLE_DASH_YELLOW       |
| RoadEdgeType.UNKNOWN       | TYPE_UNKNOWN                                | (AV2 has no first-class road-edge enum)     |
| RoadEdgeType.BOUNDARY      | TYPE_ROAD_EDGE_BOUNDARY                     | (derived from DrivableArea polygon edges)   |
| RoadEdgeType.MEDIAN        | TYPE_ROAD_EDGE_MEDIAN                       | (no AV2 equivalent — likely UNKNOWN)        |

Notes / open questions for the AV2 PR:

1. ``SOLID_BLUE`` (AV2) has no canonical bucket. Default plan: map to
   ``SOLID_WHITE`` (closest by visual semantics) or extend canonical
   ``LaneMarkType`` with ``SOLID_BLUE``. Decide before AV2 parser lands.

2. ``MIXED_DASH_SOLID`` ladder (AV2 has DASH_SOLID_WHITE,
   DASH_SOLID_YELLOW, SOLID_DASH_WHITE, SOLID_DASH_YELLOW) has no
   canonical bucket today. Coarsest collapse: map to the matching
   colour's ``DASHED_*`` (loses the solid-side hint). If the consumer
   needs the side bit, extend the canonical enum.

3. ``RoadEdgeType.MEDIAN`` is Waymo-only. AV2 sensor lacks a typed
   median; if AV2 wants a road edge, it derives one from
   ``DrivableArea`` polygon boundaries and labels them
   ``BOUNDARY``. ``MEDIAN`` stays in the canonical enum for Waymo.

4. AV2 has ``DrivableArea`` as a first-class polygon (already in the
   canonical enum). Waymo has no equivalent — the field stays empty
   for Waymo.

5. AV2's ``PedestrianCrossing`` is the closest analogue to Waymo's
   ``Crosswalk``; both already collapse to ``Crosswalk`` (polygon).

When AV2 fixtures land, replace this skip with a parametric test that
walks every AV2 source enum value, calls the AV2 parser's mapping
helper, and asserts the canonical bucket matches this table.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="AV2 fixtures land in PR3")


def test_av2_enum_coverage_table_is_authoritative():
    """Placeholder. The mapping table lives in this module's docstring."""
