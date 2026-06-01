"""Guards for ``StandardFrameDataField`` — the input side of the adapter
contract (``consumes_attrs`` / ``needs_attr``).

The enum mirrors the field names of
:class:`~standard_e2e.data_structures.frame_data.StandardFrameData`. If a
field is added/renamed/removed on the dataclass without updating the enum
(or vice versa), the first test here fails — preventing the silent
gate-skip bug the enum exists to rule out.
"""

from __future__ import annotations

from standard_e2e.data_structures import StandardFrameData
from standard_e2e.enums import Modality, StandardFrameDataField


def test_enum_exactly_mirrors_standard_frame_data_fields():
    """Every ``StandardFrameData`` field has an enum member and vice versa."""
    enum_values = {member.value for member in StandardFrameDataField}
    model_fields = set(StandardFrameData.model_fields)
    assert enum_values == model_fields, (
        "StandardFrameDataField drifted from StandardFrameData:\n"
        f"  only in enum:   {sorted(enum_values - model_fields)}\n"
        f"  only in fields: {sorted(model_fields - enum_values)}"
    )


def test_each_member_value_is_a_real_attribute():
    """``getattr(frame, member)`` must resolve for every enum member."""
    # A minimally-constructed frame; defaults cover the optional fields.
    frame = StandardFrameData(
        dataset_name="d",
        segment_id="s",
        frame_id=0,
        timestamp=0.0,
        split="train",
    )
    for member in StandardFrameDataField:
        assert hasattr(frame, member), member


def test_distinct_from_modality_where_it_matters():
    """The two enums are deliberately different layers; the classic
    mismatch (input ``lidar`` vs output ``lidar_pc``) must hold so nobody
    'simplifies' one into the other."""
    assert StandardFrameDataField.LIDAR.value == "lidar"
    assert Modality.LIDAR_PC.value == "lidar_pc"
    assert StandardFrameDataField.LIDAR.value != Modality.LIDAR_PC.value
    # detections: SFD field is frame_detections_3d, modality is detections_3d
    assert StandardFrameDataField.FRAME_DETECTIONS_3D.value == "frame_detections_3d"
    assert Modality.DETECTIONS_3D.value == "detections_3d"
