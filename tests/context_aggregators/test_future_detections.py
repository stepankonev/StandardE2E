"""Tests for ``FutureDetectionsAggregator``.

Builds a two-frame mini segment with one known agent and asserts the
aggregated trajectory in the *current* ego frame matches the closed-form
SE(2) composition. The current-frame value is derived independently from
AV2's own SE3 conventions to act as ground truth for the aggregator's
in-house SE(2) math.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from standard_e2e.caching.segment_context.future_detections import (
    FutureDetectionsAggregator,
)
from standard_e2e.data_structures import (
    Detection3D,
    FrameDetections3D,
    Trajectory,
    TransformedFrameData,
)
from standard_e2e.enums import DetectionType, Modality
from standard_e2e.enums import TrajectoryComponent as TC


def _R(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def _make_frame(
    seg: str,
    frame_id: int,
    ts: float,
    ego_xyh_world: tuple[float, float, float],
    detections: Optional[list[Detection3D]] = None,
) -> TransformedFrameData:
    x, y, h = ego_xyh_world
    frame = TransformedFrameData(
        dataset_name="ds",
        split="train",
        segment_id=seg,
        frame_id=frame_id,
        timestamp=ts,
        global_position=Trajectory(
            {
                TC.TIMESTAMP: [ts],
                TC.X: [x],
                TC.Y: [y],
                TC.HEADING: [h],
            }
        ),
    )
    frame.set_modality_data(
        Modality.DETECTIONS_3D,
        FrameDetections3D(detections=detections or []),
    )
    return frame


def _save(frame: TransformedFrameData, root: Path) -> None:
    out = root / frame.filename
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_npz(str(out))


def _index(frames: list[TransformedFrameData]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": [f.segment_id for f in frames],
            "timestamp": [f.timestamp for f in frames],
            "filename": [f.filename for f in frames],
        }
    )


def test_future_detections_se2_composition_matches_closed_form(tmp_path: Path):
    # Current ego at world origin, heading 0; future ego translated and rotated.
    ego_C = (0.0, 0.0, 0.0)
    ego_F = (5.0, 2.0, 0.3)

    # Detection in the future ego frame: 10 m forward, 5 m left of ego_F,
    # heading 0 (relative to ego_F).
    det_egoF_xy = np.array([10.0, 5.0])
    det_egoF_h = 0.0

    detection = Detection3D(
        unique_agent_id="agent_42",
        detection_type=DetectionType.VEHICLE,
        trajectory=Trajectory(
            {
                TC.TIMESTAMP: [1.0],
                TC.X: [det_egoF_xy[0]],
                TC.Y: [det_egoF_xy[1]],
                TC.Z: [0.0],
                TC.HEADING: [det_egoF_h],
                TC.LENGTH: [4.0],
                TC.WIDTH: [2.0],
                TC.HEIGHT: [1.5],
            }
        ),
    )

    frames = [
        _make_frame("seg", 0, 0.0, ego_C, detections=[]),
        _make_frame("seg", 1, 1.0, ego_F, detections=[detection]),
    ]
    for f in frames:
        _save(f, tmp_path)

    FutureDetectionsAggregator(str(tmp_path)).process(_index(frames))

    # Closed-form ground truth: det_egoC = R(H_F - H_C) @ det_egoF
    #                                   + R(-H_C) @ (P_F - P_C)
    H_C, H_F = ego_C[2], ego_F[2]
    P_F = np.array([ego_F[0], ego_F[1]])
    P_C = np.array([ego_C[0], ego_C[1]])
    delta_h = H_F - H_C
    expected_xy = _R(delta_h) @ det_egoF_xy + _R(-H_C) @ (P_F - P_C)
    expected_h = det_egoF_h + delta_h

    reloaded = TransformedFrameData.from_npz(os.path.join(tmp_path, frames[0].filename))
    fd: FrameDetections3D = reloaded.get_modality_data(Modality.DETECTIONS_3D)
    assert isinstance(fd, FrameDetections3D)
    assert len(fd.detections) == 1
    agg_traj = fd.detections[0].trajectory
    np.testing.assert_allclose(agg_traj.get([TC.X, TC.Y])[0], expected_xy, atol=1e-5)
    np.testing.assert_allclose(agg_traj.get([TC.HEADING])[0, 0], expected_h, atol=1e-5)


def test_future_detections_yaw_only_world_translation(tmp_path: Path):
    # Pure translation case (no heading change) -- the rotation term vanishes
    # and the result is just the detection's egoF position offset by the
    # world-frame ego translation expressed in the *current* ego frame.
    ego_C = (10.0, -5.0, np.pi / 4)
    ego_F = (10.0 + 3.0, -5.0 + 4.0, np.pi / 4)  # 5 m world delta, no rotation
    det_egoF_xy = np.array([7.0, -1.0])

    detection = Detection3D(
        unique_agent_id="t",
        detection_type=DetectionType.VEHICLE,
        trajectory=Trajectory(
            {
                TC.TIMESTAMP: [1.0],
                TC.X: [det_egoF_xy[0]],
                TC.Y: [det_egoF_xy[1]],
                TC.HEADING: [0.5],
            }
        ),
    )
    frames = [
        _make_frame("seg2", 0, 0.0, ego_C, detections=[]),
        _make_frame("seg2", 1, 1.0, ego_F, detections=[detection]),
    ]
    for f in frames:
        _save(f, tmp_path)

    FutureDetectionsAggregator(str(tmp_path)).process(_index(frames))

    # delta_h = 0 so the rotation matrix R(delta_h) is identity.
    # delta_xy_egoC = R(-pi/4) @ (3, 4)
    expected_xy = det_egoF_xy + _R(-ego_C[2]) @ np.array([3.0, 4.0])

    reloaded = TransformedFrameData.from_npz(os.path.join(tmp_path, frames[0].filename))
    fd: FrameDetections3D = reloaded.get_modality_data(Modality.DETECTIONS_3D)
    np.testing.assert_allclose(
        fd.detections[0].trajectory.get([TC.X, TC.Y])[0], expected_xy, atol=1e-5
    )
