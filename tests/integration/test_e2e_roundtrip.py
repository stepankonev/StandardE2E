"""Numerical round-trip gates for the Waymo E2E pipeline.

Perception's ``test_perception_roundtrip`` audits ``T_world_ego @
past_states ≟ global_position``, but that gate is unavailable on E2E
because the E2E processor does not surface a per-frame pose matrix
(``aux_data["pose_matrix"]``). Without an absolute frame, we cannot do
the world-frame round-trip.

The E2E protos *do* ship internally consistent kinematics — pos / vel /
accel for past, plus pos for future — and the past+future trajectory is
expressed in the ego frame at the current frame's timestamp. That gives
us three numerical invariants to check, each of which catches a class of
ingestion bug the schema-shape PR1 gate cannot:

* **Ego-anchored origin.** ``past_states[N-1]`` is the current ego at
  ``t = 0``. In an ego-relative frame it must be at the origin. A
  regression that emits world-frame past-states (or off-by-one stride
  in the trajectory adapter) will break this immediately.

* **Internal kinematic continuity.** ``past_states[k+1].pos`` should
  equal ``past_states[k].pos + vel[k] * dt`` (forward Euler) within a
  tolerance accounting for the per-step accel update. A regression
  that swaps the meaning of vel / accel columns or coercs ``pos_y``
  into ``pos_x`` will produce position deltas that disagree with the
  velocity column.

* **Past-future continuity.** ``future_states[0].pos`` must be reachable
  from the origin via ``vel[N-1] * dt + 0.5 * accel[N-1] * dt^2`` (the
  current frame's instantaneous kinematics). A regression that loads
  past_states from one segment and future_states from another (the
  silent-mismatch failure mode of an indexed adapter) will fail this
  end-to-end check across two distinct proto fields.

E2E shards shuffle frames across segments, so we deliberately do not
rely on cross-frame same-segment alignment — these per-frame invariants
work on whatever ``raw_iter.take(N)`` returns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from standard_e2e.data_structures import TransformedFrameData
from standard_e2e.enums import Modality
from standard_e2e.enums import TrajectoryComponent as TC

WAYMO_E2E_ROOT = Path(
    "/mnt/bigdisk/datasets/waymo/waymo_open_dataset_end_to_end_camera_v_1_0_0"
)
WAYMO_E2E_TRAINING_SHARD = (
    WAYMO_E2E_ROOT / "training_202504031202_202504151040.tfrecord-00000-of-00263"
)

# 12 frames is enough to land at least one frame with non-empty
# past+future on the training split without making the fixture slow.
NUM_FRAMES = 12

# Trajectory sample spacing used by the E2E processor (matches the
# ``WaymoE2EDatasetProcessor.TRAJECTORY_DELTA_T`` constant; copied here
# so a future change there fails this test loudly rather than silently).
TRAJECTORY_DELTA_T = 0.25

# Origin tolerance: past_states[N-1] in the proto comes back as exactly
# (0, 0, 0) in the samples we inspected, so we set a tight 1mm bound.
EGO_ORIGIN_TOL_M = 1e-3

# Forward-Euler tolerance per step. Real-world acceleration changes the
# velocity within the 0.25 s sample window, so a pure-vel*dt estimate
# can drift by ~0.5 m on aggressive accel; the 0.5 m bound catches axis
# transposes and coordinate-frame leaks while tolerating real kinematics.
KINEMATIC_TOL_M = 0.5

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def e2e_training_frames(tmp_path_factory) -> list[TransformedFrameData]:
    """Process N frames from the E2E training shard.

    The training shard (rather than test) is required because the test
    split intentionally hides labels for some frames - empty
    future_states there would defeat the kinematic checks below.
    """
    if not WAYMO_E2E_TRAINING_SHARD.exists():
        pytest.skip(f"Waymo E2E shard not found at {WAYMO_E2E_TRAINING_SHARD}")

    import tensorflow as tf

    from standard_e2e.caching.src_datasets.waymo_e2e.waymo_e2e_dataset_processor import (  # noqa: E501
        WaymoE2EDatasetProcessor,
    )

    out_dir = tmp_path_factory.mktemp("e2e_roundtrip")
    processor = WaymoE2EDatasetProcessor(
        common_output_path=str(out_dir), split="training"
    )

    frames: list[TransformedFrameData] = []
    raw_iter = tf.data.TFRecordDataset(
        str(WAYMO_E2E_TRAINING_SHARD), compression_type=""
    )
    for raw in raw_iter.take(NUM_FRAMES):
        frame_data, _ = processor.process_frame(raw)
        target = Path(out_dir) / frame_data.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        frame_data.to_npz(str(target))
        frames.append(TransformedFrameData.from_npz(str(target)))
    return frames


def _past_xy(frame: TransformedFrameData) -> np.ndarray:
    """Return ``past_states`` X/Y as ``(N, 2)`` float64."""
    past = frame.get_modality_data(Modality.PAST_STATES)
    assert past is not None and past.length > 0, "past_states missing or empty"
    return past.get([TC.X, TC.Y]).astype(np.float64)


def _past_vel_xy(frame: TransformedFrameData) -> np.ndarray:
    """Return ``past_states`` velocity X/Y as ``(N, 2)`` float64."""
    past = frame.get_modality_data(Modality.PAST_STATES)
    assert past is not None and past.length > 0
    return past.get([TC.VELOCITY_X, TC.VELOCITY_Y]).astype(np.float64)


def _past_accel_xy(frame: TransformedFrameData) -> np.ndarray:
    """Return ``past_states`` acceleration X/Y as ``(N, 2)`` float64."""
    past = frame.get_modality_data(Modality.PAST_STATES)
    assert past is not None and past.length > 0
    return past.get([TC.ACCELERATION_X, TC.ACCELERATION_Y]).astype(np.float64)


def _future_xy(frame: TransformedFrameData) -> np.ndarray:
    """Return ``future_states`` X/Y as ``(M, 2)`` float64."""
    future = frame.get_modality_data(Modality.FUTURE_STATES)
    assert future is not None and future.length > 0, "future_states missing or empty"
    return future.get([TC.X, TC.Y]).astype(np.float64)


def test_past_states_anchor_at_ego_origin(
    e2e_training_frames: list[TransformedFrameData],
):
    """``past_states[N-1]`` (the current frame entry) sits at the ego origin.

    The Waymo E2E proto encodes past trajectories in the vehicle frame at
    the prediction (current) timestamp. The last sample (``timestamp = 0``
    in the trajectory) is therefore the ego itself and must be at
    ``(0, 0)``. A regression that leaks the world-frame past trajectory
    into the cache (e.g. by skipping the proto's frame conversion) will
    produce non-zero offsets here that scale with segment-level world
    coordinates (typically km).
    """
    for frame in e2e_training_frames:
        past_xy = _past_xy(frame)
        # Last entry = current frame (timestamp 0 in trajectory).
        last_xy = past_xy[-1]
        norm = float(np.linalg.norm(last_xy))
        assert norm < EGO_ORIGIN_TOL_M, (
            f"frame {frame.frame_id}: past_states[N-1] {last_xy.tolist()} "
            f"is {norm:.3f} m from origin; expected ego-relative frame."
        )


def test_past_states_kinematic_continuity(
    e2e_training_frames: list[TransformedFrameData],
):
    """``pos[k+1] ≈ pos[k] + vel[k] * dt`` for every consecutive past pair.

    Audits internal numerical consistency of past_states across all
    three columns (pos, vel, accel implicit via the tolerance). A
    regression that swaps ``vel_x`` <-> ``pos_y`` or transposes the
    component dict will fail this check on every frame, even though
    the per-component shape gate stays green.
    """
    audited_pairs = 0
    for frame in e2e_training_frames:
        past_xy = _past_xy(frame)
        past_vel = _past_vel_xy(frame)
        # Need >= 2 samples to form a step.
        if past_xy.shape[0] < 2:
            continue
        for k in range(past_xy.shape[0] - 1):
            predicted = past_xy[k] + past_vel[k] * TRAJECTORY_DELTA_T
            actual = past_xy[k + 1]
            err = float(np.linalg.norm(actual - predicted))
            assert err < KINEMATIC_TOL_M, (
                f"frame {frame.frame_id}: kinematic break at step {k} -> "
                f"{k + 1}: predicted {predicted.tolist()}, actual "
                f"{actual.tolist()}, error {err:.3f} m. Indicates "
                "vel / pos column drift in the past_states adapter."
            )
            audited_pairs += 1
    assert audited_pairs > 0, (
        "no past_states pairs were exercised; fixture may be stale "
        "or all frames had < 2 past samples."
    )


def test_past_future_continuity_at_current_frame(
    e2e_training_frames: list[TransformedFrameData],
):
    """``future_states[0]`` is reachable from origin via current kinematics.

    This is the cross-modality continuity check: past_states' last
    velocity / acceleration (the *current* state at t=0) must forward-
    predict future_states' first sample (the *next* state at t=+dt) up
    to within the kinematic tolerance.

    A regression that loads past_states from one segment and
    future_states from another (silent-mismatch indexer bug) will fail
    here because the disconnected kinematics will not line up. The
    individual modality-shape gate cannot catch this.
    """
    audited_frames = 0
    for frame in e2e_training_frames:
        past_xy = _past_xy(frame)
        past_vel = _past_vel_xy(frame)
        past_accel = _past_accel_xy(frame)
        future_xy = _future_xy(frame)
        if future_xy.shape[0] < 1:
            continue
        # Current state lives at past_states[N-1]; future_states[0] is the
        # next sample at +TRAJECTORY_DELTA_T.
        current_pos = past_xy[-1]  # expected to be ~0 by the prior test
        current_vel = past_vel[-1]
        current_accel = past_accel[-1]
        dt = TRAJECTORY_DELTA_T
        predicted = (
            current_pos + current_vel * dt + 0.5 * current_accel * (dt**2)
        )
        actual = future_xy[0]
        err = float(np.linalg.norm(actual - predicted))
        assert err < KINEMATIC_TOL_M, (
            f"frame {frame.frame_id}: past->future discontinuity at the "
            f"current frame: predicted {predicted.tolist()} from past "
            f"kinematics, future_states[0] = {actual.tolist()}, error "
            f"{err:.3f} m. Indicates mismatched past/future loading "
            "or column-name drift between adapters."
        )
        audited_frames += 1
    assert audited_frames > 0, (
        "no frames had non-empty future_states; the fixture may be "
        "drawing from the test-split shard (where labels are masked) "
        "or NUM_FRAMES is too small."
    )
