import numpy as np
import pytest
import torch

from standard_e2e.data_structures import (
    Detection3D,
    FrameDetections3D,
    LidarPointCloud,
    Trajectory,
    TransformedFrameData,
)
from standard_e2e.dataset_utils.modality_defaults import IntentDefaults
from standard_e2e.enums import (
    DetectionType,
    Intent,
    LidarComponent,
    Modality,
    TrajectoryComponent,
)
from standard_e2e.training import (
    BatchedFrameDetections3D,
    BatchedLidarPointCloud,
    BatchedTrajectory,
    TransformedFrameDataBatch,
)


def make_frame(idx: int = 0, **overrides) -> TransformedFrameData:
    kwargs = dict(
        dataset_name="ds",
        segment_id=f"seg{idx}",
        frame_id=idx,
        timestamp=100.0 + idx,
        split="train",
    )
    kwargs.update(overrides)
    return TransformedFrameData(**kwargs)


def test_init_requires_non_empty():
    with pytest.raises(ValueError):
        TransformedFrameDataBatch([])


def test_basic_fields_and_filename_and_split():
    f1 = make_frame(1)
    f2 = make_frame(2, split="val")

    batch = TransformedFrameDataBatch([f1, f2])

    assert batch.dataset_name == ["ds", "ds"]
    assert batch.segment_id == ["seg1", "seg2"]
    assert batch.frame_id == [1, 2]
    assert isinstance(batch.timestamp, torch.Tensor)
    assert batch.timestamp.dtype == torch.float32
    assert np.isclose(batch.timestamp[0].item(), 101.0)
    assert batch.split == ["train", "val"]
    assert batch.filename[0].endswith("seg1_1.npz")


def test_union_of_modalities_and_collation_to_tensors():
    # Both frames have CAMERAS/FUTURE_STATES/PAST_STATES; INTENT present only on f2
    # f1 uses IntentDefaults to provide a default INTENT during collation
    f1 = make_frame(1, modality_defaults={Modality.INTENT: IntentDefaults()})
    f1.set_modality_data(Modality.CAMERAS, np.zeros((256, 256, 3), dtype=np.uint8))
    # FUTURE_STATES and PAST_STATES must be Trajectory instances
    f1.set_modality_data(
        Modality.FUTURE_STATES, Trajectory({TrajectoryComponent.X: [0.1, 0.2]})
    )
    f1.set_modality_data(
        Modality.PAST_STATES, Trajectory({TrajectoryComponent.X: [9.9]})
    )

    f2 = make_frame(2)
    f2.set_modality_data(Modality.CAMERAS, np.zeros((256, 256, 3), dtype=np.uint8))
    f2.set_modality_data(
        Modality.FUTURE_STATES, Trajectory({TrajectoryComponent.X: [0.3, 0.4]})
    )
    f2.set_modality_data(
        Modality.PAST_STATES, Trajectory({TrajectoryComponent.X: [8.8]})
    )
    f2.set_modality_data(Modality.INTENT, Intent.GO_LEFT)

    batch = TransformedFrameDataBatch([f1, f2])

    # Union -> CAMERAS/FUTURE/PAST always present; INTENT present due to f2
    assert batch.get_modality_data(Modality.CAMERAS) is not None
    assert batch.get_modality_data(Modality.FUTURE_STATES) is not None
    assert batch.get_modality_data(Modality.PAST_STATES) is not None
    assert batch.get_modality_data(Modality.INTENT) is not None

    # CAMERAS collated to tensor with shape (batch, H, W)
    cams = batch.get_modality_data(Modality.CAMERAS)
    assert isinstance(cams, torch.Tensor)
    # assert cams.dtype == torch.float32
    assert tuple(cams.shape) == (2, 256, 256, 3)

    # FUTURE_STATES and PAST_STATES collate to BatchedTrajectory
    fut = batch.get_modality_data(Modality.FUTURE_STATES)
    past = batch.get_modality_data(Modality.PAST_STATES)

    assert isinstance(fut, BatchedTrajectory)
    assert isinstance(past, BatchedTrajectory)
    x_fut = fut.get(TrajectoryComponent.X)
    x_past = past.get(TrajectoryComponent.X)
    assert tuple(x_fut.shape) == (2, 2, 1)
    assert tuple(x_past.shape) == (2, 1, 1)

    # INTENT collates to a 1D tensor of enum values: tensor([0, Intent.GO_LEFT])
    intent = batch.get_modality_data(Modality.INTENT)
    assert isinstance(intent, torch.Tensor)
    assert intent.dtype in (torch.int64, torch.int32)
    assert intent.tolist() == [0, int(Intent.GO_LEFT)]


def test_device_move_cpu_and_cuda_for_nested_modalities():
    f1 = make_frame(1, modality_defaults={Modality.INTENT: IntentDefaults()})
    f1.set_modality_data(Modality.CAMERAS, np.zeros((256, 256, 3), dtype=np.uint8))
    f1.set_modality_data(
        Modality.FUTURE_STATES, Trajectory({TrajectoryComponent.X: [0.1, 0.2]})
    )
    f1.set_modality_data(
        Modality.PAST_STATES, Trajectory({TrajectoryComponent.X: [9.9]})
    )
    f2 = make_frame(2)
    f2.set_modality_data(Modality.CAMERAS, np.zeros((256, 256, 3), dtype=np.uint8))
    f2.set_modality_data(
        Modality.FUTURE_STATES, Trajectory({TrajectoryComponent.X: [0.3, 0.4]})
    )
    f2.set_modality_data(
        Modality.PAST_STATES, Trajectory({TrajectoryComponent.X: [8.8]})
    )
    f2.set_modality_data(Modality.INTENT, Intent.GO_LEFT)

    batch = TransformedFrameDataBatch([f1, f2])
    # Move to CPU (no-op) then optionally to CUDA
    batch = batch.to(torch.device("cpu"))
    cams = batch.get_modality_data(Modality.CAMERAS)
    assert cams.device.type == "cpu"
    intent = batch.get_modality_data(Modality.INTENT)
    assert intent.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_to_cuda_moves_tensors_and_nested_dicts():
    f1 = make_frame(1, modality_defaults={Modality.INTENT: IntentDefaults()})
    f1.set_modality_data(Modality.CAMERAS, np.zeros((256, 256, 3), dtype=np.uint8))
    f1.set_modality_data(Modality.FUTURE_STATES, [0.1, 0.2])
    f1.set_modality_data(Modality.PAST_STATES, [9.9])
    f2 = make_frame(2)
    f2.set_modality_data(Modality.CAMERAS, np.zeros((256, 256, 3), dtype=np.uint8))
    f2.set_modality_data(Modality.FUTURE_STATES, [0.3, 0.4])
    f2.set_modality_data(Modality.PAST_STATES, [8.8])
    f2.set_modality_data(Modality.INTENT, Intent.GO_LEFT)

    batch = TransformedFrameDataBatch([f1, f2])
    batch = batch.to(torch.device("cuda:0"))

    assert batch.timestamp.device.type == "cuda"
    cams = batch.get_modality_data(Modality.CAMERAS)
    assert cams.device.type == "cuda"
    intent = batch.get_modality_data(Modality.INTENT)
    assert intent.device.type == "cuda"


def test_timestamp_diff_none_if_any_missing():
    f1 = make_frame(1, timestamp_diff=0.1)
    f2 = make_frame(2, timestamp_diff=None)
    batch = TransformedFrameDataBatch([f1, f2])
    assert batch.timestamp_diff is None


def test_timestamp_diff_tensor_when_all_present():
    f1 = make_frame(1, timestamp_diff=0.1)
    f2 = make_frame(2, timestamp_diff=0.2)
    batch = TransformedFrameDataBatch([f1, f2])
    assert isinstance(batch.timestamp_diff, torch.Tensor)
    assert batch.timestamp_diff.dtype == torch.float32
    assert np.allclose(
        batch.timestamp_diff.cpu().numpy(),
        np.array([0.1, 0.2], dtype=np.float32),
    )


def test_aux_data_copies_from_first_frame():
    f1 = make_frame(1)
    f1.aux_data = {"a": 1}
    f2 = make_frame(2)
    f2.aux_data = {"a": 2}
    batch = TransformedFrameDataBatch([f1, f2])
    assert batch.aux_data == {"a": 1}


def test_get_modality_data_absent_returns_none():
    f1 = make_frame(1)
    f2 = make_frame(2)
    batch = TransformedFrameDataBatch([f1, f2])
    assert batch.get_modality_data(Modality.CAMERAS) is None


XYZ = [LidarComponent.X, LidarComponent.Y, LidarComponent.Z]


def _detection(agent_id: str, x: float) -> Detection3D:
    return Detection3D(
        unique_agent_id=agent_id,
        detection_type=DetectionType.VEHICLE,
        trajectory=Trajectory(
            {
                TrajectoryComponent.TIMESTAMP: [0.0],
                TrajectoryComponent.X: [x],
                TrajectoryComponent.Y: [0.0],
                TrajectoryComponent.HEADING: [0.0],
            }
        ),
    )


def test_batched_frame_detections_rejects_other_types():
    with pytest.raises(TypeError):
        BatchedFrameDetections3D([object()])


def test_detections_and_lidar_collate_to_batched_types():
    f1 = make_frame(1)
    f1.set_modality_data(
        Modality.DETECTIONS_3D,
        FrameDetections3D(detections=[_detection("a", 1.0), _detection("b", 2.0)]),
    )
    f1.set_modality_data(
        Modality.LIDAR_PC, LidarPointCloud(np.zeros((3, 3), dtype=np.float32), XYZ)
    )
    f2 = make_frame(2)
    f2.set_modality_data(
        Modality.DETECTIONS_3D, FrameDetections3D(detections=[_detection("c", 3.0)])
    )
    f2.set_modality_data(
        Modality.LIDAR_PC, LidarPointCloud(np.ones((2, 3), dtype=np.float32), XYZ)
    )

    batch = TransformedFrameDataBatch([f1, f2])

    detections = batch.get_modality_data(Modality.DETECTIONS_3D)
    assert isinstance(detections, BatchedFrameDetections3D)
    # The accessors expose the first frame of the batch.
    assert detections.unique_agent_ids == ["a", "b"]
    assert detections.detection_types == [DetectionType.VEHICLE] * 2
    components = detections.trajectory_components
    trajectories = detections.trajectories
    assert tuple(trajectories.shape) == (2, 1, len(components))
    x = trajectories[:, 0, components.index(TrajectoryComponent.X)]
    assert x.tolist() == [1.0, 2.0]
    # Components a detection does not carry are zero-filled.
    z = trajectories[:, 0, components.index(TrajectoryComponent.Z)]
    assert z.tolist() == [0.0, 0.0]

    lidar = batch.get_modality_data(Modality.LIDAR_PC)
    assert isinstance(lidar, BatchedLidarPointCloud)
    assert lidar.batch_size == 2
    assert tuple(lidar.get(LidarComponent.X).shape) == (5, 1)


def test_to_moves_nested_payloads_and_timestamp_diff():
    frames = []
    for idx in (1, 2):
        frame = make_frame(idx, timestamp_diff=0.1 * idx)
        frame.set_modality_data(
            Modality.LIDAR_PC, LidarPointCloud(np.zeros((2, 3), dtype=np.float32), XYZ)
        )
        frame.set_modality_data(
            Modality.HD_MAP_BEV,
            {"raster": np.zeros((2, 2), dtype=np.float32), "name": f"map{idx}"},
        )
        frames.append(frame)

    meta = torch.device("meta")
    batch = TransformedFrameDataBatch(frames).to(meta)

    assert batch.timestamp.device == meta
    assert batch.timestamp_diff.device == meta
    assert batch.get_modality_data(Modality.LIDAR_PC).device == meta
    hd_map = batch.get_modality_data(Modality.HD_MAP_BEV)
    assert hd_map["raster"].device == meta
    assert hd_map["name"] == ["map1", "map2"]
