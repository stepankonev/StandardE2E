import numpy as np
import pytest
import torch

from standard_e2e.data_structures import (
    CameraData,
    Detection3D,
    FrameDetections3D,
    LidarPointCloud,
    Trajectory,
    TransformedFrameData,
)
from standard_e2e.dataset_utils.modality_defaults import IntentDefaults
from standard_e2e.enums import (
    CameraDirection,
    DetectionType,
    Intent,
    LidarComponent,
    Modality,
    TrajectoryComponent,
)
from standard_e2e.training import (
    BatchedCameraData,
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


def test_modality_none_in_every_frame_collates_to_none():
    frames = [make_frame(1), make_frame(2)]
    for frame in frames:
        frame.set_modality_data(Modality.INTENT, None)
    batch = TransformedFrameDataBatch(frames)
    assert batch.get_modality_data(Modality.INTENT) is None


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
    # One row per detection of the batch; batch_idx maps rows to frames.
    assert detections.batch_size == 2
    assert detections.batch_idx.tolist() == [0, 0, 1]
    assert detections.unique_agent_ids == ["a", "b", "c"]
    assert detections.detection_types == [DetectionType.VEHICLE] * 3
    components = detections.trajectory_components
    trajectories = detections.trajectories
    assert tuple(trajectories.shape) == (3, 1, len(components))
    x = trajectories[:, 0, components.index(TrajectoryComponent.X)]
    assert x.tolist() == [1.0, 2.0, 3.0]
    # Components a detection does not carry are zero-filled.
    z = trajectories[:, 0, components.index(TrajectoryComponent.Z)]
    assert z.tolist() == [0.0, 0.0, 0.0]
    assert detections.is_valid.tolist() == [[True]] * 3

    lidar = batch.get_modality_data(Modality.LIDAR_PC)
    assert isinstance(lidar, BatchedLidarPointCloud)
    assert lidar.batch_size == 2
    assert tuple(lidar.get(LidarComponent.X).shape) == (5, 1)


def test_detections_pad_in_time_and_skip_frames_without_detections():
    f1 = make_frame(1)
    f1.set_modality_data(Modality.DETECTIONS_3D, FrameDetections3D(detections=[]))
    f2 = make_frame(2)
    f2.set_modality_data(Modality.DETECTIONS_3D, None)
    f3 = make_frame(3)
    # An aggregated future track: two steps, no TIMESTAMP.
    track = Detection3D(
        unique_agent_id="b",
        detection_type=DetectionType.PEDESTRIAN,
        trajectory=Trajectory({TrajectoryComponent.X: [1.0, 2.0]}),
    )
    f3.set_modality_data(
        Modality.DETECTIONS_3D,
        FrameDetections3D(detections=[_detection("a", 1.0), track]),
    )

    batch = TransformedFrameDataBatch([f1, f2, f3])

    detections = batch.get_modality_data(Modality.DETECTIONS_3D)
    assert detections.batch_size == 3
    # The frames without detections, and the one lacking the modality, have
    # no rows.
    assert detections.batch_idx.tolist() == [2, 2]
    assert detections.unique_agent_ids == ["a", "b"]
    assert detections.detection_types == [
        DetectionType.VEHICLE,
        DetectionType.PEDESTRIAN,
    ]
    components = detections.trajectory_components
    assert tuple(detections.trajectories.shape) == (2, 2, len(components))
    x = detections.trajectories[..., components.index(TrajectoryComponent.X)]
    assert x.tolist() == [[1.0, 0.0], [1.0, 2.0]]
    assert detections.is_valid.tolist() == [[True, False], [True, True]]

    empty = BatchedFrameDetections3D([FrameDetections3D(detections=[])])
    assert empty.batch_size == 1
    assert tuple(empty.batch_idx.shape) == (0,)
    assert tuple(empty.trajectories.shape) == (0, 0, len(components))
    assert tuple(empty.is_valid.shape) == (0, 0)


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
        frame.set_modality_data(
            Modality.DETECTIONS_3D, FrameDetections3D(detections=[_detection("a", 1.0)])
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
    detections = batch.get_modality_data(Modality.DETECTIONS_3D)
    assert detections.device == meta
    assert detections.trajectories.device == meta
    assert detections.is_valid.device == meta
    assert detections.batch_idx.device == meta


def test_modality_none_in_some_frames_batches_as_empty_element():
    # The frame lacking the modalities comes first: torch would dispatch on
    # its None.
    f1 = make_frame(1)
    f1.set_modality_data(Modality.FUTURE_STATES, None)
    f1.set_modality_data(Modality.LIDAR_PC, None)
    f2 = make_frame(2)
    f2.set_modality_data(
        Modality.FUTURE_STATES, Trajectory({TrajectoryComponent.X: [0.5, 0.25]})
    )
    f2.set_modality_data(
        Modality.LIDAR_PC, LidarPointCloud(np.ones((2, 3), dtype=np.float32), XYZ)
    )

    batch = TransformedFrameDataBatch([f1, f2])

    future = batch.get_modality_data(Modality.FUTURE_STATES)
    assert isinstance(future, BatchedTrajectory)
    assert future.is_empty_mask.tolist() == [True, False]
    assert future.get(TrajectoryComponent.X)[:, :, 0].tolist() == [[0, 0], [0.5, 0.25]]
    lidar = batch.get_modality_data(Modality.LIDAR_PC)
    assert isinstance(lidar, BatchedLidarPointCloud)
    assert lidar.batch_size == 2
    assert lidar.batch_idx.tolist() == [1, 1]
    assert lidar.components == XYZ


def test_modality_none_in_some_frames_without_empty_element_raises():
    f1 = make_frame(1)
    f1.set_modality_data(Modality.HD_MAP_BEV, None)
    f2 = make_frame(2)
    f2.set_modality_data(Modality.HD_MAP_BEV, np.zeros((1, 2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="ModalityDefaults") as info:
        TransformedFrameDataBatch([f1, f2])
    # The traceback names the modality and, per payload, the frames' files.
    note = "\n".join(info.value.__notes__)
    assert "HD_MAP_BEV" in note
    assert "1 with None: ds/train/seg1_1.npz" in note
    assert "1 with ndarray float32 (1, 2, 2): ds/train/seg2_2.npz" in note


def test_arrays_of_different_shapes_raise_with_the_policy():
    f1 = make_frame(1)
    f1.set_modality_data(Modality.CAMERAS, np.zeros((4, 12, 3), dtype=np.uint8))
    f2 = make_frame(2)
    f2.set_modality_data(Modality.CAMERAS, np.zeros((3, 12, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="not resize at batch time") as info:
        TransformedFrameDataBatch([f1, f2])
    note = "\n".join(info.value.__notes__)
    assert "CAMERAS" in note
    assert "1 with ndarray uint8 (4, 12, 3): ds/train/seg1_1.npz" in note
    assert "1 with ndarray uint8 (3, 12, 3): ds/train/seg2_2.npz" in note


def test_batched_trajectory_resample_with_none_entries():
    batch = BatchedTrajectory(
        [
            None,
            Trajectory(
                {
                    TrajectoryComponent.TIMESTAMP: [0.0, 1.0],
                    TrajectoryComponent.X: [0.0, 1.0],
                }
            ),
        ]
    )
    resampled = batch.resample([0.0, 0.5, 1.0])
    assert resampled.is_empty_mask.tolist() == [True, False]
    assert resampled.get(TrajectoryComponent.X)[1, :, 0].tolist() == [0.0, 0.5, 1.0]


def _camera(direction, height=4, width=6, channels=3, distortion=None) -> CameraData:
    return CameraData(
        camera_direction=direction,
        image=np.ones((height, width, channels), dtype=np.uint8),
        intrinsics=np.eye(3),
        extrinsics=np.eye(4),
        distortion=distortion,
    )


def test_identity_cameras_collate_to_padded_batched_camera_data():
    front, rear = CameraDirection.FRONT, CameraDirection.REAR
    # The frame without cameras comes first: unlike torch's default collate,
    # the dispatch must not depend on the first sample's type.
    f1 = make_frame(1)
    f1.set_modality_data(Modality.CAMERAS, None)
    f2 = make_frame(2)
    f2.set_modality_data(
        Modality.CAMERAS, {front: _camera(front, 4, 6, distortion=[1, 2, 3, 4, 5])}
    )
    f3 = make_frame(3)
    f3.set_modality_data(
        Modality.CAMERAS, {rear: _camera(rear), front: _camera(front, 3, 8)}
    )

    batch = TransformedFrameDataBatch([f1, f2, f3])

    cameras = batch.get_modality_data(Modality.CAMERAS)
    # The union of the frames' directions, in CameraDirection order.
    assert list(cameras) == [front, rear]
    batched = cameras[front]
    assert isinstance(batched, BatchedCameraData)
    assert batched.is_empty_mask.tolist() == [True, False, False]
    # Images sit top-left, zero-padded to the largest height and width.
    expected_image = torch.zeros((3, 4, 8, 3), dtype=torch.uint8)
    expected_image[1, :4, :6] = 1
    expected_image[2, :3, :8] = 1
    assert torch.equal(batched.image, expected_image)
    assert batched.size.tolist() == [[0, 0], [4, 6], [3, 8]]
    assert not batched.intrinsics[0].any()
    assert torch.equal(batched.intrinsics[2], torch.eye(3))
    assert tuple(batched.extrinsics.shape) == (3, 4, 4)
    # A camera without a distortion is zero-filled.
    assert batched.distortion.tolist() == [[0] * 5, [1, 2, 3, 4, 5], [0] * 5]
    assert batched.is_fisheye.tolist() == [False, False, False]
    assert cameras[rear].is_empty_mask.tolist() == [True, True, False]
    assert cameras[rear].distortion is None

    meta = torch.device("meta")
    batch.to(meta)
    assert batched.image.device == meta
    assert batched.distortion.device == meta


def test_batched_camera_data_rejects_invalid_batches():
    front = CameraDirection.FRONT
    with pytest.raises(ValueError):  # 4-term fisheye with 5-term Brown-Conrady
        BatchedCameraData(
            [_camera(front, distortion=[0.1] * 4), _camera(front, distortion=[0.1] * 5)]
        )
    with pytest.raises(ValueError):  # grayscale with RGB
        BatchedCameraData([_camera(front), _camera(front, channels=1)])
    with pytest.raises(TypeError):
        BatchedCameraData([_camera(front), object()])
    with pytest.raises(ValueError):
        BatchedCameraData([None, None])
