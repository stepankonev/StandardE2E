import numpy as np
import pandas as pd
import pytest

from standard_e2e.data_structures.containers import CameraData, LidarData
from standard_e2e.enums import CameraDirection


def make_camera_data(**overrides) -> CameraData:
    kwargs = dict(
        camera_direction=CameraDirection.FRONT,
        image=np.zeros((32, 32, 3), dtype=np.uint8),
        intrinsics=np.eye(3, dtype=np.float32),
        extrinsics=np.eye(4, dtype=np.float32),
    )
    kwargs.update(overrides)
    return CameraData(**kwargs)


def test_camera_data_happy_path_and_coercion():
    # list intrinsics should be coerced to np.float32 array
    cam = make_camera_data(intrinsics=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert cam.intrinsics.shape == (3, 3)
    assert cam.extrinsics.shape == (4, 4)
    assert cam.image.dtype == np.uint8


@pytest.mark.parametrize(
    "bad_intrinsics",
    [np.zeros((2, 3), dtype=np.float32), np.zeros((3, 4), dtype=np.float32)],
)
def test_camera_data_bad_intrinsics_shape_raises(bad_intrinsics):
    with pytest.raises(ValueError):
        make_camera_data(intrinsics=bad_intrinsics)


@pytest.mark.parametrize(
    "bad_extrinsics",
    [np.zeros((3, 4), dtype=np.float32), np.zeros((4, 3), dtype=np.float32)],
)
def test_camera_data_bad_extrinsics_shape_raises(bad_extrinsics):
    with pytest.raises(ValueError):
        make_camera_data(extrinsics=bad_extrinsics)


def test_camera_data_bad_image_shape_raises():
    with pytest.raises(ValueError):
        make_camera_data(image=np.zeros((32, 32), dtype=np.uint8))  # 2D


def test_camera_data_bad_image_dtype_raises():
    with pytest.raises(ValueError):
        make_camera_data(image=np.zeros((32, 32, 3), dtype=np.float32))


def test_camera_data_runtime_assignment_validation():
    cam = make_camera_data()
    # Changing to wrong dtype image should raise at assignment
    with pytest.raises(ValueError):
        cam.image = np.zeros((32, 32, 3), dtype=np.float32)  # type: ignore
    # Correct assignment ok
    cam.image = np.ones((32, 32, 3), dtype=np.uint8)
    # Wrong intrinsics shape
    with pytest.raises(ValueError):
        cam.intrinsics = np.zeros((2, 2), dtype=np.float32)  # type: ignore


def test_camera_data_size_inferred_and_properties():
    cam = make_camera_data()
    assert cam.size == (32, 32)
    assert cam.H == 32 and cam.W == 32
    # shape consistency
    assert cam.shape[:2] == cam.size


def test_camera_data_size_explicit_matches():
    cam = make_camera_data(size=(32, 32))
    assert cam.size == (32, 32)


def test_camera_data_size_mismatch_raises():
    with pytest.raises(ValueError):
        make_camera_data(size=(16, 32))


def test_camera_data_size_inferred_from_image_shape():
    cam = make_camera_data(image=np.zeros((20, 40, 3), dtype=np.uint8))
    assert cam.size == (20, 40)


def test_camera_data_invalid_size_type_raises():
    with pytest.raises(ValueError):
        make_camera_data(size=(32, 32, 3))  # type: ignore[arg-type]


def test_camera_data_distortion_expand_three_to_five():
    cam = make_camera_data(distortion=[0.1, 0.2, 0.3])
    assert cam.distortion.shape == (5,)
    assert np.allclose(cam.distortion, [0.1, 0.2, 0.0, 0.0, 0.3])


def test_camera_data_distortion_accepts_five():
    vals = [0.1, 0.2, 0.01, 0.02, 0.3]
    cam = make_camera_data(distortion=vals)
    assert np.allclose(cam.distortion, vals)


def test_camera_data_distortion_accepts_four_fisheye():
    vals = [0.1, 0.2, 0.3, 0.4]
    cam = make_camera_data(distortion=vals)
    assert np.allclose(cam.distortion, vals)


@pytest.mark.parametrize("bad", [[0.1, 0.2], [0.1] * 6])
def test_camera_data_distortion_invalid_lengths_raise(bad):
    with pytest.raises(ValueError):
        make_camera_data(distortion=bad)


# ---------------------- LidarData tests ----------------------


def test_lidar_data_happy_path():
    df = pd.DataFrame({"x": [0.0], "y": [1.0], "z": [2.0]})
    lidar = LidarData(points=df)
    assert lidar.points.equals(df)


def test_lidar_data_missing_columns_raises():
    for cols in [{"x": [0]}, {"x": [0], "y": [1]}]:
        with pytest.raises(ValueError):
            LidarData(points=pd.DataFrame(cols))


def test_lidar_data_non_dataframe_raises():
    with pytest.raises(ValueError):
        LidarData(points=[{"x": 0, "y": 1, "z": 2}])  # type: ignore


def test_lidar_data_runtime_assignment_validation():
    df = pd.DataFrame({"x": [0.0], "y": [1.0], "z": [2.0]})
    lidar = LidarData(points=df)
    with pytest.raises(ValueError):
        lidar.points = pd.DataFrame({"x": [0.0], "y": [1.0]})  # type: ignore
