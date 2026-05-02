import numpy as np
import pandas as pd
import pytest

from standard_e2e.caching.adapters import (
    CamerasIdentityAdapter,
    FutureStatesIdentityAdapter,
    HDMapBEVAdapter,
    IntentIdentityAdapter,
    LidarAdapter,
    LidarBEVAdapter,
    PanoImageAdapter,
    PastStatesIdentityAdapter,
    PreferenceTrajectoryAdapter,
    get_adapters_from_config,
)
from standard_e2e.caching.adapters.abstract_adapter import AbstractAdapter
from standard_e2e.constants import PREFERENCE_TRAJECTORIES_KEY
from standard_e2e.data_structures import (
    CameraData,
    HDMap,
    LidarData,
    LidarPointCloud,
    MapElement,
    StandardFrameData,
    Trajectory,
)
from standard_e2e.enums import (
    CameraDirection,
    LidarComponent,
    MapElementType,
    Modality,
)

# --- Helpers -----------------------------------------------------------------


def make_frame(**overrides) -> StandardFrameData:
    kwargs = dict(
        timestamp=0.0,
        frame_id=0,
        segment_id="seg0",
        dataset_name="ds",
        split="train",
    )
    kwargs.update(overrides)
    return StandardFrameData(**kwargs)


def make_camera(direction: CameraDirection, h=20, w=10, c=3) -> CameraData:
    image = (np.random.randint(0, 255, size=(h, w, c))).astype(np.uint8)
    intrinsics = np.eye(3, dtype=np.float32)
    extrinsics = np.eye(4, dtype=np.float32)
    return CameraData(
        camera_direction=direction,
        image=image,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )


# --- AbstractAdapter type validation -----------------------------------------


def test_transform_rejects_none():
    adapter = CamerasIdentityAdapter()
    with pytest.raises(ValueError):
        adapter.transform(None)  # type: ignore[arg-type]


def test_transform_rejects_wrong_type():
    adapter = CamerasIdentityAdapter()
    with pytest.raises(TypeError):
        adapter.transform(123)  # type: ignore[arg-type]


# --- Identity Adapters --------------------------------------------------------


def test_cameras_identity_adapter_returns_cameras_mapping():
    cam_dir = CameraDirection.FRONT
    frame = make_frame(cameras={cam_dir: make_camera(cam_dir)})
    adapter = CamerasIdentityAdapter()
    out = adapter.transform(frame)
    assert set(out.keys()) == {Modality.CAMERAS}
    assert out[Modality.CAMERAS][cam_dir].image.shape[0] == 20


def test_future_states_identity_adapter():
    traj = Trajectory()
    frame = make_frame(future_states=traj)
    adapter = FutureStatesIdentityAdapter()
    out = adapter.transform(frame)
    assert out == {Modality.FUTURE_STATES: traj}


def test_past_states_identity_adapter_missing_returns_empty():
    frame = make_frame()  # no past_states
    adapter = PastStatesIdentityAdapter()
    out = adapter.transform(frame)
    assert out == {}


def test_intent_identity_adapter_value_none_included():
    frame = make_frame(intent=None)
    adapter = IntentIdentityAdapter()
    out = adapter.transform(frame)
    # intent attribute exists so we expect a mapping even if value None
    assert len(out.keys()) == 0


# --- PreferenceTrajectoryAdapter ---------------------------------------------


def test_preference_trajectory_adapter_missing_key_returns_empty():
    frame = make_frame(aux_data={})
    adapter = PreferenceTrajectoryAdapter()
    out = adapter.transform(frame)
    assert out == {}


def test_preference_trajectory_adapter_with_key():
    trajectories = [Trajectory(), Trajectory()]
    frame = make_frame(aux_data={PREFERENCE_TRAJECTORIES_KEY: trajectories})
    adapter = PreferenceTrajectoryAdapter()
    out = adapter.transform(frame)
    assert out == {Modality.PREFERENCE_TRAJECTORY: trajectories}


# --- PanoImageAdapter ---------------------------------------------------------


def test_pano_image_adapter_concatenates_order_and_crops():
    # Create 3 cameras with different width to verify ordering
    h1, w1 = 20, 11
    h2, w2 = 20, 13
    h3, w3 = 20, 17
    max_size = 256
    top_cut_frac = 0.25
    cam_front_left = make_camera(CameraDirection.FRONT_LEFT, h=h1, w=w1)
    cam_front = make_camera(CameraDirection.FRONT, h=h2, w=w2)
    cam_front_right = make_camera(CameraDirection.FRONT_RIGHT, h=h3, w=w3)
    frame = make_frame(
        cameras={
            CameraDirection.FRONT_LEFT: cam_front_left,
            CameraDirection.FRONT: cam_front,
            CameraDirection.FRONT_RIGHT: cam_front_right,
        }
    )
    res_h = h1
    res_w = w1 + w2 + w3
    assert res_w >= res_h  # Sanity check
    adapter = PanoImageAdapter(top_cut_frac=top_cut_frac, max_size=max_size)
    res_h, res_w = int(np.ceil(res_h * max_size / res_w)), max_size
    res_h = int(np.ceil(res_h * (1 - top_cut_frac)))  # after crop
    out = adapter.transform(frame)
    assert set(out.keys()) == {Modality.CAMERAS}
    pano = out[Modality.CAMERAS]
    assert pano.shape[0] == res_h
    assert pano.shape[1] == res_w
    # Channels unchanged
    assert pano.shape[2] == 3


# --- get_adapters_from_config -------------------------------------------------


def test_get_adapters_from_config_success():
    configs = [
        {"name": "cameras_identity_adapter"},
        {"name": "future_states_identity_adapter"},
        {"name": "pano_adapter", "params": {"top_cut_frac": 0.1, "max_size": 128}},
    ]
    adapters = get_adapters_from_config(configs)
    assert len(adapters) == 3
    assert isinstance(adapters[0], CamerasIdentityAdapter)
    assert isinstance(adapters[1], FutureStatesIdentityAdapter)
    assert isinstance(adapters[2], PanoImageAdapter)


def test_get_adapters_from_config_unknown_name_raises():
    with pytest.raises(ValueError):
        get_adapters_from_config([{"name": "nope"}])


# --- Sanity: all adapters are subclasses of AbstractAdapter ------------------


def test_all_concrete_adapters_subclass_abstract():
    for cls in [
        CamerasIdentityAdapter,
        FutureStatesIdentityAdapter,
        IntentIdentityAdapter,
        PastStatesIdentityAdapter,
        PreferenceTrajectoryAdapter,
        PanoImageAdapter,
        LidarAdapter,
        LidarBEVAdapter,
        HDMapBEVAdapter,
    ]:
        assert issubclass(cls, AbstractAdapter)


# --- LidarAdapter ------------------------------------------------------------


def _lidar_frame(n: int) -> StandardFrameData:
    df = pd.DataFrame(
        np.arange(n * 3, dtype=np.float32).reshape(n, 3),
        columns=[c.value for c in LidarComponent],
    )
    return make_frame(lidar=LidarData(points=df))


def test_lidar_adapter_passthrough():
    out = LidarAdapter().transform(_lidar_frame(5))
    assert set(out.keys()) == {Modality.LIDAR_PC}
    pc = out[Modality.LIDAR_PC]
    assert isinstance(pc, LidarPointCloud)
    assert pc.num_points == 5
    assert pc.components == [LidarComponent.X, LidarComponent.Y, LidarComponent.Z]


def test_lidar_adapter_stride_subsample_is_deterministic():
    # 10 points, max_points=3 -> step = 10 // 3 = 3 -> rows [0, 3, 6].
    pc = LidarAdapter(max_points=3).transform(_lidar_frame(10))[Modality.LIDAR_PC]
    assert pc.num_points == 3
    expected = np.array([[0, 1, 2], [9, 10, 11], [18, 19, 20]], dtype=np.float32)
    assert np.array_equal(pc.points, expected)


def test_lidar_adapter_no_subsample_when_below_cap():
    pc = LidarAdapter(max_points=100).transform(_lidar_frame(7))[Modality.LIDAR_PC]
    assert pc.num_points == 7


def test_lidar_adapter_missing_lidar_returns_empty():
    assert LidarAdapter().transform(make_frame()) == {}


def test_lidar_adapter_invalid_max_points_raises():
    with pytest.raises(ValueError):
        LidarAdapter(max_points=0)
    with pytest.raises(ValueError):
        LidarAdapter(max_points=-1)


# --- LidarBEVAdapter ---------------------------------------------------------


def test_lidar_bev_adapter_default_shape_and_dtype():
    out = LidarBEVAdapter().transform(_lidar_frame(50))
    assert set(out.keys()) == {Modality.LIDAR_BEV}
    bev = out[Modality.LIDAR_BEV]
    assert bev.dtype == np.float32
    assert bev.shape == (1, 256, 256)


def test_lidar_bev_adapter_use_ground_plane_adds_channel():
    bev = LidarBEVAdapter(use_ground_plane=True).transform(_lidar_frame(50))[
        Modality.LIDAR_BEV
    ]
    assert bev.shape == (2, 256, 256)


def test_lidar_bev_adapter_count_cap_saturates():
    pts = np.zeros((20, 3), dtype=np.float32)
    pts[:, 2] = 1.0  # above default split_height
    df = pd.DataFrame(pts, columns=[c.value for c in LidarComponent])
    bev = LidarBEVAdapter(count_cap=5).transform(
        make_frame(lidar=LidarData(points=df))
    )[Modality.LIDAR_BEV]
    # 20 points clipped at 5, divided by 5 -> max == 1.0
    assert bev.max() == 1.0


def test_lidar_bev_adapter_missing_lidar_returns_empty():
    assert LidarBEVAdapter().transform(make_frame()) == {}


def test_lidar_bev_adapter_invalid_args_raise():
    with pytest.raises(ValueError):
        LidarBEVAdapter(min_x=10.0, max_x=10.0)
    with pytest.raises(ValueError):
        LidarBEVAdapter(pixels_per_meter=0.5)
    with pytest.raises(ValueError):
        LidarBEVAdapter(count_cap=0)


# --- HDMapBEVAdapter ---------------------------------------------------------


def _hdmap_frame(elements: list[MapElement]) -> StandardFrameData:
    return make_frame(hd_map=HDMap(elements=elements))


def _elem(
    type_: MapElementType,
    points: list[list[float]],
    is_closed: bool = False,
    id_: str = "x",
) -> MapElement:
    return MapElement(
        id=id_,
        type=type_,
        points=np.asarray(points, dtype=np.float32),
        is_closed=is_closed,
    )


def test_hdmap_bev_adapter_default_shape_and_empty_map():
    out = HDMapBEVAdapter().transform(_hdmap_frame([]))
    bev = out[Modality.HD_MAP_BEV]
    assert bev.dtype == np.float32
    # Default: 11 channels (one per MapElementType), 64 m square at 4 ppm.
    assert bev.shape == (len(MapElementType), 256, 256)
    assert bev.max() == 0.0


def test_hdmap_bev_adapter_renders_polyline_on_correct_channel():
    # Horizontal line along vehicle x at y=0; LANE_CENTER channel should be hot.
    line = _elem(MapElementType.LANE_CENTER, [[-10, 0, 0], [10, 0, 0]])
    out = HDMapBEVAdapter(
        include_types=[MapElementType.LANE_CENTER, MapElementType.CROSSWALK]
    ).transform(_hdmap_frame([line]))
    bev = out[Modality.HD_MAP_BEV]
    assert bev.shape == (2, 256, 256)
    assert bev[0].max() == 1.0  # LANE_CENTER drawn
    assert bev[1].max() == 0.0  # CROSSWALK channel untouched


def test_hdmap_bev_adapter_polygon_is_filled():
    # 5 m square crosswalk; fill should produce far more pixels than outline.
    poly = _elem(
        MapElementType.CROSSWALK,
        [[0, 0, 0], [5, 0, 0], [5, 5, 0], [0, 5, 0]],
        is_closed=True,
    )
    out = HDMapBEVAdapter(include_types=[MapElementType.CROSSWALK]).transform(
        _hdmap_frame([poly])
    )
    bev = out[Modality.HD_MAP_BEV]
    # 5m * 5m at 4 ppm = 20*20 = 400 pixels filled (interior).
    assert (bev[0] > 0).sum() > 300


def test_hdmap_bev_adapter_point_renders_as_circle():
    point = _elem(MapElementType.STOP_SIGN, [[0, 0, 0]])
    out = HDMapBEVAdapter(
        include_types=[MapElementType.STOP_SIGN], polyline_thickness=3
    ).transform(_hdmap_frame([point]))
    bev = out[Modality.HD_MAP_BEV]
    # Circle of radius 3 → ~29 pixels (>0).
    assert 1 < (bev[0] > 0).sum() < 100


def test_hdmap_bev_adapter_excluded_type_dropped():
    elem = _elem(MapElementType.SPEED_BUMP, [[-5, 0, 0], [5, 0, 0]])
    out = HDMapBEVAdapter(include_types=[MapElementType.LANE_CENTER]).transform(
        _hdmap_frame([elem])
    )
    bev = out[Modality.HD_MAP_BEV]
    assert bev.shape == (1, 256, 256)
    assert bev.max() == 0.0


def test_hdmap_bev_adapter_missing_hd_map_returns_empty():
    assert HDMapBEVAdapter().transform(make_frame()) == {}


def test_hdmap_bev_adapter_invalid_args_raise():
    with pytest.raises(ValueError):
        HDMapBEVAdapter(min_x=10.0, max_x=10.0)
    with pytest.raises(ValueError):
        HDMapBEVAdapter(pixels_per_meter=0.5)
    with pytest.raises(ValueError):
        HDMapBEVAdapter(polyline_thickness=0)
