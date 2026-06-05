"""Source-dataset processor for the Argoverse 2 (AV2) sensor dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
from av2.geometry.interpolate import (
    NUM_CENTERLINE_INTERP_PTS,
    compute_midpoint_line,
)
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.sweep import Sweep
from av2.utils.io import read_img

from standard_e2e.caching import SourceDatasetProcessor
from standard_e2e.caching.adapters import (
    AbstractAdapter,
    Detections3DIdentityAdapter,
    HDMapBEVAdapter,
    LidarAdapter,
    PanoImageAdapter,
)
from standard_e2e.caching.segment_context import (
    FutureDetectionsAggregator,
    FuturePastStatesFromMatricesAggregator,
    SegmentContextAggregator,
)
from standard_e2e.data_structures import (
    CameraData,
    Detection3D,
    FrameDetections3D,
    HDMap,
    LidarData,
    MapElement,
    StandardFrameData,
    Trajectory,
)
from standard_e2e.enums import (
    CameraDirection,
    DetectionType,
    LidarComponent,
    MapElementType,
    StandardFrameDataField,
)
from standard_e2e.enums import TrajectoryComponent as TC
from standard_e2e.indexing import IndexDataGenerator
from standard_e2e.utils import (
    intrinsics_matrix,
    matrix_to_xyz_heading,
    quat_wxyz_to_rotmat,
    se3,
)

# AV2 ring cameras → our canonical CameraDirection. Stereo cameras are
# excluded: they run at 5 Hz and do not synchronise 1:1 with the 10 Hz
# lidar sweeps that anchor each frame.
_RING_CAMERA_TO_DIRECTION: dict[str, CameraDirection] = {
    "ring_front_center": CameraDirection.FRONT,
    "ring_front_left": CameraDirection.FRONT_LEFT,
    "ring_front_right": CameraDirection.FRONT_RIGHT,
    "ring_side_left": CameraDirection.SIDE_LEFT,
    "ring_side_right": CameraDirection.SIDE_RIGHT,
    "ring_rear_left": CameraDirection.REAR_LEFT,
    "ring_rear_right": CameraDirection.REAR_RIGHT,
}

# AV2 cuboid category → our canonical DetectionType.
_AV2_CATEGORY_TO_DETECTION_TYPE: dict[str, DetectionType] = {
    "REGULAR_VEHICLE": DetectionType.VEHICLE,
    "LARGE_VEHICLE": DetectionType.VEHICLE,
    "BUS": DetectionType.VEHICLE,
    "BOX_TRUCK": DetectionType.VEHICLE,
    "TRUCK": DetectionType.VEHICLE,
    "VEHICULAR_TRAILER": DetectionType.VEHICLE,
    "TRUCK_CAB": DetectionType.VEHICLE,
    "SCHOOL_BUS": DetectionType.VEHICLE,
    "ARTICULATED_BUS": DetectionType.VEHICLE,
    "MESSAGE_BOARD_TRAILER": DetectionType.VEHICLE,
    "RAILED_VEHICLE": DetectionType.VEHICLE,
    "MOTORCYCLE": DetectionType.VEHICLE,
    "MOTORCYCLIST": DetectionType.VEHICLE,
    "BICYCLE": DetectionType.BICYCLE,
    "BICYCLIST": DetectionType.BICYCLE,
    "WHEELED_DEVICE": DetectionType.BICYCLE,
    "WHEELED_RIDER": DetectionType.BICYCLE,
    "PEDESTRIAN": DetectionType.PEDESTRIAN,
    "WHEELCHAIR": DetectionType.PEDESTRIAN,
    "STROLLER": DetectionType.PEDESTRIAN,
    "OFFICIAL_SIGNALER": DetectionType.PEDESTRIAN,
    "ANIMAL": DetectionType.PEDESTRIAN,
    "DOG": DetectionType.PEDESTRIAN,
    "SIGN": DetectionType.SIGN,
    "STOP_SIGN": DetectionType.SIGN,
    "CONSTRUCTION_CONE": DetectionType.UNKNOWN,
    "CONSTRUCTION_BARREL": DetectionType.UNKNOWN,
    "BOLLARD": DetectionType.UNKNOWN,
    "MOBILE_PEDESTRIAN_CROSSING_SIGN": DetectionType.UNKNOWN,
    "TRAFFIC_LIGHT_TRAILER": DetectionType.UNKNOWN,
}

# AV2 LaneType.{VEHICLE, BIKE, BUS} -> our normalised lane_type string.
# `str(LaneType.VEHICLE) == "LaneType.VEHICLE"` so we key on the full string.
_AV2_LANE_TYPE_TO_NORMALISED: dict[str, str] = {
    "LaneType.VEHICLE": "vehicle",
    "LaneType.BIKE": "bike",
    "LaneType.BUS": "bus",
}

# AV2 LaneMarkType -> (paint_color, paint_pattern).
# NONE and UNKNOWN map to (None, None) -> no LANE_BOUNDARY element emitted.
# Side-of-solid for SOLID_DASH_* / DASH_SOLID_* is preserved only via
# `attrs.paint_subtype_raw`; the 2-tuple is the normalised form. See
# docs/lane_paint_comparison.md.
_AV2_MARK_TYPE_TO_PAINT: dict[str, tuple[Optional[str], Optional[str]]] = {
    "LaneMarkType.DASHED_WHITE": ("white", "dashed"),
    "LaneMarkType.SOLID_WHITE": ("white", "solid"),
    "LaneMarkType.DOUBLE_SOLID_WHITE": ("white", "double_solid"),
    "LaneMarkType.DOUBLE_DASH_WHITE": ("white", "double_dashed"),
    "LaneMarkType.SOLID_DASH_WHITE": ("white", "solid_dashed"),
    "LaneMarkType.DASH_SOLID_WHITE": ("white", "solid_dashed"),
    "LaneMarkType.DASHED_YELLOW": ("yellow", "dashed"),
    "LaneMarkType.SOLID_YELLOW": ("yellow", "solid"),
    "LaneMarkType.DOUBLE_SOLID_YELLOW": ("yellow", "double_solid"),
    "LaneMarkType.DOUBLE_DASH_YELLOW": ("yellow", "double_dashed"),
    "LaneMarkType.SOLID_DASH_YELLOW": ("yellow", "solid_dashed"),
    "LaneMarkType.DASH_SOLID_YELLOW": ("yellow", "solid_dashed"),
    "LaneMarkType.SOLID_BLUE": ("blue", "solid"),
    "LaneMarkType.NONE": (None, None),
    "LaneMarkType.UNKNOWN": (None, None),
}


def _quat_wxyz_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """AV2 stores Hamilton quaternions as (qw, qx, qy, qz)."""
    return quat_wxyz_to_rotmat((qw, qx, qy, qz)).astype(np.float32)


def _se3_from_quat_translation(
    qw: float, qx: float, qy: float, qz: float, tx: float, ty: float, tz: float
) -> np.ndarray:
    return se3(_quat_wxyz_to_rotmat(qw, qx, qy, qz), (tx, ty, tz), dtype=np.float32)


class Av2SensorDatasetProcessor(SourceDatasetProcessor):
    """Processor for the Argoverse 2 sensor dataset.

    A "frame" is a single lidar sweep timestamp; ring-camera images are
    matched by nearest timestamp per camera, and 3D box annotations are
    filtered for that exact sweep timestamp (AV2 annotates one cuboid
    per sweep, not per camera frame). One log = one segment.

    Per-log state (calibration, ego-pose table, full annotations table,
    static map) is read once and reused across every sweep of the log;
    the cache is keyed by ``log_dir`` so workers that hop between logs
    only reload when they actually transition.
    """

    DATASET_NAME = "av2_sensor"

    def __init__(
        self,
        common_output_path: str,
        split: str,
        index_data_generator: IndexDataGenerator | None = None,
        adapters: list[AbstractAdapter] | None = None,
        context_aggregators: list[SegmentContextAggregator] | None = None,
    ):
        super().__init__(
            common_output_path=common_output_path,
            split=split,
            index_data_generator=index_data_generator,
            adapters=adapters,
            context_aggregators=context_aggregators,
        )
        self._cached_log_dir: Optional[Path] = None
        self._camera_intrinsics: dict[str, np.ndarray] = {}
        self._camera_distortion: dict[str, np.ndarray] = {}
        self._camera_extrinsics: dict[str, np.ndarray] = {}
        self._camera_image_paths: dict[str, list[tuple[int, Path]]] = {}
        self._ego_pose_by_ts: dict[int, np.ndarray] = {}
        self._annotations_by_ts: dict[int, pd.DataFrame] = {}
        self._map: Optional[ArgoverseStaticMap] = None

    @property
    def allowed_splits(self) -> list[str]:
        return ["train", "val", "test"]

    def _get_default_adapters(self) -> list[AbstractAdapter]:
        return [
            PanoImageAdapter(),
            LidarAdapter(),
            HDMapBEVAdapter(),
            Detections3DIdentityAdapter(),
        ]

    def _get_default_context_aggregators(self):
        return [
            FuturePastStatesFromMatricesAggregator(self.output_path),
            FutureDetectionsAggregator(self.output_path),
        ]

    @property
    def dataset_name(self) -> str:
        return self.DATASET_NAME

    # --- per-log cache --------------------------------------------------------

    def _refresh_log_cache(self, log_dir: Path) -> None:
        if self._cached_log_dir == log_dir:
            return
        logging.info("Loading per-log AV2 state for %s", log_dir.name)

        intrinsics_df = pd.read_feather(log_dir / "calibration" / "intrinsics.feather")
        extrinsics_df = pd.read_feather(
            log_dir / "calibration" / "egovehicle_SE3_sensor.feather"
        )
        self._camera_intrinsics.clear()
        self._camera_distortion.clear()
        self._camera_extrinsics.clear()
        for raw_row in intrinsics_df.itertuples(index=False):
            row = cast(Any, raw_row)
            K = intrinsics_matrix(row.fx_px, row.fy_px, row.cx_px, row.cy_px)
            self._camera_intrinsics[row.sensor_name] = K
            # AV2 ships 3 radial distortion coefficients (k1, k2, k3); CameraData
            # expands them into the 5-term Brown-Conrady form with zero tangential.
            self._camera_distortion[row.sensor_name] = np.array(
                [row.k1, row.k2, row.k3], dtype=np.float32
            )
        for raw_row in extrinsics_df.itertuples(index=False):
            row = cast(Any, raw_row)
            self._camera_extrinsics[row.sensor_name] = _se3_from_quat_translation(
                row.qw, row.qx, row.qy, row.qz, row.tx_m, row.ty_m, row.tz_m
            )

        self._camera_image_paths.clear()
        cameras_root = log_dir / "sensors" / "cameras"
        for cam_name in _RING_CAMERA_TO_DIRECTION:
            cam_dir = cameras_root / cam_name
            if not cam_dir.is_dir():
                continue
            entries = sorted(
                ((int(p.stem), p) for p in cam_dir.glob("*.jpg")), key=lambda x: x[0]
            )
            self._camera_image_paths[cam_name] = entries

        ego_df = pd.read_feather(log_dir / "city_SE3_egovehicle.feather")
        self._ego_pose_by_ts = {}
        for raw_row in ego_df.itertuples(index=False):
            row = cast(Any, raw_row)
            self._ego_pose_by_ts[int(row.timestamp_ns)] = _se3_from_quat_translation(
                row.qw, row.qx, row.qy, row.qz, row.tx_m, row.ty_m, row.tz_m
            )

        ann_df = pd.read_feather(log_dir / "annotations.feather")
        self._annotations_by_ts = {
            int(cast(Any, ts)): group for ts, group in ann_df.groupby("timestamp_ns")
        }

        self._map = ArgoverseStaticMap.from_map_dir(log_dir / "map", build_raster=False)
        self._cached_log_dir = log_dir

    # --- per-frame helpers ---------------------------------------------------

    @staticmethod
    def _nearest_image_path(
        sweep_ts_ns: int, sorted_entries: list[tuple[int, Path]]
    ) -> Path:
        timestamps = np.fromiter(
            (ts for ts, _ in sorted_entries), dtype=np.int64, count=len(sorted_entries)
        )
        idx = int(np.argmin(np.abs(timestamps - sweep_ts_ns)))
        return sorted_entries[idx][1]

    def _build_camera_dict(
        self, log_dir: Path, sweep_ts_ns: int
    ) -> dict[CameraDirection, CameraData]:
        cameras: dict[CameraDirection, CameraData] = {}
        for cam_name, direction in _RING_CAMERA_TO_DIRECTION.items():
            entries = self._camera_image_paths.get(cam_name)
            if not entries:
                continue
            img_path = self._nearest_image_path(sweep_ts_ns, entries)
            image = read_img(img_path, channel_order="RGB")
            cameras[direction] = CameraData(
                image=np.asarray(image, dtype=np.uint8),
                camera_direction=direction,
                intrinsics=self._camera_intrinsics[cam_name],
                extrinsics=self._camera_extrinsics[cam_name],
                distortion=self._camera_distortion[cam_name],
            )
        return cameras

    def _build_detections(
        self, sweep_ts_ns: int, timestamp_s: float
    ) -> list[Detection3D]:
        rows = self._annotations_by_ts.get(sweep_ts_ns)
        if rows is None or len(rows) == 0:
            return []
        detections: list[Detection3D] = []
        for raw_row in rows.itertuples(index=False):
            row = cast(Any, raw_row)
            R = _quat_wxyz_to_rotmat(row.qw, row.qx, row.qy, row.qz)
            heading = float(np.arctan2(R[1, 0], R[0, 0]))
            detections.append(
                Detection3D(
                    unique_agent_id=str(row.track_uuid),
                    detection_type=_AV2_CATEGORY_TO_DETECTION_TYPE.get(
                        str(row.category), DetectionType.UNKNOWN
                    ),
                    trajectory=Trajectory(
                        {
                            TC.TIMESTAMP: [timestamp_s],
                            TC.X: [float(row.tx_m)],
                            TC.Y: [float(row.ty_m)],
                            TC.Z: [float(row.tz_m)],
                            TC.HEADING: [heading],
                            TC.LENGTH: [float(row.length_m)],
                            TC.WIDTH: [float(row.width_m)],
                            TC.HEIGHT: [float(row.height_m)],
                        }
                    ),
                )
            )
        return detections

    def _build_hd_map(self, T_city_from_ego: np.ndarray) -> Optional[HDMap]:
        if self._map is None:
            return None
        # See docs/map_element_translation.md §5.2 for the per-source rules.
        # 2D city->ego affine: AV2's vector Z is sometimes NaN outside the
        # 5 m drivable-area ROI; slicing to XY first sidesteps the NaN
        # entirely and matches AV2's own rasterizer convention
        # (DrivableAreaMapLayer.from_vector_data at map_api.py:251).
        T_ego_from_city = np.linalg.inv(T_city_from_ego).astype(np.float32)
        R_xy = T_ego_from_city[:2, :2]
        t_xy = T_ego_from_city[:2, 3]

        def _to_ego_xy(points_xyz_city: np.ndarray) -> np.ndarray:
            xy_city = np.asarray(points_xyz_city, dtype=np.float32)[:, :2]
            return xy_city @ R_xy.T + t_xy

        elements: list[MapElement] = []

        # ---- LANE_CENTER + LANE_BOUNDARY (paint only) per lane segment ----
        for ls in self._map.vector_lane_segments.values():
            lane_type = _AV2_LANE_TYPE_TO_NORMALISED.get(str(ls.lane_type))
            common_lane_attrs: dict[str, Any] = {
                "is_intersection": bool(ls.is_intersection),
            }
            if lane_type is not None:
                common_lane_attrs["lane_type"] = lane_type

            # Centerline derived via the same algorithm `get_lane_segment_centerline`
            # uses (compute_midpoint_line: arc-length-resample both boundaries
            # to NUM_CENTERLINE_INTERP_PTS, take midpoints) BUT with XY-only
            # inputs. Calling the public API on the raw (N, 3) boundaries
            # would propagate NaN Z values into the centerline XY.
            left_xy = np.asarray(ls.left_lane_boundary.xyz, dtype=np.float64)[:, :2]
            right_xy = np.asarray(ls.right_lane_boundary.xyz, dtype=np.float64)[:, :2]
            centerline_xy, _ = compute_midpoint_line(
                left_xy, right_xy, num_interp_pts=NUM_CENTERLINE_INTERP_PTS
            )
            # centerline is already in city XY; transform to ego frame
            centerline_xy_ego = centerline_xy.astype(np.float32) @ R_xy.T + t_xy
            elements.append(
                MapElement(
                    id=f"lane_center_{ls.id}",
                    type=MapElementType.LANE_CENTER,
                    points=centerline_xy_ego,
                    is_closed=False,
                    successor_ids=[str(s) for s in ls.successors],
                    predecessor_ids=[str(p) for p in ls.predecessors],
                    left_neighbor_id=(
                        str(ls.left_neighbor_id)
                        if ls.left_neighbor_id is not None
                        else None
                    ),
                    right_neighbor_id=(
                        str(ls.right_neighbor_id)
                        if ls.right_neighbor_id is not None
                        else None
                    ),
                    attrs=common_lane_attrs,
                )
            )

            # Paint elements only when mark_type != NONE
            for side, boundary, mark_type in (
                ("left", ls.left_lane_boundary, ls.left_mark_type),
                ("right", ls.right_lane_boundary, ls.right_mark_type),
            ):
                paint_color, paint_pattern = _AV2_MARK_TYPE_TO_PAINT.get(
                    str(mark_type), (None, None)
                )
                if paint_color is None and paint_pattern is None:
                    # NONE / UNKNOWN -> don't emit a LANE_BOUNDARY element
                    continue
                paint_attrs: dict[str, Any] = {
                    "paint_color": paint_color,
                    "paint_pattern": paint_pattern,
                    "paint_subtype_raw": str(mark_type).split(".")[-1],
                }
                elements.append(
                    MapElement(
                        id=f"lane_paint_{ls.id}_{side}",
                        type=MapElementType.LANE_BOUNDARY,
                        points=_to_ego_xy(boundary.xyz),
                        is_closed=False,
                        attrs=paint_attrs,
                    )
                )

            # INTERSECTION polygon for is_intersection=True lane segments
            if ls.is_intersection:
                elements.append(
                    MapElement(
                        id=f"intersection_{ls.id}",
                        type=MapElementType.INTERSECTION,
                        points=_to_ego_xy(ls.polygon_boundary),
                        is_closed=True,
                        attrs={"source_lane_id": str(ls.id)},
                    )
                )

        # ---- DRIVABLE_AREA ----
        for da in self._map.vector_drivable_areas.values():
            elements.append(
                MapElement(
                    id=f"drivable_{da.id}",
                    type=MapElementType.DRIVABLE_AREA,
                    points=_to_ego_xy(da.xyz),
                    is_closed=True,
                )
            )

        # ---- CROSSWALK ----
        for pc in self._map.vector_pedestrian_crossings.values():
            elements.append(
                MapElement(
                    id=f"crosswalk_{pc.id}",
                    type=MapElementType.CROSSWALK,
                    points=_to_ego_xy(pc.polygon),
                    is_closed=True,
                )
            )

        return HDMap(elements=elements)

    # --- main entry point ----------------------------------------------------

    def _prepare_standardized_frame_data(
        self, raw_frame_data: Any
    ) -> StandardFrameData:
        log_dir, sweep_ts_ns = raw_frame_data
        self._refresh_log_cache(log_dir)
        timestamp_s = sweep_ts_ns / 1e9

        if self.needs_attr(StandardFrameDataField.LIDAR):
            sweep_path = log_dir / "sensors" / "lidar" / f"{sweep_ts_ns}.feather"
            sweep = Sweep.from_feather(sweep_path)
            lidar: Optional[LidarData] = LidarData(
                points=pd.DataFrame(
                    np.asarray(sweep.xyz, dtype=np.float32),
                    columns=[c.value for c in LidarComponent],
                )
            )
        else:
            lidar = None

        T_city_from_ego = self._ego_pose_by_ts[sweep_ts_ns]
        x, y, z, heading = matrix_to_xyz_heading(T_city_from_ego)

        cameras = (
            self._build_camera_dict(log_dir, sweep_ts_ns)
            if self.needs_attr(StandardFrameDataField.CAMERAS)
            else {}
        )
        detections = (
            self._build_detections(sweep_ts_ns, timestamp_s)
            if self.needs_attr(StandardFrameDataField.FRAME_DETECTIONS_3D)
            else []
        )
        hd_map = (
            self._build_hd_map(T_city_from_ego)
            if self.needs_attr(StandardFrameDataField.HD_MAP)
            else None
        )

        return StandardFrameData(
            dataset_name=self.dataset_name,
            segment_id=log_dir.name,
            frame_id=int(sweep_ts_ns),
            timestamp=timestamp_s,
            split=self._split,
            global_position=Trajectory(
                {
                    TC.TIMESTAMP: [timestamp_s],
                    TC.X: [x],
                    TC.Y: [y],
                    TC.Z: [z],
                    TC.HEADING: [heading],
                }
            ),
            cameras=cameras,
            lidar=lidar,
            hd_map=hd_map,
            frame_detections_3d=FrameDetections3D(detections=detections),
            aux_data={"pose_matrix": T_city_from_ego},
            extra_index_data={"log_id": log_dir.name},
        )
