"""Source-dataset processor for MUHOS.

MUHOS (M3S Kuura, University of Oulu; ICSA-C 2026, CC BY 4.0) records a Toyota RAV4
around Oulu, Finland, in seven driving sets -- city centre, residential, gravel and
bumpy gravel, paved roads at 60 and 80 km/h and a highway -- with a roof-mounted
MultiSense S27 (colour camera + rectified stereo pair, 25 Hz), a Hesai OT128 lidar
(10 Hz, with an IMU), a PCAN-GPS receiver and the car's CAN bus. It ships no annotations
and no calibration.

Mapping to ``StandardFrameData``:

* One **frame** = one lidar sweep with the colour image taken during it (see
  :mod:`.muhos_dataset_converter`); its timestamp is the image's capture time on the
  sensor clock. One **segment** = one driving set.
* **cameras**: the S27 colour camera as ``CameraDirection.FRONT`` -- the raw wide-angle
  image (``is_fisheye``; Kannala-Brandt ``D``), with the intrinsics and extrinsics
  StandardE2E estimated from the data (:mod:`._muhos_calibration`).
* **lidar**: the sweep in the ego frame (the lidar frame levelled, origin on the road
  below the lidar), each point moved from its own capture time to the frame time with
  the ego trajectory; missing returns and returns from the car's own body dropped.
* **global_position** / ``aux_data["pose_matrix"]``: the fused ego pose
  (:mod:`._muhos_ego`, the CAN/GPS logs mapped onto the sensor clock) in a per-set local
  east/north frame, z = 0; speed from the scaled wheel speed. Past/future ego
  trajectories come from :class:`FuturePastStatesFromMatricesAggregator`.
  Latitude/longitude and the original stamps go to the index.

Not ingested: the stereo pair (no StandardE2E slot yet; used for the calibration), the
stereo depth and cost images (the depth was saved as a per-image min-max-normalised
colour map, so it is not metric), the thermal cameras (absent from the release), the
IMUs (the lidar's only aligns the clocks) and the other CAN signals. MUHOS has no 3D
boxes, so ``frame_detections_3d`` is empty.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from pyproj import Transformer

from standard_e2e.caching import SourceDatasetProcessor
from standard_e2e.caching.adapters import (
    AbstractAdapter,
    CamerasIdentityAdapter,
    LidarAdapter,
)
from standard_e2e.caching.segment_context import (
    FuturePastStatesFromMatricesAggregator,
    SegmentContextAggregator,
)
from standard_e2e.caching.src_datasets.muhos import _muhos_calibration as calib
from standard_e2e.caching.src_datasets.muhos import _muhos_io as io
from standard_e2e.caching.src_datasets.muhos.muhos_dataset_converter import FrameRef
from standard_e2e.data_structures import (
    CameraData,
    FrameDetections3D,
    LidarData,
    StandardFrameData,
    Trajectory,
)
from standard_e2e.enums import CameraDirection, LidarComponent, StandardFrameDataField
from standard_e2e.enums import TrajectoryComponent as TC
from standard_e2e.indexing import IndexDataGenerator
from standard_e2e.utils import se3, transform_points


def world_from_ego(x: float, y: float, heading: float) -> np.ndarray:
    c, s = np.cos(heading), np.sin(heading)
    return se3(
        np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), np.array([x, y, 0.0])
    )


def motion_compensate(
    points_ego: np.ndarray, point_times: np.ndarray, ref: FrameRef
) -> np.ndarray:
    """Move each ego-frame point from its capture time to the frame time (planar motion
    from the trajectory slice; times outside the slice use its ends)."""
    t, x, y, h = ref.trajectory_slice.T
    xp, yp, hp = (
        np.interp(point_times, t, x),
        np.interp(point_times, t, y),
        np.interp(point_times, t, h),
    )
    c, s = np.cos(hp), np.sin(hp)
    wx = xp + c * points_ego[:, 0] - s * points_ego[:, 1]
    wy = yp + s * points_ego[:, 0] + c * points_ego[:, 1]
    xf, yf, hf, _ = ref.pose
    cf, sf = np.cos(hf), np.sin(hf)
    dx, dy = wx - xf, wy - yf
    return np.stack([cf * dx + sf * dy, -sf * dx + cf * dy, points_ego[:, 2]], axis=1)


class MuhosDatasetProcessor(SourceDatasetProcessor):
    """Processor for the MUHOS dataset."""

    DATASET_NAME = "muhos"

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
        self._to_wgs84: Optional[Transformer] = None
        self._to_wgs84_anchor: Optional[tuple[float, float]] = None

    @property
    def dataset_name(self) -> str:
        return self.DATASET_NAME

    @property
    def allowed_splits(self) -> list[str]:
        # MUHOS has no canonical train/val/test split; ``split`` is a passthrough
        # output label (as for comma2k19 and NATIX).
        return ["all", "train", "val", "test"]

    def _get_default_adapters(self) -> list[AbstractAdapter]:
        return [CamerasIdentityAdapter(), LidarAdapter()]

    def _get_default_context_aggregators(self) -> list[SegmentContextAggregator]:
        return [FuturePastStatesFromMatricesAggregator(self.output_path)]

    def _lat_lon(self, ref: FrameRef) -> tuple[float, float]:
        anchor = (ref.anchor_lat, ref.anchor_lon)
        if self._to_wgs84_anchor != anchor:
            self._to_wgs84 = Transformer.from_crs(
                f"+proj=aeqd +lat_0={anchor[0]} +lon_0={anchor[1]} "
                "+datum=WGS84 +units=m",
                "EPSG:4326",
                always_xy=True,
            )
            self._to_wgs84_anchor = anchor
        assert self._to_wgs84 is not None
        lon, lat = self._to_wgs84.transform(ref.pose[0], ref.pose[1])
        return float(lat), float(lon)

    def _build_cameras(self, ref: FrameRef) -> dict[CameraDirection, CameraData]:
        return {
            CameraDirection.FRONT: CameraData(
                camera_direction=CameraDirection.FRONT,
                image=io.read_image(ref.image_path),
                intrinsics=calib.AUX_INTRINSICS,
                distortion=calib.AUX_DISTORTION,
                extrinsics=calib.T_EGO_FROM_AUX.astype(np.float32),
                is_fisheye=True,
            )
        }

    def _build_lidar(self, ref: FrameRef) -> LidarData:
        xyz_lidar, _intensity, point_times = io.read_lidar(ref.lidar_path)
        xyz_ego = transform_points(calib.T_EGO_FROM_LIDAR, xyz_lidar)
        (x0, x1), (y0, y1), (z0, z1) = calib.EGO_BODY_BOX
        on_car = (
            (xyz_ego[:, 0] > x0)
            & (xyz_ego[:, 0] < x1)
            & (xyz_ego[:, 1] > y0)
            & (xyz_ego[:, 1] < y1)
            & (xyz_ego[:, 2] > z0)
            & (xyz_ego[:, 2] < z1)
        )
        xyz = motion_compensate(xyz_ego[~on_car], point_times[~on_car], ref)
        return LidarData(
            points=pd.DataFrame(
                xyz.astype(np.float32), columns=[c.value for c in LidarComponent]
            )
        )

    def _prepare_standardized_frame_data(
        self, raw_frame_data: Any
    ) -> StandardFrameData:
        ref: FrameRef = raw_frame_data
        x, y, heading, speed = ref.pose
        latitude, longitude = self._lat_lon(ref)
        cameras = (
            self._build_cameras(ref)
            if self.needs_attr(StandardFrameDataField.CAMERAS)
            else {}
        )
        lidar = (
            self._build_lidar(ref)
            if self.needs_attr(StandardFrameDataField.LIDAR)
            else None
        )
        return StandardFrameData(
            dataset_name=self.dataset_name,
            segment_id=ref.set_id,
            frame_id=ref.frame_index,
            timestamp=ref.timestamp,
            split=self._split,
            global_position=Trajectory(
                {
                    TC.TIMESTAMP: [ref.timestamp],
                    TC.X: [x],
                    TC.Y: [y],
                    TC.Z: [0.0],
                    TC.HEADING: [heading],
                    TC.SPEED: [speed],
                }
            ),
            cameras=cameras,
            lidar=lidar,
            # MUHOS ships no 3D object boxes; detections are always empty.
            frame_detections_3d=FrameDetections3D(detections=[]),
            aux_data={"pose_matrix": world_from_ego(x, y, heading)},
            extra_index_data={
                "set": ref.set_id,
                "latitude": latitude,
                "longitude": longitude,
                "lidar_stamp_ns": ref.lidar_stamp_ns,
                "image_stamp_ns": ref.image_stamp_ns,
            },
        )
