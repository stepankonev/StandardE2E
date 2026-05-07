from enum import IntEnum, StrEnum


class Modality(StrEnum):
    """
    Enumeration of different data modalities used in the system.
    """

    CAMERAS = "cameras"
    LIDAR_BEV = "lidar_bev"
    LIDAR_PC = "lidar_pc"
    HD_MAP = "hd_map"
    HD_MAP_BEV = "hd_map_bev"
    SPEED = "speed"
    INTENT = "intent"
    FUTURE_STATES = "future_states"
    PAST_STATES = "past_states"
    PREFERENCE_TRAJECTORY = "preference_trajectory"
    DETECTIONS_3D = "detections_3d"
    DETECTIONS_3D_BEV = "detections_3d_bev"


class CameraDirection(StrEnum):
    """Enumeration of camera directions."""

    FRONT = "front"
    FRONT_LEFT = "front_left"
    FRONT_RIGHT = "front_right"
    SIDE_LEFT = "side_left"
    SIDE_RIGHT = "side_right"
    REAR = "rear"
    REAR_LEFT = "rear_left"
    REAR_RIGHT = "rear_right"


class Intent(IntEnum):
    """Intent enum for frame data."""

    UNKNOWN = 0
    GO_STRAIGHT = 1
    GO_LEFT = 2
    GO_RIGHT = 3


class LidarComponent(StrEnum):
    """Enumeration of LiDAR point-cloud per-point components."""

    X = "x"
    Y = "y"
    Z = "z"


class MapElementType(StrEnum):
    """Universal taxonomy for HD map elements across datasets.

    Each element is either a polyline (open), a polygon (closed; set
    ``MapElement.is_closed=True``), or a point (single-row ``points``).
    Dataset-specific subtypes (e.g. lane-mark style, lane vehicle/bike
    designation) live in ``MapElement.attrs``, not in this enum.
    """

    LANE_CENTER = "lane_center"
    LANE_BOUNDARY = "lane_boundary"
    ROAD_EDGE = "road_edge"
    CROSSWALK = "crosswalk"
    INTERSECTION = "intersection"
    DRIVABLE_AREA = "drivable_area"
    STOP_LINE = "stop_line"
    STOP_SIGN = "stop_sign"
    SPEED_BUMP = "speed_bump"
    DRIVEWAY = "driveway"
    WALKWAY = "walkway"
    TRAFFIC_LIGHT = "traffic_light"
    UNKNOWN = "unknown"


class TrajectoryComponent(StrEnum):
    """Enumeration of trajectory components."""

    TIMESTAMP = "timestamp"
    IS_VALID = "is_valid"
    X = "x"
    Y = "y"
    Z = "z"
    SPEED = "speed"
    HEADING = "heading"
    VELOCITY_X = "velocity_x"
    VELOCITY_Y = "velocity_y"
    ACCELERATION_X = "acceleration_x"
    ACCELERATION_Y = "acceleration_y"

    LENGTH = "length"
    WIDTH = "width"
    HEIGHT = "height"


class DetectionType(StrEnum):
    """Enumeration of agent types."""

    UNKNOWN = "unknown"
    PEDESTRIAN = "pedestrian"
    VEHICLE = "vehicle"
    BICYCLE = "bicycle"

    # https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/labeling_specifications.md#sign-labeling-specifications
    SIGN = "sign"
