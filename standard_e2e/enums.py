from enum import IntEnum, StrEnum


class Modality(StrEnum):
    """
    Enumeration of different data modalities used in the system.
    """

    CAMERAS = "cameras"
    LIDAR_BEV = "lidar_bev"
    LIDAR_PC = "lidar_pc"
    HD_MAP = "hd_map"
    SPEED = "speed"
    INTENT = "intent"
    FUTURE_STATES = "future_states"
    PAST_STATES = "past_states"
    PREFERENCE_TRAJECTORY = "preference_trajectory"
    DETECTIONS_3D = "detections_3d"


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


class LaneType(StrEnum):
    """Coarse lane semantic class.

    Source-side enums (Waymo ``LaneType``, AV2 ``LaneType``) collapse into
    this small set; granularity is intentionally lossy — it is the
    consumer-facing canonical taxonomy.
    """

    UNKNOWN = "unknown"
    VEHICLE = "vehicle"
    BIKE = "bike"
    BUS = "bus"


class LaneMarkType(StrEnum):
    """Coarse lane-boundary marking type."""

    UNKNOWN = "unknown"
    SOLID_WHITE = "solid_white"
    SOLID_YELLOW = "solid_yellow"
    DASHED_WHITE = "dashed_white"
    DASHED_YELLOW = "dashed_yellow"
    DOUBLE_SOLID_WHITE = "double_solid_white"
    DOUBLE_SOLID_YELLOW = "double_solid_yellow"
    PASSING_DOUBLE_DASH = "passing_double_dash"
    NONE = "none"


class RoadEdgeType(StrEnum):
    """Coarse road-edge type (Waymo-only modality at the source level)."""

    UNKNOWN = "unknown"
    BOUNDARY = "boundary"
    MEDIAN = "median"
