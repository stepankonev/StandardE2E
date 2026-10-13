"""Camera decoding for the Waymo datasets.

Kept apart from :mod:`standard_e2e.utils.image_utils` because it needs
TensorFlow, which only the Waymo environments install (``datasets/waymo_*/``).
"""

import numpy as np
import tensorflow as tf

from standard_e2e.data_structures import CameraData
from standard_e2e.enums import CameraDirection

# pylint: disable=no-name-in-module
from standard_e2e.third_party.waymo_open_dataset.dataset_pb2 import Frame as WaymoFrame

WAYMO_CAMERAS_ORDER = {
    CameraDirection.FRONT: 1,
    CameraDirection.FRONT_LEFT: 2,
    CameraDirection.FRONT_RIGHT: 3,
    CameraDirection.SIDE_LEFT: 4,
    CameraDirection.SIDE_RIGHT: 5,
    CameraDirection.REAR_LEFT: 6,
    CameraDirection.REAR: 7,
    CameraDirection.REAR_RIGHT: 8,
}


def waymo_fetch_images_from_frame(
    frame: WaymoFrame,
) -> dict[CameraDirection, CameraData]:
    """Fetch images from a Waymo frame."""
    camera_data = {}
    camera_idx_to_direction = {v: k for k, v in WAYMO_CAMERAS_ORDER.items()}
    for image_idx, image in enumerate(frame.images):
        camera_index = image.name
        camera_direction = camera_idx_to_direction.get(camera_index)
        if camera_direction is not None:
            # https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto#L98-L107
            # 1d Array of [f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}]
            intrinsics_array = np.array(
                frame.context.camera_calibrations[image_idx].intrinsic
            )
            intrinsics_matrix = np.array(
                [
                    [intrinsics_array[0], 0, intrinsics_array[2]],
                    [0, intrinsics_array[1], intrinsics_array[3]],
                    [0, 0, 1],
                ]
            )

            # https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto#L78
            # 4x4 row major transform matrix that tranforms 3d points from one frame to
            # another.
            extrinsics_matrix = np.array(
                frame.context.camera_calibrations[image_idx].extrinsic.transform
            ).reshape(4, 4)

            distortion_array = np.array(
                frame.context.camera_calibrations[image_idx].intrinsic
            )[4:]

            camera_data[camera_direction] = CameraData(
                image=tf.io.decode_image(image.image).numpy(),
                camera_direction=camera_direction,
                intrinsics=intrinsics_matrix,
                extrinsics=extrinsics_matrix,
                distortion=distortion_array,
            )
    return camera_data
