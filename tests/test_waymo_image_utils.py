"""Camera calibration mapping in :mod:`standard_e2e.utils.waymo_image_utils`."""

from __future__ import annotations

import numpy as np
import pytest

# TensorFlow (image decoding) and protobuf (the vendored Waymo protos) are
# installed in the Waymo environments only (datasets/waymo_*/).
tf = pytest.importorskip("tensorflow")

from standard_e2e.enums import CameraDirection  # noqa: E402

# pylint: disable=no-name-in-module
from standard_e2e.third_party.waymo_open_dataset.dataset_pb2 import (  # noqa: E402
    Frame as WaymoFrame,
)
from standard_e2e.utils import FLU_FROM_OPTICAL  # noqa: E402
from standard_e2e.utils.waymo_image_utils import (  # noqa: E402
    waymo_fetch_images_from_frame,
)


def test_extrinsics_are_in_the_optical_frame():
    """A camera whose Waymo extrinsic is the identity looks along ego +x; the
    stored ``T_ego_from_camera`` must say so with optical axes (z forward)."""
    frame = WaymoFrame()
    image = frame.images.add()
    image.name = 1  # CameraName.FRONT
    image.image = tf.io.encode_png(np.zeros((4, 6, 3), dtype=np.uint8)).numpy()
    calib = frame.context.camera_calibrations.add()
    calib.name = 1
    calib.intrinsic.extend([2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    calib.extrinsic.transform.extend(np.eye(4).flatten().tolist())

    cam = waymo_fetch_images_from_frame(frame)[CameraDirection.FRONT]
    np.testing.assert_allclose(cam.extrinsics, FLU_FROM_OPTICAL, atol=1e-6)
    rotation = cam.extrinsics[:3, :3]
    np.testing.assert_allclose(rotation @ [0, 0, 1], [1, 0, 0], atol=1e-6)  # lens
    np.testing.assert_allclose(rotation @ [0, 1, 0], [0, 0, -1], atol=1e-6)  # down
    np.testing.assert_allclose(cam.intrinsics, [[2, 0, 4], [0, 3, 5], [0, 0, 1]])
    assert cam.image.shape == (4, 6, 3)
