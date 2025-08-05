# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .multi_camera_cfg import MultiCameraCfg
from .multi_tiled_camera import MultiTiledCamera


@configclass
class MultiTiledCameraCfg(MultiCameraCfg):
    """Configuration for multiple tiled rendering-based camera sensors per environment."""

    class_type: type = MultiTiledCamera

    return_latest_camera_pose: bool = False
    """Whether to return the latest camera pose when fetching the camera's data. Defaults to False.

    If True, the latest camera pose is returned in the camera's data which will slow down performance
    due to the use of :class:`XformPrimView`.
    If False, the pose of the camera during initialization is returned.
    """
