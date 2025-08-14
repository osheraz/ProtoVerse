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
