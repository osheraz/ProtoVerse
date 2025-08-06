# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the arbove copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate
import random
from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from scipy import ndimage
from pydelatin import Delatin
import pyfqmr
from scipy.ndimage import binary_dilation

from legged_gym.utils.draw_utils import agt_color
from legged_gym.utils.draw_utils import *
import torch


class MeshTerrain:
    def __init__(self, heigthmap_data, device):
        self.border_size = 20
        self.border = 500
        self.sample_extent_x = 300
        self.sample_extent_y = 300
        self.vertical_scale = 1
        self.device = device

        self.heightsamples = torch.from_numpy(heigthmap_data["heigthmap"]).to(device)
        self.walkable_map = torch.from_numpy(heigthmap_data["walkable_map"]).to(device)
        self.cam_pos = torch.from_numpy(heigthmap_data["cam_pos"])
        self.x_scale = heigthmap_data["x_scale"]
        self.y_scale = heigthmap_data["y_scale"]

        self.y_shape, self.x_shape = self.heightsamples.shape
        self.x_c = (self.x_shape / 2) / self.x_scale
        self.y_c = (self.y_shape / 2) / self.y_scale

        coord_y, coord_x = torch.where(
            self.walkable_map == 1
        )  # Image coordinates, need to flip y and x
        coord_x, coord_y = coord_x.float(), coord_y.float()
        self.coord_x_scale = coord_x / self.x_scale - self.x_c
        self.coord_y_scale = coord_y / self.y_scale - self.y_c

        self.coord_x_scale += self.cam_pos[0]
        self.coord_y_scale += self.cam_pos[1]

        self.num_samples = self.coord_x_scale.shape[0]

    def sample_valid_locations(self, num_envs, env_ids):
        num_envs = env_ids.shape[0]
        idxes = np.random.randint(0, self.num_samples, size=num_envs)
        valid_locs = torch.stack(
            [self.coord_x_scale[idxes], self.coord_y_scale[idxes]], dim=-1
        )
        return valid_locs

    def world_points_to_map(self, points):
        points[..., 0] -= self.cam_pos[0] - self.x_c
        points[..., 1] -= self.cam_pos[1] - self.y_c
        points[..., 0] *= self.x_scale
        points[..., 1] *= self.y_scale
        points = (points).long()

        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)

        px = torch.clip(
            px, 0, self.heightsamples.shape[0] - 2
        )  # image, so sampling 1 is for x
        py = torch.clip(py, 0, self.heightsamples.shape[1] - 2)
        return px, py

    def sample_height_points(
        self,
        points,
        root_states=None,
        root_points=None,
        env_ids=None,
        num_group_people=512,
        group_ids=None,
    ):

        B, N, C = points.shape
        px, py = self.world_points_to_map(points)
        heightsamples = self.heightsamples.clone()
        device = points.device
        if env_ids is None:
            env_ids = torch.arange(B).to(points).long()

        if not root_points is None:
            # Adding human root points to the height field
            max_num_envs, num_root_points, _ = root_points.shape
            root_px, root_py = self.world_points_to_map(root_points)
            num_groups = int(root_points.shape[0] / num_group_people)
            heightsamples_group = heightsamples[None,].repeat(num_groups, 1, 1)

            root_px, root_py = root_px.view(
                -1, num_group_people * num_root_points
            ), root_py.view(-1, num_group_people * num_root_points)
            px, py = px.view(-1, N), py.view(-1, N)
            heights = torch.zeros(px.shape).to(px.device)

            if not root_states is None:
                linear_vel = root_states[
                    :, 7:10
                ]  # This contains ALL the linear velocities
                root_rot = root_states[:, 3:7]
                heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
                velocity_map = torch.zeros([px.shape[0], px.shape[1], 2]).to(
                    root_states
                )
                velocity_map_group = torch.zeros(heightsamples_group.shape + (3,)).to(
                    points
                )

            for idx in range(num_groups):
                heightsamples_group[idx][root_px[idx], root_py[idx]] += torch.tensor(
                    1.7 / self.vertical_scale
                )
                # heightsamples_group[idx][root_px[idx] + 1,root_py[idx] + 1] += torch.tensor(1.7 / self.vertical_scale)
                group_mask_env_ids = (
                    group_ids[env_ids] == idx
                )  # agents to select for this group from the current env_ids
                # if sum(group_mask) == 0:
                #     continue
                group_px, group_py = px[group_mask_env_ids].view(-1), py[
                    group_mask_env_ids
                ].view(-1)
                heights1 = heightsamples_group[idx][group_px, group_py]
                heights2 = heightsamples_group[idx][group_px + 1, group_py + 1]

                heights_group = torch.min(heights1, heights2)
                heights[group_mask_env_ids] = heights_group.view(-1, N)

                if not root_states is None:
                    # First update the map with the velocity
                    group_mask_all = group_ids == idx
                    env_ids_in_group = env_ids[group_mask_env_ids]
                    group_linear_vel = linear_vel[group_mask_all]

                    velocity_map_group[
                        idx, root_px[idx], root_py[idx], :
                    ] = group_linear_vel.repeat(1, root_points.shape[1]).view(
                        -1, 3
                    )  # Make sure that the order is correct.

                    # Then sampling the points
                    vel_group = velocity_map_group[idx][group_px, group_py]
                    vel_group = vel_group.view(-1, N, 3)
                    vel_group -= linear_vel[
                        env_ids_in_group, None
                    ]  # this is one-to-one substraction of the agents in the group to mark the static terrain with relative velocity
                    group_heading_rot = heading_rot[env_ids_in_group]

                    group_vel_idv = torch_utils.my_quat_rotate(
                        group_heading_rot.repeat(1, N).view(-1, 4),
                        vel_group.view(-1, 3),
                    )  # Global velocity transform. for ALL of the elements in the group.
                    group_vel_idv = group_vel_idv.view(-1, N, 3)[..., :2]
                    velocity_map[group_mask_env_ids] = group_vel_idv
            # import matplotlib.pyplot as plt; plt.imshow(heights[0].reshape(32, 32).cpu().numpy()); plt.show()
            if root_states is None:
                return heights * self.vertical_scale
            else:
                heights = (heights * self.vertical_scale).view(B, -1, 1)
                return torch.cat([heights, velocity_map], dim=-1)
        else:
            heights1 = heightsamples[px, py]
            heights2 = heightsamples[px + 1, py + 1]
            heights = torch.min(heights1, heights2)
            if root_states is None:
                return heights * self.vertical_scale
            else:
                velocity_map = torch.zeros((B, N, 2)).to(points)
                linear_vel = root_states[env_ids, 7:10]
                root_rot = root_states[env_ids, 3:7]
                heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
                linear_vel_ego = torch_utils.my_quat_rotate(heading_rot, linear_vel)
                velocity_map[:] = (
                    velocity_map[:] - linear_vel_ego[:, None, :2]
                )  # Flip velocity to be in agent's point of view
                heights = (heights * self.vertical_scale).view(B, -1, 1)
                return torch.cat([heights, velocity_map], dim=-1)
