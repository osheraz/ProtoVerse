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
import sys
from random import randint

import numpy as np
import torch
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
import cv2
import itertools

class Terrain_mix:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        print(f"the num of robots is {num_robots}")
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane', 'trimesh', 'heightfield']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        cfg.terrain_proportions = np.array(cfg.terrain_proportions) / np.sum(cfg.terrain_proportions)
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))
        # self.env_slope_vec = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.goals = np.zeros((cfg.num_rows, cfg.num_cols, cfg.num_goals, 3))
        self.num_goals = cfg.num_goals

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            if hasattr(cfg, "max_difficulty"):
                self.curiculum(random=True, max_difficulty=cfg.max_difficulty)
            else:
                self.curiculum(random=True)
            # self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type in ["cross"]:
            print("Converting heightmap to trimesh for cross-country setting...")
            if cfg.hf2mesh_method == "grid":
                self.vertices, self.triangles, self.x_edge_mask = convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                                self.cfg.horizontal_scale,
                                                                                                self.cfg.vertical_scale,
                                                                                                self.cfg.slope_treshold)
                half_edge_width = int(self.cfg.edge_width_thresh / self.cfg.horizontal_scale)
                structure = np.ones((half_edge_width*2+1, 1))
                self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure)
                if self.cfg.simplify_grid:
                    mesh_simplifier = pyfqmr.Simplify()
                    mesh_simplifier.setMesh(self.vertices, self.triangles)
                    mesh_simplifier.simplify_mesh(target_count = int(0.05*self.triangles.shape[0]), aggressiveness=7, preserve_border=True, verbose=10)

                    self.vertices, self.triangles, normals = mesh_simplifier.getMesh()
                    self.vertices = self.vertices.astype(np.float32)
                    self.triangles = self.triangles.astype(np.uint32)
            else:
                assert cfg.hf2mesh_method == "fast", "Height field to mesh method must be grid or fast"
                self.vertices, self.triangles = convert_heightfield_to_trimesh_delatin(self.height_field_raw, self.cfg.horizontal_scale, self.cfg.vertical_scale, max_error=cfg.max_error)
            print("Created {} vertices".format(self.vertices.shape[0]))
            print("Created {} triangles".format(self.triangles.shape[0]))

      

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            print(f"the number of sub terrians is {self.cfg.num_sub_terrains}")
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            # difficulty = np.random.choice([0.5, 0.75, 0.9])
            difficulty = np.random.uniform(-0.2, 1.2)
            terrain = self.make_terrain(choice, difficulty)
            # terrain = self.make_mix_terrain_locomotion(choice,difficulty)
            self.add_terrain_to_map(terrain, i, j)

 

        
    def curiculum(self, random=False, max_difficulty=False):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / max(1, (self.cfg.num_rows-1))
                choice = j / self.cfg.num_cols + 0.001
                step_height = 0.05 + 0.175 * difficulty
                if random:
                    if max_difficulty:
                        terrain = self.make_terrain(choice, np.random.uniform(0.7, 1))
                    else:
                        terrain = self.make_terrain(choice, np.random.uniform(0, 1))
                else:
                    terrain = self.make_terrain(choice, difficulty)

                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.length_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def add_roughness(self, terrain, difficulty=1):
        max_height = (self.cfg.height[1] - self.cfg.height[0]) * difficulty + self.cfg.height[0]
        height = random.uniform(self.cfg.height[0], max_height)
        terrain_utils.random_uniform_terrain(terrain, min_height=-height, max_height=height, step=0.005, downsampled_scale=self.cfg.downsampled_scale)

 

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.length_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        #general properties
        slope = difficulty * 0.7
        step_height = 0.05 + 0.175 * difficulty
        discrete_obstacles_height = 0.03 + difficulty * 0.15
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        #general properties

        if choice < self.proportions[0]:
            idx = 0
            if choice < self.proportions[0]/ 2:
                idx = 1
                slope *= -1
            #create all availiable terrains
            num_terrains = 9 # num of terrains in one env
            cross_country_mini(terrain, difficulty=difficulty, num_terrains=num_terrains,traj_direction=self.cfg.traj_direction)
        elif choice < self.proportions[2]:
            idx = 2
            if choice<self.proportions[1]:
                idx = 3
                slope *= -1
            #create cross_country in random order
            num_terrains = 9
            cross_country_mini_random(terrain, difficulty=difficulty, num_terrains=num_terrains,traj_direction=self.cfg.traj_direction)
            # self.add_roughness(terrain)
        elif choice < self.proportions[4]:
            idx = 4
            if choice<self.proportions[3]:
                idx = 5
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
            self.add_roughness(terrain)
        elif choice < self.proportions[5]:
            idx = 6
            num_rectangles = 40
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
            self.add_roughness(terrain)
        elif choice < self.proportions[6]:
            idx = 7
            stones_size = 1.5 - 1.2*difficulty
            # terrain_utils.stepping_stones_terrain(terrain, stone_size=stones_size, stone_distance=0.1, stone_distance_rand=0, max_height=0.04*difficulty, platform_size=2.)
            half_sloped_terrain(terrain, wall_width=4, start2center=0.5, max_height=0.00)
            stepping_stones_terrain(terrain, stone_size=1.5-0.2*difficulty, stone_distance=0.0+0.4*difficulty, max_height=0.2*difficulty, platform_size=1.2)
            self.add_roughness(terrain)
        elif choice < self.proportions[7]:
            idx = 8
            # gap_size = random.uniform(self.cfg.gap_size[0], self.cfg.gap_size[1])
            gap_parkour_terrain(terrain, difficulty, platform_size=4)
            self.add_roughness(terrain)
        elif choice < self.proportions[8]:
            idx = 9
            self.add_roughness(terrain)
            # pass
        elif choice < self.proportions[9]:
            idx = 10
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        elif choice < self.proportions[10]:
            idx = 11
            if self.cfg.all_vertical:
                half_slope_difficulty = 1.0
            else:
                difficulty *= 1.3
                if not self.cfg.no_flat:
                    difficulty -= 0.1
                if difficulty > 1:
                    half_slope_difficulty = 1.0
                elif difficulty < 0:
                    self.add_roughness(terrain)
                    terrain.slope_vector = np.array([1, 0., 0]).astype(np.float32)
                    return terrain
                else:
                    half_slope_difficulty = difficulty
            wall_width = 4 - half_slope_difficulty * 4
            # terrain_utils.wall_terrain(terrain, height=1, start2center=0.7)
            # terrain_utils.tanh_terrain(terrain, height=1.0, start2center=0.7)
            if self.cfg.flat_wall:
                half_sloped_terrain(terrain, wall_width=4, start2center=0.5, max_height=0.00)
            else:
                half_sloped_terrain(terrain, wall_width=wall_width, start2center=0.5, max_height=1.5)
            max_height = terrain.height_field_raw.max()
            top_mask = terrain.height_field_raw > max_height - 0.05
            self.add_roughness(terrain, difficulty=1)
            terrain.height_field_raw[top_mask] = max_height
        elif choice < self.proportions[11]:
            idx = 12
            # half platform terrain
            half_platform_terrain(terrain, max_height=0.1 + 0.4 * difficulty )
            self.add_roughness(terrain, difficulty=1)
        elif choice < self.proportions[13]:
            idx = 13
            height = 0.1 + 0.3 * difficulty
            if choice < self.proportions[12]:
                idx = 14
                height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=1., step_height=height, platform_size=3.)
            self.add_roughness(terrain)
        elif choice < self.proportions[14]:
            x_range = [-0.1, 0.1+0.3*difficulty]  # offset to stone_len
            y_range = [0.2, 0.3+0.1*difficulty]
            stone_len = [0.9 - 0.3*difficulty, 1 - 0.2*difficulty]#2 * round((0.6) / 2.0, 1)
            incline_height = 0.25*difficulty
            last_incline_height = incline_height + 0.1 - 0.1*difficulty
            parkour_terrain(terrain,
                            num_stones=self.num_goals - 2,
                            x_range=x_range,
                            y_range=y_range,
                            incline_height=incline_height,
                            stone_len=stone_len,
                            stone_width=1.0,
                            last_incline_height=last_incline_height,
                            pad_height=0,
                            pit_depth=[0.2, 1])
            idx = 15
            # terrain.height_field_raw[:] = 0
            self.add_roughness(terrain)
        elif choice < self.proportions[15]:
            idx = 16
            parkour_hurdle_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   stone_len=0.1+0.3*difficulty,
                                   hurdle_height_range=[0.1+0.1*difficulty, 0.15+0.25*difficulty],
                                   pad_height=0,
                                   x_range=[1.2, 2.2],
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.4, 0.8],
                                   )
            # terrain.height_field_raw[:] = 0
            self.add_roughness(terrain)
        elif choice < self.proportions[16]:
            idx = 17
            parkour_hurdle_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   stone_len=0.1+0.3*difficulty,
                                   hurdle_height_range=[0.1+0.1*difficulty, 0.15+0.15*difficulty],
                                   pad_height=0,
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.45, 1],
                                   flat=True
                                   )
            self.add_roughness(terrain)
        elif choice < self.proportions[17]:
            idx = 18
            parkour_step_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   step_height=0.1 + 0.35*difficulty,
                                   x_range=[0.3,1.5],
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.5, 1],
                                   pad_height=0,
                                   )
            self.add_roughness(terrain)
        elif choice < self.proportions[18]:
            idx = 19
            parkour_gap_terrain(terrain,
                                num_gaps=self.num_goals - 2,
                                gap_size=0.1 + 0.7 * difficulty,
                                gap_depth=[0.2, 1],
                                pad_height=0,
                                x_range=[0.8, 1.5],
                                y_range=self.cfg.y_range,
                                half_valid_width=[0.6, 1.2],
                                # flat=True
                                )
            self.add_roughness(terrain)
        elif choice < self.proportions[19]:
            idx = 20
            demo_terrain(terrain)
            #mix_terrain(terrain)
            self.add_roughness(terrain)
        # np.set_printoptions(precision=2)
        # print(np.array(self.proportions), choice)

        #poles
        elif choice < self.proportions[20]:
            idx = 21
            incline_height = 0.25*difficulty

            step_height = 0.05 + 0.175 * difficulty

            step_height *= -1

            poles_terrain(terrain=terrain, difficulty=difficulty )
            self.add_roughness(terrain)





        #poles
        elif choice < self.proportions[21]:
            idx=22
            num_terrains = 2
            cross_country_large_random(terrain, difficulty=difficulty, num_terrains=num_terrains,traj_direction=self.cfg.traj_direction)
            self.add_roughness(terrain)





        terrain.idx = idx
        return terrain



    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        # env_origin_x = (i + 0.5) * self.env_length
        env_origin_x = i * self.env_length + 1.0
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 0.5) / terrain.horizontal_scale) # within 1 meter square range
        x2 = int((self.env_length/2. + 0.5) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 0.5) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 0.5) / terrain.horizontal_scale)
        if self.cfg.origin_zero_z:
            env_origin_z = 0
        else:
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.terrain_type[i, j] = terrain.idx
        self.goals[i, j, :, :2] = terrain.goals + [i * self.env_length, j * self.env_width]
        # self.env_slope_vec[i, j] = terrain.slope_vector






def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def gap_parkour_terrain(terrain, difficulty, platform_size=2.):
    gap_size = 0.1 + 0.3 * difficulty
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -400
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

    slope_angle = 0.1 + difficulty * 1
    offset = 1 + 9 * difficulty#10
    scale = 15
    wall_center_x = [center_x - x1, center_x, center_x + x1]
    wall_center_y = [center_y - y1, center_y, center_y + y1]

    # for i in range(center_y + y1, center_y + y2):
    #     for j in range(center_x-x1, center_x + x1):
    #         for w in wall_center_x:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[j, i] < height:
    #                 terrain.height_field_raw[j, i] = int(height)
    
    # for i in range(center_y - y2, center_y - y1):
    #     for j in range(center_x-x1, center_x + x1):
    #         for w in wall_center_x:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[j, i] < height:
    #                 terrain.height_field_raw[j, i] = int(height)

    # for i in range(center_x + x1, center_x + x2):
    #     for j in range(center_y-y1, center_y + y1):
    #         for w in wall_center_y:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[i, j] < height:
    #                 terrain.height_field_raw[i, j] = int(height)
    
    # for i in range(center_x - x2, center_x - x1):
    #     for j in range(center_y-y1, center_y + y1):
    #         for w in wall_center_y:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[i, j] < height:
    #                 terrain.height_field_raw[i, j] = int(height)

def parkour_terrain(terrain, 
                    platform_len=2.5, 
                    platform_height=0., 
                    num_stones=8, 
                    x_range=[1.8, 1.9], 
                    y_range=[0., 0.1], 
                    z_range=[-0.2, 0.2],
                    stone_len=1.0,
                    stone_width=0.6,
                    pad_width=0.1,
                    pad_height=0.5,
                    incline_height=0.1,
                    last_incline_height=0.6,
                    last_stone_len=1.6,
                    pit_depth=[0.5, 1.]):
    # 1st dimension: x, 2nd dimension: y
    goals = np.zeros((num_stones+2, 2))
    terrain.height_field_raw[:] = -round(np.random.uniform(pit_depth[0], pit_depth[1]) / terrain.vertical_scale)
    
    mid_y = terrain.length // 2  # length is actually y width
    stone_len = np.random.uniform(*stone_len)
    stone_len = 2 * round(stone_len / 2.0, 1)
    stone_len = round(stone_len / terrain.horizontal_scale)
    dis_x_min = stone_len + round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = stone_len + round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)
    dis_z_min = round(z_range[0] / terrain.vertical_scale)
    dis_z_max = round(z_range[1] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_width = round(stone_width / terrain.horizontal_scale)
    last_stone_len = round(last_stone_len / terrain.horizontal_scale)

    incline_height = round(incline_height / terrain.vertical_scale)
    last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len - np.random.randint(dis_x_min, dis_x_max) + stone_len // 2
    goals[0] = [platform_len -  stone_len // 2, mid_y]
    left_right_flag = np.random.randint(0, 2)
    # dis_z = np.random.randint(dis_z_min, dis_z_max)
    dis_z = 0
    
    for i in range(num_stones):
        dis_x += np.random.randint(dis_x_min, dis_x_max)
        pos_neg = round(2*(left_right_flag - 0.5))
        dis_y = mid_y + pos_neg * np.random.randint(dis_y_min, dis_y_max)
        if i == num_stones - 1:
            dis_x += last_stone_len // 4
            heights = np.tile(np.linspace(-last_incline_height, last_incline_height, stone_width), (last_stone_len, 1)) * pos_neg
            terrain.height_field_raw[dis_x-last_stone_len//2:dis_x+last_stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
        else:
            heights = np.tile(np.linspace(-incline_height, incline_height, stone_width), (stone_len, 1)) * pos_neg
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
        
        goals[i+1] = [dis_x, dis_y]

        left_right_flag = 1 - left_right_flag
    final_dis_x = dis_x + 2*np.random.randint(dis_x_min, dis_x_max)
    final_platform_start = dis_x + last_stone_len // 2 + round(0.05 // terrain.horizontal_scale)
    terrain.height_field_raw[final_platform_start:, :] = platform_height
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

    return terrain # for mix_terrain, if ori ->no need to return


def parkour_gap_terrain_re(terrain,
                    platform_len=2.5,
                    platform_height=0.,
                    num_gaps=8,
                    gap_size=0.3,
                    x_range=[1.6, 2.4],
                    y_range=[-1.2, 1.2],
                    half_valid_width=[0.6, 1.2],
                    gap_depth=-200,
                    pad_width=0.1,
                    pad_height=0.5,
                    flat=False):
    goals = np.zeros((num_gaps + 2, 2))
    # terrain.height_field_raw[:] = -200
    # import ipdb; ipdb.set_trace()
    mid_y = terrain.length // 2  # length is actually y width

    # dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    # dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    gap_depth = -round(np.random.uniform(gap_depth[0], gap_depth[1]) / terrain.vertical_scale)

    # half_gap_width = round(np.random.uniform(0.6, 1.2) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    # terrain.height_field_raw[:, :mid_y-half_valid_width] = gap_depth
    # terrain.height_field_raw[:, mid_y+half_valid_width:] = gap_depth

    terrain.height_field_raw[0:platform_len, :] = platform_height

    gap_size = round(gap_size / terrain.horizontal_scale)
    dis_x_min = round(x_range[0] / terrain.horizontal_scale) + gap_size
    dis_x_max = round(x_range[1] / terrain.horizontal_scale) + gap_size

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_gaps):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if not flat:
            # terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
            # terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2,
            #                          gap_center-half_gap_width:gap_center+half_gap_width] = gap_depth
            terrain.height_field_raw[dis_x - gap_size // 2: dis_x + gap_size // 2, :] = gap_depth

        terrain.height_field_raw[last_dis_x:dis_x, :mid_y + rand_y - half_valid_width] = gap_depth
        terrain.height_field_raw[last_dis_x:dis_x, mid_y + rand_y + half_valid_width:] = gap_depth

        last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]

    terrain.goals = goals * terrain.horizontal_scale

    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

    return terrain
    
def parkour_gap_terrain(terrain,
                           platform_len=2.5, 
                           platform_height=0., 
                           num_gaps=8,
                           gap_size=0.3,
                           x_range=[1.6, 2.4],
                           y_range=[-1.2, 1.2],
                           half_valid_width=[0.6, 1.2],
                           gap_depth=-200,
                           pad_width=0.1,
                           pad_height=0.5,
                           flat=False):
    goals = np.zeros((num_gaps+2, 2))
    # terrain.height_field_raw[:] = -200
    # import ipdb; ipdb.set_trace()
    mid_y = terrain.length // 2  # length is actually y width

    # dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    # dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    gap_depth = -round(np.random.uniform(gap_depth[0], gap_depth[1]) / terrain.vertical_scale)
    
    # half_gap_width = round(np.random.uniform(0.6, 1.2) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    # terrain.height_field_raw[:, :mid_y-half_valid_width] = gap_depth
    # terrain.height_field_raw[:, mid_y+half_valid_width:] = gap_depth
    
    terrain.height_field_raw[0:platform_len, :] = platform_height

    gap_size = round(gap_size / terrain.horizontal_scale)
    dis_x_min = round(x_range[0] / terrain.horizontal_scale) + gap_size
    dis_x_max = round(x_range[1] / terrain.horizontal_scale) + gap_size

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_gaps):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if not flat:
            # terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
            # terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, 
            #                          gap_center-half_gap_width:gap_center+half_gap_width] = gap_depth
            terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, :] = gap_depth

        terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = gap_depth
        terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = gap_depth
        
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height


def parkour_hurdle_terrain_re(terrain,
                           platform_len=2.5,
                           platform_height=0.,
                           num_stones=8,
                           stone_len=0.3,
                           x_range=[1.5, 2.4],
                           y_range=[-0.4, 0.4],
                           half_valid_width=[0.4, 0.8],
                           hurdle_height_range=[0.2, 0.3],
                           pad_width=0.1,
                           pad_height=0.5,
                           flat=False):
    goals = np.zeros((num_stones + 2, 2))
    # terrain.height_field_raw[:] = -200

    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    # half_valid_width = round(np.random.uniform(y_range[1]+0.2, y_range[1]+1) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_len = round(stone_len / terrain.horizontal_scale)
    # stone_width = round(stone_width / terrain.horizontal_scale)

    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x
        if not flat:
            terrain.height_field_raw[dis_x - stone_len // 2:dis_x + stone_len // 2, ] = np.random.randint(
                hurdle_height_min, hurdle_height_max)
            terrain.height_field_raw[dis_x - stone_len // 2:dis_x + stone_len // 2,
            :mid_y + rand_y - half_valid_width] = 0
            terrain.height_field_raw[dis_x - stone_len // 2:dis_x + stone_len // 2,
            mid_y + rand_y + half_valid_width:] = 0
        last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]

    terrain.goals = goals * terrain.horizontal_scale

    # terrain.height_field_raw[:, :max(mid_y-half_valid_width, 0)] = 0
    # terrain.height_field_raw[:, min(mid_y+half_valid_width, terrain.height_field_raw.shape[1]):] = 0
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

    return  terrain

def parkour_hurdle_terrain(terrain,
                           platform_len=2.5, 
                           platform_height=0., 
                           num_stones=8,
                           stone_len=0.3,
                           x_range=[1.5, 2.4],
                           y_range=[-0.4, 0.4],
                           half_valid_width=[0.4, 0.8],
                           hurdle_height_range=[0.2, 0.3],
                           pad_width=0.1,
                           pad_height=0.5,
                           flat=False):
    goals = np.zeros((num_stones+2, 2))
    # terrain.height_field_raw[:] = -200
    
    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    # half_valid_width = round(np.random.uniform(y_range[1]+0.2, y_range[1]+1) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_len = round(stone_len / terrain.horizontal_scale)
    # stone_width = round(stone_width / terrain.horizontal_scale)
    
    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x
        if not flat:
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, :mid_y+rand_y-half_valid_width] = 0
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, mid_y+rand_y+half_valid_width:] = 0
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # terrain.height_field_raw[:, :max(mid_y-half_valid_width, 0)] = 0
    # terrain.height_field_raw[:, min(mid_y+half_valid_width, terrain.height_field_raw.shape[1]):] = 0
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def parkour_step_terrain(terrain,
                           platform_len=2.5, 
                           platform_height=0., 
                           num_stones=8,
                        #    x_range=[1.5, 2.4],
                            x_range=[0.2, 0.4],
                           y_range=[-0.15, 0.15],
                           half_valid_width=[0.45, 0.5],
                           step_height = 0.2,
                           pad_width=0.1,
                           pad_height=0.5):
    goals = np.zeros((num_stones+2, 2))
    # terrain.height_field_raw[:] = -200
    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round( (x_range[0] + step_height) / terrain.horizontal_scale)
    dis_x_max = round( (x_range[1] + step_height) / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    step_height = round(step_height / terrain.vertical_scale)

    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    # stone_width = round(stone_width / terrain.horizontal_scale)
    
    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len
    last_dis_x = dis_x
    stair_height = 0
    goals[0] = [platform_len - round(1 / terrain.horizontal_scale), mid_y]
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if i < num_stones // 2:
            stair_height += step_height
        elif i > num_stones // 2:
            stair_height -= step_height
        terrain.height_field_raw[dis_x:dis_x+rand_x, ] = stair_height
        dis_x += rand_x
        terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = 0
        terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = 0
        
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y+rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # terrain.height_field_raw[:, :max(mid_y-half_valid_width, 0)] = 0
    # terrain.height_field_raw[:, min(mid_y+half_valid_width, terrain.height_field_raw.shape[1]):] = 0
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

#terrains from real assets

def mesh_ground_real_assets(self):

    print(f"TBD")

#terrains from real assets
def poles_terrain(terrain, difficulty=1):
    """
    Generate stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    start_x = 0*terrain.width//32#traning is 0
    stop_x = 28*terrain.width//32 #training is 7/8
    start_y =0
    stop_y = terrain.length

    img = np.zeros((terrain.length, terrain.length), dtype=int)
    # disk, circle, curve, poly, ellipse
    base_prob = 1 / 2
    # probs = np.array([0.7, 0.7, 0.4, 0.5, 0.5]) * ((1 - base_prob) * difficulty + base_prob)
    probs = np.array([0.8, 0.4, 0.5, 0.5]) * ((1 - base_prob) * difficulty + base_prob)
    low, high = 75, 450 # training is 20-500; testing is 75-450
    num_mult = int(stop_x // 80)
    for i in range(len(probs)):
        p = probs[i]
        if i == 0:
            for _ in range(10 * num_mult):
                if np.random.binomial(1, p):
                    img += draw_disk(img_size=terrain.length, max_r = 3) * int(np.random.uniform(low, high))
        elif i == 1 and np.random.binomial(1, p):
            for _ in range(3 * num_mult):
                if np.random.binomial(1, p):
                    img += draw_curve(img_size=terrain.length) * int(np.random.uniform(low, high))
        elif i == 2 and np.random.binomial(1, p):
            for _ in range(1 * num_mult):
                if np.random.binomial(1, p):
                    img += draw_polygon(img_size=terrain.length, max_sides=2) * int(np.random.uniform(low, high))
        elif i == 3 and np.random.binomial(1, p):
            for _ in range(5 * num_mult):
                if np.random.binomial(1, p):
                    img += draw_ellipse(img_size=terrain.length,
                                        max_size=5) * int(
                                            np.random.uniform(low, high))
    size_img = cv2.resize(img.astype('float32'), (terrain.length, 7*terrain.width//8))
    terrain.height_field_raw[start_x: stop_x, start_y:stop_y] = size_img

    #set goals

    goals = np.ones((8, 2))
    mid_y = terrain.length//2
    goal_start_x = terrain.width//8
    goal_positions =[]

    # Find non-pole positions
    non_pole_postions_start = terrain.width//4
    non_pole_positions = np.argwhere(terrain.height_field_raw[non_pole_postions_start:stop_x, :mid_y] == 0)
    # Randomly select 6 goal positions from non-pole positions, based on x-axis
    non_pole_positions = non_pole_positions + [non_pole_postions_start,0]
    base_on_x = non_pole_positions[:,0]
    base_on_x_unique = np.unique(base_on_x)
    num_goals_to_sample = min(6, len(non_pole_positions))
    goal_positions_indices = np.random.choice(base_on_x_unique, size=num_goals_to_sample, replace=False)

    for idx in goal_positions_indices:
        pick_x_location = np.where(non_pole_positions[:,0] == idx)[0]
        random_index = np.random.choice(pick_x_location)
        pick_one_x = non_pole_positions[random_index]
        goal_positions.append(pick_one_x)

    # goal_positions = [tuple(non_pole_positions[np.random.choice(np.where(non_pole_positions[:,0]== idx), size =1, replace = False)]) for idx in goal_positions_indices]
    sort_goal_positions = sorted(goal_positions, key=lambda x: x[0])
    final_dis_x = int(7*terrain.width//8+2*terrain.width//32)
    # find non-poles for goal_0
    start_goal_candidates_y = np.where(terrain.height_field_raw[goal_start_x,] == 0)

    the_candidate_index_y = np.random.choice(start_goal_candidates_y[0])
    # find non-poles for goal_0
    goals[0] = [goal_start_x, the_candidate_index_y]
    goals[-1] = [final_dis_x,mid_y]
    for i in range(6):
        current_x = sort_goal_positions[i]
        goals[i+1] = current_x

    terrain.goals= goals
    terrain.goals = goals * terrain.horizontal_scale

    return terrain


def cross_country_large_random(terrain, difficulty=1, num_terrains=4, traj_direction = "uni_random"):
    '''
    try to make multiple terrains in one env from the subset of all possible types, concat them along x-axis, random combination from cross_country_mini's heightfiled
    '''
    tot_terrain_width = terrain.width
    tot_terrain_length = terrain.length
    per_terrain_width = int( terrain.width / num_terrains)
    per_terrain_length = terrain.length

    ####### total is based on parkour/gap terrain #######################
    #coin_flip_terrain_base =
    coin_flip_terrain_base = np.random.randint(0, 3)
    if coin_flip_terrain_base == 0:

        x_range = [-0.1, 0.1 + 0.3 * difficulty]  # offset to stone_len
        y_range = [0.2, 0.3 + 0.1 * difficulty]
        stone_len = [0.9 - 0.3 * difficulty, 1 - 0.2 * difficulty]  # 2 * round((0.6) / 2.0, 1)
        incline_height = 0.25 * difficulty
        last_incline_height = incline_height + 0.1 - 0.1 * difficulty

        #
        # x_range = [-0.3, 0.1 + 0.3 * difficulty]  # offset to stone_len
        # y_range = [0.3, 0.3 + 0.1 * difficulty]
        # stone_len = [0.9 - 0.3 * difficulty, 1. - 0.2 * difficulty]  # 2 * round((0.6) / 2.0, 1)
        # incline_height = 0.25 * difficulty
        # last_incline_height = incline_height + 0.1 - 0.1 * difficulty

        terrain.height_field_raw = parkour_terrain(terrain,
                                                   num_stones=6,
                                                   x_range=x_range,
                                                   y_range=y_range,
                                                   incline_height=incline_height,
                                                   stone_len=stone_len,
                                                   stone_width=1.,
                                                   last_incline_height=last_incline_height,
                                                   pad_height=0,
                                                   pit_depth=[0.2, 1]).height_field_raw



    elif coin_flip_terrain_base == 1:

        terrain.height_field_raw = parkour_gap_terrain_re(terrain,
                                                          num_gaps=6,
                                                          gap_size=0.1 + 0.45 * difficulty, #0.1/0.45(or 0.21)
                                                          gap_depth=[0.2, 1],
                                                          pad_height=0.,
                                                          x_range=[0.8, 1.5],
                                                          y_range=[-0.4, 0.4],
                                                          half_valid_width=[0.6, 1.2], ).height_field_raw

    elif coin_flip_terrain_base == 2:
        terrain.height_field_raw = parkour_hurdle_terrain_re(terrain,
                                                             num_stones=6,
                                                             stone_len=0.5 + 0.3 * difficulty,
                                                             hurdle_height_range=[-0.25 + 0.1 * difficulty,
                                                                                  0.25 + 0.25 * difficulty],
                                                             pad_height=0,
                                                             x_range=[1.2, 2.2],
                                                             half_valid_width=[0.4, 0.8], ).height_field_raw

    # need to opt
    heightfield = np.zeros((tot_terrain_width, tot_terrain_length), dtype=int)
    random_noise = random.uniform(0, 1)


    # 1st terrain
    terrain_1 = terrain_utils.SubTerrain("terrain_1",
                                         width=per_terrain_width,
                                         length=per_terrain_length,
                                         vertical_scale=0.5,
                                         horizontal_scale=0.5)
    heightfield[0:per_terrain_width, :] += terrain_utils.sloped_terrain(terrain_1,slope=.75 + 1.5 * random_noise).height_field_raw


    terrain_5 = terrain_utils.SubTerrain("terrain_5",
                                         width=per_terrain_width,
                                         length=per_terrain_length,
                                         vertical_scale=0.25,
                                         horizontal_scale=0.5)

    heightfield[0 * per_terrain_width: per_terrain_width, :] += terrain_utils.pyramid_stairs_terrain(terrain_5,
                                                                                                       step_width=4.5,
                                                                                                       step_height=3.5 + random_noise * difficulty).height_field_raw

    ##### high gap

    max_height_terrain_3 = np.argmax(terrain_1.height_field_raw)
    max_height_terrain_3_idx_unravel = np.unravel_index(max_height_terrain_3, terrain_1.height_field_raw.shape)
    max_height_terrain_3_value = terrain_1.height_field_raw[max_height_terrain_3_idx_unravel]
    end_height_1 = max_height_terrain_3_value

    max_height_terrain_5 = np.argmax(terrain_5.height_field_raw)
    max_height_terrain_5_idx_unravel = np.unravel_index(max_height_terrain_5, terrain_5.height_field_raw.shape)
    max_height_terrain_5_value = terrain_5.height_field_raw[max_height_terrain_5_idx_unravel]
    end_height_1_1 = max_height_terrain_5_value

    end_height_1 += end_height_1_1

    ##### high gap

    terrain_8 = terrain_utils.SubTerrain("terrain_8",
                                         width=per_terrain_width,
                                         length=per_terrain_length,
                                         vertical_scale=0.25 + 0.9 * random_noise,
                                         horizontal_scale=0.005)

    heightfield[0:per_terrain_width, :] += terrain_utils.wave_terrain(terrain_8,
                                                                      num_waves=5 + round(3 * random_noise),
                                                                      amplitude=5 + random_noise).height_field_raw


    terrain_4 = terrain_utils.SubTerrain("terrain_4",
                                         width=per_terrain_width,
                                         length=per_terrain_length,
                                         vertical_scale=0.25,
                                         horizontal_scale=0.75)

    heightfield[0*per_terrain_width:per_terrain_width, :] += poles_terrain(terrain=terrain_4, difficulty=difficulty ).height_field_raw


    ###################################################
    # 2nd terrain
    terrain_2 = terrain_utils.SubTerrain("terrain_2",
                                         width=per_terrain_width,
                                         length=per_terrain_length,
                                         vertical_scale=0.5 + random_noise + difficulty * 0.1,
                                         horizontal_scale=0.75)
    heightfield[per_terrain_width:2 * per_terrain_width, :] = terrain_utils.stairs_terrain(terrain_2,
                                                                                               step_width=1,
                                                                                               step_height=0.7 + 0.2 * difficulty).height_field_raw




    max_height_terrain_2 = np.argmax(terrain_2.height_field_raw)
    max_height_terrain_2_idx_unravel = np.unravel_index(max_height_terrain_2, terrain_2.height_field_raw.shape)
    max_height_terrain_2_value = terrain_2.height_field_raw[max_height_terrain_2_idx_unravel]
    end_height_2 = max_height_terrain_2_value

    # if coin_flip_terrain_base ==2:
    #     # 2nd terrain
    #     terrain_9 = terrain_utils.SubTerrain("terrain_9",
    #                                          width=per_terrain_width,
    #                                          length=per_terrain_length,
    #                                          vertical_scale=0.5 + random_noise + difficulty * 0.1,
    #                                          horizontal_scale=0.95)
    #     heightfield[per_terrain_width:2 * per_terrain_width, :] += terrain_utils.pyramid_sloped_terrain(terrain_9,
    #                                                                                                    slope=1.5 - 2 * random_noise).height_field_raw


    #2nd terrain





    #random order
    heightfield_raw_random = np.zeros((tot_terrain_width, tot_terrain_length), dtype=int)
    cache_index =list(range(num_terrains))
    random.shuffle(cache_index)
    print(f"{cache_index}")
    if cache_index[0]==0:
        end_height = end_height_1
        terrain_18 = terrain_utils.SubTerrain("terrain_18",
                                             width=per_terrain_width,
                                             length=per_terrain_length,
                                             vertical_scale=0.25 + 0.9 * random_noise,
                                             horizontal_scale=0.005)

        heightfield_raw_random[per_terrain_width:2*per_terrain_width,:] += terrain_utils.wave_terrain(terrain_18,
                                                                          num_waves=9 + round(3 * random_noise),
                                                                          amplitude=7 + random_noise).height_field_raw

        terrain_12 = terrain_utils.SubTerrain("terrain_12",
                                             width=per_terrain_width,
                                             length=per_terrain_length,
                                             vertical_scale=0.5 + random_noise + difficulty * 0.1,
                                             horizontal_scale=0.75)
        heightfield[per_terrain_width:2 * per_terrain_width, :] += terrain_utils.pyramid_sloped_terrain(terrain_12,
                                                                                                       slope=-1.5 - 2 * random_noise).height_field_raw
    elif cache_index[0] ==1:
        end_height = end_height_2


    for i in range(num_terrains):
        heightfield_raw_random[i*per_terrain_width:(i+1)*per_terrain_width,:] += heightfield[cache_index[i]*per_terrain_width:(cache_index[i]+1)*per_terrain_width,:]
        if i >0:
            heightfield_raw_random[i*per_terrain_width:(i+1)*per_terrain_width,:] += 7*end_height//8



    terrain.height_field_raw += heightfield_raw_random


    if coin_flip_terrain_base == 1 or 2:
        # set goals
        #goals = np.zeros((8, 2))
        goals = terrain.goals//terrain.horizontal_scale
        goals[-1]= int(14*tot_terrain_width//16), int(10*tot_terrain_length//16)

        terrain.goals = goals * terrain.horizontal_scale

    else:
        # goals = np.zeros((8, 2))
        # goals = terrain.goals
        # goals[-1] = int(15 * tot_terrain_width // 16), int(28 * tot_terrain_length // 32)
        terrain.goals = terrain.goals


    #
    # # set goals
    # goals = np.zeros((8, 2))
    # goals[-1]= tot_terrain_width, int(tot_terrain_length//2)
    #
    # terrain.goals = goals * terrain.horizontal_scale

def cross_country_mini_random(terrain, difficulty=1, num_terrains=9, traj_direction = "uni_random"):
    '''
    try to make multiple terrains in one env from the subset of all possible types, concat them along x-axis, random combination from cross_country_mini's heightfiled
    '''
    tot_terrain_width = terrain.width
    tot_terrain_length = terrain.length
    per_terrain_width = int( terrain.width / 9)
    per_terrain_length = terrain.length

    # need to opt
    heightfield = np.zeros((tot_terrain_width, tot_terrain_length), dtype=int)
    random_noise = random.uniform(0, 1)
    # 1st terrain
    terrain_1 = terrain_utils.SubTerrain("terrain_1",
                                         width=per_terrain_width,
                                         length=per_terrain_length,
                                         vertical_scale=0.5,
                                         horizontal_scale=0.5)
    heightfield[0:per_terrain_width, :] = terrain_utils.random_uniform_terrain(terrain_1,
                                                                               min_height=-0.75,
                                                                               max_height=3.5,
                                                                               step=0.5 + random_noise + difficulty * 0.1,
                                                                               downsampled_scale=0.5 - 0.5 * random_noise).height_field_raw
    # 2nd terrain
    terrain_2 = terrain_utils.SubTerrain("terrain_2",
                                         width=per_terrain_width,
                                         length=per_terrain_length,
                                         vertical_scale=0.5 + random_noise + difficulty * 0.1,
                                         horizontal_scale=0.75)
    heightfield[per_terrain_width:2 * per_terrain_width, :] = terrain_utils.pyramid_sloped_terrain(terrain_2,
                                                                                                   slope=-5.5 - 2 * random_noise).height_field_raw

    # 3rd terrain
    terrain_3 = terrain_utils.SubTerrain("terrain_3",
                                         width=per_terrain_width,
                                         length=per_terrain_length,
                                         vertical_scale=0.75,
                                         horizontal_scale=0.5)
    heightfield[2 * per_terrain_width:3 * per_terrain_width, :] = terrain_utils.sloped_terrain(terrain_3,
                                                                                               slope=2.5 + 1.5 * random_noise).height_field_raw
    max_height_terrain_5 = np.argmax(heightfield[3 * per_terrain_width:4 * per_terrain_width, :])

    max_height_terrain_3 = np.argmax(terrain_3.height_field_raw)
    max_height_terrain_3_idx_unravel = np.unravel_index(max_height_terrain_3, terrain_3.height_field_raw.shape)
    max_height_terrain_3_value = terrain_3.height_field_raw[max_height_terrain_3_idx_unravel]
    # 4th terrain -- up stairs
    terrain_4 = terrain_utils.SubTerrain("terrain_4",
                                         width=per_terrain_width,
                                         length=per_terrain_length,
                                         vertical_scale=0.15 + 0.1 * difficulty,
                                         horizontal_scale=0.05 + 0.00 * random_noise)
    heightfield[3 * per_terrain_width:4 * per_terrain_width, :] = terrain_utils.stairs_terrain(terrain_4,
                                                                                               step_width=.5,
                                                                                               step_height=4.5 + 0.2 * difficulty).height_field_raw
    max_height_terrain_4 = np.argmax(terrain_4.height_field_raw)
    max_height_terrain_4_idx_unravel = np.unravel_index(max_height_terrain_4, terrain_4.height_field_raw.shape)
    max_height_terrain_4_value = terrain_4.height_field_raw[max_height_terrain_4_idx_unravel]
    gap_between_34_terrains = abs(max_height_terrain_4_value - max_height_terrain_3_value)
    heightfield[3 * per_terrain_width:4 * per_terrain_width, :] += gap_between_34_terrains

    terrain_5 = terrain_utils.SubTerrain("terrain_5",
                                         width=per_terrain_width,
                                         length=per_terrain_length,
                                         vertical_scale=0.25,
                                         horizontal_scale=0.45)

    heightfield[4 * per_terrain_width:5 * per_terrain_width, :] = terrain_utils.pyramid_stairs_terrain(terrain_5,
                                                                                                       step_width=3.5,
                                                                                                       step_height=-3.5 + random_noise * difficulty).height_field_raw

    # max_height_terrain_5 = np.argmax(terrain_5.height_field_raw)
    # max_height_terrain_5_idx_unravel = np.unravel_index(max_height_terrain_5, terrain_5.height_field_raw.shape)
    # max_height_terrain_5_value = terrain_5.height_field_raw[max_height_terrain_5_idx_unravel]
    #
    # gap_between_45_terrains = abs(max_height_terrain_4_value - max_height_terrain_5_value)
    # heightfield[4 * per_terrain_width:5 * per_terrain_width, :] += gap_between_34_terrains + gap_between_45_terrains

    # 6th terrain down stairs
    terrain_6 = terrain_utils.SubTerrain("terrain_6",
                                         width=per_terrain_width,
                                         length=per_terrain_length,
                                         vertical_scale=-(0.15 + 0.1 * difficulty),
                                         horizontal_scale=0.05)
    heightfield[5 * per_terrain_width:6 * per_terrain_width, :] = terrain_utils.stairs_terrain(terrain_6,
                                                                                               step_width=.5,
                                                                                               step_height=(
                                                                                                           3.5 + 0.2 * difficulty)).height_field_raw#3.5

    terrain_7 = terrain_utils.SubTerrain("terrain_7",
                                         width=per_terrain_width,
                                         length=per_terrain_length,
                                         vertical_scale=2.5,
                                         horizontal_scale=0.5)
    x_range = [-0.05, 0.5 + 0.3 * difficulty]  # offset to stone_len
    y_range = [0.2, 0.3 + 0.1 * difficulty]
    stone_len = [0.9 - 0.3 * difficulty, 1 - 0.2 * difficulty]  # 2 * round((0.6) / 2.0, 1)
    incline_height = -0.95 * difficulty * 5
    last_incline_height = incline_height + 0.1 - 0.1 * difficulty
    last_stone_len = 3.0

    heightfield[6 * per_terrain_width:7 * per_terrain_width, :] = parkour_terrain(terrain_7,
                                                                                  num_stones=4,
                                                                                  x_range=x_range,
                                                                                  y_range=y_range,
                                                                                  incline_height=incline_height,
                                                                                  stone_len=stone_len,
                                                                                  stone_width=1.,
                                                                                  last_incline_height=last_incline_height,
                                                                                  last_stone_len=last_stone_len,
                                                                                  pad_height=0,
                                                                                  pit_depth=[-0.9,
                                                                                             20.]).height_field_raw

    # 8th terrain
    terrain_8 = terrain_utils.SubTerrain("terrain_8",
                                         width=per_terrain_width,
                                         length=per_terrain_length,
                                         vertical_scale=0.25 + 0.9 * random_noise,
                                         horizontal_scale=0.005)
    heightfield[7 * per_terrain_width:8 * per_terrain_width, :] = terrain_utils.wave_terrain(terrain_8,
                                                                                             num_waves=9 + round(
                                                                                                 3 * random_noise),
                                                                                             amplitude=9 + random_noise).height_field_raw

    # 9th terrain
    terrain_9 = terrain_utils.SubTerrain("terrain_9",
                                         width=per_terrain_width,
                                         length=per_terrain_length,
                                         vertical_scale=0.25,
                                         horizontal_scale=0.25)

    max_height_terrain_9 = 7.5 + 1 * random_noise
    stone_size = 3.2 + 2 * random_noise
    heightfield[8 * per_terrain_width:9 * per_terrain_width, :] = terrain_utils.stepping_stones_terrain(terrain_9,
                                                                                                        stone_size=stone_size,
                                                                                                        stone_distance=2.5 + 1 * random_noise,
                                                                                                        max_height=max_height_terrain_9,
                                                                                                        platform_size=0.5
                                                                                                        ).height_field_raw


    #random order
    heightfield_raw_random = np.zeros((tot_terrain_width, tot_terrain_length), dtype=int)
    cache_index =list(range(9))
    random.shuffle(cache_index)
    for i in range(num_terrains):
        heightfield_raw_random[i*per_terrain_width:(i+1)*per_terrain_width,:] = heightfield[cache_index[i]*per_terrain_width:(cache_index[i]+1)*per_terrain_width,:]
        if cache_index[i] == 8:
            mark_index = i
    terrain.height_field_raw = heightfield_raw_random
    #random order

    # set goals
    goals = np.zeros((8, 2))
    br_before = 10# ori is 10
    br = min(terrain.length // 2, br_before)
    start_cross_y = round(terrain.length // 2 + random.uniform(-br, br))
    start_cross_x = round(1.5 * per_terrain_width + random.uniform(br // 2, br))
    end_cross_y = round(terrain.length // 2 + random.uniform(-br, br))
    end_cross_x = terrain.width
    # last goals
    if mark_index < 8:
        goals[-1] = [end_cross_x, end_cross_y]
     #last goals
    size_scale = int(stone_size//2)
    diff_gap = 18

    small_noise_x = random.uniform(2, 8)
    small_noise_y = random.uniform(2, 8)

    # 8th goal need to be on stones
    cache_y_offset = (per_terrain_length // 8) #//8 default
    cache_y_offset_end = int(per_terrain_length-stone_size*2) # default os per_terrain_length - stone_size*2
    cache_x_offset_end  = round(8.5*per_terrain_width)
    default_coor = [per_terrain_width // 4, terrain.length // 4]
    height_th_for_goal = max_height_terrain_9//2
    select_from_height = np.argwhere(heightfield[8 * per_terrain_width:cache_x_offset_end, cache_y_offset:cache_y_offset_end] > height_th_for_goal)
    cache_height_anchor= random.choice(select_from_height) if len(select_from_height)!=0 else default_coor
    cache_height_anchor_x = cache_height_anchor[0]
    cache_height_anchor_y = cache_height_anchor[1] + cache_y_offset

    while not mid_sub_walkable(terrain, mid_sub_x=cache_height_anchor_x, mid_sub_y= cache_height_anchor_y,
                               terrain_index=6, size_scale=size_scale, per_terrain_width=per_terrain_width, diff_gap=2):
        if np.array_equal(cache_height_anchor, default_coor):
            break
        else:
            delete_index = np.argwhere(select_from_height == cache_height_anchor)[0]
            select_from_height = np.delete(select_from_height, delete_index,axis=0)
            if len(select_from_height)!=0:
                cache_height_anchor=random.choice(select_from_height)
                cache_height_anchor_x = cache_height_anchor[0]
                cache_height_anchor_y = cache_height_anchor[1] + cache_y_offset
            else:
                cache_height_anchor = default_coor
                cache_height_anchor_x =cache_height_anchor[0]
                cache_height_anchor_y = cache_height_anchor[1] + cache_y_offset
                break




    #8th goals need to be on stones

    if traj_direction == "mid":

        # traj-1 : around mid points
        for k in range(mark_index-2):
            mid_sub_x = round(per_terrain_width // 2)
            mid_sub_walkable(terrain, mid_sub_x=mid_sub_x, mid_sub_y= round(terrain.length//2), terrain_index=k, size_scale=5,
                             per_terrain_width=per_terrain_width, diff_gap=diff_gap)
            start_corss_sub_y = round(terrain.length // 2 + small_noise_y)
            start_cross_sub_x = per_terrain_width * (k + 3) - (mid_sub_x if mid_sub_walkable else -small_noise_x)
            goals[k + 1] = [start_cross_sub_x, start_corss_sub_y]
        for k in range(mark_index-1,num_terrains-3):
            mid_sub_x = round(per_terrain_width // 2)
            mid_sub_walkable(terrain, mid_sub_x=mid_sub_x, mid_sub_y=round(terrain.length // 2), terrain_index=k,
                             size_scale=5,
                             per_terrain_width=per_terrain_width, diff_gap=diff_gap)
            start_corss_sub_y = round(terrain.length // 2 + small_noise_y)
            start_cross_sub_x = per_terrain_width * (k + 3) - (mid_sub_x if mid_sub_walkable else -small_noise_x)
            goals[k + 1] = [start_cross_sub_x, start_corss_sub_y]
    elif traj_direction in ["uni_random", "gauss"]:
        # traj-2/3:num_terrains random/gaussian
        if mark_index > 1:
            left_mark_range = range(1, mark_index)
            right_mark_range = range(mark_index+1, num_terrains-2)
            range_waypoint_list = itertools.chain(left_mark_range, right_mark_range)
        else:
            range_waypoint_list = range(1, num_terrains-2)

        for k in range_waypoint_list:
            sampled_target_x = random.uniform(small_noise_x, per_terrain_width)
            if traj_direction == "uni_random":
                sampled_target_y_plus = random.uniform(small_noise_y, round(per_terrain_length // 2))
                sampled_target_y_sub = random.uniform(-round(per_terrain_length // 2), small_noise_y)
                sampled_target_y = np.random.choice([sampled_target_y_plus, sampled_target_y_sub])

            elif traj_direction == "gauss":
                sampled_target_y = random.gauss(per_terrain_length // 4, round(small_noise_y // 2))

            start_cross_sub_x = round(per_terrain_width * (k+1) + sampled_target_x)
            start_cross_sub_y = round(terrain.length // 2 + sampled_target_y)
            goals[k] = [start_cross_sub_x, start_cross_sub_y]

    # reset goal-steping-stone
    goals[0] =[start_cross_x,start_cross_y]
    if mark_index in [0,1]:
        goals[0] = [mark_index*per_terrain_width + cache_height_anchor_x, cache_height_anchor_y]
    elif mark_index == 8:
        goals[-1] = [mark_index * per_terrain_width + cache_height_anchor_x, cache_height_anchor_y]
    else:
        goals[mark_index-1] = [mark_index * per_terrain_width + cache_height_anchor_x, cache_height_anchor_y]
    #reset goal-steping_stone
    terrain.goals = goals * terrain.horizontal_scale

    #need to opt




def cross_country_mini(terrain, difficulty=1, num_terrains=9, traj_direction = "uni_random"):

    '''
    try to make 9 terrains in one env from the subset of all possible types, concat them along x-axis, need to modify into random combination
    '''
    tot_terrain_width = terrain.width
    tot_terrain_length = terrain.length
    per_terrain_width = int( terrain.width / num_terrains)
    per_terrain_length = terrain.length
    per_terrain_horizontal_scale = terrain.horizontal_scale
    per_terrain_vertical_scale = terrain.vertical_scale

    heightfield = np.zeros((tot_terrain_width,tot_terrain_length), dtype=int)
    random_noise = random.uniform(0, 1)
    # 1st terrain
    terrain_1 = terrain_utils.SubTerrain("terrain_1",
                              width=per_terrain_width,
                              length=per_terrain_length,
                              vertical_scale=0.5,
                              horizontal_scale=0.5)
    heightfield[0:per_terrain_width,:] = terrain_utils.random_uniform_terrain(terrain_1,
                                                                              min_height=-0.75,
                                                                              max_height=3.5,
                                                                              step=0.5+random_noise+difficulty*0.1,
                                                                              downsampled_scale=0.5-0.5*random_noise).height_field_raw
    # 2nd terrain
    terrain_2 = terrain_utils.SubTerrain("terrain_2",
                                         width=per_terrain_width,
                                         length=per_terrain_length,
                                         vertical_scale=0.5+random_noise+difficulty*0.1,
                                         horizontal_scale=0.75)
    heightfield[per_terrain_width:2*per_terrain_width,:] = terrain_utils.pyramid_sloped_terrain(terrain_2,slope=-5.5-2*random_noise).height_field_raw

    # 3rd terrain
    terrain_3 = terrain_utils.SubTerrain("terrain_3",
                                         width=per_terrain_width,
                                         length=per_terrain_length,
                                         vertical_scale=0.75,
                                         horizontal_scale=0.5)
    heightfield[2*per_terrain_width:3*per_terrain_width,:] = terrain_utils.sloped_terrain(terrain_3, slope=2.5+1.5*random_noise).height_field_raw
    max_height_terrain_5 = np.argmax(heightfield[3*per_terrain_width:4*per_terrain_width,:])

    max_height_terrain_3 = np.argmax(terrain_3.height_field_raw)
    max_height_terrain_3_idx_unravel =np.unravel_index(max_height_terrain_3,terrain_3.height_field_raw.shape)
    max_height_terrain_3_value = terrain_3.height_field_raw[max_height_terrain_3_idx_unravel]
    # 4th terrain -- up stairs
    terrain_4 = terrain_utils.SubTerrain("terrain_4",
                                         width=per_terrain_width,
                                         length=per_terrain_length,
                                         vertical_scale=0.15+0.1*difficulty,
                                         horizontal_scale=0.05+0.00*random_noise)
    heightfield[3*per_terrain_width:4*per_terrain_width,:] = terrain_utils.stairs_terrain(terrain_4,
                                                                                        step_width=.5,
                                                                                        step_height=6.5+0.2*difficulty).height_field_raw
    max_height_terrain_4 = np.argmax(terrain_4.height_field_raw)
    max_height_terrain_4_idx_unravel =np.unravel_index(max_height_terrain_4,terrain_4.height_field_raw.shape)
    max_height_terrain_4_value = terrain_4.height_field_raw[max_height_terrain_4_idx_unravel]
    gap_between_34_terrains = abs(max_height_terrain_4_value-max_height_terrain_3_value)
    heightfield[3*per_terrain_width:4*per_terrain_width,:]+=gap_between_34_terrains

    terrain_5 = terrain_utils.SubTerrain("terrain_5",
                                         width=per_terrain_width,
                                         length=per_terrain_length,
                                         vertical_scale=0.25,
                                         horizontal_scale=0.45)

    heightfield[4 * per_terrain_width:5 * per_terrain_width, :] =  terrain_utils.pyramid_stairs_terrain(terrain_5,
                                                                                                       step_width=3.5,
                                                                                                       step_height=-7.5+ random_noise*difficulty).height_field_raw

    max_height_terrain_5 = np.argmax(terrain_5.height_field_raw)
    max_height_terrain_5_idx_unravel = np.unravel_index(max_height_terrain_5,terrain_5.height_field_raw.shape)
    max_height_terrain_5_value = terrain_5.height_field_raw[max_height_terrain_5_idx_unravel]

    gap_between_45_terrains = abs(max_height_terrain_4_value - max_height_terrain_5_value)
    heightfield[4*per_terrain_width:5*per_terrain_width, :] += gap_between_34_terrains + gap_between_45_terrains


    # 6th terrain down stairs
    terrain_6 = terrain_utils.SubTerrain("terrain_6",
                                         width=per_terrain_width,
                                         length=per_terrain_length,
                                         vertical_scale=-(0.15+0.1*difficulty),
                                         horizontal_scale=0.05)
    heightfield[5*per_terrain_width:6*per_terrain_width,:] = terrain_utils.stairs_terrain(terrain_6,
                                                                                        step_width=.5,
                                                                                        step_height=(6.5+0.2*difficulty)).height_field_raw
    max_height_terrain_6 = np.argmax(terrain_6.height_field_raw)
    max_height_terrain_6_idx_unravel =np.unravel_index(max_height_terrain_6,terrain_4.height_field_raw.shape)
    max_height_terrain_6_value = terrain_6.height_field_raw[max_height_terrain_6_idx_unravel]
    gap_between_56_terrains = abs(max_height_terrain_6_value-max_height_terrain_5_value)
    heightfield[5*per_terrain_width:6*per_terrain_width,:]+=gap_between_56_terrains +max_height_terrain_5_value + gap_between_34_terrains +gap_between_45_terrains

    # 7th terrain
    # find min height of terrain-6
    min_height_terrain_6 = np.argmin(terrain_6.height_field_raw)
    min_height_terrain_6_idx_unravel = np.unravel_index(min_height_terrain_6,terrain_6.height_field_raw.shape)
    min_height_terrain_6_value = terrain_6.height_field_raw[min_height_terrain_6_idx_unravel]
    gap_between_67_terrains = min_height_terrain_6_value + gap_between_56_terrains + max_height_terrain_5_value +gap_between_45_terrains
    # find min height of terrain-6

    terrain_7 = terrain_utils.SubTerrain("terrain_7",
                                         width=per_terrain_width,
                                         length=per_terrain_length,
                                         vertical_scale=2.5,
                                         horizontal_scale=0.5)
    x_range = [-0.05, 0.5 + 0.3 * difficulty]  # offset to stone_len
    y_range = [0.2, 0.3 + 0.1 * difficulty]
    stone_len = [0.9 - 0.3 * difficulty, 1 - 0.2 * difficulty]  # 2 * round((0.6) / 2.0, 1)
    incline_height = -0.95 * difficulty*5
    last_incline_height = incline_height + 0.1 - 0.1 * difficulty
    last_stone_len = 3.0

    heightfield[6*per_terrain_width:7*per_terrain_width,:] = parkour_terrain(terrain_7,
                                                                             num_stones=2,
                                                                             x_range=x_range,
                                                                             y_range=y_range,
                                                                             incline_height=incline_height,
                                                                             stone_len=stone_len,
                                                                             stone_width=1.,
                                                                             last_incline_height=last_incline_height,
                                                                             last_stone_len=last_stone_len,
                                                                             pad_height=gap_between_67_terrains//2,
                                                                             pit_depth=[-0.9,20.]).height_field_raw

    heightfield[6*per_terrain_width:7*per_terrain_width,:] += terrain_utils.discrete_obstacles_terrain(terrain_7,
                                                                                                    max_height=gap_between_67_terrains*3.5,
                                                                                                    min_size=.6,
                                                                                                       max_size=18.,
                                                                                                       num_rects=20).height_field_raw

    # 8th terrain
    terrain_8 = terrain_utils.SubTerrain("terrain_8",
                                         width=per_terrain_width,
                                         length=per_terrain_length,
                                         vertical_scale=0.25+0.9*random_noise,
                                         horizontal_scale=0.005)
    heightfield[7*per_terrain_width:8*per_terrain_width,:] = terrain_utils.wave_terrain(terrain_8,num_waves=9+round(3*random_noise), amplitude=9+ random_noise).height_field_raw





    #9th terrain
    stone_size_9 = 1.5+2*random_noise
    max_height_terrain_9 = 7.5 +1*random_noise
    terrain_9 = terrain_utils.SubTerrain("terrain_9",
                                         width=per_terrain_width,
                                         length=per_terrain_length,
                                         vertical_scale=0.25,
                                         horizontal_scale=0.25)
    heightfield[8*per_terrain_width:9*per_terrain_width,:]= terrain_utils.stepping_stones_terrain(terrain_9,
                                                                                                  stone_size=stone_size_9,
                                                                                                  stone_distance=1.5+ 1*random_noise,
                                                                                                  max_height=max_height_terrain_9,
                                                                                                  platform_size=0.5
                                                                                                  ).height_field_raw

    terrain.height_field_raw = heightfield
    # 8th goal need to be on stones
    height_th_for_goal = round(max_height_terrain_9//2)
    select_from_height = np.argwhere(heightfield[8 * per_terrain_width:9 * per_terrain_width, :] > height_th_for_goal)
    cache_height_anchor = random.choice(select_from_height)
    cache_height_anchor_x_offset = cache_height_anchor[0]
    cache_height_anchor_y_offset = cache_height_anchor[1]
    cache_height_anchor_x = round(cache_height_anchor_x_offset)
    cache_height_anchor_y = round(cache_height_anchor_y_offset)

    while not mid_sub_walkable(terrain, mid_sub_x=cache_height_anchor_x, mid_sub_y= cache_height_anchor_y,
                               terrain_index=6, size_scale=int(stone_size_9//2), per_terrain_width=per_terrain_width, diff_gap=1):

         select_from_height = np.delete(select_from_height, cache_height_anchor)
         cache_height_anchor = random.choice(select_from_height)
         cache_height_anchor_x_offset = cache_height_anchor[0]
         cache_height_anchor_y_offset = cache_height_anchor[1]
         cache_height_anchor_x = round(cache_height_anchor_x_offset)
         cache_height_anchor_y = round(cache_height_anchor_y_offset)






    #set goals
    goals = np.zeros((8, 2))
    br_before =10
    br = min(terrain.length//2, br_before)
    start_corss_y = round(terrain.length//2 + random.uniform(-br,br))
    start_corss_x = round(2*per_terrain_width + random.uniform(br//2,br))
    end_corss_y = round(terrain.length//2 + random.uniform(-br,br))
    end_corss_x =terrain.width + random.uniform(br,2*br)
    goals[0] = [start_corss_x,start_corss_y]
    goals[-1] = [end_corss_x,end_corss_y]
    size_scale = 5
    diff_gap = 18

    small_noise_x = random.uniform(2,8)
    small_noise_y = random.uniform(2,8)


    if traj_direction == "mid":

        #traj-1 : around mid points
        for i in range(6):

            mid_sub_x = round(per_terrain_width//2)
            mid_sub_walkable(terrain, mid_sub_x=mid_sub_x, mid_sub_y=round(terrain.length//2), terrain_index=i,size_scale=size_scale, per_terrain_width=per_terrain_width,diff_gap=diff_gap)
            start_cross_sub_x = per_terrain_width * (i+4) - (mid_sub_x if mid_sub_walkable else -small_noise_x)
            start_corss_sub_y = terrain.length//2 + small_noise_y
            goals[i+1] = [start_cross_sub_x,start_corss_sub_y]
    elif traj_direction in ["uni_random", "gauss"]:
        #traj-2:
        for i in range(6):
            sampled_target_x = random.uniform(small_noise_x,per_terrain_width)
            if traj_direction == "uni_random":
                sampled_target_y_plus = random.uniform(small_noise_y, round(per_terrain_length//2))
                sampled_target_y_sub = random.uniform(-round(per_terrain_length//2), small_noise_y)
                sampled_target_y = np.random.choice([sampled_target_y_plus,sampled_target_y_sub])

            elif traj_direction == "gauss":
                sampled_target_y = random.gauss(per_terrain_length//4, round(small_noise_y//2))

            start_cross_sub_x = per_terrain_width * (i+3) + sampled_target_x
            start_corss_sub_y = terrain.length//2 + sampled_target_y
            goals[i+1] = [start_cross_sub_x,start_corss_sub_y]
    goals[6] = [cache_height_anchor_x + 8 * per_terrain_width, cache_height_anchor_y]
    terrain.goals = goals * terrain.horizontal_scale
def mid_sub_walkable(terrain, mid_sub_x=20, mid_sub_y = 20, terrain_index=0,size_scale=1,per_terrain_width=40, diff_gap=2):
    left_upper_re =[mid_sub_x+size_scale, mid_sub_y -size_scale]
    right_upper_re =[mid_sub_x-size_scale, mid_sub_y - size_scale]
    left_down_re = [mid_sub_x+size_scale, mid_sub_y + size_scale]
    right_down_re = [mid_sub_x-size_scale, mid_sub_y + size_scale]

    surrand_height = terrain.height_field_raw[(terrain_index+2)*per_terrain_width+left_upper_re[0],left_upper_re[1]]+ terrain.height_field_raw[(terrain_index+2)*per_terrain_width+right_upper_re[0],right_upper_re[1]]+ terrain.height_field_raw[(terrain_index+2)*per_terrain_width+left_down_re[0],left_down_re[1]]+ terrain.height_field_raw[(terrain_index+2)*per_terrain_width+right_down_re[0],right_down_re[1]]

    center_height = terrain.height_field_raw[(terrain_index+2)*per_terrain_width+mid_sub_x,round(terrain.length//2)]

    return  True if (abs(surrand_height-center_height) < diff_gap*4 )else False





def demo_terrain(terrain):
    goals = np.zeros((8, 2))
    mid_y = terrain.length // 2
    
    # hurdle
    platform_length = round(2 / terrain.horizontal_scale)
    hurdle_depth = round(np.random.uniform(0.35, 0.4) / terrain.horizontal_scale)
    hurdle_height = round(np.random.uniform(0.3, 0.36) / terrain.vertical_scale)
    hurdle_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[0] = [platform_length + hurdle_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+hurdle_depth, round(mid_y-hurdle_width/2):round(mid_y+hurdle_width/2)] = hurdle_height
    
    # step up
    platform_length += round(np.random.uniform(1.5, 2.5) / terrain.horizontal_scale)
    first_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    first_step_height = round(np.random.uniform(0.35, 0.45) / terrain.vertical_scale)
    first_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[1] = [platform_length+first_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+first_step_depth, round(mid_y-first_step_width/2):round(mid_y+first_step_width/2)] = first_step_height
    
    platform_length += first_step_depth
    second_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    second_step_height = first_step_height
    second_step_width = first_step_width
    goals[2] = [platform_length+second_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+second_step_depth, round(mid_y-second_step_width/2):round(mid_y+second_step_width/2)] = second_step_height
    
    # gap
    platform_length += second_step_depth
    gap_size = round(np.random.uniform(0.5, 0.8) / terrain.horizontal_scale)
    
    # step down
    platform_length += gap_size
    third_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    third_step_height = first_step_height
    third_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[3] = [platform_length+third_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+third_step_depth, round(mid_y-third_step_width/2):round(mid_y+third_step_width/2)] = third_step_height
    
    platform_length += third_step_depth
    forth_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    forth_step_height = first_step_height
    forth_step_width = third_step_width
    goals[4] = [platform_length+forth_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+forth_step_depth, round(mid_y-forth_step_width/2):round(mid_y+forth_step_width/2)] = forth_step_height
    
    # parkour
    platform_length += forth_step_depth
    gap_size = round(np.random.uniform(0.1, 0.4) / terrain.horizontal_scale)
    platform_length += gap_size
    
    left_y = mid_y + round(np.random.uniform(0.15, 0.3) / terrain.horizontal_scale)
    right_y = mid_y - round(np.random.uniform(0.15, 0.3) / terrain.horizontal_scale)
    
    slope_height = round(np.random.uniform(0.15, 0.22) / terrain.vertical_scale)
    slope_depth = round(np.random.uniform(0.75, 0.85) / terrain.horizontal_scale)
    slope_width = round(1.0 / terrain.horizontal_scale)
    
    platform_height = slope_height + np.random.randint(0, 0.2 / terrain.vertical_scale)

    goals[5] = [platform_length+slope_depth/2, left_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * 1
    terrain.height_field_raw[platform_length:platform_length+slope_depth, left_y-slope_width//2: left_y+slope_width//2] = heights.astype(int) + platform_height
    
    platform_length += slope_depth + gap_size
    goals[6] = [platform_length+slope_depth/2, right_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * -1
    terrain.height_field_raw[platform_length:platform_length+slope_depth, right_y-slope_width//2: right_y+slope_width//2] = heights.astype(int) + platform_height
    
    platform_length += slope_depth + gap_size + round(0.4 / terrain.horizontal_scale)
    goals[-1] = [platform_length, left_y]
    terrain.goals = goals * terrain.horizontal_scale

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def half_sloped_terrain(terrain, wall_width=4, start2center=0.7, max_height=1):
    wall_width_int = max(int(wall_width / terrain.horizontal_scale), 1)
    max_height_int = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.length // 2)
    terrain_length = terrain.length
    height2width_ratio = max_height_int / wall_width_int
    xs = np.arange(slope_start, terrain_length)
    heights = (height2width_ratio * (xs - slope_start)).clip(max=max_height_int).astype(np.int16)
    terrain.height_field_raw[slope_start:terrain_length, :] = heights[:, None]
    terrain.slope_vector = np.array([wall_width_int*terrain.horizontal_scale, 0., max_height]).astype(np.float32)
    terrain.slope_vector /= np.linalg.norm(terrain.slope_vector)
    # print(terrain.slope_vector, wall_width)
    # import matplotlib.pyplot as plt
    # plt.imsave('test.png', terrain.height_field_raw, cmap='gray')

def half_platform_terrain(terrain, start2center=2, max_height=1):
    max_height_int = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.length // 2)
    terrain_length = terrain.length
    terrain.height_field_raw[:, :] = max_height_int
    terrain.height_field_raw[-slope_start:slope_start, -slope_start:slope_start] = 0
    # print(terrain.slope_vector, wall_width)
    # import matplotlib.pyplot as plt
    # plt.imsave('test.png', terrain.height_field_raw, cmap='gray')

def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1., depth=-1):
    """
    Generate a stepping stones terrain

    Parameters:
        terrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    def get_rand_dis_int(scale):
        return np.random.randint(int(- scale / terrain.horizontal_scale + 1), int(scale / terrain.horizontal_scale))
    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height-1, max_height, step=1)

    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # fill first hole
            stop_x = max(0, start_x - stone_distance - get_rand_dis_int(0.2))
            terrain.height_field_raw[0: stop_x, start_y: stop_y] = np.random.choice(height_range)
            # fill row
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_x += stone_size + stone_distance + get_rand_dis_int(0.2)
            start_y += stone_size + stone_distance + get_rand_dis_int(0.2)
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # fill first hole
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x: stop_x, 0: stop_y] = np.random.choice(height_range)
            # fill column
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain

def convert_heightfield_to_trimesh_delatin(height_field_raw, horizontal_scale, vertical_scale, max_error=0.01):
    mesh = Delatin(np.flip(height_field_raw, axis=1).T, z_scale=vertical_scale, max_error=max_error)
    vertices = np.zeros_like(mesh.vertices)
    vertices[:, :2] = mesh.vertices[:, :2] * horizontal_scale
    vertices[:, 2] = mesh.vertices[:, 2]
    return vertices, mesh.triangles

def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles, move_x != 0


class MeshTerrain:
    def __init__(self, heigthmap_data, device):
        self.border_size = 20
        self.border = 500
        self.sample_extent_x = 300
        self.sample_extent_y = 300
        self.vertical_scale = 1
        self.device = device

        self.heightsamples = torch.from_numpy(heigthmap_data['heigthmap']).to(device)
        self.walkable_map  = torch.from_numpy(heigthmap_data['walkable_map']).to(device)
        self.cam_pos = torch.from_numpy(heigthmap_data['cam_pos'])
        self.x_scale = heigthmap_data['x_scale']
        self.y_scale = heigthmap_data['y_scale']

        self.x_shape, self.y_shape = self.heightsamples.shape
        self.x_c = (self.x_shape / 2) / self.x_scale
        self.y_c = (self.y_shape / 2) / self.y_scale

        coord_x, coord_y = torch.where(self.walkable_map == 1) # Image coordinates, need to flip y and x
        coord_x, coord_y = coord_x.float(), coord_y.float()
        self.coord_x_scale = coord_x / self.x_scale  - self.x_c
        self.coord_y_scale = coord_y / self.y_scale  - self.y_c

        self.coord_x_scale += self.cam_pos[0]
        self.coord_y_scale += self.cam_pos[1]

        self.num_samples = self.coord_x_scale.shape[0]


    def sample_valid_locations(self, num_envs, env_ids):
        num_envs = env_ids.shape[0]
        idxes = np.random.randint(0, self.num_samples, size=num_envs)
        valid_locs = torch.stack([self.coord_x_scale[idxes], self.coord_y_scale[idxes]], dim = -1)
        return valid_locs

    def world_points_to_map(self, points):
        points[..., 0] -= self.cam_pos[0] - self.x_c
        points[..., 1] -= self.cam_pos[1] - self.y_c
        points[..., 0] *= self.x_scale
        points[..., 1] *= self.y_scale
        points = (points).long()

        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)

        px = torch.clip(px, 0, self.heightsamples.shape[0] -
                        2)  # image, so sampling 1 is for x
        py = torch.clip(py, 0, self.heightsamples.shape[1] - 2)
        return px, py

    def sample_height_points(self,
                             points,
                             root_states = None,
                             root_points=None,
                             env_ids=None,
                             num_group_people=512,
                             group_ids=None):

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
            heightsamples_group = heightsamples[None, ].repeat(
                num_groups, 1, 1)


            root_px, root_py = root_px.view(-1, num_group_people * num_root_points), root_py.view(-1, num_group_people * num_root_points)
            px, py = px.view(-1, N), py.view(-1, N)
            heights = torch.zeros(px.shape).to(px.device)

            if not root_states is None:
                linear_vel = root_states[:, 7:10]  # This contains ALL the linear velocities
                root_rot = root_states[:, 3:7]
                heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
                velocity_map = torch.zeros([px.shape[0], px.shape[1],
                                            2]).to(root_states)
                velocity_map_group = torch.zeros(heightsamples_group.shape +
                                                 (3, )).to(points)

            for idx in range(num_groups):
                heightsamples_group[idx][root_px[idx],root_py[idx]] += torch.tensor(1.7 / self.vertical_scale)
                # heightsamples_group[idx][root_px[idx] + 1,root_py[idx] + 1] += torch.tensor(1.7 / self.vertical_scale)
                group_mask_env_ids = group_ids[env_ids] == idx  # agents to select for this group from the current env_ids
                # if sum(group_mask) == 0:
                #     continue
                group_px, group_py = px[group_mask_env_ids].view(-1), py[group_mask_env_ids].view(-1)
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
                        idx, root_px[idx],
                        root_py[idx], :] = group_linear_vel.repeat(
                            1, root_points.shape[1]).view(-1, 3) # Make sure that the order is correct.

                    # Then sampling the points
                    vel_group = velocity_map_group[idx][group_px, group_py]
                    vel_group = vel_group.view(-1, N, 3)
                    vel_group -= linear_vel[env_ids_in_group, None]  # this is one-to-one substraction of the agents in the group to mark the static terrain with relative velocity
                    group_heading_rot = heading_rot[env_ids_in_group]

                    group_vel_idv = torch_utils.my_quat_rotate(
                        group_heading_rot.repeat(1, N).view(-1, 4),
                        vel_group.view(-1, 3)
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
                velocity_map[:] = velocity_map[:] - linear_vel_ego[:, None, :2] # Flip velocity to be in agent's point of view
                heights = (heights * self.vertical_scale).view(B, -1, 1)
                return torch.cat([heights, velocity_map], dim = -1)



class SubTerrain:
    def __init__(self, terrain_name="terrain", width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)