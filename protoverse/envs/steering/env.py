import numpy as np
from typing import Dict, Optional

import torch
from torch import Tensor
from isaac_utils import rotations, torch_utils
from protoverse.envs.base_env.env import BaseEnv
from protoverse.simulator.base_simulator.config import (
    MarkerConfig,
    VisualizationMarker,
    MarkerState,
)


class Steering(BaseEnv):
    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config=config, device=device, *args, **kwargs)

        self._tar_speed_min = self.config.steering_params.tar_speed_min
        self._tar_speed_max = self.config.steering_params.tar_speed_max

        self._heading_change_steps_min = (
            self.config.steering_params.heading_change_steps_min
        )
        self._heading_change_steps_max = (
            self.config.steering_params.heading_change_steps_max
        )
        self._random_heading_probability = (
            self.config.steering_params.random_heading_probability
        )
        self._standard_heading_change = (
            self.config.steering_params.standard_heading_change
        )
        self._standard_speed_change = self.config.steering_params.standard_speed_change
        self._stop_probability = self.config.steering_params.stop_probability

        self.steering_obs = torch.zeros(
            (self.config.num_envs, self.config.steering_params.obs_size),
            device=device,
            dtype=torch.float,
        )

        self._heading_change_steps = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.int64
        )
        self._prev_root_pos = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float
        )

        self._tar_dir_theta = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self._tar_dir = torch.zeros(
            [self.num_envs, 2], device=self.device, dtype=torch.float
        )
        self._tar_dir[..., 0] = 1.0

        self._tar_speed = torch.ones(
            [self.num_envs], device=self.device, dtype=torch.float
        )

    def create_visualization_markers(self):
        if self.config.headless:
            return {}

        visualization_markers = super().create_visualization_markers()

        steering_markers = []
        steering_markers.append(MarkerConfig(size="regular"))
        steering_markers_cfg = VisualizationMarker(
            type="arrow", color=(0.0, 1.0, 1.0), markers=steering_markers
        )
        visualization_markers["steering_markers"] = steering_markers_cfg

        return visualization_markers

    def get_markers_state(self):
        if self.config.headless:
            return {}

        markers_state = super().get_markers_state()

        marker_root_pos = self.simulator.get_root_state().root_pos
        marker_root_pos[..., 0:2] += self._tar_dir

        heading_axis = torch.zeros_like(marker_root_pos)
        heading_axis[..., -1] = 1.0
        marker_rot = rotations.quat_from_angle_axis(
            self._tar_dir_theta, heading_axis, True
        )
        markers_state["steering_markers"] = MarkerState(
            translation=marker_root_pos.view(self.num_envs, -1, 3),
            orientation=marker_rot.view(self.num_envs, -1, 4),
        )

        return markers_state

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if len(env_ids) > 0:
            self.reset_heading_task(env_ids)
        return super().reset(env_ids)

    def post_physics_step(self):
        super().post_physics_step()
        self.check_update_task()

    def check_update_task(self):
        reset_task_mask = self.progress_buf >= self._heading_change_steps
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            self.reset_heading_task(rest_env_ids)

    def reset_heading_task(self, env_ids):
        n = len(env_ids)
        if np.random.binomial(1, self._random_heading_probability):
            dir_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
            tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(
                n, device=self.device
            ) + self._tar_speed_min
        else:
            dir_delta_theta = (
                2 * self._standard_heading_change * torch.rand(n, device=self.device)
                - self._standard_heading_change
            )
            # map tar_dir_theta back to [0, 2pi], add delta, project back into [0, 2pi] and then shift.
            dir_theta = (dir_delta_theta + self._tar_dir_theta[env_ids] + np.pi) % (
                2 * np.pi
            ) - np.pi

            speed_delta = (
                2 * self._standard_speed_change * torch.rand(n, device=self.device)
                - self._standard_speed_change
            )
            tar_speed = torch.clamp(
                speed_delta + self._tar_speed[env_ids],
                min=self._tar_speed_min,
                max=self._tar_speed_max,
            )

        tar_dir = torch.stack([torch.cos(dir_theta), torch.sin(dir_theta)], dim=-1)

        change_steps = torch.randint(
            low=self._heading_change_steps_min,
            high=self._heading_change_steps_max,
            size=(n,),
            device=self.device,
            dtype=torch.int64,
        )

        stop_probs = torch.ones(n, device=self.device) * self._stop_probability
        should_stop = torch.bernoulli(stop_probs)

        self._tar_speed[env_ids] = tar_speed * (1.0 - should_stop)
        self._tar_dir_theta[env_ids] = dir_theta
        self._tar_dir[env_ids] = tar_dir
        self._heading_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps

    def compute_observations(self, env_ids=None):
        super().compute_observations(env_ids)

        if env_ids is None:
            root_states = self.simulator.get_root_state()
            tar_dir = self._tar_dir
            tar_speed = self._tar_speed
        else:
            root_states = self.simulator.get_root_state(env_ids)
            tar_dir = self._tar_dir[env_ids]
            tar_speed = self._tar_speed[env_ids]

        obs = compute_heading_observations(root_states.root_rot, tar_dir, tar_speed)
        self.steering_obs[env_ids] = obs

    def get_obs(self):
        obs = super().get_obs()
        obs.update({"steering": self.steering_obs})
        return obs

    def compute_reward(self):

        # root_pos = self.simulator.get_root_state().root_pos
        # path_rew = compute_heading_reward(
        #     root_pos, self._prev_root_pos, self._tar_dir, self._tar_speed, self.dt
        # )
        # self._prev_root_pos[:] = root_pos

        rew_dict = self.rew_manager.compute_rewards()
        # rew_dict["path_rew"] = 2.0 * path_rew

        # scaled_rewards: Dict[str, Tensor] = {  # move to rew class
        #     k: v
        #     * 1.0  # ,getattr(self.config.reward_config.reward_scales, f"{k}_w")
        #     for k, v in rew_dict.items()
        # }

        self.rew_buf = sum(rew_dict.values())

        # logging & fun

        for rew_name, rew in rew_dict.items():
            self.log_dict[f"raw/{rew_name}_mean"] = rew.mean()
            self.log_dict[f"raw/{rew_name}_std"] = rew.std()

        # ----
        # Aux:
        # dof_state = self.simulator.get_dof_state()
        # root_states = self.simulator.get_root_state(d)
        # bodies_contact_buf = self.simulator.get_bodies_contact_buf()

        # # 2)
        # rew_lin_vel_z = torch.square(root_states.root_vel[:, 2])
        # rew_dict["rew_lin_vel_z"] = -2.0 * rew_lin_vel_z

        # 3)
        # rew_ang_vel_xy = torch.sum(torch.square(root_states.root_ang_vel[:, :2]), dim=1)
        # rew_dict["rew_ang_vel_xy"] = -0.05 * rew_ang_vel_xy

        # 4)
        # rew_torque = torch.sum(torch.square(self.simulator.torques), dim=1)
        # rew_dict["rew_torque"] = -0.00001 * rew_torque

        # 5)
        # rew_dof_acc = torch.sum(
        #     torch.square((self.last_dof_vel - dof_state.dof_vel) / self.dt), dim=1
        # )
        # rew_dict["rew_dof_acc"] = -2.5e-7 * rew_dof_acc

        # 6)
        # contact = bodies_contact_buf[:, self.feet_indices, 2] > 1.0
        # contact_filt = torch.logical_or(contact, self.last_contacts)
        # self.last_contacts = contact
        # first_contact = (self.feet_air_time > 0.0) * contact_filt
        # self.feet_air_time += self.dt
        # rew_feet_air_time = torch.sum(
        #     (self.feet_air_time - 0.5) * first_contact, dim=1
        # )  #  reward only on first contact with the ground
        # # rew_feet_air_time *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        # self.feet_air_time *= ~contact_filt
        # rew_dict["rew_feet_air_time"] = 1.0 * rew_feet_air_time

        # 7)
        # rew_collision = torch.sum(
        #     1.0
        #     * (
        #         torch.norm(bodies_contact_buf[:, self.penelized_body_ids, :], dim=-1)
        #         > 0.1
        #     ),
        #     dim=1,
        # )
        # rew_dict["rew_collision"] = -1.0 * rew_collision

        # 8)
        # rew_action_rate = torch.sum(
        #     torch.square(self.last_actions - self.actions), dim=1
        # )
        # rew_dict["rew_action_rate"] = -0.01 * rew_action_rate

        # ------------


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_heading_observations(
    root_rot: Tensor, tar_dir: Tensor, tar_speed: Tensor
) -> Tensor:
    tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, True)

    local_tar_dir = rotations.quat_rotate(heading_rot, tar_dir3d, True)
    local_tar_dir = local_tar_dir[..., 0:2]

    tar_speed = tar_speed.unsqueeze(-1)

    obs = torch.cat([local_tar_dir, tar_speed], dim=-1)
    return obs
