from isaac_utils import torch_utils, rotations
from typing import Dict, Optional

import torch
from torch import Tensor
from protoverse.envs.path_follower.path_generator import PathGenerator
from protoverse.envs.base_env.env import BaseEnv
from protoverse.simulator.base_simulator.config import (
    MarkerConfig,
    VisualizationMarker,
    MarkerState,
)


class PathFollowing(BaseEnv):
    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config=config, device=device, *args, **kwargs)

        head_body_name = self.config.robot.head_body_name
        self.head_body_id = self.simulator.build_body_ids_tensor(
            [head_body_name]
        ).item()

        self._num_traj_samples = self.config.path_follower_params.num_traj_samples
        self._traj_sample_timestep = (
            self.config.path_follower_params.traj_sample_timestep
        )

        self.path_obs = torch.zeros(
            (
                self.config.num_envs,
                self.config.path_follower_params.path_obs_size,
            ),
            device=device,
            dtype=torch.float,
        )

        self._fail_dist = 4.0
        self._fail_height_dist = 0.5
        self.follow_info_dict = {}
        self.build_path_generator()

    def create_visualization_markers(self):
        # override

        if self.config.headless and not self.config.init_viser:
            return {}

        visualization_markers = super().create_visualization_markers()

        path_markers = []
        for i in range(self.config.path_follower_params.num_traj_samples):
            path_markers.append(MarkerConfig(size="regular"))
        path_markers_cfg = VisualizationMarker(
            type="sphere", color=(1.0, 0.0, 0.0), markers=path_markers
        )
        visualization_markers["path_markers"] = path_markers_cfg

        return visualization_markers

    def get_markers_state(self):
        # override

        if self.config.headless and not self.config.init_viser:
            return {}

        markers_state = super().get_markers_state()

        traj_samples = self.fetch_path_samples().clone()
        if not self.config.path_follower_params.height_conditioned:
            traj_samples[..., 2] = 0.8  # CT hack

        ground_below_marker = self.terrain.get_ground_heights(
            traj_samples[..., :2].view(-1, 2)
        ).view(traj_samples.shape[:-1])
        traj_samples[..., 2] += ground_below_marker

        traj_samples = traj_samples.view(self.num_envs, -1, 3)
        markers_state["path_markers"] = MarkerState(
            translation=traj_samples,
            orientation=torch.zeros(
                self.num_envs, traj_samples.shape[1], 4, device=self.device
            ),
        )

        return markers_state

    ###############################################################
    # Handle resets
    ###############################################################
    def reset(self, env_ids=None):
        obs = super().reset(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        bodies_positions = self.simulator.get_bodies_state(env_ids).rigid_body_pos
        head_position = bodies_positions[:, self.head_body_id]

        if len(env_ids) > 0:
            flat_reset_head_position = head_position.view(-1, 3)
            ground_below_reset_head = self.terrain.get_ground_heights(
                head_position[..., :2]
            )
            flat_reset_head_position[..., 2] -= ground_below_reset_head.view(-1)
            self.path_generator.reset(env_ids, flat_reset_head_position)

        return obs

    ###############################################################
    # Environment step logic
    ###############################################################
    def compute_observations(self, env_ids=None):
        super().compute_observations(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        bodies_positions = self.simulator.get_bodies_state(env_ids).rigid_body_pos
        root_states = self.simulator.get_root_state(env_ids)
        ground_below_head = torch.min(bodies_positions, dim=1).values[..., 2]
        head_position = bodies_positions[:, self.head_body_id, :]

        traj_samples = self.fetch_path_samples(env_ids)

        flat_head_position = head_position.view(-1, 3)
        flat_head_position[..., 2] -= ground_below_head.view(-1)

        obs = compute_path_observations(
            root_states.root_rot,
            flat_head_position,
            traj_samples,
            self.config.path_follower_params.height_conditioned,
        )
        self.path_obs[env_ids] = obs

    def get_obs(self):
        obs = super().get_obs()
        obs.update({"path": self.path_obs})
        return obs

    def compute_reward(self):
        # override

        bodies_positions = self.simulator.get_bodies_state().rigid_body_pos
        head_position = bodies_positions[:, self.head_body_id, :]

        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self.path_generator.calc_pos(env_ids, time)

        ground_below_head = torch.min(bodies_positions, dim=1).values[..., 2]
        head_position[..., 2] -= ground_below_head.view(-1)

        path_rew = compute_path_reward(
            head_position, tar_pos, self.config.path_follower_params.height_conditioned
        )
        # self.rew_buf[:] = path_rew

        # ---------

        dof_state = self.simulator.get_dof_state()
        # dof_forces = self.simulator.get_dof_forces()
        root_states = self.simulator.get_root_state()
        # 1)
        rew_dict = {"path_rew": 10.0 * path_rew}

        # 2)
        rew_lin_vel_z = torch.square(root_states.root_vel[:, 2])
        rew_dict["rew_lin_vel_z"] = -2.0 * rew_lin_vel_z

        # 3)
        rew_ang_vel_xy = torch.sum(torch.square(root_states.root_ang_vel[:, :2]), dim=1)
        rew_dict["rew_ang_vel_xy"] = -0.05 * rew_ang_vel_xy

        # 4)
        rew_torque = torch.sum(torch.square(self.simulator.torques), dim=1)
        rew_dict["rew_torque"] = -0.00001 * rew_torque

        # power = torch.abs(torch.multiply(dof_forces, dof_state.dof_vel)).sum(dim=-1)
        # rew_dict["pow_rew"] = -0.0002 * power

        # 5)
        rew_dof_acc = torch.sum(
            torch.square((self.last_dof_vel - dof_state.dof_vel) / self.dt), dim=1
        )
        rew_dict["rew_dof_acc"] = -2.5e-7 * rew_dof_acc

        # 6)
        bodies_contact_buf = self.self_obs_cb.body_contacts.clone()
        contact = bodies_contact_buf[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_feet_air_time = torch.sum(
            (self.feet_air_time - 0.5) * first_contact, dim=1
        )  #  reward only on first contact with the ground
        # rew_feet_air_time *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        rew_dict["rew_feet_air_time"] = 1.0 * rew_feet_air_time

        # 7)
        rew_collision = torch.sum(
            1.0
            * (
                torch.norm(bodies_contact_buf[:, self.penelized_body_ids, :], dim=-1)
                > 0.1
            ),
            dim=1,
        )
        rew_dict["rew_collision"] = -1.0 * rew_collision

        # 8)
        rew_action_rate = torch.sum(
            torch.square(self.last_actions - self.actions), dim=1
        )
        rew_dict["rew_action_rate"] = -0.01 * rew_action_rate

        # ------------

        scaled_rewards: Dict[str, Tensor] = {  # move to rew class
            k: v
            * 1.0  # ,getattr(self.config.path_follow_reward_config.component_weights, f"{k}_w")
            for k, v in rew_dict.items()
        }

        self.rew_buf = sum(scaled_rewards.values())

        # logging & fun

        for rew_name, rew in rew_dict.items():
            self.log_dict[f"raw/{rew_name}_mean"] = rew.mean()
            self.log_dict[f"raw/{rew_name}_std"] = rew.std()

        # for rew_name, rew in scaled_rewards.items():
        #     self.log_dict[f"scaled/{rew_name}_mean"] = rew.mean()
        #     self.log_dict[f"scaled/{rew_name}_std"] = rew.std()

        # other_log_terms = {
        # "tracking_rew": tracking_rew,
        # "total_rew": self.rew_buf,
        # "cartesian_err": cartesian_err,
        # "gt_err": gt_err,
        # "gr_err": gr_err,
        # "gr_err_degrees": gr_err_degrees,
        # "lr_err_degrees": lr_err_degrees,
        # "max_joint_err": max_joint_err,
        # "max_lr_err_degrees": max_lr_err_degrees,
        # "max_gr_err_degrees": max_gr_err_degrees,
        # "root_height_error": rh_err,
        # }

        # for rew_name, rew in other_log_terms.items():
        #     self.log_dict[f"follow_other/{rew_name}_mean"] = rew.mean()
        #     self.log_dict[f"follow_other/{rew_name}_std"] = rew.std()

        # self.follow_info_dict.update(rew_dict)
        # self.follow_info_dict.update(other_log_terms)

    def compute_reset(self):
        # override

        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self.path_generator.calc_pos(env_ids, time)

        bodies_positions = self.simulator.get_bodies_state().rigid_body_pos
        bodies_contact_buf = self.self_obs_cb.body_contacts.clone()

        bodies_positions[..., 2] -= (
            torch.min(bodies_positions, dim=1).values[:, 2].view(-1, 1)
        )

        self.reset_buf[:], self.terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf,
            self.progress_buf,
            bodies_contact_buf,
            self.non_termination_contact_body_ids,
            bodies_positions,
            tar_pos,
            self.config.max_episode_length,
            self._fail_dist,
            self._fail_height_dist,
            self.config.enable_height_termination,
            self.config.path_follower_params.enable_path_termination,
            self.config.path_follower_params.height_conditioned,
            self.termination_heights
            + self.terrain.get_ground_heights(
                bodies_positions[:, self.head_body_id, :2]
            ),
            self.head_body_id,
        )

    ###############################################################
    # Helpers
    ###############################################################
    def build_path_generator(self):
        episode_dur = self.config.max_episode_length * self.dt
        self.path_generator = PathGenerator(
            self.config.path_follower_params.path_generator,
            self.device,
            self.num_envs,
            episode_dur,
            self.config.path_follower_params.height_conditioned,
        )

    def fetch_path_samples(self, env_ids=None):
        # 5 seconds with 0.5 second intervals, 10 samples.
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        timestep_beg = self.progress_buf[env_ids] * self.dt
        timesteps = torch.arange(
            self._num_traj_samples, device=self.device, dtype=torch.float
        )
        timesteps = timesteps * self._traj_sample_timestep
        traj_timesteps = (
            timestep_beg.unsqueeze(-1) + timesteps
        )  # e.g [t, t+0.5, t+1.0, ..., t+4.5] # (0.5~_traj_sample_timestep)

        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_timesteps.shape)

        traj_samples_flat = self.path_generator.calc_pos(
            env_ids_tiled.flatten(), traj_timesteps.flatten()
        )
        traj_samples = torch.reshape(
            traj_samples_flat,
            shape=(
                env_ids.shape[0],
                self._num_traj_samples,
                traj_samples_flat.shape[-1],
            ),
        )

        return traj_samples


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_path_observations(
    root_rot: Tensor,
    head_states: Tensor,
    traj_samples: Tensor,
    height_conditioned: bool,
) -> Tensor:
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, True)

    heading_rot_exp = torch.broadcast_to(
        heading_rot.unsqueeze(-2),
        (heading_rot.shape[0], traj_samples.shape[1], heading_rot.shape[1]),
    )
    heading_rot_exp = torch.reshape(
        heading_rot_exp,
        (heading_rot_exp.shape[0] * heading_rot_exp.shape[1], heading_rot_exp.shape[2]),
    )

    traj_samples_delta = traj_samples - head_states.unsqueeze(-2)

    traj_samples_delta_flat = torch.reshape(
        traj_samples_delta,
        (
            traj_samples_delta.shape[0] * traj_samples_delta.shape[1],
            traj_samples_delta.shape[2],
        ),
    )

    local_traj_pos = rotations.quat_rotate(
        heading_rot_exp, traj_samples_delta_flat, True
    )
    if not height_conditioned:
        local_traj_pos = local_traj_pos[..., 0:2]

    obs = torch.reshape(
        local_traj_pos,
        (traj_samples.shape[0], traj_samples.shape[1] * local_traj_pos.shape[1]),
    )
    return obs


@torch.jit.script
def compute_path_reward(head_pos, tar_pos, height_conditioned):
    # type: (Tensor, Tensor, bool) -> Tensor
    pos_err_scale = 2.0
    height_err_scale = 10.0

    pos_diff = tar_pos[..., 0:2] - head_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    height_diff = tar_pos[..., 2] - head_pos[..., 2]
    height_err = height_diff * height_diff

    pos_reward = torch.exp(-pos_err_scale * pos_err)
    height_reward = torch.exp(-height_err_scale * height_err)

    if height_conditioned:
        reward = (pos_reward + height_reward) * 0.5
    else:
        reward = pos_reward

    return reward


@torch.jit.script
def compute_humanoid_reset(
    reset_buf,
    progress_buf,
    contact_buf,
    non_termination_contact_body_ids,
    rigid_body_pos,
    tar_pos,
    max_episode_length,
    fail_dist,
    fail_height_dist,
    enable_early_termination,
    enable_path_termination,
    enable_height_termination,
    termination_heights,
    head_body_id,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, bool, bool, bool, Tensor, int) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, non_termination_contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, non_termination_contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= progress_buf > 1
    else:
        has_fallen = progress_buf < -1

    if enable_path_termination:
        head_pos = rigid_body_pos[..., head_body_id, :]
        tar_delta = tar_pos - head_pos
        tar_dist_sq = torch.sum(tar_delta * tar_delta, dim=-1)
        tar_overall_fail = tar_dist_sq > fail_dist * fail_dist

        if enable_height_termination:
            tar_height = tar_pos[..., 2]
            height_delta = tar_height - head_pos[..., 2]
            tar_head_dist_sq = height_delta * height_delta
            tar_height_fail = tar_head_dist_sq > fail_height_dist * fail_height_dist
            tar_height_fail *= progress_buf > 20

            tar_fail = torch.logical_or(tar_overall_fail, tar_height_fail)
        else:
            tar_fail = tar_overall_fail
    else:
        tar_fail = progress_buf < -1

    has_failed = torch.logical_or(has_fallen, tar_fail)

    terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated
    )

    return reset, terminated
