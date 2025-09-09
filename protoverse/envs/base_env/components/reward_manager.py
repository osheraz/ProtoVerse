from protoverse.envs.base_env.components.base_component import BaseComponent
from protoverse.envs.base_env.components.rewards import rewards
import torch
from typing import Dict
from loguru import logger
from termcolor import colored
from isaac_utils import rotations, torch_utils


class RewardManager(BaseComponent):
    def __init__(self, config, env):
        super().__init__(config, env)
        self.reward_scales = dict(config.reward_scales)

        self.feet_air_time = torch.zeros(
            self.env.num_envs,
            len(self.env.feet_indices),
            dtype=torch.float,
            device=self.env.device,
            requires_grad=False,
        )

        self.feet_contact_time = torch.zeros(
            self.env.num_envs,
            len(self.env.feet_indices),
            dtype=torch.float,
            device=self.env.device,
            requires_grad=False,
        )

        self.last_contacts = torch.zeros(
            self.env.num_envs,
            len(self.env.feet_indices),
            dtype=torch.bool,
            device=self.env.device,
            requires_grad=False,
        )

        self.last_actions = torch.zeros(
            self.env.num_envs,
            self.env.get_action_size(),
            dtype=torch.float,
            device=self.env.device,
            requires_grad=False,
        )

        df_state = self.env.default_state

        # TODO: Check if this params are common or sim-specific
        self.last_dof_pos = torch.zeros_like(df_state.dof_pos)
        self.last_dof_vel = torch.zeros_like(df_state.dof_vel)
        self.last_root_vel = torch.zeros_like(df_state.root_vel)
        self.last_root_pos = torch.zeros_like(
            self.env.simulator.get_root_state().root_pos
        )

        self._prepare_reward_function()

    def _prepare_reward_function(self):
        """
        Prepare reward scales from env.config.reward_config by removing
        zero or None entries. Optional dt scaling via config flag.
        """

        scale_with_dt = bool(getattr(self.config, "scale_with_dt", False))
        if scale_with_dt:
            logger.info(
                colored(
                    f"Reward scales will be multiplied by dt = {self.env.dt}", "red"
                )
            )

        # pull termination out, but still follow the same scale policy (including dt if enabled)
        term_val = float(self.reward_scales.pop("termination", 0.0))
        if scale_with_dt and term_val != 0.0:
            term_val = term_val * self.env.dt
        self.termination_scale = term_val
        if self.termination_scale != 0.0:
            logger.info(
                f"Scale: termination = {self.termination_scale}"
                f"{' (x dt)' if scale_with_dt else ''}"
            )

        # Remove zero/None scales and optionally scale by dt
        for key in list(self.reward_scales.keys()):
            val = self.reward_scales[key]
            if val is None or float(val) == 0.0:
                logger.info(f"Scale: {key} = {val} -> dropped")
                self.reward_scales.pop(key)
            else:
                if scale_with_dt:
                    new_val = float(val) * self.env.dt
                    logger.info(f"Scale: {key} = {val} -> {new_val} (x dt)")
                    self.reward_scales[key] = new_val
                else:
                    logger.info(f"Scale: {key} = {val}")

        # Optional constant reward
        self.positive_constant = float(getattr(self.config, "positive_constant", 0.0))

    def on_reset(self, env_ids):

        self.last_actions[env_ids] = 0
        self.last_contacts[env_ids] = 0
        self.last_dof_pos[env_ids] = 0
        self.last_dof_vel[env_ids] = 0
        self.feet_air_time[env_ids] = 0
        self.feet_contact_time[env_ids] = 0

    def _pre_compute(self):
        self.dof_state = self.env.simulator.get_dof_state()
        self.root_states = self.env.simulator.get_root_state()
        self.body_states = self.env.simulator.get_bodies_state()
        self.body_contacts = self.env.simulator.get_bodies_contact_buf()

    def _post_compute(self):
        self.last_actions[:] = self.env.actions[:]
        self.last_dof_pos[:] = self.dof_state.dof_pos[:]
        self.last_dof_vel[:] = self.dof_state.dof_vel[:]
        self.last_root_vel[:] = self.root_states.root_vel[:]
        self.last_root_pos[:] = self.root_states.root_pos[:]

    def compute_rewards(self):

        self._pre_compute()

        rew_dict: Dict[str, torch.Tensor] = {}

        for name, scale in self.reward_scales.items():
            reward_fn = getattr(self, f"_reward_{name}")
            rew_dict[name] = scale * reward_fn()

        if getattr(self, "positive_constant", 0.0) != 0.0:
            rew_dict["constant"] = self.positive_constant * torch.ones(
                self.env.num_envs, device=self.env.device
            )

        self._post_compute()

        return rew_dict

    def _reward_lin_vel_z(self):
        return rewards.reward_lin_vel_z(self.root_states.root_vel)

    def _reward_ang_vel_xy(self):
        return rewards.reward_ang_vel_xy(self.root_states.root_ang_vel)

    def _reward_torque(self):
        return rewards.reward_torques(self.env.simulator.get_torques())

    def _reward_dof_acc(self):
        return rewards.reward_dof_acc(
            self.dof_state.dof_vel, self.last_dof_vel, self.env.dt
        )

    def _reward_collision(self):
        return rewards.reward_collision(self.body_contacts, self.env.penelized_body_ids)

    def _reward_action_rate(self):
        return rewards.reward_action_rate(self.env.actions, self.last_actions)

    def _reward_path(self):

        return rewards.compute_heading_reward(
            self.root_states.root_pos,
            self.last_root_pos,
            self.root_states.root_rot,
            # self.body_states.rigid_body_rot[:, self.env.torso_index],
            self.env._tar_dir,
            self.env._tar_speed,
            self.env.dt,
        )

    def _reward_slippage(self):

        if self.env.config.with_foot_obs:
            forces = self.simulator.get_foot_contact_buf()  # [B, S, 3]
            names = [n.lower() for n in self.simulator.robot_config.foot_contact_links]

            left_idx = [i for i, n in enumerate(names) if "left" in n]
            right_idx = [i for i, n in enumerate(names) if "right" in n]

            left = forces[:, left_idx, :].mean(dim=1)
            right = forces[:, right_idx, :].mean(dim=1)

            foot_contact_forces = torch.stack([left, right], dim=1)  # [B, 2, 3]
        else:
            foot_contact_forces = self.body_contacts[:, self.env.feet_indices, :]

        return rewards.reward_slippage(
            self.body_states.rigid_body_vel,
            self.env.feet_indices,
            foot_contact_forces,
        )

    def _reward_feet_ori(self):
        gravity_vec = getattr(
            self.env,
            "gravity_vec",
            torch.tensor([0.0, 0.0, -1.0], device=self.env.device),
        ).repeat((self.env.num_envs, 1))

        return rewards.reward_feet_ori(
            self.body_states.rigid_body_rot,
            self.env.feet_indices,
            gravity_vec,
        )

    def _reward_base_height(self):
        # Adjust current global translations to be relative to the data origin
        measured_heights = self.env.terrain.get_ground_heights(
            self.body_states.rigid_body_pos[:, 0]
        )
        return rewards.reward_base_height(
            self.root_states.root_pos,
            measured_heights,
            float(self.config.target_base_height),
        )

    def _reward_feet_height(self):
        feet_pos = self.body_states.rigid_body_pos[
            :, self.env.feet_indices, :
        ]  # [B, F, 3]
        F = feet_pos.shape[1]

        # ground_z: [B, F]
        ground_z = torch.stack(
            [
                self.env.terrain.get_ground_heights(feet_pos[:, j, :]).squeeze(1)
                for j in range(F)
            ],
            dim=1,
        )
        return rewards.reward_feet_height(
            self.body_states.rigid_body_pos,
            self.env.feet_indices,
            ground_z,
            float(self.config.target_feet_height),  # clearance target (m)
            0.02,
        )

    def _reward_upperbody_joint_angle_freeze(self):

        default = self.env.simulator._default_dof_pos.to(self.env.device)
        default = default.expand(self.env.num_envs, -1)

        return rewards.reward_upperbody_joint_angle_freeze(
            self.dof_state.dof_pos,
            default,
            self.env.upper_dof_indices,
        )

    def _reward_feet_heading_alignment(self):
        return rewards.reward_feet_heading_alignment(
            self.body_states.rigid_body_rot,
            self.env.feet_indices,
            self.root_states.root_rot,
        )

    def _reward_penalty_close_feet_xy(self):
        return rewards.reward_penalty_close_feet_xy(
            self.body_states.rigid_body_pos,
            self.env.feet_indices,
            float(self.config.rewards.close_feet_threshold),
        )

    def _reward_penalty_ang_vel_xy_torso(self):

        return rewards.reward_penalty_ang_vel_xy_torso(
            self.body_states.rigid_body_rot,
            self.body_states.rigid_body_ang_vel,
            self.env.torso_index,
        )

    def _reward_penalty_hip_pos(self):
        # Expect a list/array of hip DOF ids where indices 1:3 and 4:6 are roll/yaw (as in the original repo)
        assert NotImplementedError
        hips_dof_id = getattr(
            self.env, "hips_dof_id", self.hips_dof_id
        )  # list or tensor
        if not torch.is_tensor(hips_dof_id):
            hips_dof_id = torch.tensor(
                hips_dof_id, device=self.env.device, dtype=torch.long
            )
        # roll+yaw for left (1:3) and right (4:6); concatenated
        roll_yaw = torch.cat([hips_dof_id[1:3], hips_dof_id[4:6]], dim=0)  # [4]
        return rewards.reward_penalty_hip_pos(
            self.dof_state.dof_pos,
            roll_yaw,
        )

    def _reward_termination(self):
        # timeout if we hit the last step this frame
        timeout_buf = self.env.progress_buf >= (self.env.max_episode_length - 1)

        # 1 where we terminated this step AND it was NOT a time-limit termination
        term_rew = (self.env.reset_buf.bool() & (~timeout_buf)).float()

        return term_rew

    def _reward_tracking_lin_vel(self):
        heading_inv = torch_utils.calc_heading_quat_inv(self.root_states.root_rot, True)
        vel_yaw = rotations.quat_rotate(
            heading_inv, self.root_states.root_vel[:, :3], True
        )

        tar_speed = self.env._tar_speed  # [B]
        cmd_xy_yaw = torch.stack(
            [tar_speed, torch.zeros_like(tar_speed)], dim=1
        )  # [B,2]

        vxf = torch.clamp(vel_yaw[:, 0], min=0.0)  # forbid backward from “helping”
        vy = vel_yaw[:, 1]

        STD = 0.3  # tighter than 0.5 so standing still hurts
        err = (cmd_xy_yaw[:, 0] - vxf) ** 2 + (cmd_xy_yaw[:, 1] - vy) ** 2
        return torch.exp(-err / (STD**2))

    def _reward_tracking_ang_vel(self):
        heading_inv = torch_utils.calc_heading_quat_inv(self.root_states.root_rot, True)
        tar_dir3 = torch.cat(
            [self.env._tar_dir, torch.zeros_like(self.env._tar_dir[:, :1])], dim=1
        )
        local_tar = rotations.quat_rotate(heading_inv, tar_dir3, True)[:, :2]

        heading_err = torch.atan2(local_tar[:, 1], local_tar[:, 0])  # (-pi, pi]
        ang_z_cmd = torch.clamp(0.5 * heading_err, -1.0, 1.0)  # smaller gain + cap

        ang_z_meas = self.root_states.root_ang_vel[:, 2]
        err = (ang_z_cmd - ang_z_meas).pow(2)
        return torch.exp(-err / (0.5**2))  # downweighted

    def _reward_progress(self):
        # finite-diff world velocity using buffers you already maintain
        vel_fd_xy = (
            self.root_states.root_pos[:, :2] - self.last_root_pos[:, :2]
        ) / self.env.dt

        # rotate FD velocity into yaw frame
        heading_inv = torch_utils.calc_heading_quat_inv(self.root_states.root_rot, True)
        vel_fd3 = torch.cat([vel_fd_xy, torch.zeros_like(vel_fd_xy[:, :1])], dim=1)
        vel_yaw_fd = rotations.quat_rotate(heading_inv, vel_fd3, True)

        # forward-only (no backward credit)
        vxf = torch.clamp(vel_yaw_fd[:, 0], min=0.0)

        # Isaac-style exponential shaping → returns (0,1], saturates as speed grows
        vel_err_scale = 0.25  # tighten/loosen if needed
        progress_reward = 1.0 - torch.exp(-vel_err_scale * vxf * vxf)
        return progress_reward

    def _reward_upperbody_joint_angle_freeze_hip(self):
        default = self.env.simulator._default_dof_pos.to(self.env.device).expand(
            self.env.num_envs, -1
        )
        return rewards.reward_upperbody_joint_angle_freeze(
            self.dof_state.dof_pos, default, self.env.hip_joint_indices
        )

    def _reward_upperbody_joint_angle_freeze_arms(self):
        default = self.env.simulator._default_dof_pos.to(self.env.device).expand(
            self.env.num_envs, -1
        )
        return rewards.reward_upperbody_joint_angle_freeze(
            self.dof_state.dof_pos, default, self.env.arm_joint_indices
        )

    def _reward_upperbody_joint_angle_freeze_torso(self):
        default = self.env.simulator._default_dof_pos.to(self.env.device).expand(
            self.env.num_envs, -1
        )
        return rewards.reward_upperbody_joint_angle_freeze(
            self.dof_state.dof_pos, default, self.env.torso_joint_indices
        )

    def _reward_orientation(self):
        g = torch.tensor([0.0, 0.0, -1.0], device=self.env.device).expand(
            self.env.num_envs, -1
        )
        projected_gravity = rotations.quat_rotate_inverse(
            self.root_states.root_rot, g, True
        )
        return rewards.reward_orientation(projected_gravity)

    def _reward_feet_air_time(self):

        if self.env.config.with_foot_obs:
            forces = self.simulator.get_foot_contact_buf()  # [B, S, 3]
            z = forces[..., 2]
            names = [n.lower() for n in self.simulator.robot_config.foot_contact_links]
            left_idx = [i for i, n in enumerate(names) if "left" in n]
            right_idx = [i for i, n in enumerate(names) if "right" in n]
            thresh = 1.0
            left_contact = (z[:, left_idx] > thresh).any(dim=1)
            right_contact = (z[:, right_idx] > thresh).any(dim=1)
            contact = torch.stack([left_contact, right_contact], dim=1)  # [B, 2]
        else:
            contact = self.body_contacts[:, self.env.feet_indices, 2] > 1.0

        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

        mode = getattr(self.config, "feet_air_time_mode", "positive_biped")
        thr = float(getattr(self.config, "feet_air_time_threshold", 0.4))
        dt = self.env.dt

        if mode == "plain":
            first_contact = contact_filt * (self.feet_air_time > 0.0)
            rew = torch.sum((self.feet_air_time - thr) * first_contact.float(), dim=1)
            rew *= self.env._tar_speed > 0.1
            self.feet_air_time = torch.where(
                contact_filt,
                torch.zeros_like(self.feet_air_time),
                self.feet_air_time + dt,
            )
            return rew

        # positive_biped
        self.feet_contact_time = torch.where(
            contact_filt,
            self.feet_contact_time + dt,
            torch.zeros_like(self.feet_contact_time),
        )
        self.feet_air_time = torch.where(
            ~contact_filt, self.feet_air_time + dt, torch.zeros_like(self.feet_air_time)
        )
        single_stance = contact_filt.int().sum(dim=1) == 1
        in_mode_time = torch.where(
            contact_filt, self.feet_contact_time, self.feet_air_time
        )
        rew = torch.min(
            torch.where(
                single_stance.unsqueeze(-1),
                in_mode_time,
                torch.zeros_like(in_mode_time),
            ),
            dim=1,
        )[0]
        rew = torch.clamp(rew, max=thr)
        rew *= self.env._tar_speed > 0.1
        return rew

    def _reward_dof_pos_limits(self):

        limits = torch.stack(
            (
                self.env.simulator._dof_limits_lower_common,
                self.env.simulator._dof_limits_upper_common,
            ),
            dim=1,
        )
        scope = getattr(self.config, "dof_pos_limits_scope", "all")
        if (
            scope == "ankles"
            and hasattr(self.env, "ankle_dof_indices")
            and self.env.ankle_dof_indices is not None
        ):
            return rewards.reward_dof_pos_limits(
                self.dof_state.dof_pos[:, self.env.ankle_dof_indices],
                limits[self.env.ankle_dof_indices],
            )
        return rewards.reward_dof_pos_limits(self.dof_state.dof_pos, limits)
