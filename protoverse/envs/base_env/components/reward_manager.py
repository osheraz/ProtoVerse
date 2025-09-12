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
        self.last_torques = torch.zeros_like(df_state.dof_pos)  # [B, ndof]
        self._fwd_vec = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(
            self.env.num_envs, -1
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
        self.last_torques[env_ids] = 0

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
        self.last_torques[:] = self.env.simulator.get_torques()

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

    def _reward_torques(self):
        return rewards.reward_torques(self.env.simulator.get_torques())

    def _reward_dof_acc(self):
        return rewards.reward_dof_acc(
            self.dof_state.dof_vel, self.last_dof_vel, self.env.dt
        )

    def _reward_dof_vel(self):
        return rewards.reward_dof_vel(self.dof_state.dof_vel)

    def _reward_collision(self):
        return rewards.reward_collision(self.body_contacts, self.env.penelized_body_ids)

    def _reward_action_rate(self):
        return rewards.reward_action_rate(self.env.actions, self.last_actions)

    def _reward_delta_torques(self):

        tau = self.env.simulator.get_torques()
        dtau = tau - self.last_torques
        return torch.sum(dtau * dtau, dim=1)

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
            float(self.config.close_feet_threshold),
        )

    def _reward_penalty_ang_vel_xy_torso(self):

        return rewards.reward_penalty_ang_vel_xy_torso(
            self.body_states.rigid_body_rot,
            self.body_states.rigid_body_ang_vel,
            self.env.torso_index,
        )


    def _reward_termination(self):
        # timeout if we hit the last step this frame
        timeout_buf = self.env.progress_buf >= (self.env.max_episode_length - 1)

        # 1 where we terminated this step AND it was NOT a time-limit termination
        term_rew = (self.env.reset_buf.bool() & (~timeout_buf)).float()

        return term_rew

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

    def _reward_tracking_lin_vel(self):
        # --- Robot base state (world -> body frame) ---
        root_rot = self.root_states.root_rot  # [B,4], world->body quat (w-last=True)
        vel_w = self.root_states.root_vel[:, :3]  # [B,3] world linear vel
        vel_b = rotations.quat_rotate_inverse(root_rot, vel_w, True)[
            ..., :2
        ]  # [B,2] body XY

        # --- Command: build desired world velocity from your steering command ---
        # v_des_w = speed * dir (world), then rotate it into the body frame like Isaac does.
        tar_dir3 = torch.cat(
            [self.env._tar_dir, torch.zeros_like(self.env._tar_dir[:, :1])], dim=1
        )  # [B,3]
        v_des_w = self.env._tar_speed.unsqueeze(-1) * tar_dir3  # [B,3]
        v_des_b = rotations.quat_rotate_inverse(root_rot, v_des_w, True)[
            ..., :2
        ]  # [B,2] body XY

        # --- IsaacLab kernel: exp( - ||cmd - meas||^2 / std^2 ) in BODY frame ---
        STD = 0.5  # IsaacLab commonly uses 0.5
        err = torch.sum((v_des_b - vel_b) ** 2, dim=-1)  # [B]
        return torch.exp(-err / (STD**2))

    def _reward_tracking_ang_vel(self):
        # --- heading error -> desired yaw rate (command) ---
        heading_inv = torch_utils.calc_heading_quat_inv(self.root_states.root_rot, True)
        tar_dir3 = torch.cat(
            [self.env._tar_dir, torch.zeros_like(self.env._tar_dir[:, :1])], dim=1
        )
        local_tar = rotations.quat_rotate(heading_inv, tar_dir3, True)[
            ..., :2
        ]  # yaw frame XY
        heading_err = torch.atan2(local_tar[:, 1], local_tar[:, 0])  # (-pi, pi]
        wz_cmd = torch.clamp(0.5 * heading_err, -1.0, 1.0)  # your existing policy

        # --- measured yaw rate in BODY frame (Isaac uses *_b) ---
        root_rot = self.root_states.root_rot
        ang_w = self.root_states.root_ang_vel  # world ω
        ang_b = rotations.quat_rotate_inverse(root_rot, ang_w, True)  # body ω
        wz_meas = ang_b[:, 2]

        # --- Isaac kernel ---
        STD = 0.5
        err = (wz_cmd - wz_meas) ** 2
        return torch.exp(-err / (STD**2))

    def _reward_dof_error(self):
        """
        sum( (q - q_default)^2 )
        """
        q = self.dof_state.dof_pos
        q0 = self.env.simulator._default_dof_pos.to(self.env.device).expand_as(q)
        err = q - q0
        return torch.sum(err * err, dim=1)

    def _reward_feet_stumble(self):
        """
        any( ||F_xy|| > 4 * |F_z| ) over feet → 1.0 else 0.0
        """
        F = self.body_contacts[:, self.env.feet_indices, :]  # [B,F,3]
        horiz = torch.norm(F[..., :2], dim=-1)
        vert = torch.abs(F[..., 2]) + 1e-8
        stumble = (horiz > 4.0 * vert).any(dim=1)
        return stumble.float()

    def _reward_tracking_goal_vel(self):
        """
        min( v_target · v_meas , cmd_speed ) / (cmd_speed + eps)

        If a relative goal is available (env.final_target_pos_rel or target_pos_rel),
        use it to form the unit target direction; otherwise fall back to steering
        command direction self.env._tar_dir.
        """
        eps = 1e-5
        # measured world XY velocity
        v_meas_xy = self.root_states.root_vel[:, :2]  # [B,2]

        # command speed (Steering uses _tar_speed)
        if hasattr(self.env, "_tar_speed"):
            cmd_speed = self.env._tar_speed
        elif hasattr(self.env, "commands"):
            cmd_speed = self.env.commands[:, 0]
        else:
            cmd_speed = torch.norm(v_meas_xy, dim=-1)  # harmless fallback

        # target direction: goal vector if present, else steering dir
        if hasattr(self.env, "final_target_pos_rel"):
            tgt = self.env.final_target_pos_rel[:, :2]
        elif hasattr(self.env, "target_pos_rel"):
            tgt = self.env.target_pos_rel[:, :2]
        elif hasattr(self.env, "_tar_dir"):
            tgt = self.env._tar_dir
        else:
            tgt = torch.zeros_like(v_meas_xy)

        tgt_unit = tgt / (torch.norm(tgt, dim=-1, keepdim=True) + eps)
        proj = torch.sum(tgt_unit * v_meas_xy, dim=-1)  # [B]
        num = torch.minimum(proj, cmd_speed)
        return num / (cmd_speed + eps)

    def _reward_tracking_yaw(self):
        """
        exp( - | yaw_target - yaw_meas | )

        yaw_target from steering dir if available; otherwise from goal vector.
        yaw_meas from base quaternion.
        """
        # forward axis in local frame, rotated to world
        fwd = self._fwd_vec  # [B,3] (already on device)
        fwd_world = rotations.quat_apply(self.root_states.root_rot, fwd, True)
        yaw_meas = torch.atan2(fwd_world[:, 1], fwd_world[:, 0])

        if hasattr(self.env, "_tar_dir"):
            yaw_tgt = torch.atan2(self.env._tar_dir[:, 1], self.env._tar_dir[:, 0])
        elif hasattr(self.env, "final_target_pos_rel"):
            v = self.env.final_target_pos_rel[:, :2]
            yaw_tgt = torch.atan2(v[:, 1], v[:, 0])
        elif hasattr(self.env, "target_pos_rel"):
            v = self.env.target_pos_rel[:, :2]
            yaw_tgt = torch.atan2(v[:, 1], v[:, 0])
        else:
            yaw_tgt = torch.zeros_like(yaw_meas)

        d = torch.abs(torch_utils.wrap_to_pi(yaw_tgt - yaw_meas))
        return torch.exp(-d)

    def _reward_pn_distance(self):
        """
        PN distance:
        1.0 if ||p_rel|| < reach
        else -0.75 * ||p_rel||   (optionally add mid-waypoint term if provided)
        If no goal available, returns zeros.
        """
        if not hasattr(self.env, "final_target_pos_rel") and not hasattr(
            self.env, "target_pos_rel"
        ):
            return torch.zeros(self.env.num_envs, device=self.env.device)

        p_rel = (
            self.env.final_target_pos_rel
            if hasattr(self.env, "final_target_pos_rel")
            else self.env.target_pos_rel
        )
        reach = float(getattr(self.config, "pn_reach_threshold", 1.0))
        norm = torch.norm(p_rel, dim=-1)

        pn_clip = bool(getattr(self.config, "pn_distance_clip", False))
        far_term = -0.75 * norm
        if pn_clip:
            far_term = torch.clamp(far_term, min=-5.0, max=0.0)

        rew = torch.where(norm < reach, torch.ones_like(norm), far_term)

        # optional mid-waypoint shaping
        if hasattr(self.env, "mid_waypoint_pos_rel") and hasattr(
            self.env, "cur_goal_idx"
        ):
            mid_norm = torch.norm(self.env.mid_waypoint_pos_rel, dim=-1)
            idx = self.env.cur_goal_idx  # [B] ints
            # add small shaping only while not at final goal
            w = (idx < 4).float() if torch.is_tensor(idx) else 0.0
            rew = rew + (-0.25) * w * mid_norm
        return rew

    def _reward_hip_pos(self):
        """
        sum( (q_hip - q_hip_default)^2 )
        """
        # pick indices from env if available
        hip_idx = getattr(self.env, "hip_joint_indices", None)
        if hip_idx is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        if not torch.is_tensor(hip_idx):
            hip_idx = torch.tensor(hip_idx, device=self.env.device, dtype=torch.long)

        q = self.dof_state.dof_pos
        q0 = self.env.simulator._default_dof_pos.to(self.env.device).expand_as(q)
        err = q[:, hip_idx] - q0[:, hip_idx]
        return torch.sum(err * err, dim=1)

    def _reward_feet_swing_height(self):
        """
        Penalize swing feet for not matching a target height.

        Defaults (matches the simple version):
        - world-Z height (no terrain subtraction)
        - contact if ||F_xyz|| > 1.0
        - squared error, only when NOT in contact, sum over feet

        Optional overrides (only if you want them):
        - config.target_feet_height: float
        - config.rewards.swing_height_mode: "exact" | "min"      (default "exact")
        - config.rewards.swing_height_ground_relative: bool      (default False)
        - config.rewards.swing_height_contact_mode: "norm" | "fz" (default "norm")
        - config.rewards.contact_threshold: float                (default 1.0)
        """
        # ---- params & defaults ----
        target = float(getattr(self.config, "target_feet_height", 0.08))
        mode = getattr(self.config, "swing_height_mode", "exact")
        ground_rel = bool(getattr(self.config, "swing_height_ground_relative", True))  
        contact_mode = getattr(self.config, "swing_height_contact_mode", "norm")     
        thresh = float(getattr(self.config, "contact_threshold", 1.0))

        # ---- feet positions ----
        feet_pos = self.body_states.rigid_body_pos[:, self.env.feet_indices, :]  # [B,F,3]
        feet_z = feet_pos[..., 2]  # [B,F]

        # ---- world-Z by default; optionally ground-relative clearance ----
        if ground_rel:
            F = feet_pos.shape[1]
            ground_z = torch.stack(
                [self.env.terrain.get_ground_heights(feet_pos[:, j, :]).squeeze(1) for j in range(F)],
                dim=1,
            )  # [B,F]
            height = feet_z - ground_z
        else:
            height = feet_z

        # ---- contact mask per foot (using body_contacts by default) ----
        if contact_mode == "fz":
            contact = (self.body_contacts[:, self.env.feet_indices, 2] > thresh)
        else:  # "norm"
            contact = (torch.norm(self.body_contacts[:, self.env.feet_indices, :], dim=-1) > thresh)

        swing = (~contact).float()  # 1 when swinging

        # ---- penalty ----
        if mode == "exact":  # (height - target)^2
            err = (height - target) ** 2
        else:                # "min": only penalize being too low
            err = torch.clamp_min(target - height, 0.0) ** 2

        return (err * swing).sum(dim=1)  # [B]


    def _reward_contact(self):
        """
        Match foot contact to a phase schedule (biped).
        - stance when phase < duty, swing otherwise
        - contact when Fz > threshold
        Returns per-env score in [0, 2] (two feet).

        Config (optional, under self.config.rewards):
        contact_period:        float, default 0.8
        contact_offset:        float, default 0.5     # right leg phase offset
        contact_duty_cycle:    float, default 0.55    # stance window
        contact_threshold:     float, default 1.0     # N, contact if Fz > thr
        """
        period = float(getattr(self.config, "contact_period", 0.8))
        offset = float(getattr(self.config, "contact_offset", 0.5))
        duty   = float(getattr(self.config, "contact_duty_cycle", 0.55))
        thr    = float(getattr(self.config, "contact_threshold", 1.0))

        # --- phase per env (like G1Robot) ---
        t = self.env.progress_buf.float() * self.env.dt                # [B]
        phase = torch.remainder(t, period) / period                    # [B] in [0,1)
        leg_phase = torch.stack([phase, torch.remainder(phase + offset, 1.0)], dim=1)  # [B,2]
        expect_contact = (leg_phase < duty)                            # [B,2] bool

        # --- contact per foot (Left/Right) ---
        B = self.env.num_envs
        device = self.env.device

        if getattr(self.env.config, "with_foot_obs", False):
            # Use dedicated foot sensors and group by name (robust on multi-sensors/links)
            forces = self.simulator.get_foot_contact_buf()  # [B,S,3]
            names = [n.lower() for n in self.simulator.robot_config.foot_contact_links]
            left_idx  = [i for i, n in enumerate(names) if "left"  in n]
            right_idx = [i for i, n in enumerate(names) if "right" in n]
            if len(left_idx) and len(right_idx):
                z = forces[..., 2]
                left_contact  = (z[:, left_idx]  > thr).any(dim=1)     # [B]
                right_contact = (z[:, right_idx] > thr).any(dim=1)     # [B]
                contact = torch.stack([left_contact, right_contact], dim=1)  # [B,2] bool
            else:
                # Fallback to body contacts if naming not available
                bc = (self.body_contacts[:, self.env.feet_indices, 2] > thr)  # [B,F]
                if bc.shape[1] >= 2:
                    contact = bc[:, :2]
                else:
                    return torch.zeros(B, device=device)
        else:
            # From body contact buffer (assume feet_indices are [L, R] for bipeds)
            bc = (self.body_contacts[:, self.env.feet_indices, 2] > thr)      # [B,F]
            if bc.shape[1] >= 2:
                contact = bc[:, :2]                                           # [B,2]
            else:
                return torch.zeros(B, device=device)

        # --- match score (1 if expected == actual) ---
        match = (contact == expect_contact).float()  # [B,2]
        return match.sum(dim=1)                      # [B], in [0,2]
