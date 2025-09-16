import math
from typing import Optional

import torch
from torch import Tensor
from isaac_utils import rotations, torch_utils

from protoverse.envs.base_env.env import BaseEnv
from protoverse.simulator.base_simulator.config import (
    MarkerConfig,
    VisualizationMarker,
    MarkerState,
)


class Locomotion(BaseEnv):
    """
    Isaac-like velocity-tracking: commands in base frame
      commands[:,0] = v_x
      commands[:,1] = v_y
      commands[:,2] = w_z
    Optionally compute w_z from a heading target (P-control).
    """

    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config=config, device=device, *args, **kwargs)

        # ---- command ranges / settings  ----
        rngs = self.config.locomotion_command_ranges
        self._lin_x_range = tuple(rngs["lin_vel_x"])
        self._lin_y_range = tuple(rngs["lin_vel_y"])
        self._wz_range = tuple(rngs["ang_vel_yaw"])
        self._heading_range = tuple(rngs.get("heading", (-math.pi, math.pi)))

        self._resample_T = float(
            self.config.locomotion_command_resampling_time
        )  # seconds
        self._heading_command = bool(
            getattr(self.config, "locomotion_heading_command", True)
        )
        self._heading_kp = float(
            getattr(self.config, "locomotion_heading_control_stiffness", 0.5)
        )
        self._rel_heading_envs = float(
            getattr(self.config, "locomotion_rel_heading_envs", 1.0)
        )
        self._rel_standing_envs = float(
            getattr(self.config, "locomotion_rel_standing_envs", 0.02)
        )
        self._debug_vis = bool(getattr(self.config, "locomotion_debug_vis", False))

        B = self.num_envs
        # ---- command buffers ----
        self.commands = torch.zeros(B, 3, device=self.device)  # [vx, vy, wz] base-frame
        self.heading_target = torch.zeros(B, device=self.device)  # world yaw targets
        self.is_heading_env = torch.zeros(B, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros(B, dtype=torch.bool, device=self.device)

        # per-env resample step
        self._cmd_next_step = torch.zeros(B, dtype=torch.long, device=self.device)

        # obs buffer like Isaac's "velocity_commands"
        self.velocity_commands_obs = torch.zeros(B, 3, device=self.device)

        # optional arrow markers
        self._vis_key = "locomotion_cmd_markers"

    # ---------- visualization (optional) ----------
    def create_visualization_markers(self):
        if self.config.headless and not self.config.init_viser:
            return {}
        markers = super().create_visualization_markers()
        if self._debug_vis:
            markers[self._vis_key] = VisualizationMarker(
                type="arrow",
                color=(0.8, 0.2, 0.6),
                markers=[MarkerConfig(size="regular")],
            )
        return markers

    def get_markers_state(self):
        if self.config.headless and not self.config.init_viser:
            return {}
        state = super().get_markers_state()
        if self._debug_vis and self._vis_key in self.visualization_markers:
            base = self.simulator.get_root_state()
            pos = base.root_pos.clone()
            pos[..., 2] += 0.45
            vxy = self.commands[:, :2]
            speed = torch.linalg.norm(vxy, dim=-1).clamp(min=1e-6)
            # scale arrow length by |v|
            scale = torch.tensor(
                self.visualization_markers[self._vis_key].markers[0].scale,
                device=self.device,
            ).repeat(self.num_envs, 1)
            scale[:, 0] *= 3.0 * speed
            # arrow facing along vxy (in base), then to world
            ang = torch.atan2(vxy[:, 1], vxy[:, 0])
            z = torch.zeros_like(ang)
            q_base = rotations.quat_from_euler_xyz(z, z, ang, True)
            q_world = rotations.quat_mul(base.root_rot, q_base, True)
            state[self._vis_key] = MarkerState(
                translation=pos.view(self.num_envs, -1, 3),
                orientation=q_world.view(self.num_envs, -1, 4),
            )
        return state

    # ---------- RL hooks ----------
    def reset(self, env_ids: Optional[Tensor] = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if len(env_ids) > 0:
            self._resample_commands(env_ids)
            self._schedule_next(env_ids)
        return super().reset(env_ids)

    def post_physics_step(self):
        super().post_physics_step()
        self._maybe_resample()
        self._update_command_obs()

    # ---------- commands ----------
    def _schedule_next(self, env_ids: Tensor):
        steps = max(1, int(self._resample_T / self.dt))
        self._cmd_next_step[env_ids] = self.progress_buf[env_ids] + steps

    def _maybe_resample(self):
        env_ids = (
            (self.progress_buf >= self._cmd_next_step).nonzero(as_tuple=False).flatten()
        )
        if len(env_ids) > 0:
            self._resample_commands(env_ids)
            self._schedule_next(env_ids)

        if self._heading_command and self.is_heading_env.any():
            active = (
                (self.is_heading_env & (~self.is_standing_env))
                .nonzero(as_tuple=False)
                .flatten()
            )
            self._apply_heading_controller(active)

    def _resample_commands(self, env_ids: Tensor):
        n = len(env_ids)
        r = torch.empty(n, device=self.device)

        self.commands[env_ids, 0] = r.uniform_(*self._lin_x_range)
        self.commands[env_ids, 1] = r.uniform_(*self._lin_y_range)
        self.commands[env_ids, 2] = r.uniform_(*self._wz_range)

        if self._heading_command:
            self.heading_target[env_ids] = r.uniform_(*self._heading_range)
            self.is_heading_env[env_ids] = (
                r.uniform_(0.0, 1.0) <= self._rel_heading_envs
            )
        else:
            self.is_heading_env[env_ids] = False

        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self._rel_standing_envs
        stand = env_ids[self.is_standing_env[env_ids]]
        if len(stand) > 0:
            self.commands[stand, :] = 0.0

        # apply heading control only to active, non-standing envs from this batch
        if self._heading_command:
            mask = self.is_heading_env[env_ids] & (~self.is_standing_env[env_ids])
            head_ids = env_ids[mask]
            self._apply_heading_controller(head_ids)

    def _apply_heading_controller(self, env_ids: Tensor):
        env_ids = env_ids.flatten()
        if env_ids.numel() == 0:
            return
        root = self.simulator.get_root_state()
        # (k,4) x (k,3) -> (k,3)
        fwd_w = rotations.quat_apply(
            root.root_rot[env_ids], self.forward_vec[env_ids], True
        )
        yaw = torch.atan2(fwd_w[:, 1], fwd_w[:, 0])
        err = torch_utils.wrap_to_pi(self.heading_target[env_ids] - yaw)
        wz = err * self._heading_kp
        wz = torch.clamp(wz, self._wz_range[0], self._wz_range[1])  # (k,)
        self.commands[env_ids, 2] = wz

    # ---------- observations ----------
    def compute_observations(self, env_ids: Optional[Tensor] = None):
        super().compute_observations(env_ids)
        if env_ids is None:
            self.velocity_commands_obs[:] = self.commands
        else:
            self.velocity_commands_obs[env_ids] = self.commands[env_ids]

        if self.config.with_foot_obs:
            foot_obs = torch.norm(self.simulator.get_foot_contact_buf(), dim=-1)
            foot_obs = torch.where(foot_obs >= 1.0, 1.0, 0.0)
            self.foot_obs[env_ids] = foot_obs[env_ids]

    def _update_command_obs(self):
        self.velocity_commands_obs[:] = self.commands

    def get_obs(self):
        obs = super().get_obs()
        obs.update({"velocity_commands": self.velocity_commands_obs})  # (B,3)
        if self.config.with_foot_obs:
            obs.update({"foot_obs": self.foot_obs})
        return obs

    def compute_reward(self):

        rew_dict = self.rew_manager.compute_rewards()

        self.rew_buf = sum(rew_dict.values())

        for rew_name, rew in rew_dict.items():
            self.log_dict[f"raw/{rew_name}_mean"] = rew.mean()
            self.log_dict[f"raw/{rew_name}_std"] = rew.std()
