# steering_env.py
import numpy as np
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


# --------------------------------------------
# JIT helper
# --------------------------------------------
@torch.jit.script
def compute_heading_observations(
    root_rot: Tensor, tar_dir: Tensor, tar_speed: Tensor
) -> Tensor:
    tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, True)
    local_tar_dir = rotations.quat_rotate(heading_rot, tar_dir3d, True)[..., 0:2]
    return torch.cat([local_tar_dir, tar_speed.unsqueeze(-1)], dim=-1)


class Steering(BaseEnv):
    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config=config, device=device, *args, **kwargs)

        # --------- style switches ----------
        self.cmd_style: str = getattr(
            self.config, "command_style", "steering"
        )  # "steering" | "isaac"
        self.obs_mode: str = getattr(
            self.config, "obs_mode", "steering"
        )  # "steering" | "isaac"

        # ====== Steering-style fields (your original) ======
        sp = self.config.steering_params
        self._tar_speed_min = sp.tar_speed_min
        self._tar_speed_max = sp.tar_speed_max
        self._heading_change_steps_min = sp.heading_change_steps_min
        self._heading_change_steps_max = sp.heading_change_steps_max
        self._random_heading_probability = sp.random_heading_probability
        self._standard_heading_change = sp.standard_heading_change
        self._standard_speed_change = sp.standard_speed_change
        self._stop_probability = sp.stop_probability

        self.steering_obs = torch.zeros(
            (self.config.num_envs, sp.obs_size), device=device, dtype=torch.float
        )
        if self.config.with_foot_obs:
            self.foot_obs = torch.zeros(
                (self.config.num_envs, self.config.foot_obs_params.obs_size),
                device=device,
                dtype=torch.float,
            )

        self._heading_change_steps = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int64
        )
        self._prev_root_pos = torch.zeros(self.num_envs, 3, device=self.device)

        self._tar_dir_theta = torch.zeros(self.num_envs, device=self.device)  # radians
        self._tar_dir = torch.zeros(self.num_envs, 2, device=self.device)
        self._tar_dir[..., 0] = 1.0
        self._tar_speed = torch.ones(self.num_envs, device=self.device)

        # ====== Isaac-style fields (commands tensor etc.) ======
        if self.cmd_style == "isaac":
            c = self.config.commands
            self.isaac_heading_command: bool = bool(c.heading_command)
            self._cmd_resample_every = max(
                1, int(float(c.resampling_time) / self.dt + 0.5)
            )

            # [vx, vy, wz, heading]
            self.commands = torch.zeros(
                self.num_envs,
                int(c.num_commands),
                device=self.device,
                dtype=torch.float,
            )

            # ranges
            r = c.ranges
            self._rng_lin_x = torch.tensor(
                r.lin_vel_x, device=self.device, dtype=torch.float
            )
            self._rng_lin_y = torch.tensor(
                r.lin_vel_y, device=self.device, dtype=torch.float
            )
            self._rng_wz = torch.tensor(
                r.ang_vel_yaw, device=self.device, dtype=torch.float
            )
            self._rng_head = torch.tensor(
                r.heading, device=self.device, dtype=torch.float
            )

            # optional override for yaw command (used by rewards when heading_command=False)
            self._wz_cmd_override: Optional[torch.Tensor] = torch.zeros(
                self.num_envs, device=self.device
            )

        else:
            self._wz_cmd_override = None  # not used in steering style

    # ---------------- vis ----------------
    def create_visualization_markers(self):
        if self.config.headless:
            return {}
        visualization_markers = super().create_visualization_markers()
        steering_markers = [MarkerConfig(size="regular")]
        visualization_markers["steering_markers"] = VisualizationMarker(
            type="arrow", color=(0.0, 1.0, 1.0), markers=steering_markers
        )
        return visualization_markers

    def get_markers_state(self):
        if self.config.headless:
            return {}
        markers_state = super().get_markers_state()
        marker_root = self.simulator.get_root_state().root_pos
        marker_root[..., 0:2] += self._tar_dir
        heading_axis = torch.zeros_like(marker_root)
        heading_axis[..., -1] = 1.0
        marker_rot = rotations.quat_from_angle_axis(
            self._tar_dir_theta, heading_axis, True
        )
        markers_state["steering_markers"] = MarkerState(
            translation=marker_root.view(self.num_envs, -1, 3),
            orientation=marker_rot.view(self.num_envs, -1, 4),
        )
        return markers_state

    # --------------- lifecycle ---------------
    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if self.cmd_style == "isaac":
            self._isaac_resample(env_ids)
            if self.isaac_heading_command:
                self._isaac_update_wz_from_heading()
            self._update_command_adapter()  # sync _tar_dir/_tar_speed
        else:
            if len(env_ids) > 0:
                self.reset_heading_task(env_ids)

        return super().reset(env_ids)

    def post_physics_step(self):
        super().post_physics_step()
        if self.cmd_style == "isaac":
            self._isaac_step_commands()
        else:
            self.check_update_task()  # steering schedule
        # keep adapter synced for rewards
        self._update_command_adapter()

    # --------------- steering schedule ---------------
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
            dth = (
                2 * self._standard_heading_change * torch.rand(n, device=self.device)
                - self._standard_heading_change
            )
            dir_theta = (dth + self._tar_dir_theta[env_ids] + np.pi) % (
                2 * np.pi
            ) - np.pi
            ds = (
                2 * self._standard_speed_change * torch.rand(n, device=self.device)
                - self._standard_speed_change
            )
            tar_speed = torch.clamp(
                ds + self._tar_speed[env_ids],
                min=self._tar_speed_min,
                max=self._tar_speed_max,
            )

        tar_dir = torch.stack([torch.cos(dir_theta), torch.sin(dir_theta)], dim=-1)
        steps = torch.randint(
            low=self._heading_change_steps_min,
            high=self._heading_change_steps_max,
            size=(n,),
            device=self.device,
            dtype=torch.int64,
        )
        should_stop = torch.bernoulli(
            torch.ones(n, device=self.device) * self._stop_probability
        )

        self._tar_speed[env_ids] = tar_speed * (1.0 - should_stop)
        self._tar_dir_theta[env_ids] = dir_theta
        self._tar_dir[env_ids] = tar_dir
        self._heading_change_steps[env_ids] = self.progress_buf[env_ids] + steps

    # --------------- isaac schedule ---------------
    def _isaac_step_commands(self):
        ids = (
            (self.progress_buf % self._cmd_resample_every == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        if len(ids) > 0:
            self._isaac_resample(ids)
        if self.isaac_heading_command:
            self._isaac_update_wz_from_heading()

    @torch.no_grad()
    def _isaac_resample(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        B = len(env_ids)
        self.commands[env_ids, 0] = self._rng_lin_x[0] + (
            self._rng_lin_x[1] - self._rng_lin_x[0]
        ) * torch.rand(B, device=self.device)
        self.commands[env_ids, 1] = self._rng_lin_y[0] + (
            self._rng_lin_y[1] - self._rng_lin_y[0]
        ) * torch.rand(B, device=self.device)
        if self.isaac_heading_command:
            self.commands[env_ids, 3] = self._rng_head[0] + (
                self._rng_head[1] - self._rng_head[0]
            ) * torch.rand(B, device=self.device)
        else:
            self.commands[env_ids, 2] = self._rng_wz[0] + (
                self._rng_wz[1] - self._rng_wz[0]
            ) * torch.rand(B, device=self.device)

        # set tiny linear commands to zero (like legged-gym)
        v = self.commands[env_ids, :2]
        keep = (torch.norm(v, dim=1) > 0.2).unsqueeze(1)
        self.commands[env_ids, :2] = v * keep

    @torch.no_grad()
    def _isaac_update_wz_from_heading(self):
        root_rot = self.simulator.get_root_state().root_rot
        fwd = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(
            self.num_envs, -1
        )
        fwd_w = rotations.quat_apply(root_rot, fwd, True)
        yaw_meas = torch.atan2(fwd_w[:, 1], fwd_w[:, 0])
        yaw_tgt = self.commands[:, 3]
        err = torch_utils.wrap_to_pi(yaw_tgt - yaw_meas)
        self.commands[:, 2] = torch.clamp(0.5 * err, -1.0, 1.0)

    # --------------- adapter (unifies interface for rewards/obs) ---------------
    @torch.no_grad()
    def _update_command_adapter(self):
        if self.cmd_style == "isaac":
            # map Isaac body-frame linear commands -> world dir/speed for your rewards
            root_rot = self.simulator.get_root_state().root_rot
            v_b = self.commands[:, :2]
            v_w3 = rotations.quat_apply(
                root_rot,
                torch.cat(
                    [v_b, torch.zeros(self.num_envs, 1, device=self.device)], dim=1
                ),
                True,
            )
            v_xy = v_w3[:, :2]
            speed = torch.norm(v_xy, dim=-1)
            dir_ = torch.zeros_like(v_xy)
            nz = speed > 1e-8
            if nz.any():
                dir_[nz] = v_xy[nz] / speed[nz].unsqueeze(-1)
            if (~nz).any():
                dir_[~nz] = torch.tensor([1.0, 0.0], device=self.device).expand(
                    (~nz).sum(), -1
                )
            self._tar_dir = dir_
            self._tar_speed = speed
            self._tar_dir_theta = torch.atan2(dir_[:, 1], dir_[:, 0])
            self._wz_cmd_override = self.commands[
                :, 2
            ]  # used by reward (matches Isaac both modes)
        else:
            # steering style: targets already set by scheduler
            self._wz_cmd_override = None

    # --------------- observations ---------------
    def compute_observations(self, env_ids=None):
        super().compute_observations(env_ids)

        if env_ids is None:
            root_states = self.simulator.get_root_state()
            tar_dir, tar_speed = self._tar_dir, self._tar_speed
            cmds = self.commands if self.cmd_style == "isaac" else None
        else:
            root_states = self.simulator.get_root_state(env_ids)
            tar_dir, tar_speed = self._tar_dir[env_ids], self._tar_speed[env_ids]
            cmds = self.commands[env_ids] if self.cmd_style == "isaac" else None

        if self.obs_mode == "isaac" and cmds is not None:
            # expose Isaac-style [vx_cmd, vy_cmd, wz_cmd]
            self.steering_obs[env_ids] = cmds[:, :3]
        else:
            # your original 3-dim obs
            self.steering_obs[env_ids] = compute_heading_observations(
                root_states.root_rot, tar_dir, tar_speed
            )

        if self.config.with_foot_obs:
            foot_obs = torch.norm(self.simulator.get_foot_contact_buf(), dim=-1)
            foot_obs = torch.where(foot_obs >= 1.0, 1.0, 0.0)
            self.foot_obs[env_ids] = foot_obs[env_ids]

    # --------------- obs access ---------------
    def get_obs(self):
        obs = super().get_obs()
        obs.update({"steering": self.steering_obs})
        if self.config.with_foot_obs:
            obs.update({"foot_obs": self.foot_obs})
        return obs

    # --------------- reward hook ---------------
    def compute_reward(self):
        # ensure adapter is synced right before rewards
        self._update_command_adapter()
        rew_dict = self.rew_manager.compute_rewards()
        self.rew_buf = sum(rew_dict.values())
        for k, v in rew_dict.items():
            self.log_dict[f"raw/{k}_mean"] = v.mean()
            self.log_dict[f"raw/{k}_std"] = v.std()
