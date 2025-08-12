from protoverse.envs.base_env.components.base_component import BaseComponent
from protoverse.envs.base_env.components.rewards import rewards
import torch
from typing import Dict
from loguru import logger
from termcolor import colored


class RewardManager(BaseComponent):
    def __init__(self, config, env):
        super().__init__(config, env)
        self.reward_scales = dict(config.reward_scales)

        self.feet_air_time = torch.zeros(
            self.env.num_envs,
            2,  # HARD code for now
            dtype=torch.float,
            device=self.env.device,
            requires_grad=False,
        )

        self.last_contacts = torch.zeros(
            self.env.num_envs,
            2,  # HARD code for now
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

        df_state = self.env.simulator.get_default_state()

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

    def _pre_compute(self):
        self.dof_state = self.env.simulator.get_dof_state()
        self.root_states = self.env.simulator.get_root_state()
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

    def _reward_feet_air_time(self):

        # TODO: to jit
        contact = self.body_contacts[:, self.env.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.env.dt
        rew_feet_air_time = torch.sum(
            (self.feet_air_time - 0.5) * first_contact, dim=1
        )  #  reward only on first contact with the ground
        # rew_feet_air_time *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        return rew_feet_air_time

    def _reward_path(self):

        return rewards.compute_heading_reward(
            self.root_states.root_pos,
            self.last_root_pos,
            self.env._tar_dir,
            self.env._tar_speed,
            self.env.dt,
        )
