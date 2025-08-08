from protoverse.envs.base_env.components.base_component import BaseComponent
from protoverse.envs.base_env.components.rewards import rewards
import torch


class RewardManager(BaseComponent):
    def __init__(self, config, env):
        super().__init__(config, env)
        self.reward_names = config.reward_names
        self.reward_scales = config.reward_scales

        self.last_contacts = torch.zeros(
            env.num_envs, env.num_feet, dtype=torch.bool, device=env.device
        )
        self.feet_air_time = torch.zeros(
            env.num_envs, env.num_feet, dtype=torch.float, device=env.device
        )

    def reset(self, env_ids):
        self.last_contacts[env_ids] = False
        self.feet_air_time[env_ids] = 0.0

    def compute_rewards(self):
        rewards = torch.zeros(self.env.num_envs, device=self.env.device)
        for name in self.reward_names:
            reward_fn = getattr(self, f"_reward_{name}")
            scale = self.reward_scales[name]
            rewards += scale * reward_fn()
        return rewards

    def _reward_torques(self):
        return rewards.reward_torques(self.env.torques)

    def _reward_lin_vel_z(self):
        return rewards.reward_lin_vel_z(self.env.base_lin_vel)

    def _reward_feet_air_time(self):
        reward, self.feet_air_time, self.last_contacts = rewards.reward_feet_air_time(
            self.env.contact_forces,
            self.last_contacts,
            self.feet_air_time,
            self.env.dt,
            self.env.commands,
            self.env.feet_indices,
        )
        return reward
