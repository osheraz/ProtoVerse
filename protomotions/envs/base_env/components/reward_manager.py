from protomotions.envs.base_env.components.base_component import BaseComponent
import torch
from protomotions.envs.base_env.components.rewards.reward_registry import (
    REWARD_REGISTRY,
)

# VINL
# tracking_lin_vel = 1.0
# tracking_ang_vel = 0.5
# lin_vel_z = -2.0
# ang_vel_xy = -0.05
# torques = -0.00001
# dof_acc = -2.5e-7
# feet_air_time = 1.0
# collision = -1.0
# action_rate = -0.01


class RewardManager(BaseComponent):
    def __init__(self, config, env):
        super().__init__(config, env)
        self.reward_names = config.reward_names
        self.reward_scales = config.reward_scales

        # Internal state for stateful rewards
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
        rewards = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.env.device
        )

        for name in self.reward_names:
            scale = self.reward_scales[name]
            reward = self._call_reward(name)

            rewards += scale * reward

        return rewards

    def _call_reward(self, name):
        fn = REWARD_REGISTRY[name]

        if name == "feet_air_time":
            reward, self.feet_air_time, self.last_contacts = fn(
                self.env.contact_forces,
                self.last_contacts,
                self.feet_air_time,
                self.env.dt,
                self.env.commands,
                self.env.feet_indices,
            )
            return reward

        return getattr(self, f"_reward_{name}")()

    def _reward_feet_stumble(self):
        return REWARD_REGISTRY["feet_stumble"](
            self.env.contact_forces, self.env.feet_indices
        )

    def _reward_stand_still(self):
        return REWARD_REGISTRY["stand_still"](
            self.env.dof_pos,
            self.env.default_dof_pos,
            self.env.commands,
        )

    def _reward_feet_contact_forces(self):
        return REWARD_REGISTRY["feet_contact_forces"](
            self.env.contact_forces,
            self.env.feet_indices,
            self.config.max_contact_force,
        )

    def _reward_feet_step(self):
        return REWARD_REGISTRY["feet_step"](
            self.env.rb_states,
            self.env.contact_forces,
            self.env.feet_indices,
            self.env.num_feet,
        )

    def _reward_lin_vel_z(self):
        return REWARD_REGISTRY["lin_vel_z"](self.env.base_lin_vel)

    def _reward_ang_vel_xy(self):
        return REWARD_REGISTRY["ang_vel_xy"](self.env.base_ang_vel)

    def _reward_orientation(self):
        return REWARD_REGISTRY["orientation"](self.env.projected_gravity)

    def _reward_base_height(self):
        return REWARD_REGISTRY["base_height"](
            self.env.root_states,
            self.env.measured_heights,
            self.config.base_height_target,
        )

    def _reward_torques(self):
        return REWARD_REGISTRY["torques"](self.env.torques)

    def _reward_dof_vel(self):
        return REWARD_REGISTRY["dof_vel"](self.env.dof_vel)

    def _reward_dof_acc(self):
        return REWARD_REGISTRY["dof_acc"](
            self.env.dof_vel, self.env.last_dof_vel, self.env.dt
        )

    def _reward_action_rate(self):
        return REWARD_REGISTRY["action_rate"](self.env.actions, self.env.last_actions)

    def _reward_tracking_lin_vel(self):
        return REWARD_REGISTRY["tracking_lin_vel"](
            self.env.commands,
            self.env.base_lin_vel,
            self.config.tracking_sigma,
        )

    def _reward_tracking_ang_vel(self):
        return REWARD_REGISTRY["tracking_ang_vel"](
            self.env.commands,
            self.env.base_ang_vel,
            self.config.tracking_sigma,
        )

    def _reward_collision(self):
        return REWARD_REGISTRY["collision"](
            self.env.contact_forces, self.env.penalised_contact_indices
        )

    def _reward_termination(self):
        return REWARD_REGISTRY["termination"](self.env.reset_buf, self.env.time_out_buf)

    def _reward_dof_pos_limits(self):
        return REWARD_REGISTRY["dof_pos_limits"](
            self.env.dof_pos, self.env.dof_pos_limits
        )

    def _reward_dof_vel_limits(self):
        return REWARD_REGISTRY["dof_vel_limits"](
            self.env.dof_vel,
            self.env.dof_vel_limits,
            self.config.soft_dof_vel_limit,
        )

    def _reward_torque_limits(self):
        return REWARD_REGISTRY["torque_limits"](
            self.env.torques,
            self.env.torque_limits,
            self.config.soft_torque_limit,
        )
