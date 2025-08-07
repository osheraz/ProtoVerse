# reward_registry.py

from typing import Dict, Callable
from torch import Tensor

from protomotions.envs.base_env.components.rewards.rewards import (
    reward_lin_vel_z,
    reward_ang_vel_xy,
    reward_orientation,
    reward_base_height,
    reward_torques,
    reward_dof_vel,
    reward_dof_acc,
    reward_action_rate,
    reward_tracking_lin_vel,
    reward_tracking_ang_vel,
    reward_collision,
    reward_termination,
    reward_dof_pos_limits,
    reward_dof_vel_limits,
    reward_torque_limits,
    reward_feet_air_time,
    reward_feet_stumble,
    reward_stand_still,
    reward_feet_contact_forces,
    reward_feet_step,
)

REWARD_REGISTRY: Dict[str, Callable[..., Tensor]] = {
    # Locomotion base rewards
    "lin_vel_z": reward_lin_vel_z,
    "ang_vel_xy": reward_ang_vel_xy,
    "orientation": reward_orientation,
    "base_height": reward_base_height,
    "torques": reward_torques,
    "dof_vel": reward_dof_vel,
    "dof_acc": reward_dof_acc,
    "action_rate": reward_action_rate,
    "tracking_lin_vel": reward_tracking_lin_vel,
    "tracking_ang_vel": reward_tracking_ang_vel,
    # Constraints
    "collision": reward_collision,
    "termination": reward_termination,
    "dof_pos_limits": reward_dof_pos_limits,
    "dof_vel_limits": reward_dof_vel_limits,
    "torque_limits": reward_torque_limits,
    # Motion / contact
    "feet_air_time": reward_feet_air_time,
    "feet_stumble": reward_feet_stumble,
    "stand_still": reward_stand_still,
    "feet_contact_forces": reward_feet_contact_forces,
    "feet_step": reward_feet_step,
    # etc.. add lego-h
}
