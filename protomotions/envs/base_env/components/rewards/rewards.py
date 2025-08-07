import torch
from torch import Tensor
from isaac_utils import rotations, torch_utils
from typing import Tuple


@torch.jit.script
def reward_lin_vel_z(base_lin_vel: Tensor) -> Tensor:
    # Penalize z axis base linear velocity

    return torch.square(base_lin_vel[:, 2])


@torch.jit.script
def reward_ang_vel_xy(base_ang_vel: Tensor) -> Tensor:
    # Penalize xy axes base angular velocity

    return torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)


@torch.jit.script
def reward_orientation(projected_gravity: Tensor) -> Tensor:
    # Penalize non flat base orientation

    return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)


@torch.jit.script
def reward_base_height(
    root_states: Tensor, measured_heights: Tensor, base_height_target: float
) -> Tensor:
    # Penalize base height away from target

    base_height = torch.mean(root_states[:, 2].unsqueeze(1) - measured_heights, dim=1)
    return torch.square(base_height - base_height_target)


@torch.jit.script
def reward_torques(torques: Tensor) -> Tensor:
    # Penalize torques

    return torch.sum(torch.square(torques), dim=1)


@torch.jit.script
def reward_dof_vel(dof_vel: Tensor) -> Tensor:
    # Penalize dof velocities

    return torch.sum(torch.square(dof_vel), dim=1)


@torch.jit.script
def reward_dof_acc(dof_vel: Tensor, last_dof_vel: Tensor, dt: float) -> Tensor:
    # Penalize dof accelerations

    return torch.sum(torch.square((last_dof_vel - dof_vel) / dt), dim=1)


@torch.jit.script
def reward_action_rate(actions: Tensor, last_actions: Tensor) -> Tensor:
    # Penalize changes in actions

    return torch.sum(torch.square(last_actions - actions), dim=1)


@torch.jit.script
def reward_collision(
    contact_forces: Tensor, penalised_contact_indices: Tensor
) -> Tensor:
    # Penalize collisions on selected bodies

    contact_subset = contact_forces.index_select(1, penalised_contact_indices)
    return torch.sum((torch.norm(contact_subset, dim=-1) > 0.1).float(), dim=1)


@torch.jit.script
def reward_termination(reset_buf: Tensor, time_out_buf: Tensor) -> Tensor:
    # Terminal reward / penalty

    return reset_buf * (~time_out_buf)


@torch.jit.script
def reward_dof_pos_limits(dof_pos: Tensor, dof_pos_limits: Tensor) -> Tensor:
    # Penalize dof positions too close to the limit

    lower_violation = -(dof_pos - dof_pos_limits[:, 0]).clamp(max=0.0)
    upper_violation = (dof_pos - dof_pos_limits[:, 1]).clamp(min=0.0)
    return torch.sum(lower_violation + upper_violation, dim=1)


@torch.jit.script
def reward_dof_vel_limits(
    dof_vel: Tensor, dof_vel_limits: Tensor, soft_dof_vel_limit: float
) -> Tensor:
    # Penalize dof velocities too close to the limit
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    return torch.sum(
        (torch.abs(dof_vel) - dof_vel_limits * soft_dof_vel_limit).clamp(
            min=0.0, max=1.0
        ),
        dim=1,
    )


@torch.jit.script
def reward_torque_limits(
    torques: Tensor, torque_limits: Tensor, soft_torque_limit: float
) -> Tensor:
    # penalize torques too close to the limit

    return torch.sum(
        (torch.abs(torques) - torque_limits * soft_torque_limit).clamp(min=0.0), dim=1
    )


### Tracking


@torch.jit.script
def reward_tracking_lin_vel(
    commands: Tensor, base_lin_vel: Tensor, tracking_sigma: float
) -> Tensor:
    # Tracking of linear velocity commands (xy axes)

    lin_vel_error = torch.sum(
        torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / tracking_sigma)


@torch.jit.script
def reward_tracking_ang_vel(
    commands: Tensor, base_ang_vel: Tensor, tracking_sigma: float
) -> Tensor:
    # Tracking of angular velocity commands (yaw)

    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    return torch.exp(-ang_vel_error / tracking_sigma)


################

#  Motion rewards

################


@torch.jit.script
def reward_feet_stumble(contact_forces: Tensor, feet_indices: Tensor) -> Tensor:
    # Penalize feet hitting vertical surfaces

    contact_xy = torch.norm(contact_forces[:, feet_indices, :2], dim=2)
    contact_z = torch.abs(contact_forces[:, feet_indices, 2])
    return torch.any(contact_xy > 5.0 * contact_z, dim=1)


@torch.jit.script
def reward_stand_still(
    dof_pos: Tensor,
    default_dof_pos: Tensor,
    commands: Tensor,
) -> Tensor:
    # Penalize motion at zero commands

    delta = torch.abs(dof_pos - default_dof_pos)
    still_mask = torch.norm(commands[:, :2], dim=1) < 0.1
    return torch.sum(delta, dim=1) * still_mask


@torch.jit.script
def reward_feet_contact_forces(
    contact_forces: Tensor,
    feet_indices: Tensor,
    max_contact_force: float,
) -> Tensor:
    # penalize high contact forces

    force_mags = torch.norm(contact_forces[:, feet_indices, :], dim=-1)
    return torch.sum((force_mags - max_contact_force).clamp(min=0.0), dim=1)


@torch.jit.script
def reward_feet_air_time(
    contact_forces: Tensor,
    last_contacts: Tensor,
    feet_air_time: Tensor,
    dt: float,
    commands: Tensor,
    feet_indices: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    # Reward long steps
    # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    contact = contact_forces[:, feet_indices, 2] > 1.0
    contact_filt = torch.logical_or(contact, last_contacts)
    first_contact = (feet_air_time > 0.0) * contact_filt

    feet_air_time = feet_air_time + dt
    rew_air_time = torch.sum((feet_air_time - 0.5) * first_contact, dim=1)
    # reward only on first contact with the ground
    command_mask = torch.norm(commands[:, :2], dim=1) > 0.1
    rew_air_time = rew_air_time * command_mask
    # no reward for zero command
    feet_air_time = feet_air_time * (~contact_filt)
    last_contacts = contact

    return rew_air_time, feet_air_time, last_contacts


@torch.jit.script
def reward_feet_step(
    rb_states: Tensor,
    contact_forces: Tensor,
    feet_indices: Tensor,
    num_feet: int,
) -> Tensor:

    feet_heights = rb_states[:, feet_indices, 2].reshape(-1)

    xy_forces = torch.norm(contact_forces[:, feet_indices, :2], dim=2).reshape(-1)
    z_forces = torch.abs(contact_forces[:, feet_indices, 2].reshape(-1))

    # contact = torch.logical_or(
    #     contact_forces[:, feet_indices, 2] > 1.0,
    #     torch.logical_or(
    #         contact_forces[:, feet_indices, 1] > 1.0,
    #         contact_forces[:, feet_indices, 0] > 1.0,
    #     ),
    # )
    # self.last_contacts = contact

    mask = feet_heights < 0.05
    xy_forces[mask] = 0.0
    z_forces[mask] = 0.0

    z_ans = z_forces.view(-1, num_feet).sum(dim=1)
    z_ans = torch.where(z_ans > 1.0, torch.ones_like(z_ans), z_ans)

    return z_ans
