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
def reward_torques(torques: Tensor) -> Tensor:
    # Penalize torques

    return torch.sum(torch.square(torques), dim=1)


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

    return torch.sum(
        1.0
        * (torch.norm(contact_forces[:, penalised_contact_indices, :], dim=-1) > 0.1),
        dim=1,
    )


@torch.jit.script
def compute_heading_reward(
    root_pos: Tensor,
    prev_root_pos: Tensor,
    tar_dir: Tensor,
    tar_speed: Tensor,
    dt: float,
) -> Tensor:
    vel_err_scale = 0.25
    tangent_err_w = 0.1

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)

    tar_dir_vel = tar_dir_speed.unsqueeze(-1) * tar_dir
    tangent_vel = root_vel[..., :2] - tar_dir_vel

    tangent_speed = torch.sum(tangent_vel, dim=-1)

    tar_vel_err = tar_speed - tar_dir_speed
    tangent_vel_err = tangent_speed
    dir_reward = torch.exp(
        -vel_err_scale
        * (
            tar_vel_err * tar_vel_err
            + tangent_err_w * tangent_vel_err * tangent_vel_err
        )
    )

    speed_mask = tar_dir_speed < -0.5
    dir_reward[speed_mask] = 0

    return dir_reward


@torch.jit.script
def reward_slippage(
    rigid_body_vel: Tensor, feet_indices: Tensor, foot_contact_forces: Tensor
) -> Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """

    foot_vel = rigid_body_vel[:, feet_indices]

    return torch.sum(
        torch.norm(foot_vel, dim=-1) * (torch.norm(foot_contact_forces, dim=-1) > 1.0),
        dim=1,
    )


@torch.jit.script
def reward_feet_ori(
    rigid_body_rot: Tensor, feet_indices: Tensor, gravity_vec: Tensor
) -> Tensor:
    # Penalize non-flat feet orientation wrt gravity
    # TODO: what about uneven terrain?
    left_quat = rigid_body_rot[:, feet_indices[0]]
    left_gravity = rotations.quat_rotate_inverse(left_quat, gravity_vec)

    right_quat = rigid_body_rot[:, feet_indices[1]]
    right_gravity = rotations.quat_rotate_inverse(right_quat, gravity_vec)

    left_pen = torch.sqrt(torch.sum(torch.square(left_gravity[:, :2]), dim=1))
    right_pen = torch.sqrt(torch.sum(torch.square(right_gravity[:, :2]), dim=1))

    return left_pen + right_pen


@torch.jit.script
def reward_dof_pos_limits(dof_pos: Tensor, dof_pos_limits: Tensor) -> Tensor:
    # Penalize dof positions too close to the limit

    lower_violation = -(dof_pos - dof_pos_limits[:, 0]).clamp(max=0.0)
    upper_violation = (dof_pos - dof_pos_limits[:, 1]).clamp(min=0.0)
    return torch.sum(lower_violation + upper_violation, dim=1)


@torch.jit.script
def reward_base_height(
    root_states: Tensor, measured_heights: Tensor, base_height_target: float
) -> Tensor:
    # Penalize base height away from target

    base_height = torch.mean(root_states[:, 2].unsqueeze(1) - measured_heights, dim=1)
    return torch.square(base_height - base_height_target)


@torch.jit.script
def reward_feet_height(
    rigid_body_pos: Tensor,  # [B, N, 3]
    feet_indices: Tensor,  # [F]
    ground_z: Tensor,  # [B, F]
    target_clearance: float,
    tol: float = 0.02,
) -> Tensor:
    feet_z = rigid_body_pos[:, feet_indices, 2]  # [B, F]
    clr = feet_z - ground_z  # [B, F]
    dif = (clr - target_clearance).abs()  # [B, F]
    return torch.clamp(dif.min(dim=1).values - tol, min=0.0)  # [B]


@torch.jit.script
def reward_upperbody_joint_angle_freeze(
    dof_pos: Tensor, default_dof_pos: Tensor, upper_dof_indices: Tensor
) -> Tensor:
    deviation = torch.abs(
        dof_pos[:, upper_dof_indices] - default_dof_pos[:, upper_dof_indices]
    )
    return torch.sum(deviation, dim=1)


@torch.jit.script
def reward_feet_heading_alignment(
    rigid_body_rot: Tensor, feet_indices: Tensor, root_rot: Tensor
) -> Tensor:
    B = rigid_body_rot.shape[0]
    fwd = torch.zeros((B, 3), dtype=rigid_body_rot.dtype, device=rigid_body_rot.device)
    fwd[:, 0] = 1.0  # TODO: check local +X as foot forward

    lq = rigid_body_rot[:, feet_indices[0]]
    rq = rigid_body_rot[:, feet_indices[1]]

    lf = rotations.quat_apply(lq, fwd, True)
    rf = rotations.quat_apply(rq, fwd, True)
    bf = rotations.quat_apply(root_rot, fwd, True)

    hl = torch.atan2(lf[:, 1], lf[:, 0])
    hr = torch.atan2(rf[:, 1], rf[:, 0])
    hb = torch.atan2(bf[:, 1], bf[:, 0])

    dl = torch.abs(torch_utils.wrap_to_pi(hl - hb))
    dr = torch.abs(torch_utils.wrap_to_pi(hr - hb))
    return dl + dr


# ----------------
# ----------------


@torch.jit.script
def reward_dof_vel(dof_vel: Tensor) -> Tensor:
    # Penalize dof velocities

    return torch.sum(torch.square(dof_vel), dim=1)


@torch.jit.script
def reward_termination(done_mask: Tensor, timeout_mask: Tensor) -> Tensor:
    # 1 where episode ended this step AND it was NOT a time-limit
    return (done_mask.to(torch.bool) & (~timeout_mask.to(torch.bool))).to(
        done_mask.dtype
    )


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


@torch.jit.script
def reward_orientation(projected_gravity: Tensor) -> Tensor:
    # Penalize non flat base orientation

    return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)


@torch.jit.script
def reward_penalty_close_feet_xy(
    rigid_body_pos: Tensor,  # [B, N_bodies, 3]
    feet_indices: Tensor,  # [2] (left, right)
    threshold: float,
) -> Tensor:
    left_xy = rigid_body_pos[:, feet_indices[0], :2]  # [B, 2]
    right_xy = rigid_body_pos[:, feet_indices[1], :2]  # [B, 2]
    dist = torch.norm(left_xy - right_xy, dim=1)  # [B]
    return (dist < threshold).float()  # [B]


@torch.jit.script
def reward_penalty_ang_vel_xy_torso(
    rigid_body_rot: Tensor,  # [B, N_bodies, 4]
    rigid_body_ang_vel: Tensor,  # [B, N_bodies, 3]
    torso_index: int,
) -> Tensor:
    # Angular velocity in the torso (local) frame; penalize x,y components
    torso_q = rigid_body_rot[:, torso_index]  # [B, 4]
    torso_wld = rigid_body_ang_vel[:, torso_index]  # [B, 3]
    torso_loc = rotations.quat_rotate_inverse(torso_q, torso_wld, True)  # [B, 3]
    return torch.sum(torso_loc[:, :2].pow(2), dim=1)  # [B]


@torch.jit.script
def reward_penalty_hip_pos(
    dof_pos: Tensor,  # [B, D]
    roll_yaw_indices: Tensor,  # [K] long, DOF indices for both hips (roll & yaw only)
) -> Tensor:
    # Penalize hip roll/yaw joint positions
    return torch.sum(dof_pos[:, roll_yaw_indices].pow(2), dim=1)  # [B]
