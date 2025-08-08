import torch

############### KY ###################
######################################


def _reward_tracking_goal_vel(self):
    norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
    target_vec_norm = self.target_pos_rel / (norm + 1e-5)
    cur_vel = self.root_states[:, 7:9]
    rew = torch.minimum(
        torch.sum(target_vec_norm * cur_vel, dim=-1), self.commands[:, 0]
    ) / (self.commands[:, 0] + 1e-5)
    return rew


def _reward_tracking_yaw(self):
    rew = torch.exp(-torch.abs(self.target_yaw - self.yaw))
    return rew


def _reward_lin_vel_z(self):
    rew = torch.square(self.base_lin_vel[:, 2])
    rew[self.env_class != 17] *= 0.5
    return rew


def _reward_ang_vel_xy(self):
    return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)


def _reward_orientation(self):
    rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    rew[self.env_class != 17] = 0.0
    return rew


def _reward_dof_acc(self):
    return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)


def _reward_collision(self):
    return torch.sum(
        1.0
        * (
            torch.norm(
                self.contact_forces[:, self.penalised_contact_indices, :], dim=-1
            )
            > 0.1
        ),
        dim=1,
    )


def _reward_action_rate(self):
    return torch.norm(self.last_actions - self.actions, dim=1)


def _reward_delta_torques(self):
    return torch.sum(torch.square(self.torques - self.last_torques), dim=1)


def _reward_torques(self):
    return torch.sum(torch.square(self.torques), dim=1)


def _reward_hip_pos(self):
    if self.cfg.asset in ["g1", "gr1t1"] and (self.cfg.env.hip_reward_roll == True):
        return torch.sum(
            torch.square(
                self.dof_pos[:, self.roll_hip_indices]
                - self.default_dof_pos[:, self.roll_hip_indices]
            ),
            dim=1,
        )
    else:
        return torch.sum(
            torch.square(
                self.dof_pos[:, self.hip_indices]
                - self.default_dof_pos[:, self.hip_indices]
            ),
            dim=1,
        )


def _reward_dof_error(self):
    dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
    return dof_error


def _reward_feet_stumble(self):
    # Penalize feet hitting vertical surfaces
    rew = torch.any(
        torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
        > 4 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
        dim=1,
    )
    return rew.float()


def _reward_feet_edge(self):
    feet_pos_xy = (
        (
            (
                self.rigid_body_states[:, self.feet_indices, :2]
                + self.terrain.cfg.border_size
            )
            / self.cfg.terrain.horizontal_scale
        )
        .round()
        .long()
    )  # (num_envs, 4, 2)
    feet_pos_xy[..., 0] = torch.clip(
        feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0] - 1
    )
    feet_pos_xy[..., 1] = torch.clip(
        feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1] - 1
    )
    feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]

    self.feet_at_edge = self.contact_filt & feet_at_edge
    rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
    return rew


def _reward_feet_swing_height(self):
    contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
    # feet_pos = ((self.rigid_body_states[:, self.feet_indices, :3] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()
    # pos_error = torch.square(feet_pos[:, :, 2] - ((0.08+self.terrain.cfg.border_size)/ self.cfg.terrain.horizontal_scale).round().long()) * ~contact
    feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
    pos_error = torch.square(feet_pos[:, :, 2] - 0.08) * ~contact
    return torch.sum(pos_error, dim=(1))


def _reward_base_height(self):
    # Penalize base height away from target
    base_height = self.root_states[:, 2]  # y
    return torch.square(base_height - self.cfg.rewards.base_height_target)


def _reward_feet_air_time(self):
    # Reward long steps
    # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
    contact_filt = torch.logical_or(contact, self.last_contacts)
    self.last_contacts = contact
    first_contact = (self.feet_air_time > 0.0) * contact_filt
    self.feet_air_time += self.dt
    rew_airTime = torch.sum(
        (self.feet_air_time - 0.5) * first_contact, dim=1
    )  # reward only on first contact with the ground
    rew_airTime *= (
        torch.norm(self.commands[:, :2], dim=1) > 0.1
    )  # no reward for zero command
    self.feet_air_time *= ~contact_filt
    return rew_airTime


def _reward_point_navigation_direction(self):  # vel
    norm = torch.norm(self.final_target_pos_rel, dim=-1, keepdim=True)
    final_target_vec_norm = self.final_target_pos_rel / (norm + 1e-5)
    cur_vel = self.root_states[:, 7:9]
    rew_pn_direction = torch.minimum(
        torch.sum(final_target_vec_norm * cur_vel, dim=-1), self.commands[:, 0]
    ) / (self.commands[:, 0] + 1e-5)
    if self.cfg.depth.Mid_waypoint_know:
        norm_res = torch.norm(self.mid_waypoint_pos_rel, dim=-1, keepdim=True)
        for i in range(rew_pn_direction.size(0)):
            if self.cur_goal_idx[i] < 4:
                rew_pn_direction[i] += (
                    0.75 * ((self.cur_goal_idx[i] + 1) / 5) * norm_res[i].item()
                )
    return rew_pn_direction


def _reward_point_navigation_distance(self):
    norm = torch.norm(self.final_target_pos_rel, dim=-1, keepdim=True)
    rew_pn_dis = torch.zeros_like(norm)
    if self.cfg.depth.Mid_waypoint_know:
        norm_res_mid = torch.norm(self.mid_waypoint_pos_rel, dim=-1, keepdim=True)
    for i_th_env_id in range(rew_pn_dis.size(0)):
        if norm[i_th_env_id, :] < 1.0:
            rew_pn_dis[i_th_env_id, :] = 1.0
        else:
            if self.cfg.rewards.pn_distance_clip:
                rew_pn_dis[i_th_env_id, :] = torch.clip(
                    -norm[i_th_env_id, :] * 0.75, -5, 0
                )
            else:
                rew_pn_dis[i_th_env_id, :] = -norm[i_th_env_id, :] * 0.75
            if self.cfg.depth.Mid_waypoint_know and (
                self.cur_goal_idx[i_th_env_id] < 4
            ):
                rew_pn_dis[i_th_env_id, :] += -norm_res_mid[i_th_env_id, :] * 0.25

    return rew_pn_dis.squeeze(1)


def _reward_lower_body_balance(self):

    range_feet = 2 if self.cfg.env.robot_asset in ["h1", "g1", "gr1t1"] else 4
    normal_vec_ref = torch.tensor(
        [0.0, 0.0, 1.0], device=self.device
    )  # Standard normal vector ,up z
    normal_vec_ref = normal_vec_ref.unsqueeze(0)
    normal_vec_ref_batch = normal_vec_ref.repeat(self.base_quat.size(0), 1)
    root_normal = torch.zeros(self.base_quat.size(0), 3, device=self.device)
    # normal_vec_ref_batch_for_feet = normal_vec_ref_batch.clone()
    # normal_vec_ref_batch_for_feet = normal_vec_ref_batch_for_feet.unsqueeze(1)
    # normal_vec_ref_batch_for_feet = normal_vec_ref_batch_for_feet.expand(-1,2,-1)

    # Compute the root's normal direction in the world frame
    for env_index in range(self.base_quat.size(0)):
        root_normal[env_index,] = rotate_vector_by_quaternion(
            self.base_quat[env_index,], normal_vec_ref_batch[env_index,]
        )
    # compute feet's normal
    feet_quat = self.rigid_body_states[:, self.feet_indices, 3:7]

    feet_normal = torch.zeros(self.base_quat.size(0), range_feet, 3, device=self.device)

    rew_1 = torch.tensor(0.0, device=self.device)

    for j in range(self.base_quat.size(0)):
        for i in range(range_feet):
            feet_normal[
                j,
                i,
            ] = rotate_vector_by_quaternion(
                feet_quat[
                    j,
                    i,
                ],
                normal_vec_ref_batch[j,],
            )

    for foot_normal in range(feet_normal.size(1)):
        if torch.linalg.norm(root_normal - foot_normal) < 0.1:
            rew_1 += torch.tensor(1.0)  # Reward value if alignment is within threshold
        else:
            rew_1 += -torch.linalg.norm(root_normal - foot_normal)

    return 0.7 * rew_1


def _reward_time_spend(self):
    ts_alpha = 0.01
    ts_beta = 0.001
    ts_tau = ts_alpha * torch.exp(ts_beta * self.episode_length_buf)
    # rew_ts = ts_tau * self.episode_length_buf
    a = self.episode_length_buf
    rew_ts = ts_tau * torch.max(
        torch.tensor(0),
        torch.ceil(torch.tensor(10.0) / self.cfg.sim.dt) - self.episode_length_buf,
    )
    return rew_ts


def _reward_dof_pos_limits(self):
    # Penalize dof positions too close to the limit
    out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(
        max=0.0
    )  # lower limit
    out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)
