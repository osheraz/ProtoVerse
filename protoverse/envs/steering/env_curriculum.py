import numpy as np
import torch
from torch import Tensor

from protoverse.envs.steering.env import Steering
from protoverse.simulator.base_simulator.robot_state import RobotState


class SteeringCurriculum(Steering):
    """
    Curriculum over start position (terrain difficulty) and heading.
    Success = no termination AND enough projected progress along the target direction.
    Difficulty goes up/down based on success streaks/failure streaks.
    """

    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config=config, device=device, *args, **kwargs)

        # ---- pull curriculum config once ----
        curr_config = config.steering_curriculum

        # Per-env difficulty in [0, 1]
        self._diff = torch.zeros(self.num_envs, device=self.device)
        self._succ_streak = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._fail_streak = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )

        # Difficulty stepping
        self._diff_step_up = float(curr_config.difficulty.diff_step_up)
        self._diff_step_down = float(curr_config.difficulty.diff_step_down)
        self._succ_needed = int(curr_config.difficulty.succ_needed)
        self._fail_needed = int(curr_config.difficulty.fail_needed)

        # Start-position sampling (flat vs rough mix)
        self._flat_mix_k = float(curr_config.starts.flat_mix_k)
        self._flat_min_prob = float(curr_config.starts.flat_min_prob)

        # Heading/goal knobs
        self._min_goal_dist = float(curr_config.heading.min_goal_dist_m)
        self._max_goal_dist = float(curr_config.heading.max_goal_dist_m)
        self._max_heading_jitter = float(
            np.deg2rad(curr_config.heading.max_heading_jitter_deg)
        )
        self._random_heading_base = float(curr_config.heading.random_heading_base)
        self._random_heading_max = float(curr_config.heading.random_heading_max)

        # Success threshold (projected progress along target dir)
        self._min_prog_at_d0 = float(curr_config.success.min_progress_m_at_d0)
        self._min_prog_at_d1 = float(curr_config.success.min_progress_m_at_d1)

        # Optional down-step on stagnation (disabled if <=0)
        self._stagnant_eps_need = int(curr_config.success.stagnant_eps_need)
        self._stagnant_eps = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )

        # Episode trackers
        self._last_root_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._ep_proj_prog = torch.zeros(
            self.num_envs, device=self.device
        )  # meters of forward progress

    # ----------------------- episode lifecycle -----------------------

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # Outcome of the episode that just ended
        just_timed_out = self.progress_buf[env_ids] >= (self.max_episode_length - 1)
        no_termination = self.terminate_buf[env_ids] == 0

        # progress requirement scales with difficulty
        d = self._diff[env_ids]
        need_prog = (
            self._min_prog_at_d0 + (self._min_prog_at_d1 - self._min_prog_at_d0) * d
        )
        moved_enough = self._ep_proj_prog[env_ids] >= need_prog

        ep_success = no_termination & moved_enough  # & just_timed_out

        # Update streaks
        self._succ_streak[env_ids] = torch.where(
            ep_success,
            self._succ_streak[env_ids] + 1,
            torch.zeros_like(self._succ_streak[env_ids]),
        )
        self._fail_streak[env_ids] = torch.where(
            ~ep_success,
            self._fail_streak[env_ids] + 1,
            torch.zeros_like(self._fail_streak[env_ids]),
        )

        # Step difficulty up/down by streaks
        inc_mask = ep_success & (self._succ_streak[env_ids] >= self._succ_needed)
        if inc_mask.any():
            self._diff[env_ids[inc_mask]] = torch.clamp(
                self._diff[env_ids[inc_mask]] + self._diff_step_up, 0.0, 1.0
            )
            self._succ_streak[env_ids[inc_mask]] = 0

        dec_mask = (~ep_success) & (self._fail_streak[env_ids] >= self._fail_needed)
        if dec_mask.any():
            self._diff[env_ids[dec_mask]] = torch.clamp(
                self._diff[env_ids[dec_mask]] - self._diff_step_down, 0.0, 1.0
            )
            self._fail_streak[env_ids[dec_mask]] = 0

        # Optional stagnation backoff (if not making progress repeatedly)
        if self._stagnant_eps_need > 0:
            is_stagnant = ~ep_success
            self._stagnant_eps[env_ids] = torch.where(
                is_stagnant,
                self._stagnant_eps[env_ids] + 1,
                torch.zeros_like(self._stagnant_eps[env_ids]),
            )
            stagnant_mask = self._stagnant_eps[env_ids] >= self._stagnant_eps_need
            if stagnant_mask.any():
                self._diff[env_ids[stagnant_mask]] = torch.clamp(
                    self._diff[env_ids[stagnant_mask]] - self._diff_step_down, 0.0, 1.0
                )
                self._stagnant_eps[env_ids[stagnant_mask]] = 0

        # Log difficulty
        self.log_dict["curriculum/diff_mean"] = self._diff.mean()
        self.log_dict["curriculum/diff_std"] = self._diff.std()

        rs = self.simulator.get_root_state(env_ids)
        self._last_root_xy[env_ids] = rs.root_pos[:, :2]
        self._ep_proj_prog[env_ids] = 0.0

        return super().reset(env_ids)

    def post_physics_step(self):
        super().post_physics_step()

        # Accumulate projected progress: project dxy onto current target direction
        rs = self.simulator.get_root_state()
        xy = rs.root_pos[:, :2]
        dxy = xy - self._last_root_xy
        # target direction per env (2D). Forward-only progress:
        step_prog = (dxy * self._tar_dir).sum(dim=-1).clamp(min=0.0)
        self._ep_proj_prog += step_prog

        self._last_root_xy[:] = xy

    # -------------------- start pose (XY) & height --------------------

    def reset_default(self, env_ids):
        """
        Same as BaseEnv.reset_default, but:
        - start XY comes from curriculum (_sample_start_xy)
        - Z is computed exactly like BaseEnv.get_envs_respawn_position(...) with rigid_body_pos
        - all in-place ops are done on cloned tensors to avoid aliasing
        """
        default_state = self.default_state

        # --- clone everything we'll mutate ---
        root_pos = default_state.root_pos[env_ids].clone()
        root_rot = default_state.root_rot[env_ids].clone()
        dof_pos = default_state.dof_pos[env_ids].clone()
        root_vel = default_state.root_vel[env_ids].clone()
        root_ang_vel = default_state.root_ang_vel[env_ids].clone()
        dof_vel = default_state.dof_vel[env_ids].clone()
        rigid_body_pos = default_state.rigid_body_pos[env_ids].clone()
        rigid_body_rot = default_state.rigid_body_rot[env_ids].clone()
        rigid_body_vel = default_state.rigid_body_vel[env_ids].clone()
        rigid_body_ang_vel = default_state.rigid_body_ang_vel[env_ids].clone()

        # ---- curriculum-based XY ----
        start_xy = self._sample_start_xy(len(env_ids), env_ids)  # [n,2] in meters

        # ---- compute Z like BaseEnv.get_envs_respawn_position(...) ----
        # Normalize RB positions around the root, then translate to start_xy
        normalized = rigid_body_pos.clone()  # [n, B, 3]
        normalized[:, :, :2] -= rigid_body_pos[:, :1, :2].clone()  # remove root XY
        normalized[:, :, :2] += start_xy.unsqueeze(1)  # add new XY

        # Query ground heights under every joint
        flat_norm = normalized.reshape(-1, 3)  # [n*B, 3] (uses first 2 cols)
        z_all = self.terrain.get_ground_heights(flat_norm)  # [n*B]
        z_all = z_all.view(normalized.shape[0], normalized.shape[1])  # [n, B]

        # Find the joint needing the largest upward shift
        z_diff = z_all - normalized[:, :, 2]  # [n, B]
        z_indices = torch.argmax(z_diff, dim=1, keepdim=True)  # [n, 1]
        z_offset = z_all.gather(1, z_indices) + self.config.ref_respawn_offset  # [n, 1]

        # ---- set final root pose ----
        root_pos[:, 0:2] = start_xy
        root_pos[:, 2:3] = z_offset

        # ---- rebase whole body to the new root ----
        rb = rigid_body_pos.clone()
        rb[:, :, :3] -= rb[:, 0:1, :3].clone()
        rb[:, :, :3] += root_pos.unsqueeze(1)

        return RobotState(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            rigid_body_pos=rb,
            rigid_body_rot=rigid_body_rot,
            rigid_body_vel=rigid_body_vel,
            rigid_body_ang_vel=rigid_body_ang_vel,
        )

    @torch.no_grad()
    def _sample_start_xy(self, n: int, env_ids: torch.Tensor) -> torch.Tensor:
        """
        Flat→rough mixture by difficulty d:
          p_flat = clamp(1 - k*d, flat_min_prob, 1)
        """
        d = self._diff[env_ids]
        p_flat = torch.clamp(1.0 - self._flat_mix_k * d, self._flat_min_prob, 1.0)

        flat_x, flat_y = self.terrain.flat_x_coords, self.terrain.flat_y_coords
        walk_x, walk_y = self.terrain.walkable_x_coords, self.terrain.walkable_y_coords

        def sample_pool(px: torch.Tensor, py: torch.Tensor, count: int) -> torch.Tensor:
            ix = torch.randint(0, px.shape[0], (count,), device=self.device)
            iy = torch.randint(0, py.shape[0], (count,), device=self.device)
            return torch.stack([px[ix], py[iy]], dim=-1)

        choose_flat = torch.rand(n, device=self.device) < p_flat
        flat_xy = sample_pool(flat_x, flat_y, n)
        rough_xy = sample_pool(walk_x, walk_y, n)
        return torch.where(choose_flat.unsqueeze(-1), flat_xy, rough_xy)

    # ----------------------- difficulty-aware heading -----------------------

    def reset_heading_task(self, env_ids):
        """
        Heading/goal sampling that scales with difficulty.
        """
        n = len(env_ids)
        if n == 0:
            return

        d = self._diff[env_ids]  # [n]

        # distance grows with difficulty (with a little noise)
        goal_dist = (
            self._min_goal_dist + (self._max_goal_dist - self._min_goal_dist) * d
        )
        goal_dist = goal_dist * (0.8 + 0.4 * torch.rand_like(goal_dist))

        # probability of fully random heading grows with difficulty
        rand_p = (
            self._random_heading_base
            + (self._random_heading_max - self._random_heading_base) * d
        )
        choose_random = torch.rand(n, device=self.device) < rand_p

        # jitter amplitude grows with difficulty
        jitter_amp = d * self._max_heading_jitter
        jitter = (2 * torch.rand(n, device=self.device) - 1.0) * jitter_amp

        prev_theta = self._tar_dir_theta[env_ids]
        new_theta = torch.where(
            choose_random,
            2 * np.pi * torch.rand(n, device=self.device) - np.pi,
            prev_theta + jitter,
        )

        tar_dir = torch.stack([torch.cos(new_theta), torch.sin(new_theta)], dim=-1)

        # speed: reuse your bounds; mildly scale with difficulty
        tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(
            n, device=self.device
        ) + self._tar_speed_min
        tar_speed = torch.clamp(
            tar_speed * (0.9 + 0.2 * d), self._tar_speed_min, self._tar_speed_max
        )

        change_steps = torch.randint(
            low=self._heading_change_steps_min,
            high=self._heading_change_steps_max,
            size=(n,),
            device=self.device,
            dtype=torch.int64,
        )

        self._tar_speed[env_ids] = tar_speed
        self._tar_dir_theta[env_ids] = new_theta
        self._tar_dir[env_ids] = tar_dir
        self._heading_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps
