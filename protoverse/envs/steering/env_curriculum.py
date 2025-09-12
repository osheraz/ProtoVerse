import numpy as np
import torch
from torch import Tensor

from protoverse.envs.steering.env_proto import Steering
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

        # outcome of the episode that just ended
        just_timed_out = self.progress_buf[env_ids] >= (self.max_episode_length - 1)
        no_termination = self.terminate_buf[env_ids] == 0

        d = self._diff[env_ids]
        need_prog = (
            self._min_prog_at_d0 + (self._min_prog_at_d1 - self._min_prog_at_d0) * d
        )
        moved_enough = self._ep_proj_prog[env_ids] >= need_prog
        ep_success = no_termination & moved_enough  # keep as you had

        # streaks
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

        self.log_dict["curriculum/diff_mean"] = self._diff.mean()
        self.log_dict["curriculum/diff_std"] = self._diff.std()

        # do the actual respawn first (calls reset_default + reset_heading_task)
        obs = super().reset(env_ids)

        # now start progress tracking from the new pose (avoid counting teleport)
        rs = self.simulator.get_root_state(env_ids)
        self._last_root_xy[env_ids] = rs.root_pos[:, :2]
        self._ep_proj_prog[env_ids] = 0.0

        return obs

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
        Start XY from curriculum sampler that uses Terrain's paired samplers.
        Height (Z) computed like BaseEnv.get_envs_respawn_position with rigid_body_pos.
        Preserve original vertical offset by ADDING the respawn vector.
        """
        default_state = self.default_state

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

        # curriculum XY using Terrain's paired samplers (validated inside Terrain)
        start_xy = self._sample_start_xy(len(env_ids), env_ids).to(
            device=root_pos.device, dtype=root_pos.dtype
        )  # [n,2]

        # compute Z via terrain under all joints (preserves motion-relative heights)
        normalized = rigid_body_pos.clone()
        normalized[:, :, :2] -= rigid_body_pos[:, :1, :2].clone()
        normalized[:, :, :2] += start_xy.unsqueeze(1)

        flat_norm = normalized.reshape(-1, 3)
        z_all = self.terrain.get_ground_heights(flat_norm)
        z_all = z_all.view(normalized.shape[0], normalized.shape[1])

        z_diff = z_all - normalized[:, :, 2]
        z_indices = torch.argmax(z_diff, dim=1, keepdim=True)
        z_offset = (
            z_all.gather(1, z_indices).view(-1, 1) + self.config.ref_respawn_offset
        )

        respawn_pos = torch.cat([start_xy, z_offset], dim=-1)  # [n,3]
        root_pos[:, :2] = 0.0
        root_pos[:, :3] += respawn_pos

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
        Per-env start XY:
        - Map difficulty d in [0,1] -> level row index in [0, terrain.env_rows-1].
        - Sample inside that row band only.
        - Mix flat vs rough per-env via p_flat = clamp(1 - k*d, flat_min_prob, 1).
        - Use terrain masks to get valid PAIRS of (row,col) within the band.
        - Fallbacks ensure we always return a coordinate.
        """
        device = self.device
        if getattr(self, "terrain", None) is None:
            return torch.zeros((n, 2), device=device, dtype=torch.float32)

        T = self.terrain
        d = self._diff[env_ids].clamp(0.0, 1.0)  # [n]
        p_flat = torch.clamp(
            1.0 - self._flat_mix_k * d, self._flat_min_prob, 1.0
        )  # [n]

        # map difficulty -> level row index
        # d=0 -> level 0, d=1 -> last level
        level_idx = torch.clamp((d * (T.env_rows - 1)).long(), 0, T.env_rows - 1)  # [n]

        # convenience
        hs = T.horizontal_scale
        rows_per_level = T.length_per_env_pixels
        start_rows = (T.border + level_idx * rows_per_level).tolist()  # python ints
        end_rows = (T.border + (level_idx + 1) * rows_per_level).tolist()

        # masks (on device)
        walkable = T.walkable_field == 0  # True where walkable
        flatmask = T.flat_field_raw == 0  # True where flat

        out = torch.zeros((n, 2), device=device, dtype=torch.float32)

        # per-env banded sampling
        for i in range(n):
            r0, r1 = int(start_rows[i]), int(end_rows[i])
            # band-restricted masks
            band_mask = torch.zeros_like(walkable)
            band_mask[r0:r1, :] = True

            # candidates inside band
            flat_band = band_mask & flatmask
            rough_band = band_mask & walkable & (~flatmask)

            # choose flat vs rough for this env
            choose_flat = torch.rand((), device=device) < p_flat[i]

            # pick candidate set (fallbacks if empty)
            cand = flat_band if choose_flat else rough_band
            if not cand.any():
                # if chosen set empty, try the other
                cand = rough_band if choose_flat else flat_band
            if not cand.any():
                # if both empty in this band, fallback: any walkable in band
                cand = band_mask & walkable
            if not cand.any():
                # if still empty (pathological), fallback: any global walkable
                cand = walkable

            rows, cols = torch.where(cand)
            j = torch.randint(0, rows.shape[0], (1,), device=device).item()

            # convert grid indices -> world XY (meters)
            x_m = rows[j].float() * hs
            y_m = cols[j].float() * hs
            out[i] = torch.stack([x_m, y_m])

        return out

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
