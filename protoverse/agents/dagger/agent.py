import torch
from torch import Tensor
import logging, time
from pathlib import Path
from typing import Tuple, Dict, Optional
from lightning.fabric import Fabric
from hydra.utils import instantiate
from omegaconf import OmegaConf
from rich.progress import track

from protoverse.agents.utils.data_utils import DictDataset, ExperienceBuffer
from protoverse.agents.ppo.agent import PPO
from protoverse.agents.ppo.model import PPOModel
from protoverse.agents.common.common import weight_init

log = logging.getLogger(__name__)


class Dagger(PPO):
    # -----------------------------
    # Initialization and Setup
    # -----------------------------
    def __init__(self, fabric: Fabric, env, config):
        super().__init__(fabric, env, config)

    def setup(self):
        # Student model + optimizer (reuse PPOModel API)
        model: PPOModel = instantiate(self.config.model)
        model.apply(weight_init)
        optimizer = instantiate(
            self.config.model.config.optimizer, params=list(model.parameters())
        )
        self.model, self.optimizer = self.fabric.setup(model, optimizer)
        self.model.mark_forward_method("act")

        # Load the expert (oracle)
        if self.config.expert_model_path is None:
            raise ValueError("DAgger requires config.expert_model_path.")
        exp_dir = Path(self.config.expert_model_path)
        ckpt = exp_dir / "score_based.ckpt"
        if not ckpt.exists():
            ckpt = exp_dir / "last.ckpt"
        log.info(f"Loading expert from: {ckpt}")

        expert_cfg = OmegaConf.load(exp_dir / "config.yaml")
        expert_model: PPOModel = instantiate(expert_cfg.agent.config.model)
        self.expert_model = self.fabric.setup(expert_model)
        self.expert_model.mark_forward_method("act")

        state = torch.load(ckpt, map_location=self.fabric.device)
        self.expert_model.load_state_dict(state["model"])
        for p in self.expert_model.parameters():
            p.requires_grad = False
        self.expert_model.eval()

    # -----------------------------
    # Experience Buffer Registration
    # -----------------------------
    def register_extra_experience_buffer_keys(self):
        pass

    # -----------------------------
    # Beta schedule
    # -----------------------------
    def _beta_at_epoch(self, epoch: int) -> float:
        cfg = self.config.dagger
        beta0 = float(getattr(cfg, "beta_start", 1.0))
        betaT = float(getattr(cfg, "beta_end", 0.0))
        T = int(getattr(cfg, "beta_anneal_epochs", 50))
        if T <= 0:
            return betaT
        frac = max(0.0, min(1.0, epoch / T))
        return beta0 + frac * (betaT - beta0)

    # -----------------------------
    # Fit: collect -> supervise (no aggregation)
    # -----------------------------
    def fit(self):
        # Per-epoch buffer (only what we need)
        self.experience_buffer = ExperienceBuffer(self.num_envs, self.num_steps).to(
            self.device
        )
        self.experience_buffer.register_key(
            "self_obs", shape=(self.env.config.robot.self_obs_size,)
        )
        self.experience_buffer.register_key("rewards")
        self.experience_buffer.register_key("extra_rewards")
        self.experience_buffer.register_key("total_rewards")
        self.experience_buffer.register_key("dones", dtype=torch.long)
        self.experience_buffer.register_key(
            "expert_actions", shape=(self.env.config.robot.number_of_actions,)
        )
        self.register_extra_experience_buffer_keys()

        if self.config.get("extra_inputs", None) is not None:
            obs0 = self.env.get_obs()
            for key in self.config.extra_inputs.keys():
                assert (
                    key in obs0
                ), f"Key {key} not found in obs returned from env: {obs0.keys()}"
                env_tensor = obs0[key]
                shape = env_tensor.shape
                dtype = env_tensor.dtype
                self.experience_buffer.register_key(key, shape=shape[1:], dtype=dtype)

        # Fit start
        done_indices = None
        if self.fit_start_time is None:
            self.fit_start_time = time.time()
        self.fabric.call("on_fit_start", self)

        while self.current_epoch < self.config.max_epochs:
            self.epoch_start_time = time.time()
            beta = self._beta_at_epoch(self.current_epoch)

            # -------- collect current rollout --------
            self.eval()
            with torch.no_grad():
                self.fabric.call("before_play_steps", self)
                for step in track(
                    range(self.num_steps),
                    description=f"Epoch {self.current_epoch} (β={beta:.2f}), collecting data...",
                ):
                    obs = self.handle_reset(done_indices)

                    self.experience_buffer.update_data(
                        "self_obs", step, obs["self_obs"]
                    )
                    if self.config.get("extra_inputs", None) is not None:
                        for key in self.config.extra_inputs:
                            self.experience_buffer.update_data(key, step, obs[key])

                    # student & expert
                    student_action = self.model.act(obs)
                    expert_action = self.expert_model.act(obs)
                    self.experience_buffer.update_data(
                        "expert_actions", step, expert_action
                    )

                    # β-mixture for env step
                    if beta <= 0.0:
                        action = student_action
                    elif beta >= 1.0:
                        action = expert_action
                    else:
                        mask = (
                            (torch.rand(self.num_envs, device=self.device) < beta)
                            .float()
                            .unsqueeze(-1)
                        )
                        action = mask * expert_action + (1.0 - mask) * student_action

                    # NaN checks (keep parity with your style)
                    for k in obs.keys():
                        if torch.isnan(obs[k]).any():
                            print(f"NaN in {k}: {obs[k]}")
                            raise ValueError("NaN in obs")
                    if torch.isnan(action).any():
                        raise ValueError(f"NaN in action: {action}")

                    # env step
                    next_obs, rewards, dones, terminated, extras = self.env_step(action)

                    all_done_indices = dones.nonzero(as_tuple=False)
                    done_indices = all_done_indices.squeeze(-1)

                    self.post_train_env_step(rewards, dones, done_indices, extras, step)
                    self.experience_buffer.update_data("rewards", step, rewards)
                    self.experience_buffer.update_data("dones", step, dones)

                    self.step_count += self.get_step_count_increment()

                # reward fields for logging parity
                rewards = self.experience_buffer.rewards
                extra_rewards = self.calculate_extra_reward()
                self.experience_buffer.batch_update_data("extra_rewards", extra_rewards)
                total_rewards = rewards + extra_rewards
                self.experience_buffer.batch_update_data("total_rewards", total_rewards)

            # -------- supervise on THIS rollout only --------
            training_log_dict = self.optimize_model()
            training_log_dict["epoch"] = self.current_epoch

            self.current_epoch += 1
            self.fabric.call("after_train", self)

            # regular checkpoint cadence
            if self.current_epoch % self.config.manual_save_every == 0:
                self.save()

            # optional eval (kept aligned with PPO flow)
            if (
                self.config.eval_metrics_every is not None
                and self.current_epoch > 0
                and self.current_epoch % self.config.eval_metrics_every == 0
            ):
                eval_log_dict, evaluated_score = self.calc_eval_metrics()
                evaluated_score = self.fabric.broadcast(evaluated_score, src=0)
                if evaluated_score is not None:
                    if (
                        self.best_evaluated_score is None
                        or evaluated_score >= self.best_evaluated_score
                    ):
                        self.best_evaluated_score = evaluated_score
                        self.save(new_high_score=True)
                training_log_dict.update(eval_log_dict)

            self.post_epoch_logging(training_log_dict)
            self.env.on_epoch_end(self.current_epoch)

            if self.should_stop:
                self.save()
                return

        self.time_report.report()
        self.save()
        self.fabric.call("on_fit_end", self)

    # -----------------------------
    # Dataset & Optimization (current rollout only)
    # -----------------------------
    @torch.no_grad()
    def process_dataset(self, dataset):
        return DictDataset(self.config.batch_size, dataset, shuffle=True)

    def optimize_model(self) -> Dict:
        dataset = self.process_dataset(self.experience_buffer.make_dict())

        self.train()
        training_log_dict = {}

        for batch_idx in track(
            range(self.max_num_batches()),
            description=f"Epoch {self.current_epoch}, supervised training...",
        ):
            iter_log_dict = {}

            dataset_idx = batch_idx % len(dataset)
            if dataset_idx == 0 and batch_idx != 0 and dataset.do_shuffle:
                dataset.shuffle()
            batch_dict = dataset[dataset_idx]

            for key in batch_dict.keys():
                if torch.isnan(batch_dict[key]).any():
                    print(f"NaN in {key}: {batch_dict[key]}")
                    raise ValueError("NaN in training")

            # supervised BC on expert_actions
            loss, loss_dict = self.model_step(batch_dict)
            iter_log_dict.update(loss_dict)
            self.optimizer.zero_grad(set_to_none=True)
            self.fabric.backward(loss)
            grad_clip_dict = self.handle_model_grad_clipping(
                self.model, self.optimizer, "model"
            )
            iter_log_dict.update(grad_clip_dict)
            self.optimizer.step()

            extra_opt_steps_dict = self.extra_optimization_steps(batch_dict, batch_idx)
            iter_log_dict.update(extra_opt_steps_dict)

            for k, v in iter_log_dict.items():
                if k in training_log_dict:
                    training_log_dict[k][0] += v
                    training_log_dict[k][1] += 1
                else:
                    training_log_dict[k] = [v, 1]

        for k, v in training_log_dict.items():
            training_log_dict[k] = v[0] / v[1]

        self.eval()
        return training_log_dict

    # -----------------------------
    # Supervised step (BC)
    # -----------------------------
    def model_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        student_actions = self.model.act(batch_dict)
        expert_actions = batch_dict["expert_actions"]

        bc_loss = torch.square(student_actions - expert_actions).mean()
        extra_loss, extra_log_dict = self.calculate_extra_loss(
            batch_dict, student_actions
        )

        loss = bc_loss + extra_loss
        log_dict = {
            "model/bc_loss": bc_loss.detach(),
            "model/extra_loss": extra_loss.detach(),
            "losses/model_loss": loss.detach(),
            "dagger/beta": torch.tensor(
                self._beta_at_epoch(self.current_epoch), device=self.device
            ),
        }
        log_dict.update(extra_log_dict)
        return loss, log_dict

    def calculate_extra_loss(self, batch_dict, actions) -> Tuple[Tensor, Dict]:
        return torch.tensor(0.0, device=self.device), {}

    # -----------------------------
    # State Saving / Loading (same pattern)
    # -----------------------------
    def get_state_dict(self, state_dict):
        extra = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "step_count": self.step_count,
            "run_start_time": self.fit_start_time,
            "episode_reward_meter": self.episode_reward_meter.state_dict(),
            "episode_length_meter": self.episode_length_meter.state_dict(),
            "best_evaluated_score": self.best_evaluated_score,
        }
        if self.config.normalize_values:
            extra["running_val_norm"] = self.running_val_norm.state_dict()
        state_dict.update(extra)
        return state_dict

    def load_parameters(self, state_dict):
        self.current_epoch = state_dict["epoch"]
        if "step_count" in state_dict:
            self.step_count = state_dict["step_count"]
        if "run_start_time" in state_dict:
            self.fit_start_time = state_dict["run_start_time"]
        self.best_evaluated_score = state_dict.get("best_evaluated_score", None)

        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

        if self.config.normalize_values:
            self.running_val_norm.load_state_dict(state_dict["running_val_norm"])

        self.episode_reward_meter.load_state_dict(state_dict["episode_reward_meter"])
        self.episode_length_meter.load_state_dict(state_dict["episode_length_meter"])

    @torch.no_grad()
    def calc_eval_metrics(self):
        """Evaluate the student policy by rolling out for a fixed horizon.
        Aggregates metrics from extras['to_log'] across envs/steps/episodes/ranks.
        """
        self.eval()

        # ---- config knobs ----
        eval_num_episodes = getattr(self.config, "eval_num_episodes", 1)
        eval_length = getattr(self.config, "eval_length", None)
        if eval_length is None and hasattr(self.env, "max_episode_length"):
            eval_length = int(self.env.max_episode_length)

        metric_keys = list(getattr(self.config, "eval_metric_keys", []))
        # which metric becomes the scalar "score" (or "mean_reward")
        score_key = getattr(self.config, "eval_score_key", "mean_reward")

        # per-rank accumulators
        total_steps = torch.zeros(1, device=self.device)
        reward_sum = torch.zeros(1, device=self.device)
        metric_sums: Dict[str, torch.Tensor] = {
            k: torch.zeros(1, device=self.device) for k in metric_keys
        }

        for _ in range(eval_num_episodes):
            # hard reset all envs
            obs = self.env.reset(
                torch.arange(0, self.num_envs, dtype=torch.long, device=self.device)
            )

            steps = 0
            while eval_length is None or steps < eval_length:
                actions = self.model.act(obs)
                obs, rewards, dones, terminated, extras = self.env_step(actions)

                # reward accumulation
                reward_sum += rewards.sum()
                total_steps += torch.tensor([rewards.numel()], device=self.device)

                # optional metrics from env -> extras["to_log"]
                tolog = extras.get("to_log", {}) if isinstance(extras, dict) else {}
                if isinstance(tolog, dict):
                    for k in metric_keys:
                        v = tolog.get(k, None)
                        if v is None:
                            continue
                        if not isinstance(v, torch.Tensor):
                            v = torch.tensor(v, device=self.device, dtype=torch.float32)
                        # sum per-step over envs
                        metric_sums[k] += v if v.ndim == 0 else v.float().sum()

                steps += 1

        # ---- distributed reduction ----
        def _sum_all_ranks(x: torch.Tensor) -> torch.Tensor:
            xs = self.fabric.all_gather(x)
            return xs.sum()

        g_total_steps = _sum_all_ranks(total_steps)
        g_reward_sum = _sum_all_ranks(reward_sum)
        g_metric_sums = {k: _sum_all_ranks(v) for k, v in metric_sums.items()}

        # ---- means ----
        denom = torch.clamp_min(g_total_steps, 1)
        mean_reward = (g_reward_sum / denom).item()

        to_log: Dict[str, float] = {"eval/mean_reward": mean_reward}
        for k in metric_keys:
            to_log[f"eval/{k}"] = (g_metric_sums[k] / denom).item()

        # ---- choose scalar score ----
        evaluated_score = (
            to_log["eval/mean_reward"]
            if score_key == "mean_reward"
            else to_log.get(f"eval/{score_key}", float("nan"))
        )

        return to_log, evaluated_score
