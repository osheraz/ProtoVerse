import time
from operator import itemgetter

from protoverse.utils.common import print_info
from rich.console import Console
from rich.panel import Panel

console = Console()


class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.time_total = 0.0
        self.num_ons = 0

    def on(self):
        assert self.start_time is None, "Timer {} is already turned on!".format(
            self.name
        )
        self.num_ons += 1
        self.start_time = time.time()

    def off(self):
        assert self.start_time is not None, "Timer {} not started yet!".format(
            self.name
        )
        self.time_total += time.time() - self.start_time
        self.start_time = None

    def report(self):
        if self.num_ons > 0:
            print_info(
                "Time report [{}]: {:.2f} {:.4f} seconds".format(
                    self.name, self.time_total, self.time_total / self.num_ons
                )
            )

    def clear(self):
        self.start_time = None
        self.time_total = 0.0


class TimeReport:
    def __init__(self):
        self.timers = {}

    def add_timer(self, name):
        assert name not in self.timers, "Timer {} already exists!".format(name)
        self.timers[name] = Timer(name=name)

    def start_timer(self, name):
        assert name in self.timers, "Timer {} does not exist!".format(name)
        self.timers[name].on()

    def end_timer(self, name):
        assert name in self.timers, "Timer {} does not exist!".format(name)
        self.timers[name].off()

    def report(self, name=None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].report()
        else:
            print_info("------------Time Report------------")

            timer_with_times = []
            for timer_name in self.timers.keys():
                timer_with_times.append(
                    (self.timers[timer_name].time_total, self.timers[timer_name])
                )
            timer_with_times.sort(key=itemgetter(0))

            for _, timer in timer_with_times:
                timer.report()
            print_info("-----------------------------------")

    def clear_timer(self, name=None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].clear()
        else:
            for timer_name in self.timers.keys():
                self.timers[timer_name].clear()

    def pop_timer(self, name=None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].report()
            del self.timers[name]
        else:
            self.report()
            self.timers = {}


def print_training_panel(
    log_dict: dict,
    it: int,
    total_epochs: int,
    step_count: int,
    fit_start_time: float,
    epoch_start_time: float,
    log_dir: str,
):
    """Pretty-print PPO training stats to the console using Rich."""
    import time

    end_time = time.time()

    # ETA
    epochs_done = it + 1
    if fit_start_time:
        avg_epoch_time = (end_time - fit_start_time) / max(1, epochs_done)
    else:
        avg_epoch_time = log_dict.get("times/last_epoch_seconds", 0.0)
    eta_secs = (
        float(avg_epoch_time * max(0, (total_epochs or 0) - epochs_done))
        if total_epochs
        else 0.0
    )

    pad = 30
    header = (
        f" \033[1mEpoch {it}/{total_epochs if total_epochs else '-'}\033[0m ".center(
            80, " "
        )
    )

    fields = [
        ("Computation (fps,last):", "times/fps_last_epoch", "{:.0f}"),
        ("Computation (fps,total):", "times/fps_total", "{:.0f}"),
        ("Iteration time:", "times/last_epoch_seconds", "{:.2f}s"),
        ("Total frames:", "info/frames", "{}"),
        ("Mean episode reward:", "info/episode_reward", "{:.3f}"),
        ("Mean episode length:", "info/episode_length", "{:.2f}"),
        ("Task reward (mean):", "rewards/task_rewards", "{:.4f}"),
        ("Total reward (mean):", "rewards/total_rewards", "{:.4f}"),
        ("Actor loss (mean):", "losses/actor_loss", "{:.5f}"),
        ("Critic loss (mean):", "losses/critic_loss", "{:.5f}"),
        ("PPO obj (mean):", "actor/ppo_loss", "{:.5f}"),
        ("Entropy (mean):", "actor/entropy", "{:.5f}"),
        ("KL (mean):", "actor/kl", "{:.5f}"),
        ("Clip frac:", "actor/clip_frac", "{:.3f}"),
        ("Policy std (mean):", "actor/mean_action_std", "{:.4f}"),
        ("Actor LR:", "actor/lr", "{:.2e}"),
        ("Critic LR:", "critic/lr", "{:.2e}"),
    ]

    def line(label, key, fmt):
        if key not in log_dict:
            return None
        try:
            return f"{label:>{pad}} {fmt.format(log_dict[key])}"
        except Exception:
            return f"{label:>{pad}} {log_dict[key]}"

    lines = [ln for (lbl, key, fmt) in fields if (ln := line(lbl, key, fmt))]

    # Show some env metrics if available
    env_keys = sorted(k for k in log_dict if k.startswith("env/"))
    if env_keys:
        lines.append("")
        lines.append("Env metrics:")
        for k in env_keys:
            val = log_dict[k]
            if abs(val) < 1e-4:
                lines.append(f"{k:>{pad}} {val:.3e}")  # scientific
            else:
                lines.append(f"{k:>{pad}} {val:.6f}")

    footer = (
        "-" * 80
        + "\n"
        + f"{'ETA:':>{pad}} {eta_secs:.1f}s\n"
        + f"{'Log dir:':>{pad}} {log_dir}"
    )

    body = "\n".join(lines)
    console.print(Panel(header + "\n\n" + body + "\n" + footer, title="Training Log"))
