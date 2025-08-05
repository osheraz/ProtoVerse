import typer
import os


def main(
    motion_file: str = "g1_walk.npy",
    simulator: str = "isaaclab",
    robot: str = "g1",
    num_envs: int = 1,
    extra_args: str = "",
):
    # command = f"python protomotions/eval_agent.py +base=[fabric,structure] +exp=deepmimic_mlp +robot={robot} +simulator={simulator} +checkpoint=null +training_max_steps=1 +motion_file={motion_file} env.config.sync_motion=True ref_respawn_offset=0 +headless=False num_envs={num_envs} {extra_args} +experiment_name=debug"
    command = (
        f"python protomotions/eval_agent.py "
        f"+base=[fabric,structure] "
        f"+exp=deepmimic_mlp "
        f"+robot={robot} "
        f"+simulator={simulator} "
        f"+checkpoint=null "
        f"+training_max_steps=1 "
        f"+motion_file={motion_file} "
        f"env.config.sync_motion=True "
        f"ref_respawn_offset=0 "
        f"+headless=False "
        f"num_envs={num_envs} "
        f"{extra_args} "
        f"+experiment_name=debug"
    )

    os.system(command)


if __name__ == "__main__":
    typer.run(main)
