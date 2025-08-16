# Train
```
HYDRA_FULL_ERROR=1 python protoverse/train_agent.py +experiment_name=h1_new_obs +exp=steering_mlp +robot=h1 +simulator=isaaclab +opt=wandb ++env.config.enable_height_termination=true ++robot.use_robot_obs=true

HYDRA_FULL_ERROR=1 python protoverse/train_agent.py +exp=path_follower_amp_mlp motion_file=protomotions/data/motions/amp_humanoid_walk.npy +simulator=isaaclab +opt=wandb +experiment_name=test_amp

HYDRA_FULL_ERROR=1 python protoverse/train_agent.py +exp=amp_mlp +robot=h1 motion_file=protoverse/data/motions/h1_walk.npy +simulator=isaaclab +opt=wandb +experiment_name=test_amp ++num_envs=512

```

# Eval
```
python protoverse/eval_agent.py +robot=h1 +simulator=isaaclab +checkpoint=results/test_amp/last.ckpt +init_viser=true

python protoverse/eval_agent.py +robot=g1 +simulator=isaaclab +checkpoint=results/g1_slow/last.ckpt +init_viser=true ++env.config.enable_height_termination=true

HYDRA_FULL_ERROR=1 python protoverse/eval_agent.py +robot=h1 +simulator=isaaclab +checkpoint=results/h1_new_obs/last.ckpt +init_viser=true ++env.config.enable_height_termination=true ++simulator.config.with_multi_viewport_camera=true

HYDRA_FULL_ERROR=1 python protoverse/eval_agent.py +robot=g1_w_foot_sensors +simulator=isaaclab +checkpoint=results/g1_slow/last.ckpt +init_viser=true ++env.config.enable_height_termination=true ++simulator.config.with_multi_viewport_camera=true
```

# Important params
```
+init_viser=true
++env.config.enable_height_termination=true
++simulator.config.with_multi_viewport_camera=true
++simulator.config.record0=true
+robot=g1_w_foot_sensors
```