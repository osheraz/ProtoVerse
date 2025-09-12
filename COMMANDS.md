# Train
```
HYDRA_FULL_ERROR=1 python protoverse/train_agent.py +experiment_name=g1_new_obs_terrain +exp=steering_mlp +robot=g1 +simulator=isaaclab +opt=wandb ++env.config.enable_height_termination=true ++robot.use_robot_obs=true

HYDRA_FULL_ERROR=1 python protoverse/train_agent.py +experiment_name=h1_verify +robot=h1 +exp=steering_mlp +simulator=isaaclab +opt=wandb ++env.config.enable_height_termination=true ++robot.use_robot_obs=true 

HYDRA_FULL_ERROR=1 python protoverse/train_agent.py +experiment_name=g1_test_1 +robot=g1 +exp=steering_mlp +simulator=isaaclab +opt=wandb ++env.config.enable_height_termination=true ++robot.use_robot_obs=true ++num_envs=1024

HYDRA_FULL_ERROR=1 python protoverse/train_agent.py +exp=path_follower_amp_mlp +robot=h1 motion_file=protoverse/data/motions/h1_walk.npy +simulator=isaacgym +experiment_name=h1_gym_am4 ++num_envs=100 ++headless=false

```

# Eval
```
python protoverse/eval_agent.py +robot=h1 +simulator=isaaclab +checkpoint=results/test_amp/last.ckpt +init_viser=true

python protoverse/eval_agent.py +robot=g1_w_foot_sensors +simulator=isaaclab +checkpoint=results/g1_slow/last.ckpt +init_viser=true ++env.config.enable_height_termination=true ++simulator.config.with_multi_viewport_camera=true ++robot.use_robot_obs=false 

todo check:
python protoverse/eval_agent.py +robot=h1_w_foot_sensors +simulator=isaaclab +checkpoint=results/h1_verify_2/last.ckpt +init_viser=true ++env.config.enable_height_termination=true ++simulator.config.with_multi_viewport_camera=true ++robot.use_robot_obs=true

HYDRA_FULL_ERROR=1 python protoverse/eval_agent.py +robot=g1 +simulator=isaaclab +checkpoint=results/g1_new_obs_terrain_foot_no_obs/last.ckpt +init_viser=true ++env.config.enable_height_termination=true ++simulator.config.with_multi_viewport_camera=true ++robot.use_robot_obs=true 

HYDRA_FULL_ERROR=1 python protoverse/eval_agent.py +robot=h1 +simulator=isaaclab +checkpoint=results/h1_test/last.ckpt +init_viser=true ++env.config.enable_height_termination=true ++simulator.config.with_multi_viewport_camera=true
```

# Important params 
```
+init_viser=true
++env.config.enable_height_termination=true
++simulator.config.with_multi_viewport_camera=true
++simulator.config.record0=true
+robot=g1_w_foot_sensors
env.force_respawn_on_flat=true
self_collisions
enable_stabilization
```


# defaults:
#   - /robot/g1_29dof_with_sensors

# Running


HYDRA_FULL_ERROR=1 python protove rse/train_agent.py +experiment_name=g1_23_t +robot=g1_29dof_anneal_23dof +exp=steering_mlp +simulator=isaaclab +opt=wandb ++env.config.enable_height_termination=true ++num_envs=512

HYDRA_FULL_ERROR=1 python protoverse/train_agent.py +experiment_name=h1_isaac_reward_a +robot=h1 +exp=steering_mlp +simulator=isaaclab +opt=wandb ++env.config.enable_height_termination=true 

HYDRA_FULL_ERROR=1 python protoverse/train_agent.py +robot=h1 +exp=steering_mlp +simulator=isaacgym ++env.config.enable_height_termination=true +opt=wandb +experiment_name=g23_off_gym


HYDRA_FULL_ERROR=1 python protoverse/train_agent.py +experiment_name=vsiew +robot=h1 +exp=steering_mlp +simulator=isaacgym ++env.config.enable_height_termination=true ++headless=false ++num_envs=100

HYDRA_FULL_ERROR=1 python protoverse/train_agent.py +experiment_name=vsi323ew +robot=g1_29dof_anneal_23dof +exp=steering_mlp +simulator=isaaclab ++env.config.enable_height_termination=true ++headless=false ++num_envs=100get_env_origin