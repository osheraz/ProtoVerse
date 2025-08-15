"""
IsaacLab and app launcher must be setup before all other imports.
"""

from isaaclab.app import AppLauncher

headless = True
app_launcher = AppLauncher({"headless": headless})
simulation_app = app_launcher.app

import torch
import time
from protoverse.simulator.isaaclab.config import (
    IsaacLabSimulatorConfig,
    IsaacLabSimParams,
)
from protoverse.simulator.isaaclab.simulator import IsaacLabSimulator
from protoverse.simulator.base_simulator.config import (
    RobotConfig,
    RobotAssetConfig,
    InitState,
    ControlConfig,
    ControlType,
    CameraConfig,
)
from protoverse.envs.base_env.env_utils.terrains.flat_terrain import Terrain
from protoverse.envs.base_env.env_utils.terrains.terrain_config import TerrainConfig
from protoverse.utils.scene_lib import (
    Scene,
    SceneObject,
    ObjectOptions,
    SceneLib,
)

with_foot_sensor = True
with_multi_viewport_camera = False
with_cam_obs = False

# Create robot asset configuration
robot_asset_config = RobotAssetConfig(
    robot_type="g1",
    # asset_file_name="urdf/g1_29dof_with_sensors.urdf",
    # usd_asset_file_name="usd/g1_29dof_with_sensors.usd",
    asset_file_name="urdf/g1.urdf",
    usd_asset_file_name="usd/g1.usd",
    self_collisions=False,
    collapse_fixed_joints=True,
)


# Create robot configuration
robot_config = RobotConfig(
    body_names=[
        "pelvis",
        "head",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "waist_yaw_link",
        "waist_roll_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
        "left_rubber_hand",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
        "right_rubber_hand",
    ],
    dof_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    dof_body_ids=list(range(1, 30)),
    joint_axis=[
        "y",
        "x",
        "z",
        "y",
        "y",
        "x",
        "y",
        "x",
        "z",
        "y",
        "y",
        "x",
        "z",
        "x",
        "y",
        "y",
        "x",
        "z",
        "y",
        "x",
        "y",
        "z",
        "y",
        "x",
        "z",
        "y",
        "x",
        "y",
        "z",
    ],
    dof_obs_size=174,  # 29 DOFs * 6
    number_of_actions=29,
    with_cam_obs=with_cam_obs,
    with_foot_sensors=with_foot_sensor,
    self_obs_max_coords_size=493,
    left_foot_name="left_ankle_pitch_link",
    right_foot_name="right_ankle_pitch_link",
    head_body_name="head",
    key_bodies=[
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_ankle_pitch_link",
        "right_ankle_pitch_link",
    ],
    foot_contact_links=[
        f"{side}_ankle_roll_link_sensor_{i}"
        for side in ["left", "right"]
        for i in range(49)  # modify to be dynamic
    ],
    non_termination_contact_bodies=[
        "left_wrist_yaw_link",
        "left_wrist_pitch_link",
        "left_wrist_roll_link",
        "right_wrist_yaw_link",
        "right_wrist_pitch_link",
        "right_wrist_roll_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
    ],
    dof_effort_limits=[
        88.0,
        88.0,
        88.0,
        139.0,
        50.0,
        50.0,
        88.0,
        88.0,
        88.0,
        139.0,
        50.0,
        50.0,
        88.0,
        50.0,
        50.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
        5.0,
        5.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
        5.0,
        5.0,
    ],
    dof_vel_limits=[
        32.0,
        32.0,
        32.0,
        20.0,
        37.0,
        37.0,
        32.0,
        32.0,
        32.0,
        20.0,
        37.0,
        37.0,
        32.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        22.0,
        22.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        22.0,
        22.0,
    ],
    dof_armatures=[0.03] * 29,
    dof_joint_frictions=[0.03] * 29,
    asset=robot_asset_config,
    init_state=InitState(
        pos=[0.0, 0.0, 0.8],
        default_joint_angles={
            "left_hip_pitch_joint": -0.1,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.1,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
    ),
    control=ControlConfig(
        control_type=ControlType.PROPORTIONAL,
        action_scale=1.0,
        clamp_actions=100.0,
        stiffness={
            "hip_yaw": 100,
            "hip_roll": 100,
            "hip_pitch": 100,
            "knee": 200,
            "ankle_pitch": 20,
            "ankle_roll": 20,
            "waist_yaw": 400,
            "waist_roll": 400,
            "waist_pitch": 400,
            "shoulder_pitch": 90,
            "shoulder_roll": 60,
            "shoulder_yaw": 20,
            "elbow": 60,
            "wrist_roll": 4.0,
            "wrist_pitch": 4.0,
            "wrist_yaw": 4.0,
        },
        damping={
            "hip_yaw": 2.5,
            "hip_roll": 2.5,
            "hip_pitch": 2.5,
            "knee": 5.0,
            "ankle_pitch": 0.2,
            "ankle_roll": 0.1,
            "waist_yaw": 5.0,
            "waist_roll": 5.0,
            "waist_pitch": 5.0,
            "shoulder_pitch": 2.0,
            "shoulder_roll": 1.0,
            "shoulder_yaw": 0.4,
            "elbow": 1.0,
            "wrist_roll": 0.2,
            "wrist_pitch": 0.2,
            "wrist_yaw": 0.2,
        },
    ),
)


# Create simulator configuration
simulator_config = IsaacLabSimulatorConfig(
    sim=IsaacLabSimParams(
        fps=200,
        decimation=4,
    ),
    headless=headless,  # Set to True for headless mode
    robot=robot_config,
    num_envs=4,  # Number of parallel environments
    experiment_name="scene_isaaclab_example",
    w_last=False,  # IsaacLab uses wxyz quaternions
    init_viser=True,
    with_cam_obs=with_cam_obs,
    with_multi_viewport_camera=with_multi_viewport_camera,
    camera=CameraConfig(),
)

device = torch.device("cuda")

# Create a flat terrain using the default config
terrain_config = TerrainConfig(
    num_terrains=7,
    num_levels=1,
    terrain_proportions=[0.2, 0.1, 0.1, 0.1, 0.05, 0.0, 0.0, 0.45],
    minimal_humanoid_spacing=0,  # We defined the terrain size, so no need for additional humanoid spacing
)
terrain = Terrain(
    config=terrain_config, num_envs=simulator_config.num_envs, device=device
)

# Create and initialize the simulator
simulator = IsaacLabSimulator(
    config=simulator_config,
    terrain=terrain,
    scene_lib=None,
    visualization_markers=None,
    device=device,
    simulation_app=simulation_app,
)
simulator.on_environment_ready()

# Get robot default state
default_state = simulator.get_default_state()
# Set the robot to a new random position above the ground
root_pos = torch.zeros(simulator_config.num_envs, 3, device=device)
xy_pos = terrain.sample_valid_locations(simulator_config.num_envs)
height = terrain.get_ground_heights(xy_pos).view(-1)
root_pos[:, :2] = xy_pos
root_pos[:, 2] = (
    height + 1.1
)  # Height determines the height of the terrain, add offset to properly spawn above ground without collisions
default_state.root_pos[:] = root_pos

# Reset the robots
simulator.reset_envs(
    default_state, env_ids=torch.arange(simulator_config.num_envs, device=device)
)

# Run the simulation loop
# try:
while True:
    actions = torch.randn(
        simulator_config.num_envs,
        simulator_config.robot.number_of_actions,
        device=device,
    )
    simulator.step(actions)
# except KeyboardInterrupt:
#     print("\nSimulation stopped by user")
# finally:
#     simulator.close()
