from __future__ import annotations

import time
from typing import Literal

import numpy as np
import tyro

import viser
from viser.extras import ViserUrdf
import os
from pathlib import Path
from protoverse.utils.urdf_loader import load_urdf
import pyroki as pk
from pyroki.pyroki_snippets.solve_ik_with_multiple_targets import (
    solve_ik_with_multiple_targets,
    solve_ik,
)

import yourdfpy


# import pyroki_snippets as pks
"""
Solves the basic IK problem.
"""
init_joint_pos = {
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
}

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "../data")

# ---- choose one of: "g1_29", "g1_29_with_sensors", "g1_23"
VARIANT = "h1"

ROBOTS = {
    "g1_29": {
        "urdf_rel": "assets/urdf/g1.urdf",
        "hands": ("left_rubber_hand", "right_rubber_hand"),
        "feet": ("left_ankle_pitch_link", "right_ankle_pitch_link"),
    },
    "g1_29_with_sensors": {
        "urdf_rel": "assets/urdf/g1_29dof_with_sensors.urdf",
        "hands": ("left_rubber_hand", "right_rubber_hand"),
        "feet": ("left_ankle_roll_link", "right_ankle_roll_link"),
    },
    "g1_23": {
        "urdf_rel": "assets/urdf/g1_29dof_anneal_23dof.urdf",
        "hands": ("left_elbow_link", "right_elbow_link"),
        "feet": ("left_ankle_roll_link", "right_ankle_roll_link"),
    },
    "g1_23_with_sensors": {
        "urdf_rel": "assets/urdf/g1_29dof_anneal_23dof_29dof_with_sensors.urdf",
        "hands": ("left_elbow_link", "right_elbow_link"),  #  23-DoF has no hand links
        "feet": ("left_ankle_roll_link", "right_ankle_roll_link"),
    },
    "h1": {
        "urdf_rel": "assets/urdf/h1.urdf",
        "hands": ("left_elbow_link", "right_elbow_link"),
        "feet": ("left_foot_link", "right_foot_link"),
    },
    "h1_29dof_with_sensors": {
        "urdf_rel": "assets/urdf/h1_29dof_with_sensors.urdf",
        "hands": ("left_elbow_link", "right_elbow_link"),
        "feet": ("left_foot_link", "right_foot_link"),
    },
    "rel3_4": {
        "urdf_rel": "assets/urdf/rel3_4.urdf",
        "hands": ("left_hand_flange_link", "right_hand_flange_link"),
        "feet": ("left_foot_ee_link", "right_foot_ee_link"),
    },
}

cfg = ROBOTS[VARIANT]
LEFT_HAND, RIGHT_HAND = cfg["hands"]
LEFT_FOOT, RIGHT_FOOT = cfg["feet"]

urdf_path = {"robot": Path(os.path.join(data_dir, cfg["urdf_rel"]))}


from yourdfpy import URDF, Robot


def extract_sub_urdf(urdf: URDF, base_link: str, end_link: str) -> URDF:
    """
    Extracts a sub-URDF from base_link to end_link (inclusive), ensuring all joints
    and links along the kinematic chain are included.
    """
    # Follow joints upward from end_link to base_link
    joints_in_chain = []
    links_in_chain = set()

    current_link = end_link
    while current_link != base_link:
        # Find joint whose child is the current link
        joint = next((j for j in urdf.robot.joints if j.child == current_link), None)
        if joint is None:
            raise ValueError(
                f"No joint found leading to {current_link} from base {base_link}"
            )
        joints_in_chain.append(joint)
        links_in_chain.add(joint.child)
        current_link = joint.parent
        links_in_chain.add(current_link)

    sub_robot = Robot(
        name=f"{urdf.robot.name}_{end_link}",
        joints=list(reversed(joints_in_chain)),
        links=[l for l in urdf.robot.links if l.name in links_in_chain],
    )
    return URDF(robot=sub_robot)


def create_robot_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []
    for joint_name, (
        lower,
        upper,
    ) in viser_urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_pos,
        )
        slider.on_update(  # When sliders move, we update the URDF configuration.
            lambda _: viser_urdf.update_cfg(
                np.array([slider.value for slider in slider_handles])
            )
        )
        slider_handles.append(slider)
        initial_config.append(initial_pos)
    return slider_handles, initial_config


def main(
    load_meshes: bool = True,
    load_collision_meshes: bool = False,
    split_robots: bool = True,
) -> None:
    # Start viser server.
    server = viser.ViserServer()

    # Load URDF.

    urdf = load_urdf(None, urdf_path["robot"])

    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf,
        load_meshes=load_meshes,
        load_collision_meshes=load_collision_meshes,
        collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
    )

    # Joint sliders.
    with server.gui.add_folder("Joint position control"):
        (slider_handles, initial_config) = create_robot_control_sliders(
            server, viser_urdf
        )

    # Visibility checkboxes.
    with server.gui.add_folder("Visibility"):
        show_meshes_cb = server.gui.add_checkbox(
            "Show meshes",
            viser_urdf.show_visual,
        )
        show_collision_meshes_cb = server.gui.add_checkbox(
            "Show collision meshes", viser_urdf.show_collision
        )

    @show_meshes_cb.on_update
    def _(_):
        viser_urdf.show_visual = show_meshes_cb.value

    @show_collision_meshes_cb.on_update
    def _(_):
        viser_urdf.show_collision = show_collision_meshes_cb.value

    show_meshes_cb.visible = load_meshes
    show_collision_meshes_cb.visible = load_collision_meshes

    # Set initial config.
    viser_urdf.update_cfg(np.array(initial_config))
    robot = pk.Robot.from_urdf(urdf)

    if not split_robots:
        robot = pk.Robot.from_urdf(urdf)
    else:
        target_link_names = [RIGHT_HAND, LEFT_HAND, RIGHT_FOOT, LEFT_FOOT]

        sub_robots = {
            link_name: pk.Robot.from_urdf(extract_sub_urdf(urdf, "pelvis", link_name))
            for link_name in target_link_names
        }

        for link_name, sub_robot in sub_robots.items():
            print(f"{link_name}: {sub_robot.joints.actuated_names}")
    # Grid.
    trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene
    server.scene.add_grid(
        "/grid",
        width=2,
        height=2,
        position=(0.0, 0.0, 0.0),
    )

    # Reset button.
    reset_button = server.gui.add_button("Reset")

    @reset_button.on_click
    def _(_):
        for s, init_q in zip(slider_handles, initial_config):
            s.value = init_q

    # IK targets (4 links).
    ik_targets = {
        RIGHT_HAND: server.scene.add_transform_controls(
            "/ik_target_right_hand",
            scale=0.2,
            position=(0.3, -0.3, 0.1),
            wxyz=(1, 0, 0, 0),
        ),
        LEFT_HAND: server.scene.add_transform_controls(
            "/ik_target_left_hand",
            scale=0.2,
            position=(0.3, 0.3, 0.1),
            wxyz=(1, 0, 0, 0),
        ),
        RIGHT_FOOT: server.scene.add_transform_controls(
            "/ik_target_right_foot",
            scale=0.2,
            position=(0.0, -0.2, -0.8),
            wxyz=(1, 0, 0, 0),
        ),
        LEFT_FOOT: server.scene.add_transform_controls(
            "/ik_target_left_foot",
            scale=0.2,
            position=(0.0, 0.2, -0.8),
            wxyz=(1, 0, 0, 0),
        ),
    }

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    viser_urdf.update_cfg(
        {j: init_joint_pos.get(j, 0.0) for j in robot.joints.actuated_names}
    )

    # Loop.
    while True:
        start_time = time.time()

        if not split_robots:
            # Full-body IK for all 4 targets
            solution = solve_ik_with_multiple_targets(
                robot=robot,
                target_link_names=target_link_names,
                target_positions=np.array([t.position for t in ik_targets.values()]),
                target_wxyzs=np.array([t.wxyz for t in ik_targets.values()]),
            )
            cfg_array = solution  # Already a np.ndarray

        else:
            # Start from initial joint config
            cfg_array = np.array(
                [init_joint_pos.get(j, 0.0) for j in robot.joints.actuated_names]
            )

            # Solve IK for hands (together)
            # hand_targets = ["left_rubber_hand", "right_rubber_hand"]
            hand_targets = [LEFT_HAND, RIGHT_HAND]
            foot_targets = [LEFT_FOOT, RIGHT_FOOT]
            hand_solution = solve_ik_with_multiple_targets(
                robot=robot,
                target_link_names=hand_targets,
                target_positions=np.array(
                    [ik_targets[n].position for n in hand_targets]
                ),
                target_wxyzs=np.array([ik_targets[n].wxyz for n in hand_targets]),
            )
            for joint_name, value in zip(robot.joints.actuated_names, hand_solution):
                idx = robot.joints.actuated_names.index(joint_name)
                cfg_array[idx] = value

            # Solve IK for each foot (separately)
            # foot_targets = ["left_ankle_roll_link", "right_ankle_roll_link"]
            for link_name in foot_targets:
                sub_robot = sub_robots[link_name]
                target = ik_targets[link_name]

                partial_solution = solve_ik(
                    robot=sub_robot,
                    target_link_name=link_name,
                    target_position=np.array(target.position),
                    target_wxyz=np.array(target.wxyz),
                )

                for joint_name, value in zip(
                    sub_robot.joints.actuated_names, partial_solution
                ):
                    if joint_name in robot.joints.actuated_names:
                        idx = robot.joints.actuated_names.index(joint_name)
                        cfg_array[idx] = value

        # Render solution
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        viser_urdf.update_cfg(cfg_array)


if __name__ == "__main__":

    tyro.cli(main)
