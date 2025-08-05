"""IK with Mimic Joints

This is a simple test to ensure that mimic joints are handled correctly in the IK solver.

We procedurally generate a "zig-zag" chain of links with mimic joints, where:
- the first joint is driven directly,
- and the remaining joints are driven indirectly via mimic joints.
The multipliers alternate between -1 and 1, and the offsets are all 0.
"""

import tempfile
import time

import numpy as np
import pyroki as pk
import viser
import yourdfpy
from viser.extras import ViserUrdf

import pyroki_snippets as pks


def create_chain_xml(length: float = 0.2, num_chains: int = 5) -> str:
    def create_link(idx):
        return f"""
                <link name="link_{idx}">
                    <visual>
                    <geometry>
                        <cylinder length="{length}" radius="0.02"/>
                    </geometry>
                    </visual>
                    <inertial>
                    <mass value="0.1"/>
                    <inertia ixx="0.0001" iyy="0.0001" izz="0.0001"/>
                    </inertial>
                </link>
            """

    def create_joint(idx, multiplier=1.0, offset=0.0):
        mimic = f'<mimic joint="joint_0" multiplier="{multiplier}" offset="{offset}"/>'
        return f"""
            <joint name="joint_{idx}" type="revolute">
                <parent link="link_{idx}"/>
                <child link="link_{idx + 1}"/>
                <axis xyz="1 0 0"/>
                <origin xyz="0 0 {length}" rpy="0 0 0"/>
                {mimic if idx != 0 else ""}
                <limit lower="-10.0" upper="10.0" velocity="1.0"/>
            </joint>
            """

    world_joint_origin_z = length / 2.0
    xml = f"""
    <?xml version="1.0"?>
    <robot name="chain_with_mimic_joints">
    <link name="world"></link>
    <joint name="world_joint" type="revolute">
        <parent link="world"/>
        <child link="link_0"/>
        <axis xyz="1 0 0"/>
        <origin xyz="0 0 {world_joint_origin_z}" rpy="0 0 0"/>
        <limit lower="-10.0" upper="10.0" velocity="1.0"/>
    </joint>
    """
    # Create the definition + first link.
    xml += create_link(0)
    xml += create_link(1)
    xml += create_joint(0)

    # Procedurally add more links.
    assert num_chains >= 2
    for idx in range(2, num_chains):
        xml += create_link(idx)
        current_offset = 0.0
        current_multiplier = 1.0 * ((-1) ** (idx % 2))
        xml += create_joint(idx - 1, current_multiplier, current_offset)

    xml += """
    </robot>
    """
    return xml


def main():
    """Main function for basic IK."""

    xml = create_chain_xml(num_chains=10, length=0.1)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf") as f:
        f.write(xml)
        f.flush()
        urdf = yourdfpy.URDF.load(f.name)

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    target_link_name_handle = server.gui.add_dropdown(
        "Target Link",
        robot.links.names,
        initial_value=robot.links.names[-1],
    )

    # Create interactive controller with initial position.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.0, 0.1, 0.1), wxyz=(0, 0, 1, 0)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    while True:
        # Solve IK.
        start_time = time.time()
        solution = pks.solve_ik(
            robot=robot,
            target_link_name=target_link_name_handle.value,
            target_position=np.array(ik_target.position),
            target_wxyz=np.array(ik_target.wxyz),
        )

        # Update timing handle.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Update visualizer.
        urdf_vis.update_cfg(solution)


if __name__ == "__main__":
    main()
