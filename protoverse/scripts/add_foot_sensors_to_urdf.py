import trimesh
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import os


def sample_foot_surface(
    mesh_path,
    n_points=20,
    z_threshold=0.00,
    sampling_method="random",
):
    mesh = trimesh.load_mesh(mesh_path)

    samples, _ = trimesh.sample.sample_surface(mesh, n_points * 5)
    z_min = samples[:, 2].min()
    bottom_samples = samples[samples[:, 2] < z_min + z_threshold]

    if sampling_method == "random":
        return random_sample(bottom_samples, n_points)
    elif sampling_method == "uniform":
        return farthest_point_sample(bottom_samples, n_points)
    elif sampling_method == "structured":
        return structured_grid_sample(
            mesh,
            grid_size=(int(n_points**0.5), int(n_points**0.5)),
            z_offset=z_threshold,
        )
    else:
        raise ValueError(f"Unknown sampling_method: {sampling_method}")


def random_sample(points, n_samples):
    if len(points) > n_samples:
        points = points[np.random.choice(len(points), n_samples, replace=False)]
    return points


def farthest_point_sample(points, n_samples):
    selected = [points[np.random.choice(len(points))]]
    for _ in range(n_samples - 1):
        dists = np.linalg.norm(
            points[:, None, :] - np.array(selected)[None, :, :], axis=2
        )
        min_dists = np.min(dists, axis=1)
        next_idx = np.argmax(min_dists)
        selected.append(points[next_idx])
    return np.array(selected)


def structured_grid_sample(mesh, grid_size=(10, 10), z_offset=0.00):
    bounds = mesh.bounds
    x_min, y_min = bounds[0][0], bounds[0][1]
    x_max, y_max = bounds[1][0], bounds[1][1]

    x_lin = np.linspace(x_min, x_max, grid_size[0])
    y_lin = np.linspace(y_min, y_max, grid_size[1])
    xx, yy = np.meshgrid(x_lin, y_lin)
    grid_points = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)

    # Correct input shape (n, 3)
    closest_points, _, _ = trimesh.proximity.closest_point(mesh, grid_points)

    # Flatten all to the same Z-plane
    z_min = mesh.vertices[:, 2].min()
    closest_points[:, 2] = z_min - z_offset

    return closest_points


def add_sensors_to_urdf(
    urdf_path,
    output_path,
    parent_links_and_points,
    sensor_radius=0.005,
    sensor_shape="sphere",  # "sphere" or "box"
):
    tree = ET.parse(urdf_path.as_posix())
    root = tree.getroot()

    for parent_link, sensor_positions in parent_links_and_points.items():
        for i, pos in enumerate(sensor_positions):
            x, y, z = pos
            z -= 0.01
            sensor_name = f"{parent_link}_sensor_{i}"

            link_elem = ET.Element("link", name=sensor_name)
            inertial = ET.SubElement(link_elem, "inertial")
            ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(inertial, "mass", value="0.001")
            ET.SubElement(
                inertial,
                "inertia",
                ixx="1e-7",
                ixy="0",
                ixz="0",
                iyy="1e-7",
                iyz="0",
                izz="1e-7",
            )

            # Visual
            visual = ET.SubElement(link_elem, "visual")
            ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
            geom = ET.SubElement(visual, "geometry")
            if sensor_shape == "sphere":
                ET.SubElement(geom, "sphere", radius=str(sensor_radius))
            elif sensor_shape == "box":
                box_size = [
                    sensor_radius * 2,
                    sensor_radius * 2,
                    0.004,
                ]
                ET.SubElement(geom, "box", size=" ".join(map(str, box_size)))
            else:
                raise ValueError("sensor_shape must be 'sphere' or 'box'")

            material = ET.SubElement(visual, "material", name="blue")
            ET.SubElement(material, "color", rgba="0 0 1 1")

            # Collision
            collision = ET.SubElement(link_elem, "collision")
            ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
            geom_col = ET.SubElement(collision, "geometry")
            if sensor_shape == "sphere":
                ET.SubElement(geom_col, "sphere", radius=str(sensor_radius))
            elif sensor_shape == "box":
                ET.SubElement(geom_col, "box", size=" ".join(map(str, box_size)))

            root.append(link_elem)

            # Joint
            joint_elem = ET.Element("joint", name=f"{sensor_name}_joint", type="fixed")
            ET.SubElement(joint_elem, "parent", link=parent_link)
            ET.SubElement(joint_elem, "child", link=sensor_name)
            ET.SubElement(joint_elem, "origin", xyz=f"{x} {y} {z}", rpy="0 0 0")
            root.append(joint_elem)

    ET.indent(tree, space="  ")
    tree.write(output_path.as_posix(), xml_declaration=True, encoding="utf-8")


# Example Usage
if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(dir_path, "../data")

    # Paths
    left_stl_path = Path(f"{data_dir}/assets/mesh/G1/left_ankle_roll_link.stl")
    right_stl_path = Path(f"{data_dir}/assets/mesh/G1/right_ankle_roll_link.stl")

    urdf_path = Path(f"{data_dir}/assets/urdf/g1.urdf")

    sampling_method = "structured"  # "random", "uniform", or "structured"

    left_samples = sample_foot_surface(
        left_stl_path, n_points=50, z_threshold=-0.01, sampling_method=sampling_method
    )
    right_samples = sample_foot_surface(
        right_stl_path, n_points=50, z_threshold=-0.01, sampling_method=sampling_method
    )

    sensor_shape = "box"  # or "sphere"

    add_sensors_to_urdf(
        urdf_path=urdf_path,
        output_path=Path(f"{data_dir}/assets/urdf/g1_29dof_with_sensors.urdf"),
        parent_links_and_points={
            "left_ankle_roll_link": left_samples,
            "right_ankle_roll_link": right_samples,
        },
        sensor_radius=0.005,
        sensor_shape=sensor_shape,
    )
    print(f"Modified URDF saved with sampling_method = {sampling_method}")
