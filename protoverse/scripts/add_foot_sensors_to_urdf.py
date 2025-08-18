import trimesh
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import os


def structured_grid_sample_old(mesh, n_points=50, z_offset=0.00, bottom_tol=1e-3):

    bounds = mesh.bounds
    (x_min, y_min, _), (x_max, y_max, _) = bounds

    # Estimate spacing from XY area so we seed enough points.
    area_xy = max((x_max - x_min) * (y_max - y_min), 1e-9)

    spacing = (area_xy / n_points) ** 0.5
    dx = spacing
    dy = spacing

    # Build a hex grid that oversamples (factor ~3) so FPS can pick well-spaced points.
    xs = []
    ys = []
    ny = int(np.ceil((y_max - y_min) / dy)) + 2
    nx = int(np.ceil((x_max - x_min) / dx)) + 2
    for j in range(ny):
        y = y_min - dy + j * dy
        shift = (dx / 2) if (j % 2) else 0.0
        for i in range(nx):
            x = x_min - dx + shift + i * dx
            if (x_min - dx) <= x <= (x_max + dx) and (y_min - dy) <= y <= (y_max + dy):
                xs.append(x)
                ys.append(y)
    seeds = np.column_stack([np.array(xs), np.array(ys), np.zeros(len(xs))])

    # Project to mesh
    closest_points, _, _ = trimesh.proximity.closest_point(mesh, seeds)

    # Keep only seeds that land near the bottom surface (avoid sides)
    z_min = mesh.vertices[:, 2].min()
    keep = closest_points[:, 2] <= z_min + bottom_tol
    pts = closest_points[keep]

    # If we lost too many (e.g., very narrow footprint), fall back to all
    if len(pts) == 0:
        pts = closest_points

    # Farthest Point Sampling to enforce spacing and exact count
    def fps(points, k):
        points = np.asarray(points)
        n = len(points)
        if n == 0:
            return points
        k = min(k, n)
        chosen_idx = [np.random.randint(0, n)]
        dists = np.full(n, np.inf)
        for _ in range(k - 1):
            # update distances to nearest chosen
            diff = points - points[chosen_idx[-1]]
            dists = np.minimum(dists, np.einsum("ij,ij->i", diff, diff))
            next_idx = int(np.argmax(dists))
            chosen_idx.append(next_idx)
        return points[chosen_idx]

    pts_xy = pts[:, :2]
    chosen = fps(pts_xy, n_points)

    # Recover the corresponding 3D (xy from fps + flatten z to a plane just below sole)
    out = np.column_stack(
        [chosen[:, 0], chosen[:, 1], np.full(len(chosen), z_min - z_offset)]
    )
    return out


def structured_grid_sample(mesh, n_points=50, z_offset=0.00, bottom_tol=1e-3):
    """
    Ordered XY-aligned grid clipped to the sole footprint (z <= z_min + bottom_tol),
    with a quota-based row picker so you always get a regular-looking pattern:
    - distribute exactly n_points across rows proportionally to available candidates
    - within each row, pick evenly spaced columns
    - output is row-major (Y then X), Z flattened to sole plane
    """
    import numpy as np

    V = mesh.vertices
    F = mesh.faces
    if V.size == 0 or F.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # ---- Sole slab (by Z) ----
    z_min = float(V[:, 2].min())
    vert_sole = V[:, 2] <= (z_min + bottom_tol)

    # Faces fully on the sole slab (all three verts inside)
    face_sole_mask = vert_sole[F].all(axis=1)
    sole_faces = F[face_sole_mask]
    if sole_faces.shape[0] == 0:
        # Relax: any-vertex-in-slab
        face_sole_mask = vert_sole[F].any(axis=1)
        sole_faces = F[face_sole_mask]
        if sole_faces.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)

    # ---- 2D footprint bounds from sole faces ----
    sole_vert_ids = np.unique(sole_faces.ravel())
    sole_xy = V[sole_vert_ids][:, :2]
    xmin, ymin = sole_xy.min(axis=0)
    xmax, ymax = sole_xy.max(axis=0)

    # ---- Choose base grid shape (nx, ny) to match aspect & oversample ----
    xspan = max(xmax - xmin, 1e-12)
    yspan = max(ymax - ymin, 1e-12)
    aspect = xspan / yspan
    nx_base = max(1, int(round((n_points * aspect) ** 0.5)))
    ny_base = max(1, int(np.ceil(n_points / nx_base)))

    oversample = 1.6
    nx = max(nx_base, int(np.ceil(nx_base * oversample)))
    ny = max(ny_base, int(np.ceil(ny_base * oversample)))

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)  # (ny, nx), row-major by Y then X
    P = np.column_stack([XX.ravel(), YY.ravel()])  # (M, 2) row-major

    # ---- Build triangle list in 2D (XY), point-in-triangle to clip to footprint ----
    tri_xy = V[sole_faces][:, :, :2]  # (T, 3, 2)
    M = P.shape[0]
    inside = np.zeros(M, dtype=bool)

    # vectorized-ish barycentric per triangle with bbox prefilter
    for tri in tri_xy:
        A, B, C = tri[0], tri[1], tri[2]
        tmin = np.minimum(np.minimum(A, B), C)
        tmax = np.maximum(np.maximum(A, B), C)
        in_bb = (
            (P[:, 0] >= tmin[0])
            & (P[:, 0] <= tmax[0])
            & (P[:, 1] >= tmin[1])
            & (P[:, 1] <= tmax[1])
        )
        idx = np.nonzero(~inside & in_bb)[0]
        if idx.size == 0:
            continue
        Q = P[idx] - A
        v0 = C - A
        v1 = B - A
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot11 = np.dot(v1, v1)
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-18:
            continue
        dot0Q = Q @ v0
        dot1Q = Q @ v1
        u = (dot11 * dot0Q - dot01 * dot1Q) / denom
        v = (dot00 * dot1Q - dot01 * dot0Q) / denom
        in_tri = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        inside[idx[in_tri]] = True

    if not inside.any():
        return np.zeros((0, 3), dtype=np.float32)

    # ---- Row-wise quotas: distribute exactly n_points across rows ----
    inside_idx = np.nonzero(inside)[0]
    # row index for each grid point in row-major layout:
    row_idx = np.arange(M) // nx
    rows = np.arange(ny)

    # count how many candidates per row are inside
    cand_per_row = np.bincount(row_idx[inside], minlength=ny).astype(int)
    total_cand = int(cand_per_row.sum())
    if total_cand == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # initial quota proportional to availability
    quotas = np.floor(n_points * (cand_per_row / total_cand)).astype(int)
    # ensure at least 1 where there are candidates
    quotas = np.where((cand_per_row > 0) & (quotas == 0), 1, quotas)

    # cap by availability
    quotas = np.minimum(quotas, cand_per_row)

    # adjust to hit exact n_points
    diff = n_points - int(quotas.sum())
    if diff > 0:
        # add one at a time to rows with largest remaining capacity
        capacity = cand_per_row - quotas
        for _ in range(diff):
            j = int(np.argmax(capacity))
            if capacity[j] <= 0:
                break
            quotas[j] += 1
            capacity[j] -= 1
    elif diff < 0:
        # remove one at a time from rows with the smallest fractional need
        for _ in range(-diff):
            # remove from the row with smallest (quota>0) and smallest capacity loss
            candidates = np.where(quotas > 0)[0]
            if candidates.size == 0:
                break
            # heuristic: remove from row having the smallest (cand_per_row) first
            j = int(candidates[np.argmin(cand_per_row[candidates])])
            quotas[j] -= 1

    # ---- Within each row, pick evenly spaced columns among inside candidates ----
    picked = []
    for j in rows:
        q = int(quotas[j])
        if q <= 0:
            continue
        # indices of row j in the flattened grid
        start = j * nx
        end = start + nx
        row_inside = np.nonzero(inside[start:end])[0]  # local column indices
        if row_inside.size == 0:
            continue
        if row_inside.size <= q:
            cols_pick = row_inside  # take all
        else:
            # pick q evenly spaced indices across available columns
            # use linspace over [0, n-1]
            idxs = np.linspace(0, row_inside.size - 1, q)
            cols_pick = row_inside[np.round(idxs).astype(int)]
        picked.extend(start + cols_pick)

    if len(picked) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # keep row-major order naturally by sorted indices
    picked = np.array(sorted(picked), dtype=int)
    if picked.size > n_points:
        picked = picked[:n_points]

    P_in = P[picked]

    # ---- Final Z (flatten to sole plane with optional offset) ----
    Z = np.full((P_in.shape[0],), z_min - z_offset, dtype=np.float32)
    out = np.column_stack([P_in[:, 0], P_in[:, 1], Z]).astype(np.float32)
    return out


def sample_foot_surface(
    mesh_path,
    n_points=20,
    z_threshold=0.002,
    sampling_method="random",
):
    mesh = trimesh.load_mesh(mesh_path)

    samples, _ = trimesh.sample.sample_surface(mesh, n_points * 100)
    z_min = samples[:, 2].min()
    bottom_samples = samples[samples[:, 2] < z_min + z_threshold]

    if sampling_method == "random":
        return random_sample(bottom_samples, n_points)
    elif sampling_method == "uniform":
        return farthest_point_sample(bottom_samples, n_points)
    elif sampling_method == "structured":
        return structured_grid_sample(mesh, n_points=n_points, z_offset=0.001)
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


# def structured_grid_sample(mesh, grid_size=(10, 10), z_offset=0.00):
#     bounds = mesh.bounds
#     x_min, y_min = bounds[0][0], bounds[0][1]
#     x_max, y_max = bounds[1][0], bounds[1][1]

#     x_lin = np.linspace(x_min, x_max, grid_size[0])
#     y_lin = np.linspace(y_min, y_max, grid_size[1])
#     xx, yy = np.meshgrid(x_lin, y_lin)
#     grid_points = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)

#     # Correct input shape (n, 3)
#     closest_points, _, _ = trimesh.proximity.closest_point(mesh, grid_points)

#     # Flatten all to the same Z-plane
#     z_min = mesh.vertices[:, 2].min()
#     closest_points[:, 2] = z_min - z_offset

#     return closest_points


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

            sensor_name = f"{parent_link}_sensor_{i}"

            link_elem = ET.Element("link", name=sensor_name)
            inertial = ET.SubElement(link_elem, "inertial")
            ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
            # ET.SubElement(inertial, "mass", value="0.001")
            # ET.SubElement(
            #     inertial,
            #     "inertia",
            #     ixx="1e-7",
            #     ixy="0",
            #     ixz="0",
            #     iyy="1e-7",
            #     iyz="0",
            #     izz="1e-7",
            # )

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
                    0.002,
                ]
                ET.SubElement(geom, "box", size=" ".join(map(str, box_size)))
            else:
                raise ValueError("sensor_shape must be 'sphere' or 'box'")

            material = ET.SubElement(visual, "material", name="white")
            ET.SubElement(material, "color", rgba="1 1 1 0.5")

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
        left_stl_path, n_points=21, sampling_method=sampling_method
    )
    right_samples = sample_foot_surface(
        right_stl_path, n_points=21, sampling_method=sampling_method
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
