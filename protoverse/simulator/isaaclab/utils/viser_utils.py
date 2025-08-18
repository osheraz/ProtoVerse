import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "../../../data/assets/")


def simple_colormap(vals, vmin=None, vmax=None):
    """Blue (low) -> Red (high), uint8 RGB."""
    vals = np.asarray(vals, dtype=np.float32)
    if vmin is None:
        vmin = float(np.min(vals)) if vals.size else 0.0
    if vmax is None:
        vmax = float(np.max(vals)) if vals.size else 1.0
    vmax = max(vmax, vmin + 1e-6)
    t = np.clip((vals - vmin) / (vmax - vmin), 0.0, 1.0)
    r = (255 * t).astype(np.uint8)
    g = np.zeros_like(r, dtype=np.uint8)
    b = (255 * (1.0 - t)).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def xy_to_img(sensor_xy, bb_min, bb_max, H, W, margin=8):
    """Map XY coords to image pixels with fixed margins, y-up -> row-down."""
    sx = (W - 2 * margin) / max(1e-8, (bb_max[0] - bb_min[0]))
    sy = (H - 2 * margin) / max(1e-8, (bb_max[1] - bb_min[1]))
    s = min(sx, sy)  # preserve aspect
    # center the footprint
    cx = (bb_min[0] + bb_max[0]) * 0.5
    cy = (bb_min[1] + bb_max[1]) * 0.5
    px = (sensor_xy[:, 0] - cx) * s + W * 0.5
    py = (sensor_xy[:, 1] - cy) * s + H * 0.5
    # image coords
    cols = np.clip(np.round(px).astype(int), 0, W - 1)
    rows = np.clip(np.round(py).astype(int), 0, H - 1)
    return rows, cols


def draw_disks(img, rows, cols, colors, radius=5):
    """Draw filled disks at (row,col) with given RGB colors."""
    H, W, _ = img.shape
    rr = np.arange(-radius, radius + 1)
    dx, dy = np.meshgrid(rr, rr, indexing="xy")
    mask = (dx * dx + dy * dy) <= radius * radius
    offsets = np.column_stack([dy[mask], dx[mask]])  # (k, 2) as (drow, dcol)
    for (r, c), col in zip(zip(rows, cols), colors):
        rs = r + offsets[:, 0]
        cs = c + offsets[:, 1]
        valid = (rs >= 0) & (rs < H) & (cs >= 0) & (cs < W)
        img[rs[valid], cs[valid]] = col


import re
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from typing import Tuple, List

_SENSOR_RE = re.compile(r".*_sensor_(\d+)$")


def load_foot_sensors_from_urdf(
    urdf_path: Path,
    left_parent_link: str = "left_ankle_roll_link",
    right_parent_link: str = "right_ankle_roll_link",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a URDF and return (left_xyz, right_xyz) arrays (N,3) of sensor positions,
    in the parent link frame, ordered by the index in child link names '*_sensor_<i>'.
    """
    urdf_path = Path(os.path.join(data_dir, urdf_path))
    print(urdf_path)
    tree = ET.parse(urdf_path.as_posix())
    root = tree.getroot()

    def collect(parent_link: str) -> np.ndarray:
        items: List[Tuple[int, np.ndarray]] = []
        for joint in root.findall("joint"):
            # Only fixed joints that attach a sensor link to this parent
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            if parent.get("link") != parent_link:
                continue

            child_name = child.get("link", "")
            m = _SENSOR_RE.match(child_name)
            if not m:
                continue  # skip non-sensor attachments under this parent

            idx = int(m.group(1))
            origin = joint.find("origin")
            if origin is None:
                continue
            xyz_str = origin.get("xyz", "0 0 0")
            x, y, z = map(float, xyz_str.strip().split())
            items.append((idx, np.array([x, y, z], dtype=np.float32)))

        items.sort(key=lambda t: t[0])
        return (
            np.vstack([p for _, p in items])
            if items
            else np.zeros((0, 3), dtype=np.float32)
        )

    left_xyz = collect(left_parent_link)
    right_xyz = collect(right_parent_link)
    return left_xyz, right_xyz
