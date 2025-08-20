import torch

import numpy as np
from pathlib import Path
from collections import deque
from copy import deepcopy
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import trimesh

try:
    import viser
    import viser.extras
except (
    Exception
) as e:  # This is needed since IsaacLab installs its own older websockets version, which is incompatible with visers.
    if e.args[0] == "No module named 'websockets.asyncio'":
        import shutil
        import websockets

        try:
            shutil.rmtree(websockets.__path__[0])
            print(
                f"Directory '{websockets.__path__[0]}' deleted successfully. Please run the script again."
            )
        except OSError as e:
            print(f"Error deleting directory '{websockets.__path__[0]}': {e}")

from protoverse.utils.urdf_loader import load_urdf
from protoverse.simulator.isaaclab.utils.camera_manager import (
    ViserCameraManager,
    FrameStore,
)
from protoverse.simulator.isaaclab.utils.camera_io import IsaacLabMultiCameraIO
from protoverse.simulator.isaaclab.utils.viser_utils import (
    simple_colormap,
    xy_to_img,
    draw_disks,
    load_foot_sensors_from_urdf,
)


class ViserLab:
    def __init__(
        self,
        config,
        cam_io: IsaacLabMultiCameraIO,
        marker_names: Optional[List[str]],
        frames: Optional[FrameStore] = None,
        frames_maxlen: int = 1,  # <- configurable, default 1
    ):
        """
        Create a Viser-based visualizer for an external simulation scene.

        Args:
        """
        self.num_envs = config.num_envs
        self.urdf_path = config.robot.asset.asset_file_name
        self.robot_config = config.robot
        self.config = config
        self.viser_server = viser.ViserServer()
        self.client = None
        self.env = 0  # Default selected environment

        self.cam_io = cam_io

        if cam_io:
            self.frames = frames or FrameStore(maxlen=frames_maxlen)

            self.available_cameras = config.camera.available_cameras
            if not len(self.available_cameras):
                print(
                    "No supported cameras found in scene.sensors (need 'viewport_camera' or 'overhead_camera')."
                )
            else:
                self.camera_source_name = self.available_cameras[0]  # default
                self.use_viewport = self.camera_source_name == "viewport_camera"
                self.cam_io.set_camera(self.camera_source_name)
                # self.view_camera = self.scene.sensors[self.camera_source_name]

        else:
            self.available_cameras = []

        self._setup_viser_scene()
        self._setup_viser_gui()
        self._handle_client_connection()

        # self.add_per_foot_snapshot_plot("left", sensor_names)
        # self.add_per_foot_snapshot_plot("right", sensor_names)

        if self.robot_config.with_foot_sensors:

            sensor_names = self.robot_config.foot_contact_links
            if (
                isinstance(sensor_names, tuple)
                and len(sensor_names) == 1
                and isinstance(sensor_names[0], list)
            ):
                sensor_names = sensor_names[0]

            self.add_per_foot_history_plot("left", sensor_names)
            self.add_per_foot_history_plot("right", sensor_names)

            left_samples, right_samples = load_foot_sensors_from_urdf(
                urdf_path=Path(self.urdf_path["robot"]),
                left_parent_link=self.robot_config.left_foot_name,
                right_parent_link=self.robot_config.right_foot_name,
            )

            self.add_feet_spatial_image(
                left_mesh=Path(config.robot.asset.left_foot_file_path),
                right_mesh=Path(config.robot.asset.right_foot_file_path),
                left_sensor_xyz_local=left_samples,  # (M_left, 3)
                right_sensor_xyz_local=right_samples,  # (M_right, 3)
                image_size=(260, 260),
                gap=8,
                rotate_each="cw",  # "cw" | "ccw" | "none"
                background=230,
            )

        self.setup_marker_toggles(marker_names)

    def _handle_client_connection(self):
        """Handle client connection setup"""
        if self.client is None:
            while self.client is None:
                self.client = (
                    self.viser_server.get_clients()[0]
                    if len(self.viser_server.get_clients()) > 0
                    else None
                )
                time.sleep(0.1)

    def _setup_viser_scene(self):
        self.base_frame = self.viser_server.scene.add_frame("/base", show_axes=False)
        if (
            getattr(self.robot_config, "init_state", None)
            and self.robot_config.init_state.pos is not None
        ):
            self.base_frame.position = tuple(self.robot_config.init_state.pos[:3])
        else:
            self.base_frame.position = (0.0, 0.0, 0.0)

        self.urdf = {}
        self.urdf_vis = {}

        self.urdf_path = {"robot": Path(self.urdf_path)}  # modify for multi agents

        for name, path in self.urdf_path.items():
            self.urdf[name] = load_urdf(None, path)
            self.urdf_vis[name] = viser.extras.ViserUrdf(
                self.viser_server, self.urdf[name], root_node_name="/base"
            )

        if self.cam_io:
            self.camera_manager = ViserCameraManager(
                self.viser_server, self.config.camera
            )
            self.camera_manager.on_camera_added = lambda key: None
            self.camera_manager.on_camera_removed = self.frames.drop

            self.use_viewport = len(self.camera_manager.frustums) == 0

        self.viser_ground_plane = self.viser_server.scene.add_grid(
            name="ground_plane",
            width_segments=50,
            height_segments=50,
            position=(0, 0, -1.0),
        )

    def _setup_viser_gui(self):

        self.plot_folder = self.viser_server.gui.add_folder("Contact Plots")
        folder = self.viser_server.gui.add_folder("Camera Viewer")

        with folder:
            self.viewport_image = self.viser_server.gui.add_image(
                np.zeros((240, 320, 3))
            )
            self.env_selector = self.viser_server.gui.add_dropdown(
                "Environment to View",
                [str(i) for i in range(self.num_envs)],
                initial_value="0",
            )

            if len(self.available_cameras):

                self.camera_selector = self.viser_server.gui.add_dropdown(
                    "Camera Source",
                    self.available_cameras,
                    initial_value=self.camera_source_name,
                )

                @self.camera_selector.on_update
                def _update_camera(_) -> None:
                    self.camera_source_name = self.camera_selector.value
                    # self.view_camera = self.scene.sensors[self.camera_source_name]
                    self.use_viewport = self.camera_source_name == "viewport_camera"
                    self.cam_io.set_camera(self.camera_source_name)

            else:
                self.camera_selector = None
                self.view_camera = None
                self.use_viewport = False

        self.env = int(self.env_selector.value)

        stats = self.viser_server.gui.add_folder("Stats")
        with stats:
            self.render_time_ms = self.viser_server.gui.add_number(
                "Render Time (ms): ", 0, disabled=True
            )
            self.sim_step_time_ms = self.viser_server.gui.add_number(
                "Simulation Step Time (ms): ", 0, disabled=True
            )
            self.save_time_ms = self.viser_server.gui.add_number(
                "Save File Time (ms): ", 0, disabled=True
            )

        controls = self.viser_server.gui.add_folder("Controls")

        @self.env_selector.on_update
        def _update_env(_) -> None:
            self.env = int(self.env_selector.value)

        if self.available_cameras:
            self.camera_manager.setup_gui(folder, controls)

            @self.camera_manager.add_camera_button.on_click
            def _add_camera(_) -> None:
                self.camera_manager.handle_add_camera(self.client)
                self.use_viewport = False

    def update_robot_configuration(
        self, base_pose, base_rot, joint_pos_dict: Dict[str, float]
    ):

        self.base_frame.position = base_pose
        self.base_frame.wxyz = base_rot
        self.urdf_vis["robot"].update_cfg(joint_pos_dict)

    # Camera

    def render_wrapped_impl(self, env_origins: Optional[torch.Tensor] = None):

        assert hasattr(self, "cam_io") or self.cam_io, "missing cam_io"

        if env_origins is None:
            env_origins = torch.zeros((self.num_envs, 3), dtype=torch.float32)

        if self.use_viewport and self.client is not None:

            cam_out = self.cam_io.render_from_viewport(
                client_position=self.client.camera.position,
                client_wxyz=self.client.camera.wxyz,
                env_origins=env_origins,
                use_first_cam_per_env=True,
            )

            self.frames.append("camera_0", cam_out)

        elif not self.use_viewport and len(self.camera_manager.frustums) > 0:

            by_key = self.cam_io.render_from_frustums(
                self.camera_manager.frustums, env_origins
            )

            for key, data in by_key.items():
                self.frames.append(key, data)

        # draw selected buffer
        if self.client is not None:
            selected = self.camera_manager.render_cam
            frame = self.frames.latest(selected)  # dict or None

            if isinstance(frame, dict):
                img = frame.get("rgb", None)
                if img is None:
                    rgba = frame.get("rgba", None)
                    if rgba is not None:
                        img = rgba[..., :3]

                if img is not None:
                    v = img[self.env]
                    self.viewport_image.image = (
                        v.detach().cpu().numpy()
                        if hasattr(v, "detach")
                        else np.asarray(v)
                    )

    # Foot as sequence

    def add_per_foot_snapshot_plot(self, side: str, sensor_names: list[str]):
        """Create a marker-only plot for one foot's contact sensors (norm values)."""
        indices = [i for i, name in enumerate(sensor_names) if side in name]
        x = np.arange(len(indices))  # ← match x to number of foot-specific sensors
        y = np.zeros(len(indices))

        with self.plot_folder:
            plot = self.viser_server.gui.add_uplot(
                data=(x, y),
                series=(
                    viser.uplot.Series(label="sensor index"),
                    viser.uplot.Series(
                        label=f"{side} foot",
                        stroke="blue" if side == "left" else "red",
                        width=0,
                        points={"show": True, "size": 6},
                    ),
                ),
                scales={
                    "x": viser.uplot.Scale(auto=True),
                    "y": viser.uplot.Scale(range=(0, 1.0)),
                },
                legend=viser.uplot.Legend(show=False),
                aspect=2.0,
            )

        setattr(self, f"{side}_foot_plot", plot)
        setattr(self, f"{side}_foot_x", x)
        setattr(self, f"{side}_foot_indices", indices)

    def update_per_foot_snapshot_plot(
        self, contact_norms: np.ndarray, side: str, env_id: int = 0
    ):
        """
        Update per-foot contact norm plot (snapshot of 1 frame).
        contact_norms: shape (num_envs, num_bodies), already ||F||
        """
        plot = getattr(self, f"{side}_foot_plot")
        x = getattr(self, f"{side}_foot_x")
        indices = getattr(self, f"{side}_foot_indices")

        raw = contact_norms[env_id, indices]
        y = raw / (raw.max() + 1e-6)
        plot.data = (x, y)

    def add_per_foot_history_plot(
        self, side: str, sensor_names: list[str], num_timesteps: int = 100
    ):
        """Create a time-series uPlot with one line per contact sensor on a foot."""
        indices = [i for i, name in enumerate(sensor_names) if side in name]
        x_data = np.linspace(0, num_timesteps / 60.0, num_timesteps)
        y_deques = [
            deque(np.zeros(num_timesteps), maxlen=num_timesteps) for _ in indices
        ]

        series = [viser.uplot.Series(label="time")]
        colors = [
            "red",
            "green",
            "blue",
            "orange",
            "purple",
            "cyan",
            "magenta",
            "brown",
        ]

        for i, idx in enumerate(indices):
            series.append(
                viser.uplot.Series(
                    label=sensor_names[idx],
                    stroke=colors[i % len(colors)],
                    width=2,
                )
            )

        with self.plot_folder:
            plot = self.viser_server.gui.add_uplot(
                data=(x_data, *[np.array(buf) for buf in y_deques]),
                series=tuple(series),
                scales={
                    "x": viser.uplot.Scale(auto=True),
                    "y": viser.uplot.Scale(range=(0, 400.0)),
                },
                legend=viser.uplot.Legend(show=False),
                aspect=2.0,
            )

        setattr(self, f"{side}_foot_history_plot", plot)
        setattr(self, f"{side}_foot_history_x", x_data)
        setattr(self, f"{side}_foot_history_y", y_deques)
        setattr(self, f"{side}_foot_indices", indices)

    def update_per_foot_history_plot(
        self, contact_norms: np.ndarray, side: str, env_id: int = 0
    ):
        """Update sliding window of contact magnitudes per foot."""
        plot = getattr(self, f"{side}_foot_history_plot")
        x = getattr(self, f"{side}_foot_history_x")
        y_deques = getattr(self, f"{side}_foot_history_y")
        indices = getattr(self, f"{side}_foot_indices")

        raw = contact_norms[env_id, indices]
        normed = raw  #  / (raw.max() + 1e-6)

        for val, buf in zip(normed, y_deques):
            buf.append(val)

        # Shift x-axis forward
        x += 1.0 / 60.0

        # Update plot data
        plot.data = (x.copy(), *[np.array(buf) for buf in y_deques])

    # Terrain

    def update_local_terrain_pointcloud(
        self,
        name: str,
        xyz_points: np.ndarray,
        color_by_height: bool = True,
        point_size: float = 0.02,
    ) -> None:
        """
        Update a point cloud in Viser for the terrain patch around the character.

        Args:
            name: Name of the point cloud (e.g., "/local_terrain_patch").
            xyz_points: (N, 3) array of [x, y, z] points.
            color_by_height: If True, colors will be height-based.
            point_size: Size of each rendered point.
        """
        if xyz_points.shape[0] == 0:
            return

        if color_by_height:
            z = xyz_points[:, 2]
            z_min, z_max = z.min(), z.max()
            normalized = (z - z_min) / (z_max - z_min + 1e-8)
            colors = np.zeros_like(xyz_points, dtype=np.uint8)
            colors[:, 0] = (normalized * 255).astype(np.uint8)  # red
            colors[:, 2] = ((1 - normalized) * 255).astype(np.uint8)  # blue
        else:
            colors = np.full_like(xyz_points, fill_value=200, dtype=np.uint8)  # gray

        if not hasattr(self, "_terrain_pcls"):
            self._terrain_pcls = {}

        if name not in self._terrain_pcls:
            self._terrain_pcls[name] = self.client.scene.add_point_cloud(
                name=name,
                points=xyz_points,
                colors=colors,
                point_size=point_size,
            )
        else:
            pcl = self._terrain_pcls[name]
            pcl.points = xyz_points
            pcl.colors = colors

    # Markers

    def setup_marker_toggles(self, marker_names: list[str]):

        if marker_names is None:
            return

        self.marker_toggle_folder = self.viser_server.gui.add_folder("Markers")
        self.marker_toggles = {}

        with self.marker_toggle_folder:
            for name in marker_names:
                toggle = self.viser_server.gui.add_checkbox(f"Show {name}", True)
                self.marker_toggles[name] = toggle

    def update_marker_group(
        self,
        name: str,
        positions: np.ndarray,
        orientations: np.ndarray = None,  # ignored unless future arrow support
        marker_type: str = "sphere",
        scale: float = 0.03,
        color: Optional[Tuple[int, int, int]] = None,
        color_by_height: bool = True,
    ) -> None:

        if positions.size == 0:
            return

        positions = positions.reshape(-1, 3)  # Ensure flat (N, 3)

        if color_by_height:
            z = positions[:, 2]
            z_min, z_max = z.min(), z.max()
            normalized = (z - z_min) / (z_max - z_min + 1e-8)

            colors = np.zeros_like(positions, dtype=np.uint8)
            colors[:, 0] = (normalized * 255).astype(np.uint8)  # red
            colors[:, 2] = ((1 - normalized) * 255).astype(np.uint8)  # blue
        else:
            if color is None:
                color = (0, 200, 0) if "goal" in name else (200, 0, 0)
            colors = np.tile(np.array(color, dtype=np.uint8), (positions.shape[0], 1))

        if not hasattr(self, "_viser_marker_dict"):
            self._viser_marker_dict = {}

        if name not in self._viser_marker_dict:
            self._viser_marker_dict[name] = self.viser_server.scene.add_point_cloud(
                name=f"/marker/{name}",
                points=positions,
                colors=colors,
                point_size=scale,
            )
        else:
            marker = self._viser_marker_dict[name]
            marker.points = positions
            marker.colors = colors

    # Foot as img

    def add_feet_spatial_image(
        self,
        left_mesh: Path,
        right_mesh: Path,
        left_sensor_xyz_local: np.ndarray,
        right_sensor_xyz_local: np.ndarray,
        image_size: Tuple[int, int] = (260, 260),
        gap: int = 8,
        rotate_each: str = "cw",  # "cw" | "ccw" | "none"
        background: int = 230,
    ):
        """Create a single combined (left+right) side-panel image."""
        # Prepare per-foot state (geometry + layout)
        self._left_foot_state = self._prepare_foot_state(
            left_mesh, left_sensor_xyz_local, image_size, background
        )
        self._right_foot_state = self._prepare_foot_state(
            right_mesh, right_sensor_xyz_local, image_size, background
        )

        H, W = image_size
        combo = np.full((H, W * 2 + gap, 3), background, dtype=np.uint8)
        with self.plot_folder:
            self._feet_combo_widget = self.viser_server.gui.add_image(
                combo, label="Feet (combined)"
            )
        self._feet_combo_cfg = {
            "gap": gap,
            "background": background,
            "rotate_each": rotate_each,
        }

    def _rotate_img(self, img: np.ndarray, mode: str) -> np.ndarray:
        if mode == "cw":
            return np.rot90(img, k=3)  # 90° clockwise
        if mode == "ccw":
            return np.rot90(img, k=1)  # 90° counter-clockwise
        if mode == "180":
            return np.rot90(img, k=2)  # 180°
        return img

    def _prepare_foot_state(
        self,
        mesh_path: Path,
        sensor_xyz_local: np.ndarray,
        image_size: Tuple[int, int],
        background: int,
    ):
        """Compute XY bounds from the sole of the mesh and store sensor XY layout."""
        mesh = trimesh.load_mesh(str(mesh_path))
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump().sum()

        z_min = float(mesh.vertices[:, 2].min())
        faces, verts = mesh.faces, mesh.vertices
        face_zmin = verts[faces].min(axis=(1, 2))
        keep = face_zmin <= (z_min + 1e-3)
        sole_verts = verts[np.unique(faces[keep].ravel())] if keep.any() else verts

        bb_min = sole_verts[:, :2].min(axis=0).astype(np.float32)
        bb_max = sole_verts[:, :2].max(axis=0).astype(np.float32)

        return {
            "H": image_size[0],
            "W": image_size[1],
            "bb_min": bb_min,
            "bb_max": bb_max,
            "radius": 5,
            "background": background,
            "sensor_xy": np.asarray(sensor_xyz_local[:, :2], dtype=np.float32),
        }

    def _render_foot_image(
        self,
        state: Dict[str, Any],
        vals,  # (E, M) or (M,)
        env_id: int = 0,
        vmin: float = 0.0,
        vmax: float | None = None,
        threshold: float | None = None,
    ):
        """Render one foot as an RGB numpy image using the stored state."""
        H, W = state["H"], state["W"]
        bb_min, bb_max = state["bb_min"], state["bb_max"]
        sensor_xy = state["sensor_xy"]
        bg = state["background"]

        if isinstance(vals, torch.Tensor):
            vals = vals.detach().cpu().numpy()
        vals = np.asarray(vals, dtype=np.float32)
        if vals.ndim == 2:
            vals = vals[env_id]

        assert (
            vals.shape[0] == sensor_xy.shape[0]
        ), f"contact length {vals.shape[0]} != sensors {sensor_xy.shape[0]}"

        if vmax is None:
            vmax = float(np.max(vals)) if vals.size else 1.0

        colors = simple_colormap(vals, vmin=vmin, vmax=vmax)

        if threshold is not None:
            colors[vals < threshold] = np.array([160, 160, 160], dtype=np.uint8)

        img = np.full((H, W, 3), bg, dtype=np.uint8)
        rows, cols = xy_to_img(sensor_xy, bb_min, bb_max, H, W)
        draw_disks(img, rows, cols, colors, radius=state["radius"])
        return img

    def update_feet_spatial_image(
        self,
        contact_norms: np.ndarray,  # (E, N_total)
        left_indices: np.ndarray,  # indices into N_total
        right_indices: np.ndarray,  # indices into N_total
        env_id: int = 0,
        threshold: float = 0.0,
        vmin: float = 0.0,
        vmax: float | None = None,
    ):
        """Update the combined image with current contact values."""
        if not hasattr(self, "_feet_combo_widget"):
            return

        # per-foot slices
        left_vals = contact_norms[:, left_indices]
        right_vals = contact_norms[:, right_indices]

        # shared vmax for consistent colors
        if vmax is None:
            l = left_vals[env_id] if left_vals.ndim == 2 else left_vals
            r = right_vals[env_id] if right_vals.ndim == 2 else right_vals
            vmax = float(
                max(1e-6, np.max(l) if l.size else 0.0, np.max(r) if r.size else 0.0)
            )

        # render both feet
        left_img = self._render_foot_image(
            self._left_foot_state, left_vals, env_id, vmin, vmax, threshold
        )
        right_img = self._render_foot_image(
            self._right_foot_state, right_vals, env_id, vmin, vmax, threshold
        )

        # rotate EACH image, then stitch left→right
        mode = self._feet_combo_cfg["rotate_each"]
        left_img = self._rotate_img(left_img, mode)
        right_img = self._rotate_img(right_img, mode)

        # canvas sized after rotation (H/W may swap)
        H, W, _ = left_img.shape
        gap = self._feet_combo_cfg["gap"]
        bg = self._feet_combo_cfg["background"]
        combo = np.full((H, W * 2 + gap, 3), bg, dtype=np.uint8)
        combo[:, :W] = left_img
        combo[:, W + gap :] = right_img

        self._feet_combo_widget.image = self._rotate_img(combo, "180")

    def update_feet_spatial_image_with_internal_indices(
        self,
        contact_norms: np.ndarray,
        env_id: int = 0,
        threshold: float = 1.0,
        vmin: float = 0.0,
        vmax: float | None = None,
    ):
        """Convenience: use the indices you already stored in add_per_foot_history_plot."""
        left_idx = getattr(self, "left_foot_indices")
        right_idx = getattr(self, "right_foot_indices")

        self.update_feet_spatial_image(
            contact_norms=contact_norms,
            left_indices=left_idx,
            right_indices=right_idx,
            env_id=env_id,
            threshold=threshold,
            vmin=vmin,
            vmax=vmax,
        )
