import torch

import numpy as np
from pathlib import Path
from collections import deque
from copy import deepcopy
import time
from typing import Dict, List, Any, Optional, Tuple

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

from protomotions.utils.urdf_loader import load_urdf
from protomotions.simulator.isaaclab.utils.camera_manager import CameraManager


# TODO: convert to simulator-independent...


class ViserLab:
    def __init__(self, scene, robot_config, marker_names):
        """
        Create a Viser-based visualizer for an external simulation scene.

        Args:
            scene: A simulation scene object with attributes:
                - sensors (dict)
                - articulations["robot"] with joint_names and data.default_joint_pos
                - env_origins (torch.Tensor)
                - num_envs (int)
            urdf_path: Dict mapping name to URDF path for each robot
        """
        self.scene = scene
        self.scene_config = scene.cfg
        self.urdf_path = robot_config.asset.asset_file_name
        self.robot_config = robot_config

        self.viser_server = viser.ViserServer()
        self.client = None
        self.env = 0  # Default selected environment

        self.available_cameras = self._get_available_cameras()
        if not len(self.available_cameras):
            print(
                "No supported cameras found in scene.sensors (need 'viewport_camera' or 'overhead_camera')."
            )
        else:
            self.camera_source_name = self.available_cameras[0]  # default
            self.view_camera = self.scene.sensors[self.camera_source_name]
            self.use_viewport = self.camera_source_name == "viewport_camera"

        self._setup_viser_scene()
        self._setup_viser_gui()
        # self._handle_client_connection()

        # body_ordering.contact_sensor_body_names
        sensor_names = self.scene["contact_sensor"].body_names
        # self.add_per_foot_snapshot_plot("left", sensor_names)
        # self.add_per_foot_snapshot_plot("right", sensor_names)

        self.add_per_foot_history_plot("left", sensor_names)
        self.add_per_foot_history_plot("right", sensor_names)

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

    def _get_available_cameras(self):
        return [
            name
            for name in ["viewport_camera", "overhead_camera"]
            if name in self.scene.sensors
        ]

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

        from pathlib import Path

        self.urdf_path = {"robot": Path(self.urdf_path)}  # modify for multi agents

        for name, path in self.urdf_path.items():
            self.urdf[name] = load_urdf(None, path)
            self.urdf_vis[name] = viser.extras.ViserUrdf(
                self.viser_server, self.urdf[name], root_node_name="/base"
            )

        # Set initial joint pose
        robot = self.scene["robot"]
        default_joint_pos_dict = {
            robot.joint_names[i]: robot.data.default_joint_pos[0][i].item()
            for i in range(len(robot.data.default_joint_pos[0]))
        }
        self.urdf_vis["robot"].update_cfg(default_joint_pos_dict)

        self.camera_manager = CameraManager(self.viser_server, self.scene)
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
                [str(i) for i in range(self.scene_config.num_envs)],
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
                    self.view_camera = self.scene.sensors[self.camera_source_name]
                    self.use_viewport = self.camera_source_name == "viewport_camera"

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
        self.camera_manager.setup_gui(folder, controls)

        @self.env_selector.on_update
        def _update_env(_) -> None:
            self.env = int(self.env_selector.value)

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

    def render_wrapped_impl(self, scene_positions: Optional[torch.Tensor] = None):
        cam = self.view_camera

        if self.use_viewport and self.client is not None:

            repeat_n = (
                self.scene_config.num_envs * cam.cfg.cams_per_env
                if getattr(cam.cfg, "cams_per_env", None) is not None
                else self.scene_config.num_envs
            )

            xyz = (
                torch.tensor(self.client.camera.position)
                .unsqueeze(0)
                .repeat(repeat_n, 1)
            )
            # xyz += self.scene.env_origins.cpu().repeat_interleave(
            #     repeat_n // self.scene_config.num_envs, dim=0
            # )
            xyz += torch.stack(scene_positions).cpu()

            wxyz = (
                torch.tensor(self.client.camera.wxyz).unsqueeze(0).repeat(repeat_n, 1)
            )
            cam.set_world_poses(xyz, wxyz, convention="ros")

            single_cam_ids = [
                cam.cfg.cams_per_env * i for i in range(self.scene_config.num_envs)
            ]
            cam_out = {
                key: cam.data.output[key][single_cam_ids] for key in cam.data.output
            }
            self.camera_manager.buffers["camera_0"].append(cam_out)

        elif not self.use_viewport and len(self.camera_manager.frustums) > 0:
            cams_per_env = getattr(cam.cfg, "cams_per_env", 1)
            repeat_n = self.scene_config.num_envs * cams_per_env

            xyzs, wxyzs = [], []
            for f in self.camera_manager.frustums:
                xyzs.append(f.position)
                wxyzs.append(f.wxyz)
            while len(xyzs) < cams_per_env:
                xyzs.append(xyzs[-1])
                wxyzs.append(wxyzs[-1])

            xyzs = torch.tensor(np.array(xyzs)).repeat(self.scene_config.num_envs, 1)
            wxyzs = torch.tensor(np.array(wxyzs)).repeat(self.scene_config.num_envs, 1)
            xyzs += self.scene.env_origins.cpu().repeat_interleave(
                repeat_n // self.scene_config.num_envs, dim=0
            )

            cam.set_world_poses(xyzs, wxyzs, convention="ros")

            indices = [
                i * cams_per_env + j
                for i in range(self.scene_config.num_envs)
                for j in range(len(self.camera_manager.frustums))
            ]
            cam_out = {key: cam.data.output[key][indices] for key in cam.data.output}

            for idx, frustum in enumerate(self.camera_manager.frustums):
                frustum_data = {
                    key: cam_out[key][idx :: len(self.camera_manager.frustums)]
                    for key in cam_out
                }
                buffer_key = frustum.name[1:]
                if buffer_key not in self.camera_manager.buffers:
                    self.camera_manager.buffers[buffer_key] = deque(maxlen=1)
                self.camera_manager.buffers[buffer_key].append(deepcopy(frustum_data))

        # Update Viser GUI
        if (
            "camera_0" in self.camera_manager.buffers
            and len(self.camera_manager.buffers["camera_0"]) > 0
            and self.client is not None
        ):
            rgb = self.camera_manager.buffers["camera_0"][0]["rgb"][self.env]
            self.viewport_image.image = rgb.clone().cpu().detach().numpy()

        # self.sim.render() # moved to pre_physics_step

    # Plots

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
                    "y": viser.uplot.Scale(range=(0, 1.0)),
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
        normed = raw / (raw.max() + 1e-6)

        for val, buf in zip(normed, y_deques):
            buf.append(val)

        # Shift x-axis forward
        x += 1.0 / 60.0

        # Update plot data
        plot.data = (x.copy(), *[np.array(buf) for buf in y_deques])

    def update_local_terrain_pointcloud(
        self,
        name: str,
        xyz_points: np.ndarray,
        color_by_height: bool = True,
        point_size: float = 0.1,
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
            colors = np.full_like(xyz_points, fill_value=180, dtype=np.uint8)  # gray

        if not hasattr(self, "pcl"):
            self.pcl = self.client.scene.add_point_cloud(
                name=name,
                points=xyz_points,
                colors=colors,
                point_size=point_size,
            )
        else:
            self.pcl.points = xyz_points
            self.pcl_colors = colors

    # markers:

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
