# protoverse/vis/bridges/isaaclab_camera_io.py
from typing import Dict, Optional, Sequence, List
import torch
import numpy as np
from loguru import logger


class IsaacLabMultiCameraIO:
    """
    Middle layer that hides IsaacLab specifics.
    Owns the scene and the active camera, and exposes high-level render calls.
    """

    def __init__(self, scene, config):
        self._scene = scene
        self.cam_cfg = config.camera
        self.config = config
        camera_name = self.cam_cfg.available_cameras[0]
        logger.info(f"Using: {camera_name}")
        assert (
            camera_name in self._scene.sensors.keys()
        ), f"{camera_name} is not available"

        self._set_camera(camera_name)

    # --- camera selection ---
    def _set_camera(self, name: str):
        self._cam = self._scene.sensors[name]
        self.cams_per_env = int(getattr(self._cam.cfg, "cams_per_env", 1))

    def set_camera(self, name: str):
        self._set_camera(name)

    # --- helpers ViserLab can query ---
    def num_envs(self) -> int:
        return int(self.config.num_envs)

    def env_origins(self) -> torch.Tensor:
        return self._scene.env_origins.detach().cpu()  # (E,3)

    def first_cam_indices(self) -> List[int]:
        E, C = self.num_envs(), self.cams_per_env
        return [i * C for i in range(E)] if C > 1 else list(range(E))

    def set_world_poses(self, xyz: torch.Tensor, wxyz: torch.Tensor):
        self._cam.set_world_poses(xyz, wxyz, convention="ros")

    def _get_output(
        self, indices: Optional[Sequence[int]] = None
    ) -> Dict[str, torch.Tensor]:
        out = {
            k: (v if indices is None else v[indices])
            for k, v in self._cam.data.output.items()
        }
        if "rgb" not in out and "rgba" in out:
            out["rgb"] = out["rgba"][..., :3]
        return out

    # --- high-level entrypoints used by ViserLab ---

    @torch.no_grad()
    def render_from_viewport(
        self,
        client_position: Sequence[float],
        client_wxyz: Sequence[float],
        env_origins: torch.Tensor,  # (E,3) CPU
        use_first_cam_per_env: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Place camera(s) at client pose for every env, return outputs for one cam per env."""

        E, C = self.num_envs(), self.cams_per_env
        repeat_n = E * C

        base_xyz = (
            torch.tensor(client_position, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(repeat_n, 1)
        )
        base_wxyz = (
            torch.tensor(client_wxyz, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(repeat_n, 1)
        )
        xyz = base_xyz + env_origins.repeat_interleave(repeat_n // E, dim=0)

        self.set_world_poses(xyz, base_wxyz)

        # in case the are multiple cams in the env (external, wrist, ..)
        indices = (
            self.first_cam_indices() if use_first_cam_per_env else list(range(repeat_n))
        )
        return self._get_output(indices)

    @torch.no_grad()
    def render_from_frustums(
        self,
        frustums: Sequence,  # Viser frustum objects with .position/.wxyz
        env_origins: torch.Tensor,  # (E,3) CPU
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Place cameras at frustum poses replicated across envs.
        Return dict: { 'camera_0': {'rgb': (E, ...), ...}, 'camera_1': {...}, ... }
        """
        E, C = self.num_envs(), self.cams_per_env
        n_frust = len(frustums)

        xyzs = [f.position for f in frustums]
        wxyzs = [f.wxyz for f in frustums]

        # pad up to cams_per_env
        while len(xyzs) < C:
            xyzs.append(xyzs[-1])
            wxyzs.append(wxyzs[-1])

        base_xyz = torch.tensor(np.array(xyzs), dtype=torch.float32)  # (C,3)
        base_wxyz = torch.tensor(np.array(wxyzs), dtype=torch.float32)  # (C,4)
        repeat_n = E * C

        xyz = base_xyz.repeat(E, 1) + env_origins.repeat_interleave(
            repeat_n // E, dim=0
        )
        wxyz = base_wxyz.repeat(E, 1)

        self.set_world_poses(xyz, wxyz)

        # gather outputs for all placed cams, then split per frustum
        indices = [i * C + j for i in range(E) for j in range(n_frust)]
        out_all = self._get_output(indices)

        by_key: Dict[str, Dict[str, torch.Tensor]] = {}
        for j, frustum in enumerate(frustums):
            key = frustum.name[1:]  # "/camera_0" -> "camera_0"
            by_key[key] = {k: v[j::n_frust] for k, v in out_all.items()}
        return by_key


class IsaacLabSingleCameraIO:
    """
    Minimal wrapper for a single IsaacLab camera (always attached to env_0).
    """

    def __init__(self, scene, camera_name: str):
        self._scene = scene
        assert camera_name in self._scene.sensors, f"{camera_name} not found"
        self._cam = self._scene.sensors[camera_name]
        self.cam_pos = None

    def set_camera_view(self, eye: np.array, target: np.array):
        """
        Initialize camera view relative to a target (e.g., robot root pos).
        """
        self._cam.set_world_poses_from_view(
            torch.tensor(eye[None], dtype=torch.float32, device=self._cam.device),
            torch.tensor(target[None], dtype=torch.float32, device=self._cam.device),
        )
        self.cam_pos = eye

    def get_camera_state(self):
        return self.cam_pos

    def capture(self) -> Dict[str, torch.Tensor]:
        """
        Grab latest camera outputs (rgb, depth, etc.).
        """
        out = dict(self._cam.data.output)
        if "rgb" not in out and "rgba" in out:
            out["rgb"] = out["rgba"][..., :3]
        return out
