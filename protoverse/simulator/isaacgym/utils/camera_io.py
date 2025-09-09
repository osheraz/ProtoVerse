from typing import Dict, Optional, Sequence, List, Literal, Union
from dataclasses import dataclass
import torch
import numpy as np

from isaacgym import gymapi, gymtorch


# =========================
# Config dataclasses
# =========================

# io = IsaacGymMultiCameraIO(
#     gym=gym, sim=sim, envs=envs, env_origins=env_origins,  # (E,3)
#     props=CameraProps(width=512, height=512, enable_depth=True, enable_seg=True),
#     create_mode=CreateWorldFixed(
#         base_xyz=(0.7, 0.0, 0.5),
#         base_xyzw=(0, 0.3827, 0, 0.9239),   # example quat (x,y,z,w)
#         add_env_origin_offset=True,
#     ),
#     device="cuda:0",
# )

# io = IsaacGymMultiCameraIO(
#     gym=gym, sim=sim, envs=envs, env_origins=env_origins,
#     props=CameraProps(),
#     create_mode=CreateLookAt(
#         eye=(0.8, 0.0, 0.6),
#         target=(0.5, 0.0, 0.25),
#         add_env_origin_offset=True,
#     ),
# )

# io = IsaacGymMultiCameraIO(
#     gym=gym, sim=sim, envs=envs, env_origins=env_origins,
#     props=CameraProps(enable_depth=True),
#     create_mode=CreateAttach(
#         local_pos=(0.25, 0.0, 0.15),
#         local_euler_zyx_rad=(0.0, 0.5, 3.14159),  # or set local_xyzw=(x,y,z,w)
#         follow_mode=gymapi.FOLLOW_TRANSFORM,
#     ),
#     attach_actor_handles=kuka_handles,  # len == num_envs
# )

# cam_io = IsaacGymSingleCameraIO(
#     gym, sim, envs[0], width=800, height=600, enable_depth=True
# )


@dataclass
class CameraProps:
    width: int = 320
    height: int = 240
    horizontal_fov: float = 70.0  # degrees
    enable_color: bool = True
    enable_depth: bool = False
    enable_seg: bool = False


@dataclass
class CreateWorldFixed:
    """Create one camera per env at the same world pose, optionally offset by env_origin."""

    mode: Literal["world_fixed"] = "world_fixed"
    base_xyz: Sequence[float] = (0.6, 0.0, 0.6)
    base_xyzw: Sequence[float] = (0.0, 0.0, 0.0, 1.0)  # (x,y,z,w)
    add_env_origin_offset: bool = True  # base_xyz + env_origin


@dataclass
class CreateLookAt:
    """Create one camera per env using look-at (position + target point)."""

    mode: Literal["lookat"] = "lookat"
    eye: Sequence[float] = (0.8, 0.0, 0.6)
    target: Sequence[float] = (0.5, 0.0, 0.2)
    add_env_origin_offset: bool = True  # add env_origin to both eye and target


@dataclass
class CreateAttach:
    """Attach one camera per env to a given actor with a local transform."""

    mode: Literal["attach"] = "attach"
    local_pos: Sequence[float] = (0.2, 0.0, 0.1)
    # local orientation as ZYX Euler (IsaacGym convenience) OR xyzw quat (set one of them)
    local_euler_zyx_rad: Optional[Sequence[float]] = (0.0, 0.5, 3.14159)
    local_xyzw: Optional[Sequence[float]] = None
    follow_mode: int = gymapi.FOLLOW_TRANSFORM  # or FOLLOW_POSITION
    track_actor: bool = True  # if False, it's a one-time placement (rare)


CreateMode = Union[CreateWorldFixed, CreateLookAt, CreateAttach]

# =========================
# Helpers
# =========================


def _t(x, device="cpu", dtype=torch.float32):
    return (
        x
        if isinstance(x, torch.Tensor)
        else torch.tensor(x, device=device, dtype=dtype)
    )


def _norm_xyzw(q: torch.Tensor, eps: float = 1e-8):
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def _ensure_rgb(out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if "rgb" not in out and "rgba" in out:
        out["rgb"] = out["rgba"][..., :3]
    return out


# =========================
# Main bridge
# =========================


class IsaacGymMultiCameraIO:
    """
    Isaac Gym multi-env camera bridge.

    - Can CREATE per-env camera handles internally (world_fixed / lookat / attach).
    - Wraps GPU image tensors **once**; per-step reads are batched via torch.stack.
    - Quaternions are (x, y, z, w).
    """

    def __init__(
        self,
        *,
        gym,
        sim,
        envs: List,
        env_origins: Optional[torch.Tensor] = None,  # (E,3) if you want origin offsets
        props: CameraProps = CameraProps(),
        create_mode: CreateMode = CreateWorldFixed(),
        attach_actor_handles: Optional[List[int]] = None,  # required for CreateAttach
        device: str = "cuda:0",
    ):
        self.gym = gym
        self.sim = sim
        self.envs = envs
        self._E = len(envs)
        self.device = device
        self.env_origins = (
            None
            if env_origins is None
            else _t(env_origins, device=device).view(self._E, 3)
        )

        self.props = props
        self.create_mode = create_mode
        self.camera_handles: List[int] = []

        # Create per-env cameras
        if isinstance(create_mode, CreateAttach):
            assert (
                attach_actor_handles is not None
                and len(attach_actor_handles) == self._E
            ), "attach_actor_handles must be provided (len == num_envs) for CreateAttach"
            self._create_cameras_attach(attach_actor_handles, create_mode)
        elif isinstance(create_mode, CreateLookAt):
            self._create_cameras_lookat(create_mode)
        else:
            self._create_cameras_world_fixed(create_mode)

        # Wrap GPU tensors once
        self._color_tensors: List[torch.Tensor] = []
        self._depth_tensors: List[torch.Tensor] = []
        self._seg_tensors: List[torch.Tensor] = []
        self._init_image_views()

    # ----------- creation paths -----------

    def _make_props(self) -> gymapi.CameraProperties:
        p = gymapi.CameraProperties()
        p.width = int(self.props.width)
        p.height = int(self.props.height)
        p.horizontal_fov = float(self.props.horizontal_fov)
        p.enable_tensors = True
        return p

    def _create_cameras_world_fixed(self, cfg: CreateWorldFixed):
        for i in range(self._E):
            props = self._make_props()
            cam = self.gym.create_camera_sensor(self.envs[i], props)

            # world transform
            xyz = _t(cfg.base_xyz, device="cpu").clone()
            if cfg.add_env_origin_offset and self.env_origins is not None:
                xyz = xyz + self.env_origins[i].cpu()

            q = _norm_xyzw(_t(cfg.base_xyzw)).cpu()

            t = gymapi.Transform()
            t.p = gymapi.Vec3(float(xyz[0]), float(xyz[1]), float(xyz[2]))
            t.r = gymapi.Quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            self.gym.set_camera_transform(cam, self.envs[i], t)

            self.camera_handles.append(cam)

    def _create_cameras_lookat(self, cfg: CreateLookAt):
        for i in range(self._E):
            props = self._make_props()
            cam = self.gym.create_camera_sensor(self.envs[i], props)

            eye = _t(cfg.eye, device="cpu").clone()
            tgt = _t(cfg.target, device="cpu").clone()
            if cfg.add_env_origin_offset and self.env_origins is not None:
                eye = eye + self.env_origins[i].cpu()
                tgt = tgt + self.env_origins[i].cpu()

            self.gym.set_camera_location(
                cam,
                self.envs[i],
                gymapi.Vec3(float(eye[0]), float(eye[1]), float(eye[2])),
                gymapi.Vec3(float(tgt[0]), float(tgt[1]), float(tgt[2])),
            )
            self.camera_handles.append(cam)

    def _create_cameras_attach(self, actor_handles: List[int], cfg: CreateAttach):
        for i in range(self._E):
            props = self._make_props()
            cam = self.gym.create_camera_sensor(self.envs[i], props)

            local_tf = gymapi.Transform()
            lp = _t(cfg.local_pos, device="cpu")
            local_tf.p = gymapi.Vec3(float(lp[0]), float(lp[1]), float(lp[2]))

            if cfg.local_xyzw is not None:
                q = _norm_xyzw(_t(cfg.local_xyzw)).cpu().tolist()
                local_tf.r = gymapi.Quat(q[0], q[1], q[2], q[3])
            else:
                # ZYX Euler (roll about z last) — IsaacGym helper
                rz, ry, rx = cfg.local_euler_zyx_rad
                local_tf.r = gymapi.Quat.from_euler_zyx(rz, ry, rx)

            self.gym.attach_camera_to_body(
                cam,
                self.envs[i],
                actor_handles[i],
                local_tf,
                cfg.follow_mode,
            )
            self.camera_handles.append(cam)

    # ----------- GPU views -----------

    def _init_image_views(self):
        # ensure valid buffers
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        self.gym.start_access_image_tensors(self.sim)

        if self.props.enable_color:
            for i in range(self._E):
                t = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR
                )
                self._color_tensors.append(gymtorch.wrap_tensor(t))

        if self.props.enable_depth:
            for i in range(self._E):
                t = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_DEPTH
                )
                self._depth_tensors.append(gymtorch.wrap_tensor(t))

        if self.props.enable_seg:
            for i in range(self._E):
                t = self.gym.get_camera_image_gpu_tensor(
                    self.sim,
                    self.envs[i],
                    self.camera_handles[i],
                    gymapi.IMAGE_SEGMENTATION,
                )
                self._seg_tensors.append(gymtorch.wrap_tensor(t))

        self.gym.end_access_image_tensors(self.sim)

    # ----------- pose updates -----------

    def set_world_poses(self, xyz: torch.Tensor, xyzw: torch.Tensor):
        """Update world poses for per-env cameras (xyzw quats)."""
        xyz = _t(xyz, device="cpu").view(self._E, 3)
        q = _norm_xyzw(_t(xyzw, device="cpu")).view(self._E, 4)
        for i in range(self._E):
            t = gymapi.Transform()
            t.p = gymapi.Vec3(float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]))
            t.r = gymapi.Quat(
                float(q[i, 0]), float(q[i, 1]), float(q[i, 2]), float(q[i, 3])
            )
            self.gym.set_camera_transform(self.camera_handles[i], self.envs[i], t)

    # ----------- batched capture -----------

    def _begin_access(self):
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

    def _end_access(self):
        self.gym.end_access_image_tensors(self.sim)

    def _batched_output(
        self, indices: Optional[Sequence[int]]
    ) -> Dict[str, torch.Tensor]:
        idx = list(range(self._E)) if indices is None else list(indices)
        out: Dict[str, torch.Tensor] = {}

        if self.props.enable_color and len(self._color_tensors):
            rgba = torch.stack(self._color_tensors, dim=0)[idx]
            out["rgba"] = rgba.clone()
        if self.props.enable_depth and len(self._depth_tensors):
            depth = torch.stack(self._depth_tensors, dim=0)[idx]
            out["depth"] = depth.clone()
        if self.props.enable_seg and len(self._seg_tensors):
            seg = torch.stack(self._seg_tensors, dim=0)[idx]
            out["seg"] = seg.clone()

        return _ensure_rgb(out)

    # ----------- IsaacLab-like APIs -----------

    @torch.no_grad()
    def render_from_viewport(
        self,
        client_position: Sequence[float],
        client_xyzw: Sequence[float],
        use_first_cam_per_env: bool = True,  # kept for parity (no-op)
        return_cpu: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Put each env's camera at client pose (plus env_origin if provided), then batch read.
        """
        E = self._E
        base_xyz = _t(client_position, device=self.device).view(1, 3).repeat(E, 1)
        base_q = _t(client_xyzw, device=self.device).view(1, 4).repeat(E, 1)
        if self.env_origins is not None:
            xyz = base_xyz + self.env_origins.to(self.device)
        else:
            xyz = base_xyz

        self.set_world_poses(xyz, base_q)

        self._begin_access()
        out = self._batched_output(indices=None)
        self._end_access()

        return {k: v.cpu() for k, v in out.items()} if return_cpu else out

    @torch.no_grad()
    def render_from_frustums(
        self,
        frustums: Sequence,  # each: .position (3,), .xyzw (4,), .name (str)
        return_cpu: bool = False,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        For each incoming frustum, apply pose across envs (plus env_origin if available) and capture.
        """
        results: Dict[str, Dict[str, torch.Tensor]] = {}
        for fr in frustums:
            key = (
                fr.name[1:]
                if getattr(fr, "name", "").startswith("/")
                else getattr(fr, "name", "camera")
            )

            E = self._E
            base_xyz = _t(fr.position, device=self.device).view(1, 3).repeat(E, 1)
            base_q = _t(fr.xyzw, device=self.device).view(1, 4).repeat(E, 1)

            if self.env_origins is not None:
                xyz = base_xyz + self.env_origins.to(self.device)
            else:
                xyz = base_xyz

            self.set_world_poses(xyz, base_q)

            self._begin_access()
            out = self._batched_output(indices=None)
            self._end_access()

            results[key] = {k: v.cpu() for k, v in out.items()} if return_cpu else out
        return results


class IsaacGymSingleCameraIO:
    """
    Minimal wrapper for a single Isaac Gym camera (always attached to env_0).
    """

    def __init__(
        self,
        gym,
        sim,
        env,
        *,
        width: int = 320,
        height: int = 240,
        horizontal_fov: float = 70.0,
        enable_color: bool = True,
        enable_depth: bool = False,
        enable_seg: bool = False,
    ):
        self.gym = gym
        self.sim = sim
        self.env = env

        # create camera
        props = gymapi.CameraProperties()
        props.width = width
        props.height = height
        props.horizontal_fov = horizontal_fov
        props.enable_tensors = True
        self._cam = self.gym.create_camera_sensor(self.env, props)

        self._enable_color = enable_color
        self._enable_depth = enable_depth
        self._enable_seg = enable_seg

        # wrap GPU tensors
        self._rgba = None
        self._depth = None
        self._seg = None
        self._init_views()

        self.cam_pos = None
        self.cam_target = None

    def _init_views(self):
        # warm-up rendering
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        self.gym.start_access_image_tensors(self.sim)
        if self._enable_color:
            t = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.env, self._cam, gymapi.IMAGE_COLOR
            )
            self._rgba = gymtorch.wrap_tensor(t)
        if self._enable_depth:
            t = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.env, self._cam, gymapi.IMAGE_DEPTH
            )
            self._depth = gymtorch.wrap_tensor(t)
        if self._enable_seg:
            t = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.env, self._cam, gymapi.IMAGE_SEGMENTATION
            )
            self._seg = gymtorch.wrap_tensor(t)
        self.gym.end_access_image_tensors(self.sim)

    def set_camera_view(self, eye: Sequence[float], target: Sequence[float]):
        """
        Initialize camera view relative to a target (world coords).
        """
        self.gym.set_camera_location(
            self._cam,
            self.env,
            gymapi.Vec3(float(eye[0]), float(eye[1]), float(eye[2])),
            gymapi.Vec3(float(target[0]), float(target[1]), float(target[2])),
        )
        self.cam_pos = np.array(eye, dtype=np.float32)
        self.cam_target = np.array(target, dtype=np.float32)

    def get_camera_state(self):
        return {"pos": self.cam_pos, "target": self.cam_target}

    def capture(self, return_cpu: bool = True) -> Dict[str, torch.Tensor]:
        """
        Grab latest camera outputs (rgb, depth, seg).
        """
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        out: Dict[str, torch.Tensor] = {}
        if self._rgba is not None:
            out["rgba"] = self._rgba.clone()
            out["rgb"] = out["rgba"][..., :3]
        if self._depth is not None:
            out["depth"] = self._depth.clone()
        if self._seg is not None:
            out["seg"] = self._seg.clone()

        self.gym.end_access_image_tensors(self.sim)
        if return_cpu:
            out = {k: v.cpu() for k, v in out.items()}
        return out
