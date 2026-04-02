from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
import uuid

import blenderproc as bproc
import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector
from blenderproc.python.utility.GlobalStorage import GlobalStorage

from blendforge.blender_runtime.camera.RealsenseProfileLoader import RealSenseProfile
from blendforge.blender_runtime.camera import RealsenseProjectorUtility as projector_util


@dataclass(frozen=True)
class ProjectorOverrides:
    proj_tx: float = 0.0
    proj_ty: float = 0.0
    proj_tz: float = 0.0
    proj_yaw_deg: float = 0.0
    proj_pitch_deg: float = 0.0
    proj_roll_deg: float = 0.0
    proj_fov_x_deg: Optional[float] = None
    proj_fov_y_deg: Optional[float] = None
    flip_u: bool = False
    flip_v: bool = False
    dot_sigma_px: Optional[float] = None
    projector_energy: Optional[float] = None
    projector_color_rgb: Optional[tuple[float, float, float]] = None

    def has_transform_override(self) -> bool:
        return any(
            abs(float(v)) > 1e-12
            for v in (
                self.proj_tx,
                self.proj_ty,
                self.proj_tz,
                self.proj_yaw_deg,
                self.proj_pitch_deg,
                self.proj_roll_deg,
            )
        )


@dataclass(frozen=True)
class ProjectorRuntimeConfig:
    source: str
    pattern_width: int
    pattern_height: int
    pattern_path: Optional[str]
    pattern_base_dir: Optional[str]
    pattern_seed: Optional[int]
    pattern_min_sep_px: Optional[float]
    pattern_dot_radius_px: Optional[float]
    pattern_dot_sigma_px: Optional[float]
    dot_count: int
    energy: float
    color_rgb: tuple[float, float, float]
    fov_h_deg: float
    fov_v_deg: float
    mount_frame: str
    mount_mode: str
    local_transform_blender_4x4_base: np.ndarray
    local_transform_blender_4x4_final: np.ndarray
    flip_u_profile_default: bool = False
    flip_v_profile_default: bool = False
    flip_u: bool = False
    flip_v: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _override_value(overrides: Optional[Any], name: str, default: Any) -> Any:
    if overrides is None:
        return default
    return getattr(overrides, name, default)


def _normalize_color_rgb(value: Optional[Any], default=(1.0, 1.0, 1.0)) -> tuple[float, float, float]:
    raw = default if value is None else value
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"Projector color must contain exactly 3 values, got {arr.tolist()}")
    arr = np.clip(arr, 0.0, 1.0)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def matrix_from_translation_euler_xyz(
    tx: float,
    ty: float,
    tz: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    euler = Euler(
        (
            np.deg2rad(float(roll_deg)),
            np.deg2rad(float(pitch_deg)),
            np.deg2rad(float(yaw_deg)),
        ),
        "XYZ",
    )
    rot = euler.to_matrix().to_4x4()
    tr = Matrix.Translation(Vector((float(tx), float(ty), float(tz))))
    return np.asarray((tr @ rot), dtype=np.float64)


def get_stream_world_pose_from_anchor(
    rs: RealSenseProfile,
    stream_name: str,
    anchor_stream: str,
    anchor_world_pose: np.ndarray,
) -> np.ndarray:
    if stream_name == anchor_stream:
        return np.asarray(anchor_world_pose, dtype=np.float64).copy()
    return np.asarray(anchor_world_pose, dtype=np.float64) @ rs.get_T_blender(stream_name, anchor_stream)


def get_mount_world_pose(
    rs: RealSenseProfile,
    *,
    anchor_stream: str,
    secondary_stream: str,
    anchor_world_pose: np.ndarray,
) -> np.ndarray:
    mount_frame = rs.get_projector_mount_frame() if rs.has_projector() else anchor_stream
    if mount_frame == anchor_stream:
        return np.asarray(anchor_world_pose, dtype=np.float64).copy()
    if mount_frame == secondary_stream:
        return get_stream_world_pose_from_anchor(rs, secondary_stream, anchor_stream, anchor_world_pose)
    if rs.has_stream(mount_frame):
        return get_stream_world_pose_from_anchor(rs, mount_frame, anchor_stream, anchor_world_pose)
    if mount_frame == "COLOR" and rs.has_stream("COLOR"):
        return get_stream_world_pose_from_anchor(rs, "COLOR", anchor_stream, anchor_world_pose)
    if mount_frame == "DEPTH":
        return np.asarray(anchor_world_pose, dtype=np.float64).copy()
    return np.asarray(anchor_world_pose, dtype=np.float64).copy()


def resolve_projector_runtime_config(
    rs: RealSenseProfile,
    overrides: Optional[Any] = None,
    *,
    fallback_left_stream: str,
) -> ProjectorRuntimeConfig:
    if rs.has_projector():
        pr = rs.get_projector()
        projector_metadata = getattr(pr, "metadata", {}) or {}
        base_local = np.asarray(rs.get_projector_local_transform_blender(), dtype=np.float64)
        cfg = ProjectorRuntimeConfig(
            source="profile",
            pattern_width=int(pr.pattern_w),
            pattern_height=int(pr.pattern_h),
            pattern_path=getattr(pr, "pattern_path", None),
            pattern_base_dir=getattr(rs, "profile_base_dir", None),
            pattern_seed=getattr(pr, "pattern_seed", None),
            pattern_min_sep_px=getattr(pr, "pattern_min_sep_px", None),
            pattern_dot_radius_px=getattr(pr, "pattern_dot_radius_px", None),
            pattern_dot_sigma_px=getattr(pr, "pattern_dot_sigma_px", None),
            dot_count=int(pr.dot_count),
            energy=float(pr.energy),
            color_rgb=_normalize_color_rgb(projector_metadata.get("color_rgb", (1.0, 1.0, 1.0))),
            fov_h_deg=float(np.rad2deg(pr.fov_h_rad)),
            fov_v_deg=float(np.rad2deg(pr.fov_v_rad)),
            mount_frame=rs.get_projector_mount_frame(),
            mount_mode=rs.get_projector_mount_mode(),
            local_transform_blender_4x4_base=base_local,
            local_transform_blender_4x4_final=base_local.copy(),
            flip_u_profile_default=bool(projector_metadata.get("flip_u", False)),
            flip_v_profile_default=bool(projector_metadata.get("flip_v", False)),
        )
    else:
        s_left = rs.get_stream(fallback_left_stream)
        fov_h_rad, fov_v_rad = projector_util.fov_from_K(s_left.K, s_left.width, s_left.height)
        base_local = np.eye(4, dtype=np.float64)
        cfg = ProjectorRuntimeConfig(
            source="legacy_fallback",
            pattern_width=int(s_left.width),
            pattern_height=int(s_left.height),
            pattern_path=None,
            pattern_base_dir=None,
            pattern_seed=0,
            pattern_min_sep_px=None,
            pattern_dot_radius_px=1.0,
            pattern_dot_sigma_px=None,
            dot_count=25600,
            energy=3000.0,
            color_rgb=(1.0, 1.0, 1.0),
            fov_h_deg=float(np.rad2deg(fov_h_rad)),
            fov_v_deg=float(np.rad2deg(fov_v_rad)),
            mount_frame=fallback_left_stream,
            mount_mode="legacy_frame_lock",
            local_transform_blender_4x4_base=base_local,
            local_transform_blender_4x4_final=base_local.copy(),
        )

    delta = matrix_from_translation_euler_xyz(
        _override_value(overrides, "proj_tx", 0.0),
        _override_value(overrides, "proj_ty", 0.0),
        _override_value(overrides, "proj_tz", 0.0),
        _override_value(overrides, "proj_roll_deg", 0.0),
        _override_value(overrides, "proj_pitch_deg", 0.0),
        _override_value(overrides, "proj_yaw_deg", 0.0),
    )
    final = delta @ np.asarray(cfg.local_transform_blender_4x4_base, dtype=np.float64)

    projector_energy = _override_value(overrides, "projector_energy", None)
    projector_color_rgb = _override_value(overrides, "projector_color_rgb", None)
    dot_sigma_px = _override_value(overrides, "dot_sigma_px", None)
    fov_x_override = _override_value(overrides, "proj_fov_x_deg", None)
    fov_y_override = _override_value(overrides, "proj_fov_y_deg", None)

    return ProjectorRuntimeConfig(
        source=cfg.source,
        pattern_width=cfg.pattern_width,
        pattern_height=cfg.pattern_height,
        pattern_path=cfg.pattern_path,
        pattern_base_dir=cfg.pattern_base_dir,
        pattern_seed=cfg.pattern_seed,
        pattern_min_sep_px=cfg.pattern_min_sep_px,
        pattern_dot_radius_px=cfg.pattern_dot_radius_px,
        pattern_dot_sigma_px=cfg.pattern_dot_sigma_px if dot_sigma_px is None else float(dot_sigma_px),
        dot_count=cfg.dot_count,
        energy=cfg.energy if projector_energy is None else float(projector_energy),
        color_rgb=cfg.color_rgb if projector_color_rgb is None else _normalize_color_rgb(projector_color_rgb),
        fov_h_deg=cfg.fov_h_deg if fov_x_override is None else float(fov_x_override),
        fov_v_deg=cfg.fov_v_deg if fov_y_override is None else float(fov_y_override),
        mount_frame=cfg.mount_frame,
        mount_mode=cfg.mount_mode,
        local_transform_blender_4x4_base=np.asarray(cfg.local_transform_blender_4x4_base, dtype=np.float64),
        local_transform_blender_4x4_final=np.asarray(final, dtype=np.float64),
        flip_u_profile_default=cfg.flip_u_profile_default,
        flip_v_profile_default=cfg.flip_v_profile_default,
        flip_u=bool(cfg.flip_u_profile_default or _override_value(overrides, "flip_u", False)),
        flip_v=bool(cfg.flip_v_profile_default or _override_value(overrides, "flip_v", False)),
    )


def _read_render_result_colors() -> Optional[np.ndarray]:
    img = bpy.data.images.get("Render Result")
    if img is None:
        return None
    width = int(img.size[0])
    height = int(img.size[1])
    if width <= 0 or height <= 0:
        return None
    try:
        pixels = np.asarray(img.pixels[:], dtype=np.float32)
    except Exception:
        return None
    if pixels.size != width * height * 4:
        return None
    rgba = pixels.reshape(height, width, 4)
    return np.asarray(rgba[:, :, :3], dtype=np.float32).copy()


def _unregister_output_key(output_key: str) -> None:
    if not GlobalStorage.is_in_storage("output"):
        return
    outputs = [
        item for item in GlobalStorage.get("output")
        if str(item.get("key", "")) != str(output_key)
    ]
    GlobalStorage.set("output", outputs)


def render_single_stream_with_projector(
    rs: RealSenseProfile,
    stream_name: str,
    world_pose: np.ndarray,
    mount_obj,
    mount_world_pose: np.ndarray,
    projector_cfg: ProjectorRuntimeConfig | Dict[str, Any],
    *,
    socket_name: str = "rs_projector_socket_runtime",
    output_dir: Optional[str] = None,
    file_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    bproc.utility.reset_keyframes()
    rs.set_bproc_intrinsics(stream_name)
    bproc.camera.add_camera_pose(np.asarray(world_pose, dtype=np.float64), frame=0)
    projector_util.animate_mount_from_cam2world(mount_obj, [np.asarray(mount_world_pose, dtype=np.float64)])
    projector, _pattern_img = projector_util.create_projector_from_runtime_config(
        mount_obj,
        projector_cfg,
        socket_name=socket_name,
    )
    render_output_key = None
    try:
        projector_util.activate_only_projector_light(projector)
        render_uid = uuid.uuid4().hex[:12]
        render_output_key = f"colors_active_stereo_{render_uid}"
        render_file_prefix = f"{file_prefix or 'rgb_'}{socket_name}_{stream_name.lower()}_{render_uid}_"
        render_kwargs: Dict[str, Any] = {
            "file_prefix": render_file_prefix,
            "output_key": render_output_key,
            "load_keys": {render_output_key, "depth"},
        }
        if bpy.context.scene.render.film_transparent:
            render_kwargs["keys_with_alpha_channel"] = {render_output_key}
        if output_dir is not None:
            render_kwargs["output_dir"] = output_dir
        data = bproc.renderer.render(**render_kwargs)
        file_colors = None
        if render_output_key in data and data[render_output_key]:
            file_colors = np.asarray(data[render_output_key][0]).copy()
        render_result_colors = _read_render_result_colors()
        if file_colors is not None:
            colors = file_colors
        elif render_result_colors is not None:
            colors = render_result_colors
        else:
            raise RuntimeError(
                "Active stereo render did not return color data from either the registered output "
                f"'{render_output_key}' or Blender Render Result."
            )
    finally:
        try:
            if render_output_key is not None:
                _unregister_output_key(render_output_key)
        except Exception:
            pass
        projector_util.cleanup_projector_pattern_images(projector)
        projector.delete()
    return {
        "colors": colors,
        "depth": None if "depth" not in data else np.asarray(data["depth"][0], dtype=np.float32).copy(),
    }


def render_active_stereo_pair(
    rs: RealSenseProfile,
    *,
    left_stream: str,
    right_stream: str,
    overrides: Optional[Any] = None,
    left_world_pose: Optional[np.ndarray] = None,
    mount_name: str = "rs_projector_mount_runtime",
    socket_name: str = "rs_projector_socket_runtime",
) -> Dict[str, Any]:
    left_pose = np.eye(4, dtype=np.float64) if left_world_pose is None else np.asarray(left_world_pose, dtype=np.float64)
    right_pose = get_stream_world_pose_from_anchor(rs, right_stream, left_stream, left_pose)
    mount_world_pose = get_mount_world_pose(
        rs,
        anchor_stream=left_stream,
        secondary_stream=right_stream,
        anchor_world_pose=left_pose,
    )

    render_uid = uuid.uuid4().hex[:12]
    scoped_mount_name = f"{mount_name}_{render_uid}"
    scoped_socket_name = f"{socket_name}_{render_uid}"
    mount = projector_util.get_or_create_mount_empty(scoped_mount_name)
    projector_cfg = resolve_projector_runtime_config(
        rs,
        overrides,
        fallback_left_stream=left_stream,
    )

    try:
        left_render = render_single_stream_with_projector(
            rs,
            left_stream,
            left_pose,
            mount,
            mount_world_pose,
            projector_cfg,
            socket_name=scoped_socket_name,
        )
        right_render = render_single_stream_with_projector(
            rs,
            right_stream,
            right_pose,
            mount,
            mount_world_pose,
            projector_cfg,
            socket_name=scoped_socket_name,
        )
    finally:
        socket_obj = bpy.data.objects.get(scoped_socket_name)
        if socket_obj is not None:
            bpy.data.objects.remove(socket_obj, do_unlink=True)
        mount_obj = bpy.data.objects.get(scoped_mount_name)
        if mount_obj is not None:
            bpy.data.objects.remove(mount_obj, do_unlink=True)

    return {
        "left_colors": left_render["colors"],
        "right_colors": right_render["colors"],
        "left_depth_m": left_render["depth"],
        "projector_cfg": projector_cfg.to_dict(),
        "projector_pattern_rgba": left_render["pattern_rgba"],
        "left_world_pose": left_pose,
        "right_world_pose": right_pose,
        "mount_world_pose": mount_world_pose,
    }


__all__ = [
    "ProjectorOverrides",
    "ProjectorRuntimeConfig",
    "matrix_from_translation_euler_xyz",
    "get_stream_world_pose_from_anchor",
    "get_mount_world_pose",
    "resolve_projector_runtime_config",
    "render_single_stream_with_projector",
    "render_active_stereo_pair",
]
