#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import blenderproc as bproc


import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import bpy
import cv2
import numpy as np
from mathutils import Euler, Matrix, Vector


REPO_ROOT = Path(__file__).resolve().parents[2]
BLENDFORGE_SRC = REPO_ROOT / "blendforge" / "src"
if str(BLENDFORGE_SRC) not in sys.path:
    sys.path.insert(0, str(BLENDFORGE_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blendforge.blender_runtime.camera.ProjectorPatternUtility import load_or_generate_projector_pattern
from blendforge.blender_runtime.camera.RealsenseProfileLoader import RealSenseProfile
from blendforge.blender_runtime.camera import RealsenseProjectorUtility as projector_util
from blendforge.blender_runtime.camera.ActiveStereoProjectorRuntime import (
    render_active_stereo_pair as shared_render_active_stereo_pair,
    resolve_projector_runtime_config as shared_resolve_projector_runtime_config,
)
from blendforge.blender_runtime.stereo.StereoMatching import stereo_global_matching_rectified
from blendforge.blender_runtime.stereo.StereoRectify import build_rectify_maps, rectify_pair
from blendforge.blender_runtime.stereo.ActiveStereoIRUtility import (
    build_rectify_from_rs as shared_build_rectify_from_rs,
    rgb_to_intensity_u8 as shared_rgb_to_intensity_u8,
    stereo_from_ir_pair as shared_stereo_from_ir_pair,
)


DEFAULT_LEFT_STREAM = "IR_LEFT"
DEFAULT_RIGHT_STREAM = "IR_RIGHT"
DEFAULT_RENDER_SAMPLES = 64
DEFAULT_PLANE_WIDTH_M = 3.5
DEFAULT_PLANE_HEIGHT_M = 3.5
DEFAULT_COMPARE_P_LO = 1.0
DEFAULT_COMPARE_P_HI = 99.0
DEFAULT_SUPPORT_BLUR_SIGMA_PX = 9.0
DEFAULT_SUPPORT_THRESHOLD_REL = 0.22
DEFAULT_AMBIENT_WORLD_STRENGTH = 0.004
DEFAULT_AMBIENT_LIGHT_ENERGY = 2.5
DEFAULT_AMBIENT_LIGHT_X = 0.0
DEFAULT_AMBIENT_LIGHT_Y = -0.08
DEFAULT_AMBIENT_LIGHT_Z = -0.35


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

    def has_transform_override(self) -> bool:
        return any(
            abs(v) > 1e-12
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
class AmbientConfig:
    enabled: bool = False
    world_strength: float = DEFAULT_AMBIENT_WORLD_STRENGTH
    light_energy: float = DEFAULT_AMBIENT_LIGHT_ENERGY
    light_x: float = DEFAULT_AMBIENT_LIGHT_X
    light_y: float = DEFAULT_AMBIENT_LIGHT_Y
    light_z: float = DEFAULT_AMBIENT_LIGHT_Z


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standalone RealSense-like IR stereo/projector calibration harness."
    )
    p.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=("render", "compare", "stereo_debug", "fov_sweep"),
        help="Execution mode.",
    )
    p.add_argument(
        "--camera_profile_json",
        type=str,
        required=True,
        help="Runtime RealSense camera/projector profile JSON.",
    )
    p.add_argument(
        "--session_dir",
        type=str,
        default=None,
        help="Path to wall_capture/session_* directory. Required for compare mode.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for debug outputs.",
    )
    p.add_argument(
        "--plane_distance_m",
        type=float,
        default=None,
        help="Plane distance from IR_LEFT camera center in meters.",
    )
    p.add_argument(
        "--plane_width_m",
        type=float,
        default=DEFAULT_PLANE_WIDTH_M,
        help="Plane width in meters.",
    )
    p.add_argument(
        "--plane_height_m",
        type=float,
        default=DEFAULT_PLANE_HEIGHT_M,
        help="Plane height in meters.",
    )
    p.add_argument(
        "--use_stage1_outputs",
        action="store_true",
        help="Use prepared projector_stage1 observations when available.",
    )
    p.add_argument(
        "--profile_stream_left",
        type=str,
        default=DEFAULT_LEFT_STREAM,
        help="Left stream name inside profile.",
    )
    p.add_argument(
        "--profile_stream_right",
        type=str,
        default=DEFAULT_RIGHT_STREAM,
        help="Right stream name inside profile.",
    )
    p.add_argument(
        "--render_samples",
        type=int,
        default=DEFAULT_RENDER_SAMPLES,
        help="Cycles samples for synthetic render.",
    )
    p.add_argument(
        "--compare_percentile_low",
        type=float,
        default=DEFAULT_COMPARE_P_LO,
        help="Low percentile for compare normalization.",
    )
    p.add_argument(
        "--compare_percentile_high",
        type=float,
        default=DEFAULT_COMPARE_P_HI,
        help="High percentile for compare normalization.",
    )
    p.add_argument("--proj_tx", type=float, default=0.0)
    p.add_argument("--proj_ty", type=float, default=0.0)
    p.add_argument("--proj_tz", type=float, default=0.0)
    p.add_argument("--proj_yaw_deg", type=float, default=0.0)
    p.add_argument("--proj_pitch_deg", type=float, default=0.0)
    p.add_argument("--proj_roll_deg", type=float, default=0.0)
    p.add_argument("--proj_fov_x_deg", type=float, default=None)
    p.add_argument("--proj_fov_y_deg", type=float, default=None)
    p.add_argument("--flip_u", action="store_true")
    p.add_argument("--flip_v", action="store_true")
    p.add_argument("--dot_sigma_px", type=float, default=None)
    p.add_argument("--projector_energy", type=float, default=None)
    p.add_argument("--ambient_enabled", action="store_true")
    p.add_argument("--ambient_world_strength", type=float, default=DEFAULT_AMBIENT_WORLD_STRENGTH)
    p.add_argument("--ambient_light_energy", type=float, default=DEFAULT_AMBIENT_LIGHT_ENERGY)
    p.add_argument("--ambient_light_x", type=float, default=DEFAULT_AMBIENT_LIGHT_X)
    p.add_argument("--ambient_light_y", type=float, default=DEFAULT_AMBIENT_LIGHT_Y)
    p.add_argument("--ambient_light_z", type=float, default=DEFAULT_AMBIENT_LIGHT_Z)
    p.add_argument("--proj_fov_x_deg_min", type=float, default=None)
    p.add_argument("--proj_fov_x_deg_max", type=float, default=None)
    p.add_argument("--proj_fov_x_deg_step", type=float, default=None)
    p.add_argument("--proj_fov_y_deg_min", type=float, default=None)
    p.add_argument("--proj_fov_y_deg_max", type=float, default=None)
    p.add_argument("--proj_fov_y_deg_step", type=float, default=None)
    parsed = p.parse_args(_strip_blender_args(argv))
    if parsed.mode == "compare" and not parsed.session_dir:
        p.error("--session_dir is required in compare mode")
    if parsed.mode == "fov_sweep":
        required = (
            parsed.proj_fov_x_deg_min,
            parsed.proj_fov_x_deg_max,
            parsed.proj_fov_x_deg_step,
            parsed.proj_fov_y_deg_min,
            parsed.proj_fov_y_deg_max,
            parsed.proj_fov_y_deg_step,
        )
        if any(v is None for v in required):
            p.error("fov_sweep requires all proj_fov_*_{min,max,step} arguments")
        if parsed.proj_fov_x_deg_step <= 0.0 or parsed.proj_fov_y_deg_step <= 0.0:
            p.error("fov_sweep steps must be > 0")
    return parsed


def _strip_blender_args(argv: list[str]) -> list[str]:
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return argv


def load_profile(profile_json: str) -> RealSenseProfile:
    return RealSenseProfile.from_json(profile_json)


def infer_plane_distance(
    args: argparse.Namespace,
    session_dir: Optional[Path],
) -> float:
    if args.plane_distance_m is not None:
        return float(args.plane_distance_m)
    if session_dir is None:
        raise ValueError("--plane_distance_m is required when session_dir is not provided")
    stage1_params = session_dir / "projector_stage1" / "camera_and_plane_params.json"
    if stage1_params.is_file():
        cfg = json.loads(stage1_params.read_text(encoding="utf-8"))
        for key in ("plane_distance_m", "wall_distance_m", "depth_plane_m"):
            if key in cfg:
                return float(cfg[key])
    raise ValueError(
        "Could not infer plane distance from session. Provide --plane_distance_m explicitly."
    )


def ensure_output_dir(path: str) -> Path:
    out_dir = Path(path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def configure_renderer(samples: int, ambient_cfg: AmbientConfig) -> None:
    bpy.context.scene.render.engine = "CYCLES"
    bproc.renderer.set_max_amount_of_samples(max(1, int(samples)))
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    scene = bpy.context.scene
    scene.render.film_transparent = False
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg = next((n for n in world.node_tree.nodes if n.type == "BACKGROUND"), None)
    if bg is not None:
        bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        bg.inputs["Strength"].default_value = (
            float(max(0.0, ambient_cfg.world_strength)) if ambient_cfg.enabled else 0.0
        )


def _cleanup_default_scene() -> None:
    for obj in list(bpy.data.objects):
        if obj.type in {"MESH", "LIGHT"} and obj.name in {"Cube", "Light"}:
            bpy.data.objects.remove(obj, do_unlink=True)


def build_ambient_config(args: argparse.Namespace) -> AmbientConfig:
    return AmbientConfig(
        enabled=bool(args.ambient_enabled),
        world_strength=float(args.ambient_world_strength),
        light_energy=float(args.ambient_light_energy),
        light_x=float(args.ambient_light_x),
        light_y=float(args.ambient_light_y),
        light_z=float(args.ambient_light_z),
    )


def setup_ambient_lighting(cfg: AmbientConfig):
    if not cfg.enabled:
        return None
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_energy(float(max(0.0, cfg.light_energy)))
    light.set_location([float(cfg.light_x), float(cfg.light_y), float(cfg.light_z)])
    light.set_color([1.0, 1.0, 1.0])
    return light


def create_neutral_plane_material(name: str = "calibration_plane_material"):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    node_tree = material.node_tree
    bsdf = next((n for n in node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        raise RuntimeError("Could not find Principled BSDF for calibration plane material.")
    bsdf.inputs["Base Color"].default_value = (0.82, 0.82, 0.82, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.92
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.05
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.05
    return material


def create_calibration_plane(
    *,
    distance_m: float,
    width_m: float,
    height_m: float,
):
    plane = bproc.object.create_primitive(
        "PLANE",
        scale=[float(width_m) * 0.5, float(height_m) * 0.5, 1.0],
        location=[0.0, 0.0, -float(distance_m)],
        rotation=[0.0, 0.0, 0.0],
    )
    plane.set_name("calibration_plane")
    material = create_neutral_plane_material()
    blender_plane = plane.blender_obj
    blender_plane.data.materials.clear()
    blender_plane.data.materials.append(material)
    return plane


def _matrix_from_translation_euler_xyz(
    tx: float,
    ty: float,
    tz: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    euler = Euler(
        (
            math.radians(float(roll_deg)),
            math.radians(float(pitch_deg)),
            math.radians(float(yaw_deg)),
        ),
        "XYZ",
    )
    rot = euler.to_matrix().to_4x4()
    tr = Matrix.Translation(Vector((float(tx), float(ty), float(tz))))
    return np.asarray((tr @ rot), dtype=np.float64)


def get_left_anchor_pose() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def get_stream_world_pose_from_left_anchor(
    rs: RealSenseProfile,
    stream_name: str,
    left_stream: str,
    left_world_pose: np.ndarray,
) -> np.ndarray:
    if stream_name == left_stream:
        return np.asarray(left_world_pose, dtype=np.float64).copy()
    return np.asarray(left_world_pose, dtype=np.float64) @ rs.get_T_blender(stream_name, left_stream)


def get_mount_world_pose(
    rs: RealSenseProfile,
    *,
    left_stream: str,
    right_stream: str,
    left_world_pose: np.ndarray,
) -> np.ndarray:
    mount_frame = rs.get_projector_mount_frame() if rs.has_projector() else left_stream
    if mount_frame == left_stream:
        return np.asarray(left_world_pose, dtype=np.float64).copy()
    if mount_frame == right_stream:
        return get_stream_world_pose_from_left_anchor(rs, right_stream, left_stream, left_world_pose)
    if rs.has_stream(mount_frame):
        return get_stream_world_pose_from_left_anchor(rs, mount_frame, left_stream, left_world_pose)
    if mount_frame == "COLOR" and rs.has_stream("COLOR"):
        return get_stream_world_pose_from_left_anchor(rs, "COLOR", left_stream, left_world_pose)
    if mount_frame == "DEPTH":
        return np.asarray(left_world_pose, dtype=np.float64).copy()
    return np.asarray(left_world_pose, dtype=np.float64).copy()


def build_rectify_from_rs(
    rs: RealSenseProfile,
    left: str,
    right: str,
    *,
    use_distortion: bool = False,
) -> Dict[str, Any]:
    return shared_build_rectify_from_rs(rs, left=left, right=right, use_distortion=use_distortion)


def _rgb_to_intensity_u8(img: np.ndarray) -> np.ndarray:
    return shared_rgb_to_intensity_u8(img, mode="bt601")


def apply_pattern_overrides(
    pattern_rgba: np.ndarray,
    *,
    flip_u: bool,
    flip_v: bool,
    dot_sigma_px: Optional[float],
) -> np.ndarray:
    out = np.asarray(pattern_rgba).copy()
    if flip_u:
        out = np.ascontiguousarray(np.flip(out, axis=1))
    if flip_v:
        out = np.ascontiguousarray(np.flip(out, axis=0))
    if dot_sigma_px is not None and float(dot_sigma_px) > 1e-8:
        sigma = float(dot_sigma_px)
        k = max(3, int(math.ceil(sigma * 6.0)))
        if k % 2 == 0:
            k += 1
        out = cv2.GaussianBlur(out, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    return out.astype(np.uint8, copy=False)


def resolve_projector_pattern_reference(
    rs: RealSenseProfile,
    pattern_path: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    if pattern_path in (None, ""):
        return None, None

    raw = Path(str(pattern_path)).expanduser()
    profile_base_dir = getattr(rs, "profile_base_dir", None)
    if raw.is_absolute() and raw.exists():
        return str(raw), None

    if profile_base_dir in (None, ""):
        return str(pattern_path), None

    base = Path(str(profile_base_dir)).expanduser()

    if raw.is_absolute():
        # Some profiles were already normalized into a broken absolute path like:
        #   .../projector_stage3/projector_stage2/projector_texture.png
        # while the real file lives one level выше at session/projector_stage2/...
        tail_candidates = []
        if len(raw.parts) >= 2:
            tail_candidates.append(Path(*raw.parts[-2:]))
        if len(raw.parts) >= 3:
            tail_candidates.append(Path(*raw.parts[-3:]))
        tail_candidates.append(Path(raw.name))

        for rel_tail in tail_candidates:
            direct = base / rel_tail
            if direct.exists():
                return str(direct), None
            for root in [*list(base.parents[:4])]:
                candidate = root / rel_tail
                if candidate.exists():
                    return str(candidate), None
        return str(pattern_path), None

    direct = base / raw
    if direct.exists():
        return str(pattern_path), str(base)

    # Stage3 exports may live in projector_stage3/ while the texture sits in a
    # sibling projector_stage2/ directory. Walk up a few ancestors and reuse the
    # first base directory where the same relative path resolves successfully.
    for root in [*list(base.parents[:4])]:
        candidate = root / raw
        if candidate.exists():
            return str(pattern_path), str(root)

    return str(pattern_path), str(base)


def resolve_projector_runtime_config(
    rs: RealSenseProfile,
    overrides: ProjectorOverrides,
    *,
    fallback_left_stream: str,
) -> Dict[str, Any]:
    return shared_resolve_projector_runtime_config(
        rs,
        overrides,
        fallback_left_stream=fallback_left_stream,
    ).to_dict()


def create_projector_for_debug(
    rs: RealSenseProfile,
    mount_obj,
    config: Dict[str, Any],
):
    return projector_util.create_projector_from_runtime_config(
        mount_obj,
        config,
        socket_name="rs_projector_socket_debug",
    )


def render_single_stream(
    rs: RealSenseProfile,
    stream_name: str,
    world_pose: np.ndarray,
    mount_obj,
    mount_world_pose: np.ndarray,
    projector_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    bproc.utility.reset_keyframes()
    rs.set_bproc_intrinsics(stream_name)
    bproc.camera.add_camera_pose(np.asarray(world_pose, dtype=np.float64), frame=0)
    projector_util.animate_mount_from_cam2world(mount_obj, [np.asarray(mount_world_pose, dtype=np.float64)])
    projector, pattern_img = create_projector_for_debug(rs, mount_obj, projector_cfg)
    try:
        data = bproc.renderer.render()
    finally:
        projector.delete()
    return {
        "colors": data["colors"][0],
        "depth": None if "depth" not in data else data["depth"][0],
        "pattern_rgba": pattern_img,
    }


def render_synthetic_pair(
    rs: RealSenseProfile,
    *,
    left_stream: str,
    right_stream: str,
    overrides: ProjectorOverrides,
) -> Dict[str, Any]:
    pair_render = shared_render_active_stereo_pair(
        rs,
        left_stream=left_stream,
        right_stream=right_stream,
        overrides=overrides,
        mount_name="rs_projector_mount_debug",
        socket_name="rs_projector_socket_debug",
    )
    left_ir = _rgb_to_intensity_u8(pair_render["left_colors"])
    right_ir = _rgb_to_intensity_u8(pair_render["right_colors"])
    left_lcn_u8 = build_synthetic_lcn_u8(left_ir)
    right_lcn_u8 = build_synthetic_lcn_u8(right_ir)
    left_support = build_synthetic_support_artifacts(left_ir)
    right_support = build_synthetic_support_artifacts(right_ir)

    return {
        "left_ir_u8": left_ir,
        "right_ir_u8": right_ir,
        "left_lcn_u8": left_lcn_u8,
        "right_lcn_u8": right_lcn_u8,
        "left_support_mask": left_support["mask"],
        "right_support_mask": right_support["mask"],
        "left_support_map": left_support["support_map"],
        "right_support_map": right_support["support_map"],
        "left_depth_m": pair_render["left_depth_m"],
        "projector_cfg": pair_render["projector_cfg"],
        "projector_pattern_rgba": pair_render["projector_pattern_rgba"],
        "left_world_pose": pair_render["left_world_pose"],
        "right_world_pose": pair_render["right_world_pose"],
        "mount_world_pose": pair_render["mount_world_pose"],
    }


def list_pngs(path: Path) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])


def read_gray_image(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def median_image_from_paths(paths: list[Path]) -> np.ndarray:
    if not paths:
        raise FileNotFoundError("No image files found for median computation.")
    frames = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frames.append(img.astype(np.float32))
    stack = np.stack(frames, axis=0)
    return np.median(stack, axis=0).astype(np.float32)


def load_real_observations_from_stage1(session_dir: Path) -> Dict[str, np.ndarray]:
    obs_path = session_dir / "projector_stage1" / "observations" / "pattern_observations.npz"
    if not obs_path.is_file():
        raise FileNotFoundError(f"Stage1 observation archive not found: {obs_path}")
    data = np.load(obs_path)
    obs_dir = session_dir / "projector_stage1" / "observations"
    left_lcn = read_gray_image(obs_dir / "left_diff_lcn.png")
    right_lcn = read_gray_image(obs_dir / "right_diff_lcn.png")
    left_mask = read_gray_image(obs_dir / "left_dense_support_mask.png")
    right_mask = read_gray_image(obs_dir / "right_dense_support_mask.png")
    return {
        "left_diff": np.asarray(data["diff_left"], dtype=np.float32),
        "right_diff": np.asarray(data["diff_right"], dtype=np.float32),
        "left_median_on": np.asarray(data["med_left_on"], dtype=np.uint8),
        "left_median_off": np.asarray(data["med_left_off"], dtype=np.uint8),
        "right_median_on": np.asarray(data["med_right_on"], dtype=np.uint8),
        "right_median_off": np.asarray(data["med_right_off"], dtype=np.uint8),
        "left_diff_lcn": None if left_lcn is None else np.asarray(left_lcn, dtype=np.uint8),
        "right_diff_lcn": None if right_lcn is None else np.asarray(right_lcn, dtype=np.uint8),
        "left_dense_support_mask": None if left_mask is None else np.asarray(left_mask > 0, dtype=bool),
        "right_dense_support_mask": None if right_mask is None else np.asarray(right_mask > 0, dtype=bool),
    }


def load_real_observations_from_raw_capture(session_dir: Path) -> Dict[str, np.ndarray]:
    left_on = median_image_from_paths(list_pngs(session_dir / "left_on"))
    left_off = median_image_from_paths(list_pngs(session_dir / "left_off"))
    right_on = median_image_from_paths(list_pngs(session_dir / "right_on"))
    right_off = median_image_from_paths(list_pngs(session_dir / "right_off"))
    obs = {
        "left_diff": left_on - left_off,
        "right_diff": right_on - right_off,
        "left_median_on": np.clip(left_on, 0.0, 255.0).astype(np.uint8),
        "left_median_off": np.clip(left_off, 0.0, 255.0).astype(np.uint8),
        "right_median_on": np.clip(right_on, 0.0, 255.0).astype(np.uint8),
        "right_median_off": np.clip(right_off, 0.0, 255.0).astype(np.uint8),
        "left_diff_lcn": None,
        "right_diff_lcn": None,
        "left_dense_support_mask": None,
        "right_dense_support_mask": None,
    }
    _merge_stage1_auxiliary_observations(obs, session_dir)
    return obs


def load_real_observations(session_dir: Path, use_stage1_outputs: bool) -> Dict[str, np.ndarray]:
    if use_stage1_outputs:
        stage1_path = session_dir / "projector_stage1" / "observations" / "pattern_observations.npz"
        if stage1_path.is_file():
            return load_real_observations_from_stage1(session_dir)
    return load_real_observations_from_raw_capture(session_dir)


def _merge_stage1_auxiliary_observations(obs: Dict[str, Any], session_dir: Path) -> None:
    obs_dir = session_dir / "projector_stage1" / "observations"
    left_lcn = read_gray_image(obs_dir / "left_diff_lcn.png")
    right_lcn = read_gray_image(obs_dir / "right_diff_lcn.png")
    left_mask = read_gray_image(obs_dir / "left_dense_support_mask.png")
    right_mask = read_gray_image(obs_dir / "right_dense_support_mask.png")
    if left_lcn is not None:
        obs["left_diff_lcn"] = np.asarray(left_lcn, dtype=np.uint8)
    if right_lcn is not None:
        obs["right_diff_lcn"] = np.asarray(right_lcn, dtype=np.uint8)
    if left_mask is not None:
        obs["left_dense_support_mask"] = np.asarray(left_mask > 0, dtype=bool)
    if right_mask is not None:
        obs["right_dense_support_mask"] = np.asarray(right_mask > 0, dtype=bool)


def normalize_signal_for_compare(
    img: np.ndarray,
    *,
    percentile_low: float,
    percentile_high: float,
) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.percentile(x, percentile_low))
    hi = float(np.percentile(x, percentile_high))
    if hi <= lo + 1e-6:
        hi = lo + 1e-6
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def resize_like(src: np.ndarray, ref_shape: Tuple[int, int]) -> np.ndarray:
    h, w = int(ref_shape[0]), int(ref_shape[1])
    if src.shape[:2] == (h, w):
        return src
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)


def compute_ncc(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    aa = aa - float(np.mean(aa))
    bb = bb - float(np.mean(bb))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def compute_peak_overlap(a: np.ndarray, b: np.ndarray, percentile: float = 99.5) -> Dict[str, float]:
    thr_a = float(np.percentile(a, percentile))
    thr_b = float(np.percentile(b, percentile))
    mask_a = a >= thr_a
    mask_b = b >= thr_b
    inter = int(np.count_nonzero(mask_a & mask_b))
    union = int(np.count_nonzero(mask_a | mask_b))
    a_count = int(np.count_nonzero(mask_a))
    b_count = int(np.count_nonzero(mask_b))
    return {
        "percentile": float(percentile),
        "iou": 0.0 if union == 0 else float(inter / union),
        "precision_vs_real": 0.0 if a_count == 0 else float(inter / a_count),
        "recall_vs_real": 0.0 if b_count == 0 else float(inter / b_count),
    }


def build_synthetic_lcn_u8(img_u8: np.ndarray) -> np.ndarray:
    x = np.asarray(img_u8, dtype=np.float32) / 255.0
    mu = cv2.GaussianBlur(x, (0, 0), sigmaX=5.0, sigmaY=5.0, borderType=cv2.BORDER_REPLICATE)
    mu2 = cv2.GaussianBlur(x * x, (0, 0), sigmaX=5.0, sigmaY=5.0, borderType=cv2.BORDER_REPLICATE)
    sigma = np.sqrt(np.maximum(mu2 - mu * mu, 1e-6))
    lcn = (x - mu) / sigma
    return to_u8_preview(lcn, percentile_low=1.0, percentile_high=99.0)


def build_synthetic_support_artifacts(
    img_u8: np.ndarray,
    *,
    blur_sigma_px: float = DEFAULT_SUPPORT_BLUR_SIGMA_PX,
    threshold_rel: float = DEFAULT_SUPPORT_THRESHOLD_REL,
) -> Dict[str, np.ndarray]:
    norm = normalize_signal_for_compare(img_u8, percentile_low=1.0, percentile_high=99.5)
    support_map = cv2.GaussianBlur(
        norm.astype(np.float32),
        (0, 0),
        sigmaX=float(blur_sigma_px),
        sigmaY=float(blur_sigma_px),
        borderType=cv2.BORDER_REPLICATE,
    )
    max_v = float(np.max(support_map)) if support_map.size else 0.0
    if max_v <= 1e-8:
        mask = np.zeros_like(support_map, dtype=bool)
    else:
        threshold = float(max(0.04, threshold_rel * max_v))
        mask = support_map >= threshold
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_u8 = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel_open)
        mask = mask_u8 > 0
    return {
        "support_map": support_map.astype(np.float32),
        "mask": mask,
    }


def mask_to_u8(mask: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(mask, dtype=bool), 255, 0).astype(np.uint8)


def build_mask_overlay_rgb(synth_mask: np.ndarray, real_mask: np.ndarray) -> np.ndarray:
    synth_u8 = mask_to_u8(synth_mask)
    real_u8 = mask_to_u8(real_mask)
    return np.stack([synth_u8, real_u8, np.zeros_like(synth_u8)], axis=2)


def compute_mask_metrics(synth_mask: np.ndarray, real_mask: np.ndarray) -> Dict[str, float]:
    synth = np.asarray(synth_mask, dtype=bool)
    real = np.asarray(real_mask, dtype=bool)
    inter = int(np.count_nonzero(synth & real))
    synth_count = int(np.count_nonzero(synth))
    real_count = int(np.count_nonzero(real))
    union = int(np.count_nonzero(synth | real))
    return {
        "iou": 0.0 if union == 0 else float(inter / union),
        "dice": 0.0 if (synth_count + real_count) == 0 else float((2.0 * inter) / (synth_count + real_count)),
        "precision": 0.0 if synth_count == 0 else float(inter / synth_count),
        "recall": 0.0 if real_count == 0 else float(inter / real_count),
    }


def resize_mask_like(src_mask: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    src = np.asarray(src_mask, dtype=np.uint8)
    if src.shape[:2] == ref_mask.shape[:2]:
        return src > 0
    resized = cv2.resize(src, (int(ref_mask.shape[1]), int(ref_mask.shape[0])), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def build_overlay_rgb(synth_norm: np.ndarray, real_norm: np.ndarray) -> np.ndarray:
    r = np.clip(synth_norm * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    g = np.clip(real_norm * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    b = np.zeros_like(r)
    return np.stack([r, g, b], axis=2)


def compare_images(
    synthetic_img: np.ndarray,
    real_img: np.ndarray,
    *,
    percentile_low: float,
    percentile_high: float,
) -> Dict[str, Any]:
    synth = resize_like(np.asarray(synthetic_img, dtype=np.float32), real_img.shape[:2])
    real = np.asarray(real_img, dtype=np.float32)
    synth_norm = normalize_signal_for_compare(
        synth,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )
    real_norm = normalize_signal_for_compare(
        real,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )
    diff = synth_norm - real_norm
    abs_diff = np.abs(diff)
    return {
        "synthetic_norm": synth_norm,
        "real_norm": real_norm,
        "abs_error": abs_diff,
        "overlay_rgb": build_overlay_rgb(synth_norm, real_norm),
        "metrics": {
            "mse": float(np.mean(diff * diff)),
            "mae": float(np.mean(abs_diff)),
            "ncc": compute_ncc(synth_norm, real_norm),
            "peak_overlap": compute_peak_overlap(synth_norm, real_norm),
        },
    }


def compare_render_to_real(
    render_data: Dict[str, Any],
    real_obs: Dict[str, Any],
    *,
    percentile_low: float,
    percentile_high: float,
) -> Dict[str, Any]:
    diff_left = compare_images(
        render_data["left_ir_u8"],
        real_obs["left_diff"],
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )
    diff_right = compare_images(
        render_data["right_ir_u8"],
        real_obs["right_diff"],
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )
    result: Dict[str, Any] = {
        "diff": {
            "left": diff_left,
            "right": diff_right,
            "metrics": {
                "mean_mse": float(0.5 * (diff_left["metrics"]["mse"] + diff_right["metrics"]["mse"])),
                "mean_mae": float(0.5 * (diff_left["metrics"]["mae"] + diff_right["metrics"]["mae"])),
                "mean_ncc": float(0.5 * (diff_left["metrics"]["ncc"] + diff_right["metrics"]["ncc"])),
            },
        },
        "lcn": None,
        "support_mask": None,
    }
    if real_obs.get("left_diff_lcn") is not None and real_obs.get("right_diff_lcn") is not None:
        lcn_left = compare_images(
            render_data["left_lcn_u8"],
            real_obs["left_diff_lcn"],
            percentile_low=percentile_low,
            percentile_high=percentile_high,
        )
        lcn_right = compare_images(
            render_data["right_lcn_u8"],
            real_obs["right_diff_lcn"],
            percentile_low=percentile_low,
            percentile_high=percentile_high,
        )
        result["lcn"] = {
            "left": lcn_left,
            "right": lcn_right,
            "metrics": {
                "mean_mse": float(0.5 * (lcn_left["metrics"]["mse"] + lcn_right["metrics"]["mse"])),
                "mean_mae": float(0.5 * (lcn_left["metrics"]["mae"] + lcn_right["metrics"]["mae"])),
                "mean_ncc": float(0.5 * (lcn_left["metrics"]["ncc"] + lcn_right["metrics"]["ncc"])),
            },
        }
    if real_obs.get("left_dense_support_mask") is not None and real_obs.get("right_dense_support_mask") is not None:
        left_synth_mask = resize_mask_like(render_data["left_support_mask"], real_obs["left_dense_support_mask"])
        right_synth_mask = resize_mask_like(render_data["right_support_mask"], real_obs["right_dense_support_mask"])
        left_mask_metrics = compute_mask_metrics(
            left_synth_mask,
            real_obs["left_dense_support_mask"],
        )
        right_mask_metrics = compute_mask_metrics(
            right_synth_mask,
            real_obs["right_dense_support_mask"],
        )
        result["support_mask"] = {
            "left": left_mask_metrics,
            "right": right_mask_metrics,
            "mean_iou": float(0.5 * (left_mask_metrics["iou"] + right_mask_metrics["iou"])),
            "mean_dice": float(0.5 * (left_mask_metrics["dice"] + right_mask_metrics["dice"])),
            "overlay_left": build_mask_overlay_rgb(left_synth_mask, real_obs["left_dense_support_mask"]),
            "overlay_right": build_mask_overlay_rgb(right_synth_mask, real_obs["right_dense_support_mask"]),
        }
    return result


def save_u8_png(path: Path, img: np.ndarray) -> None:
    arr = np.asarray(img)
    if arr.ndim == 2:
        data = arr
    elif arr.ndim == 3 and arr.shape[2] == 3:
        data = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        data = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    else:
        raise ValueError(f"Unsupported image shape for save_u8_png: {arr.shape}")
    cv2.imwrite(str(path), data)


def to_u8_preview(
    img: np.ndarray,
    *,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    norm = normalize_signal_for_compare(
        img,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )
    return np.clip(norm * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def save_npy(path: Path, arr: np.ndarray) -> None:
    np.save(path, np.asarray(arr))


def serialize(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]
    return obj


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(serialize(payload), indent=2, ensure_ascii=True), encoding="utf-8")


def save_render_outputs(
    out_dir: Path,
    render_data: Dict[str, Any],
) -> None:
    save_u8_png(out_dir / "synthetic_left.png", render_data["left_ir_u8"])
    save_u8_png(out_dir / "synthetic_right.png", render_data["right_ir_u8"])
    save_u8_png(out_dir / "synthetic_left_lcn.png", render_data["left_lcn_u8"])
    save_u8_png(out_dir / "synthetic_right_lcn.png", render_data["right_lcn_u8"])
    save_u8_png(out_dir / "synthetic_left_support_mask.png", mask_to_u8(render_data["left_support_mask"]))
    save_u8_png(out_dir / "synthetic_right_support_mask.png", mask_to_u8(render_data["right_support_mask"]))
    save_u8_png(out_dir / "synthetic_left_support_map.png", to_u8_preview(render_data["left_support_map"], percentile_low=0.0, percentile_high=99.5))
    save_u8_png(out_dir / "synthetic_right_support_map.png", to_u8_preview(render_data["right_support_map"], percentile_low=0.0, percentile_high=99.5))
    save_u8_png(out_dir / "projector_pattern.png", render_data["projector_pattern_rgba"])
    if render_data["left_depth_m"] is not None:
        save_npy(out_dir / "synthetic_left_depth.npy", np.asarray(render_data["left_depth_m"], dtype=np.float32))
        save_u8_png(
            out_dir / "synthetic_left_depth_preview.png",
            to_u8_preview(np.asarray(render_data["left_depth_m"], dtype=np.float32)),
        )


def save_compare_outputs(
    out_dir: Path,
    real_obs: Dict[str, np.ndarray],
    compare_bundle: Dict[str, Any],
) -> None:
    save_u8_png(out_dir / "real_left_median_on.png", real_obs["left_median_on"])
    save_u8_png(out_dir / "real_left_median_off.png", real_obs["left_median_off"])
    save_u8_png(out_dir / "real_right_median_on.png", real_obs["right_median_on"])
    save_u8_png(out_dir / "real_right_median_off.png", real_obs["right_median_off"])
    save_u8_png(out_dir / "real_left_diff.png", to_u8_preview(real_obs["left_diff"]))
    save_u8_png(out_dir / "real_right_diff.png", to_u8_preview(real_obs["right_diff"]))
    diff_left = compare_bundle["diff"]["left"]
    diff_right = compare_bundle["diff"]["right"]
    save_u8_png(out_dir / "error_left.png", to_u8_preview(diff_left["abs_error"], percentile_low=0.0, percentile_high=99.0))
    save_u8_png(out_dir / "error_right.png", to_u8_preview(diff_right["abs_error"], percentile_low=0.0, percentile_high=99.0))
    save_u8_png(out_dir / "overlay_left.png", diff_left["overlay_rgb"])
    save_u8_png(out_dir / "overlay_right.png", diff_right["overlay_rgb"])
    save_u8_png(out_dir / "synthetic_left_compare_norm.png", to_u8_preview(diff_left["synthetic_norm"], percentile_low=0.0, percentile_high=100.0))
    save_u8_png(out_dir / "synthetic_right_compare_norm.png", to_u8_preview(diff_right["synthetic_norm"], percentile_low=0.0, percentile_high=100.0))
    save_u8_png(out_dir / "real_left_compare_norm.png", to_u8_preview(diff_left["real_norm"], percentile_low=0.0, percentile_high=100.0))
    save_u8_png(out_dir / "real_right_compare_norm.png", to_u8_preview(diff_right["real_norm"], percentile_low=0.0, percentile_high=100.0))
    if real_obs.get("left_diff_lcn") is not None and real_obs.get("right_diff_lcn") is not None and compare_bundle.get("lcn") is not None:
        lcn_left = compare_bundle["lcn"]["left"]
        lcn_right = compare_bundle["lcn"]["right"]
        save_u8_png(out_dir / "real_left_diff_lcn.png", np.asarray(real_obs["left_diff_lcn"], dtype=np.uint8))
        save_u8_png(out_dir / "real_right_diff_lcn.png", np.asarray(real_obs["right_diff_lcn"], dtype=np.uint8))
        save_u8_png(out_dir / "error_left_lcn.png", to_u8_preview(lcn_left["abs_error"], percentile_low=0.0, percentile_high=99.0))
        save_u8_png(out_dir / "error_right_lcn.png", to_u8_preview(lcn_right["abs_error"], percentile_low=0.0, percentile_high=99.0))
        save_u8_png(out_dir / "overlay_left_lcn.png", lcn_left["overlay_rgb"])
        save_u8_png(out_dir / "overlay_right_lcn.png", lcn_right["overlay_rgb"])
    if real_obs.get("left_dense_support_mask") is not None and real_obs.get("right_dense_support_mask") is not None and compare_bundle.get("support_mask") is not None:
        save_u8_png(out_dir / "real_left_dense_support_mask.png", mask_to_u8(real_obs["left_dense_support_mask"]))
        save_u8_png(out_dir / "real_right_dense_support_mask.png", mask_to_u8(real_obs["right_dense_support_mask"]))
        save_u8_png(out_dir / "support_overlay_left.png", compare_bundle["support_mask"]["overlay_left"])
        save_u8_png(out_dir / "support_overlay_right.png", compare_bundle["support_mask"]["overlay_right"])


def run_stereo_debug(
    rs: RealSenseProfile,
    *,
    left_stream: str,
    right_stream: str,
    left_ir_u8: np.ndarray,
    right_ir_u8: np.ndarray,
    left_depth_gt_m: Optional[np.ndarray],
    plane_distance_m: float,
) -> Dict[str, Any]:
    return shared_stereo_from_ir_pair(
        rs,
        left_stream=left_stream,
        right_stream=right_stream,
        left_ir_u8=left_ir_u8,
        right_ir_u8=right_ir_u8,
        left_depth_gt_m=left_depth_gt_m,
        plane_distance_m=plane_distance_m,
        use_distortion=False,
    )


def save_stereo_outputs(out_dir: Path, stereo_data: Dict[str, Any]) -> None:
    save_u8_png(out_dir / "synthetic_left_rect.png", stereo_data["left_rect_u8"])
    save_u8_png(out_dir / "synthetic_right_rect.png", stereo_data["right_rect_u8"])
    save_npy(out_dir / "synthetic_disp.npy", stereo_data["disp_rect_px"])
    save_npy(out_dir / "synthetic_depth.npy", stereo_data["depth_rect_m"])
    save_u8_png(out_dir / "synthetic_disp_preview.png", to_u8_preview(stereo_data["disp_rect_px"]))
    depth = np.asarray(stereo_data["depth_rect_m"], dtype=np.float32)
    inv_depth = np.where(depth > 0.0, 1.0 / np.maximum(depth, 1e-6), 0.0)
    save_u8_png(out_dir / "synthetic_depth_preview.png", to_u8_preview(inv_depth))


def build_run_summary(
    args: argparse.Namespace,
    rs: RealSenseProfile,
    render_data: Dict[str, Any],
    plane_distance_m: float,
) -> Dict[str, Any]:
    left_stream = args.profile_stream_left
    right_stream = args.profile_stream_right
    return {
        "mode": args.mode,
        "camera_profile_json": str(Path(args.camera_profile_json).resolve()),
        "session_dir": None if args.session_dir is None else str(Path(args.session_dir).resolve()),
        "plane": {
            "distance_m": float(plane_distance_m),
            "width_m": float(args.plane_width_m),
            "height_m": float(args.plane_height_m),
        },
        "streams": {
            "left": left_stream,
            "right": right_stream,
            "left_K": np.asarray(rs.get_stream(left_stream).K, dtype=np.float64),
            "right_K": np.asarray(rs.get_stream(right_stream).K, dtype=np.float64),
            "baseline_m": float(rs.baseline_m),
            "T_right_from_left_cv": np.asarray(rs.get_T_cv(left_stream, right_stream), dtype=np.float64),
        },
        "projector": render_data["projector_cfg"],
        "camera_world_poses": {
            "left": np.asarray(render_data["left_world_pose"], dtype=np.float64),
            "right": np.asarray(render_data["right_world_pose"], dtype=np.float64),
            "mount": np.asarray(render_data["mount_world_pose"], dtype=np.float64),
        },
        "overrides": {
            "proj_tx": float(args.proj_tx),
            "proj_ty": float(args.proj_ty),
            "proj_tz": float(args.proj_tz),
            "proj_yaw_deg": float(args.proj_yaw_deg),
            "proj_pitch_deg": float(args.proj_pitch_deg),
            "proj_roll_deg": float(args.proj_roll_deg),
            "proj_fov_x_deg": args.proj_fov_x_deg,
            "proj_fov_y_deg": args.proj_fov_y_deg,
            "flip_u": bool(args.flip_u),
            "flip_v": bool(args.flip_v),
            "dot_sigma_px": args.dot_sigma_px,
            "projector_energy": args.projector_energy,
        },
        "ambient": {
            "enabled": bool(args.ambient_enabled),
            "world_strength": float(args.ambient_world_strength),
            "light_energy": float(args.ambient_light_energy),
            "light_x": float(args.ambient_light_x),
            "light_y": float(args.ambient_light_y),
            "light_z": float(args.ambient_light_z),
        },
    }


def build_compare_summary(
    run_summary: Dict[str, Any],
    compare_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    diff_payload = {
        "left": compare_bundle["diff"]["left"]["metrics"],
        "right": compare_bundle["diff"]["right"]["metrics"],
        **compare_bundle["diff"]["metrics"],
    }
    summary = {
        **run_summary,
        "diff_metrics": diff_payload,
        "compare_metrics": diff_payload,
    }
    if compare_bundle.get("lcn") is not None:
        summary["lcn_metrics"] = {
            "left": compare_bundle["lcn"]["left"]["metrics"],
            "right": compare_bundle["lcn"]["right"]["metrics"],
            **compare_bundle["lcn"]["metrics"],
        }
    if compare_bundle.get("support_mask") is not None:
        summary["support_mask_metrics"] = {
            "left": compare_bundle["support_mask"]["left"],
            "right": compare_bundle["support_mask"]["right"],
            "mean_iou": compare_bundle["support_mask"]["mean_iou"],
            "mean_dice": compare_bundle["support_mask"]["mean_dice"],
        }
    return summary


def run_single_configuration(
    *,
    args: argparse.Namespace,
    rs: RealSenseProfile,
    out_dir: Path,
    plane_distance_m: float,
    overrides: ProjectorOverrides,
    real_obs: Optional[Dict[str, Any]],
    enable_stereo_debug: bool,
) -> Dict[str, Any]:
    render_data = render_synthetic_pair(
        rs,
        left_stream=args.profile_stream_left,
        right_stream=args.profile_stream_right,
        overrides=overrides,
    )
    save_render_outputs(out_dir, render_data)
    run_summary = build_run_summary(args, rs, render_data, plane_distance_m)
    save_json(out_dir / "render_config_used.json", run_summary)

    compare_summary = None
    compare_bundle = None
    if real_obs is not None:
        compare_bundle = compare_render_to_real(
            render_data,
            real_obs,
            percentile_low=float(args.compare_percentile_low),
            percentile_high=float(args.compare_percentile_high),
        )
        save_compare_outputs(out_dir, real_obs, compare_bundle)
        compare_summary = build_compare_summary(run_summary, compare_bundle)
        save_json(out_dir / "compare_summary.json", compare_summary)

    stereo_summary = None
    if enable_stereo_debug:
        stereo_data = run_stereo_debug(
            rs,
            left_stream=args.profile_stream_left,
            right_stream=args.profile_stream_right,
            left_ir_u8=render_data["left_ir_u8"],
            right_ir_u8=render_data["right_ir_u8"],
            left_depth_gt_m=render_data["left_depth_m"],
            plane_distance_m=plane_distance_m,
        )
        save_stereo_outputs(out_dir, stereo_data)
        stereo_summary = {
            **run_summary,
            "stereo": {
                "depth_stats": stereo_data["depth_stats"],
                "rectify_cfg": stereo_data["rectify_cfg"],
            },
        }
        save_json(out_dir / "stereo_summary.json", stereo_summary)

    return {
        "run_summary": run_summary,
        "compare_summary": compare_summary,
        "compare_bundle": compare_bundle,
        "stereo_summary": stereo_summary,
    }


def frange_inclusive(start: float, stop: float, step: float) -> list[float]:
    values = []
    current = float(start)
    stop = float(stop)
    step = float(step)
    eps = abs(step) * 1e-9 + 1e-12
    while current <= stop + eps:
        values.append(round(current, 10))
        current += step
    return values


def format_fov_dirname(fov_x_deg: float, fov_y_deg: float) -> str:
    fx = f"{float(fov_x_deg):.3f}".replace("-", "m").replace(".", "p")
    fy = f"{float(fov_y_deg):.3f}".replace("-", "m").replace(".", "p")
    return f"fov_x_{fx}__fov_y_{fy}"


def sweep_row_from_result(
    *,
    fov_x_deg: float,
    fov_y_deg: float,
    variant_dir: Path,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "fov_x_deg": float(fov_x_deg),
        "fov_y_deg": float(fov_y_deg),
        "output_dir": str(variant_dir),
    }
    compare_summary = result.get("compare_summary")
    if compare_summary is not None:
        diff_metrics = compare_summary.get("diff_metrics", {})
        row["diff_mean_mse"] = diff_metrics.get("mean_mse")
        row["diff_mean_mae"] = diff_metrics.get("mean_mae")
        row["diff_mean_ncc"] = diff_metrics.get("mean_ncc")
        if "left" in diff_metrics and "right" in diff_metrics:
            row["diff_left_ncc"] = diff_metrics["left"].get("ncc")
            row["diff_right_ncc"] = diff_metrics["right"].get("ncc")
        support_metrics = compare_summary.get("support_mask_metrics")
        if support_metrics is not None:
            row["support_mean_iou"] = support_metrics.get("mean_iou")
            row["support_mean_dice"] = support_metrics.get("mean_dice")
            row["support_left_iou"] = support_metrics["left"].get("iou")
            row["support_right_iou"] = support_metrics["right"].get("iou")
        lcn_metrics = compare_summary.get("lcn_metrics")
        if lcn_metrics is not None:
            row["lcn_mean_ncc"] = lcn_metrics.get("mean_ncc")
            row["lcn_left_ncc"] = lcn_metrics["left"].get("ncc")
            row["lcn_right_ncc"] = lcn_metrics["right"].get("ncc")
    return row


def save_sweep_summary_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: serialize(v) for k, v in row.items()})


def run_fov_sweep(
    *,
    args: argparse.Namespace,
    rs: RealSenseProfile,
    out_dir: Path,
    plane_distance_m: float,
    base_overrides: ProjectorOverrides,
    real_obs: Optional[Dict[str, Any]],
) -> None:
    x_values = frange_inclusive(args.proj_fov_x_deg_min, args.proj_fov_x_deg_max, args.proj_fov_x_deg_step)
    y_values = frange_inclusive(args.proj_fov_y_deg_min, args.proj_fov_y_deg_max, args.proj_fov_y_deg_step)
    rows = []
    for fov_x_deg in x_values:
        for fov_y_deg in y_values:
            variant_dir = out_dir / format_fov_dirname(fov_x_deg, fov_y_deg)
            variant_dir.mkdir(parents=True, exist_ok=True)
            variant_overrides = replace(
                base_overrides,
                proj_fov_x_deg=float(fov_x_deg),
                proj_fov_y_deg=float(fov_y_deg),
            )
            result = run_single_configuration(
                args=args,
                rs=rs,
                out_dir=variant_dir,
                plane_distance_m=plane_distance_m,
                overrides=variant_overrides,
                real_obs=real_obs,
                enable_stereo_debug=False,
            )
            rows.append(
                sweep_row_from_result(
                    fov_x_deg=float(fov_x_deg),
                    fov_y_deg=float(fov_y_deg),
                    variant_dir=variant_dir,
                    result=result,
                )
            )
    save_json(
        out_dir / "sweep_summary.json",
        {
            "mode": "fov_sweep",
            "camera_profile_json": str(Path(args.camera_profile_json).resolve()),
            "session_dir": None if args.session_dir is None else str(Path(args.session_dir).resolve()),
            "rows": rows,
        },
    )
    save_sweep_summary_csv(out_dir / "sweep_summary.csv", rows)


def main(argv: Optional[list[str]] = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)

    bproc.init()
    bproc.utility.reset_keyframes()
    _cleanup_default_scene()

    out_dir = ensure_output_dir(args.output_dir)
    session_dir = None if args.session_dir is None else Path(args.session_dir).resolve()
    rs = load_profile(args.camera_profile_json)
    plane_distance_m = infer_plane_distance(args, session_dir)
    ambient_cfg = build_ambient_config(args)
    overrides = ProjectorOverrides(
        proj_tx=float(args.proj_tx),
        proj_ty=float(args.proj_ty),
        proj_tz=float(args.proj_tz),
        proj_yaw_deg=float(args.proj_yaw_deg),
        proj_pitch_deg=float(args.proj_pitch_deg),
        proj_roll_deg=float(args.proj_roll_deg),
        proj_fov_x_deg=args.proj_fov_x_deg,
        proj_fov_y_deg=args.proj_fov_y_deg,
        flip_u=bool(args.flip_u),
        flip_v=bool(args.flip_v),
        dot_sigma_px=args.dot_sigma_px,
        projector_energy=args.projector_energy,
    )

    if not rs.has_stream(args.profile_stream_left):
        raise KeyError(f"Left stream '{args.profile_stream_left}' not found in profile.")
    if not rs.has_stream(args.profile_stream_right):
        raise KeyError(f"Right stream '{args.profile_stream_right}' not found in profile.")

    configure_renderer(args.render_samples, ambient_cfg)
    create_calibration_plane(
        distance_m=plane_distance_m,
        width_m=float(args.plane_width_m),
        height_m=float(args.plane_height_m),
    )
    setup_ambient_lighting(ambient_cfg)

    real_obs: Optional[Dict[str, Any]] = None
    if session_dir is not None and (args.mode in {"compare", "fov_sweep"}):
        real_obs = load_real_observations(session_dir, bool(args.use_stage1_outputs))

    if args.mode == "fov_sweep":
        run_fov_sweep(
            args=args,
            rs=rs,
            out_dir=out_dir,
            plane_distance_m=plane_distance_m,
            base_overrides=overrides,
            real_obs=real_obs,
        )
        return

    run_single_configuration(
        args=args,
        rs=rs,
        out_dir=out_dir,
        plane_distance_m=plane_distance_m,
        overrides=overrides,
        real_obs=real_obs,
        enable_stereo_debug=(args.mode == "stereo_debug"),
    )


if __name__ == "__main__":
    main()
