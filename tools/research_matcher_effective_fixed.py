#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import blenderproc as bproc
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
BLENDFORGE_SRC = REPO_ROOT / "blendforge" / "src"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BLENDFORGE_SRC) not in sys.path:
    sys.path.insert(0, str(BLENDFORGE_SRC))

from blendforge.host.FiletoDict import Config
from blendforge.blender_runtime.CustomLightSetting import TemperatureToRGBConverter
from blendforge.blender_runtime.utils import build_lookat_pose_cam
from blendforge.blender_runtime.camera.RealsenseProfileLoader import RealSenseProfile
from blendforge.blender_runtime.camera.ActiveStereoProjectorRuntime import (
    ProjectorOverrides,
    render_active_stereo_pair,
)
from blendforge.blender_runtime.stereo.ActiveStereoIRUtility import (
    rgb_to_intensity_u8,
    stereo_from_ir_pair,
)
from blendforge.blender_runtime.stereo.StereoRectify import build_rectify_maps
from blendforge.blender_runtime.stereo.utils.PadCropUtility import rectify_single_channel
from single_depth_gen_effective_projector import (
    DEFAULT_PROFILE_JSON,
    _ensure_output_dir,
    _save_png,
    _depth_preview,
    _disp_preview,
    _prepare_objects,
    _sample_effective_energy_config,
    _capture_render_lighting_state,
    _apply_lighting_energy,
    _restore_render_lighting_state,
)


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Fixed-scene matcher research for the effective projector pipeline."
    )
    p.add_argument("--config_file", type=str, required=True)
    p.add_argument("--camera_profile_json", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--texture_name", type=str, default=None)
    return p.parse_args(argv)


def _cfg_get(section, key: str, default):
    if section is None:
        return default
    if isinstance(section, dict):
        return section.get(key, default)
    return getattr(section, key, default)


def _get_matcher_section(cfg):
    return getattr(cfg, "matcher_data", getattr(cfg, "matcher_research", None))


def _sample_scalar_or_range(value, *, cast):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return cast(value)

    vals = list(value)
    if len(vals) == 0:
        raise ValueError("Numeric range must not be empty")
    if len(vals) == 1:
        return cast(vals[0])

    lo, hi = vals[0], vals[1]
    if cast is int:
        lo = int(round(lo))
        hi = int(round(hi))
        if hi < lo:
            lo, hi = hi, lo
        return int(np.random.randint(lo, hi + 1))

    lo = float(lo)
    hi = float(hi)
    if hi < lo:
        lo, hi = hi, lo
    return cast(np.random.uniform(lo, hi))


def _normalize_matcher_value(name: str, value):
    if value is None:
        return None

    if name == "num_disparities":
        v = max(16, int(round(value)))
        return int(((v + 15) // 16) * 16)

    if name == "block_size":
        v = max(1, int(round(value)))
        if v % 2 == 0:
            v += 1
        return int(v)

    if name in {
        "min_disparity",
        "fill_iters",
        "uniqueness_ratio",
        "disp12_max_diff",
        "pre_filter_cap",
    }:
        return int(round(value))

    if name in {
        "depth_min",
        "depth_max",
        "lr_thresh_px",
        "lr_min_keep_ratio",
        "p1_scale",
        "p2_scale",
    }:
        return float(value)

    return value


def _matcher_numeric_param(section, key: str, default, *, cast):
    raw = _cfg_get(section, key, default)
    sampled = _sample_scalar_or_range(raw, cast=cast)
    return _normalize_matcher_value(key, sampled)


def _load_cc_material(texture_root: Path, texture_name: str):
    mats = bproc.loader.load_ccmaterials(
        str(texture_root),
        used_assets=[texture_name],
        use_all_materials=True,
        skip_transparent_materials=True,
    )
    if not mats:
        raise RuntimeError(f"Could not load CC material '{texture_name}' from {texture_root}")
    return mats[0]


def _build_fixed_room_and_lights(cfg):
    room_planes = [
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1]),
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[0, -3, 3], rotation=[-1.570796, 0, 0]),
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[0, 3, 3], rotation=[1.570796, 0, 0]),
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[3, 0, 3], rotation=[0, -1.570796, 0]),
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[-3, 0, 3], rotation=[0, 1.570796, 0]),
    ]
    for plane in room_planes:
        plane.enable_rigidbody(
            False,
            collision_shape="BOX",
            friction=100.0,
            linear_damping=0.99,
            angular_damping=0.99,
        )

    fixed_scene = getattr(cfg, "fixed_scene", None)

    light_plane = bproc.object.create_primitive("PLANE", scale=[1, 1, 1], location=[0, 0, 5])
    light_plane.set_name("light_plane")
    light_plane_material = bproc.material.create("light_material")
    light_plane_material.make_emissive(
        emission_strength=float(_cfg_get(fixed_scene, "light_plane_emission_strength", 4.0)),
        emission_color=np.asarray(
            _cfg_get(fixed_scene, "light_plane_emission_color", [0.9, 0.9, 0.9, 1.0]),
            dtype=np.float64,
        ),
    )
    light_plane.replace_materials(light_plane_material)

    diap_tem = [5500, 6500]
    colour = [TemperatureToRGBConverter(tem) for tem in diap_tem]

    light_point = bproc.types.Light()
    light_point.set_energy(250.0)
    light_point.set_color(
        np.asarray(
            _cfg_get(fixed_scene, "light_color", np.random.uniform(colour[0], colour[1])),
            dtype=np.float64,
        )
    )
    light_point.set_location(
        np.asarray(
            _cfg_get(fixed_scene, "light_location", [0.2, -0.35, 4.1]),
            dtype=np.float64,
        )
    )
    return room_planes, light_point, light_plane_material


def _fixed_camera_poses(rs: RealSenseProfile, cfg):
    fixed_scene = getattr(cfg, "fixed_scene", None)
    cam_loc = np.asarray(_cfg_get(fixed_scene, "camera_location", [0.95, -1.2, 0.75]), dtype=np.float64)
    poi = np.asarray(_cfg_get(fixed_scene, "point_of_interest", [0.0, 0.0, 0.1]), dtype=np.float64)
    rotation_factor = float(_cfg_get(fixed_scene, "rotation_factor", 5.0))

    T_w_irL = build_lookat_pose_cam(cam_loc, poi, rotation_factor=rotation_factor)

    T_color_from_ir_left = rs.get_T_blender("IR_LEFT", "COLOR")
    T_color_from_ir_right = rs.get_T_blender("IR_RIGHT", "COLOR")
    T_ir_left_from_color = np.linalg.inv(T_color_from_ir_left)
    T_w_color = T_w_irL @ T_ir_left_from_color
    T_w_irR = T_w_color @ T_color_from_ir_right
    return (
        np.asarray(T_w_irL, dtype=np.float64),
        np.asarray(T_w_irR, dtype=np.float64),
        np.asarray(T_w_color, dtype=np.float64),
    )


def _resolve_sgbm_mode(value):
    if value is None:
        return cv2.STEREO_SGBM_MODE_HH
    if isinstance(value, int):
        return int(value)

    mode = str(value).strip().upper()
    mapping = {
        "SGBM": getattr(cv2, "STEREO_SGBM_MODE_SGBM", 0),
        "HH": getattr(cv2, "STEREO_SGBM_MODE_HH", 1),
        "3WAY": getattr(cv2, "STEREO_SGBM_MODE_SGBM_3WAY", 2),
        "HH4": getattr(cv2, "STEREO_SGBM_MODE_HH4", 3),
    }
    if mode not in mapping:
        raise ValueError(f"Unsupported sgbm_mode: {value}")
    return int(mapping[mode])


def _build_matcher_kwargs(cfg, rs: RealSenseProfile):
    section = _get_matcher_section(cfg)
    return {
        "depth_min": _matcher_numeric_param(section, "depth_min_m", rs.depth_min_m, cast=float),
        "depth_max": _matcher_numeric_param(section, "depth_max_m", rs.depth_max_m, cast=float),
        "min_disparity": _matcher_numeric_param(section, "min_disparity", 0, cast=int),
        "num_disparities": _matcher_numeric_param(
            section,
            "num_disparities",
            rs.recommend_num_disparities(stream="IR_LEFT"),
            cast=int,
        ),
        "block_size": _matcher_numeric_param(section, "block_size", 7, cast=int),
        "preprocess": str(_cfg_get(section, "preprocess", "clahe")),
        "use_wls": bool(_cfg_get(section, "use_wls", False)),
        "lr_check": bool(_cfg_get(section, "lr_check", False)),
        "lr_thresh_px": _matcher_numeric_param(section, "lr_thresh_px", 1.0, cast=float),
        "lr_min_keep_ratio": _matcher_numeric_param(section, "lr_min_keep_ratio", 0.02, cast=float),
        "speckle_filter": bool(_cfg_get(section, "speckle_filter", False)),
        "fill_mode": str(_cfg_get(section, "fill_mode", "none")),
        "fill_iters": _matcher_numeric_param(section, "fill_iters", 0, cast=int),
        "depth_completion": bool(_cfg_get(section, "depth_completion", False)),
        "sgbm_mode": _resolve_sgbm_mode(_cfg_get(section, "sgbm_mode", "HH")),
        "uniqueness_ratio": _matcher_numeric_param(section, "uniqueness_ratio", 10, cast=int),
        "disp12_max_diff": _matcher_numeric_param(section, "disp12_max_diff", 1, cast=int),
        "pre_filter_cap": _matcher_numeric_param(section, "pre_filter_cap", 63, cast=int),
        "p1_scale": _matcher_numeric_param(section, "p1_scale", 8.0, cast=float),
        "p2_scale": _matcher_numeric_param(section, "p2_scale", 32.0, cast=float),
        "use_geom_mask_from_gt": bool(_cfg_get(section, "use_geom_mask_from_gt", False)),
    }


def _jsonify(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def main(argv=None):
    bproc.init()
    bproc.utility.reset_keyframes()

    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    cfg = Config(args.config_file)

    camera_profile_json = args.camera_profile_json
    if camera_profile_json is None:
        camera_profile_json = str(getattr(cfg, "camera_profile_json", str(DEFAULT_PROFILE_JSON)))
    rs = RealSenseProfile.from_json(camera_profile_json)

    fixed_scene = getattr(cfg, "fixed_scene", None)
    scene_seed = int(_cfg_get(fixed_scene, "scene_seed", 1234))
    np.random.seed(scene_seed)
    random.seed(scene_seed)

    max_spp = int(getattr(cfg, "max_amount_of_samples", 50))
    bproc.renderer.set_max_amount_of_samples(max_spp)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)

    room_planes, light_point, light_plane_material = _build_fixed_room_and_lights(cfg)
    lighting_base_state = None

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(getattr(cfg, "output_dir", str(TOOLS_DIR / "output" / "matcher_research_fixed")))
    out_dir = _ensure_output_dir(output_dir)

    texture_section = getattr(cfg, "fixed_texture", None)
    texture_name = str(args.texture_name or _cfg_get(texture_section, "texture_name", "Tiles074"))
    texture_root = Path(_cfg_get(texture_section, "texture_root", REPO_ROOT / "resources" / "textures_2k_plate_valid")).resolve()
    cc_material = _load_cc_material(texture_root, texture_name)

    energy_cfg = _sample_effective_energy_config(cfg)
    overrides = ProjectorOverrides(projector_energy=float(energy_cfg["projector_energy"]))
    matcher_section = _get_matcher_section(cfg)
    plane_distance_raw = _cfg_get(matcher_section, "plane_distance_m", None)
    plane_distance_m = None if plane_distance_raw is None else float(_sample_scalar_or_range(plane_distance_raw, cast=float))
    rgb_to_intensity_mode = str(_cfg_get(matcher_section, "rgb_to_intensity_mode", "lcn"))
    matcher_kwargs = _build_matcher_kwargs(cfg, rs)

    try:
        _prepare_objects(cfg)
        lighting_base_state = _capture_render_lighting_state(light_point, light_plane_material)
        left_pose, _right_pose, color_pose = _fixed_camera_poses(rs, cfg)

        for plane in room_planes:
            plane.replace_materials(cc_material)

        print(
            "[Matcher Research] "
            f"seed={scene_seed} | "
            f"texture={texture_name} | "
            f"projector_energy={float(energy_cfg['projector_energy']):.3f} | "
            f"rgb_light_energy={float(energy_cfg['rgb_light_energy']):.3f} | "
            f"ir_light_energy={float(energy_cfg['ir_light_energy']):.3f}"
        )

        _apply_lighting_energy(lighting_base_state, float(energy_cfg["ir_light_energy"]))
        try:
            render_data = render_active_stereo_pair(
                rs,
                left_stream="IR_LEFT",
                right_stream="IR_RIGHT",
                overrides=overrides,
                left_world_pose=left_pose,
                mount_name="rs_projector_mount_matcher_research",
                socket_name="rs_projector_socket_matcher_research",
            )
        finally:
            _apply_lighting_energy(lighting_base_state, float(energy_cfg["rgb_light_energy"]))

        left_ir = rgb_to_intensity_u8(render_data["left_colors"], mode=rgb_to_intensity_mode)
        right_ir = rgb_to_intensity_u8(render_data["right_colors"], mode=rgb_to_intensity_mode)

        stereo = stereo_from_ir_pair(
            rs,
            left_stream="IR_LEFT",
            right_stream="IR_RIGHT",
            left_ir_u8=left_ir,
            right_ir_u8=right_ir,
            left_depth_gt_m=render_data["left_depth_m"],
            plane_distance_m=None if plane_distance_m is None else float(plane_distance_m),
            **matcher_kwargs,
        )

        bproc.utility.reset_keyframes()
        rs.set_bproc_intrinsics("COLOR")
        bproc.camera.add_camera_pose(color_pose, frame=0)
        data_rgb = bproc.renderer.render()
        rgb = np.asarray(data_rgb["colors"][0])

        depth_gt_left_m = np.asarray(render_data["left_depth_m"], dtype=np.float32)
        depth_rect_m = np.asarray(stereo["depth_rect_m"], dtype=np.float32)
        disp_rect_px = np.asarray(stereo["disp_rect_px"], dtype=np.float32)
        left_rect_u8 = np.asarray(stereo["left_rect_u8"], dtype=np.uint8)
        right_rect_u8 = np.asarray(stereo["right_rect_u8"], dtype=np.uint8)

        rectify_maps = build_rectify_maps(stereo["rectify_cfg"], stereo["rectify_cfg"]["image_size"])
        depth_gt_rect_m = rectify_single_channel(
            depth_gt_left_m,
            rectify_maps,
            interp=cv2.INTER_NEAREST,
            border_val=0.0,
        ).astype(np.float32, copy=False)

        stem = "000000"
        _save_png(out_dir / f"{stem}_rgb.png", rgb)
        _save_png(out_dir / f"{stem}_left_ir.png", left_ir)
        _save_png(out_dir / f"{stem}_right_ir.png", right_ir)
        _save_png(out_dir / f"{stem}_left_ir_rect.png", left_rect_u8)
        _save_png(out_dir / f"{stem}_right_ir_rect.png", right_rect_u8)
        _save_png(out_dir / f"{stem}_depth_gt_left_preview.png", _depth_preview(depth_gt_left_m))
        _save_png(out_dir / f"{stem}_depth_gt_rect_preview.png", _depth_preview(depth_gt_rect_m))
        _save_png(out_dir / f"{stem}_depth_rect_preview.png", _depth_preview(depth_rect_m))
        _save_png(out_dir / f"{stem}_disp_rect_preview.png", _disp_preview(disp_rect_px))

        np.save(out_dir / f"{stem}_depth_gt_left.npy", depth_gt_left_m)
        np.save(out_dir / f"{stem}_depth_gt_rect.npy", depth_gt_rect_m)
        np.save(out_dir / f"{stem}_depth_rect.npy", depth_rect_m)
        np.save(out_dir / f"{stem}_disp_rect.npy", disp_rect_px)
        _save_png(out_dir / "projector_pattern.png", np.asarray(render_data["projector_pattern_rgba"], dtype=np.uint8))

        manifest = {
            "camera_profile_json": str(camera_profile_json),
            "texture_root": str(texture_root),
            "texture_name": texture_name,
            "scene_seed": scene_seed,
            "max_amount_of_samples": max_spp,
            "rgb_to_intensity_mode": rgb_to_intensity_mode,
            "plane_distance_m": plane_distance_m,
            "fixed_scene": _jsonify(getattr(cfg, "fixed_scene", {})),
            "effective_projector_render": _jsonify(energy_cfg),
            "matcher_data": _jsonify(_build_matcher_kwargs(cfg, rs)),
            "depth_stats": _jsonify(stereo["depth_stats"]),
            "projector_cfg": _jsonify(render_data["projector_cfg"]),
            "left_world_pose": _jsonify(render_data["left_world_pose"]),
            "right_world_pose": _jsonify(render_data["right_world_pose"]),
            "mount_world_pose": _jsonify(render_data["mount_world_pose"]),
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        print(
            "[Matcher Research] "
            f"saved={out_dir} | "
            f"valid_fraction={stereo['depth_stats']['valid_fraction']:.4f}"
        )

    finally:
        try:
            _restore_render_lighting_state(lighting_base_state)
        except Exception:
            pass
        if light_point is not None:
            try:
                light_point.delete()
            except Exception:
                pass


if __name__ == "__main__":
    main()
