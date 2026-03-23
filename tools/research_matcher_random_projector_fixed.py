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
from blendforge.blender_runtime.camera.RealsenseProfileLoader import RealSenseProfile
from blendforge.blender_runtime.camera.ActiveStereoProjectorRuntime import (
    ProjectorOverrides,
    get_mount_world_pose,
    get_stream_world_pose_from_anchor,
    render_single_stream_with_projector,
    resolve_projector_runtime_config,
)
from blendforge.blender_runtime.camera import RealsenseProjectorUtility as projector_util
from blendforge.blender_runtime.stereo.ActiveStereoIRUtility import (
    rgb_to_intensity_u8,
    stereo_from_ir_pair,
)
from blendforge.blender_runtime.stereo.StereoRectify import build_rectify_maps
from blendforge.blender_runtime.stereo.utils.PadCropUtility import rectify_single_channel
from research_matcher_effective_fixed import (
    _cfg_get,
    _get_matcher_section,
    _load_cc_material,
    _build_fixed_room_and_lights,
    _fixed_camera_poses,
    _build_matcher_kwargs,
    _sample_scalar_or_range,
    _jsonify,
)
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
        description="Fixed-scene matcher research for a random-projector branch using effective-profile geometry."
    )
    p.add_argument("--config_file", type=str, required=True)
    p.add_argument("--camera_profile_json", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--texture_name", type=str, default=None)
    p.add_argument("--stereo_branch", type=str, default=None)
    return p.parse_args(argv)


def _sample_from_range(value, *, cast=float):
    if isinstance(value, (int, float)):
        return cast(value)
    vals = list(value)
    if len(vals) == 0:
        raise ValueError("Range must not be empty")
    if len(vals) == 1:
        return cast(vals[0])
    lo = vals[0]
    hi = vals[1]
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


def _resolve_non_overlap_min_sep(cfg_section, dot_radius_px: float, requested_min_sep_px: float) -> tuple[float, dict]:
    enforce_non_overlap = bool(_cfg_get(cfg_section, "enforce_non_overlap", True))
    non_overlap_margin_px = float(_cfg_get(cfg_section, "non_overlap_margin_px", 0.0))
    drawn_radius_px = max(1, int(round(float(dot_radius_px))))

    effective_min_sep_px = float(requested_min_sep_px)
    if enforce_non_overlap:
        effective_min_sep_px = max(
            effective_min_sep_px,
            float(2 * drawn_radius_px) + float(non_overlap_margin_px),
        )

    info = {
        "enforce_non_overlap": enforce_non_overlap,
        "non_overlap_margin_px": float(non_overlap_margin_px),
        "drawn_radius_px": int(drawn_radius_px),
        "requested_min_sep_px": float(requested_min_sep_px),
        "effective_min_sep_px": float(effective_min_sep_px),
    }
    return float(effective_min_sep_px), info


def _build_random_projector_config(rs: RealSenseProfile, overrides: ProjectorOverrides, cfg) -> tuple[dict, dict]:
    base_cfg = resolve_projector_runtime_config(rs, overrides, fallback_left_stream="IR_LEFT").to_dict()
    section = getattr(cfg, "random_projector_pattern", None)

    seed_mode = str(_cfg_get(section, "seed_mode", "random")).strip().lower()
    if seed_mode == "fixed":
        pattern_seed = int(_cfg_get(section, "seed_value", 12345))
    elif seed_mode == "random":
        pattern_seed = int(random.SystemRandom().randrange(0, 2**31 - 1))
    else:
        raise ValueError(f"Unsupported random_projector_pattern.seed_mode: {seed_mode}")

    dot_count = _sample_from_range(_cfg_get(section, "dot_count_range", [4000, 7000]), cast=int)
    dot_radius_px = _sample_from_range(_cfg_get(section, "dot_radius_px_range", [1.0, 2.0]), cast=float)
    requested_min_sep_px = _sample_from_range(_cfg_get(section, "min_sep_px_range", [3.0, 5.0]), cast=float)
    min_sep_px, non_overlap_info = _resolve_non_overlap_min_sep(section, dot_radius_px, requested_min_sep_px)

    dot_sigma_range = _cfg_get(section, "dot_sigma_px_range", None)
    dot_sigma_px = None
    if dot_sigma_range not in (None, ""):
        dot_sigma_px = _sample_from_range(dot_sigma_range, cast=float)

    base_cfg["pattern_path"] = None
    base_cfg["pattern_base_dir"] = None
    base_cfg["pattern_seed"] = int(pattern_seed)
    base_cfg["dot_count"] = int(dot_count)
    base_cfg["pattern_dot_radius_px"] = float(dot_radius_px)
    base_cfg["pattern_min_sep_px"] = float(min_sep_px)
    base_cfg["pattern_dot_sigma_px"] = dot_sigma_px

    sampled = {
        "seed_mode": seed_mode,
        "pattern_seed": int(pattern_seed),
        "dot_count": int(dot_count),
        "dot_radius_px": float(dot_radius_px),
        "min_sep_px": float(min_sep_px),
        "dot_sigma_px": None if dot_sigma_px is None else float(dot_sigma_px),
        "pattern_width": int(base_cfg["pattern_width"]),
        "pattern_height": int(base_cfg["pattern_height"]),
        "fov_h_deg": float(base_cfg["fov_h_deg"]),
        "fov_v_deg": float(base_cfg["fov_v_deg"]),
    }
    sampled.update(non_overlap_info)
    return base_cfg, sampled


def _render_stereo_branch_pair(
    rs: RealSenseProfile,
    *,
    left_pose: np.ndarray,
    overrides: ProjectorOverrides,
    stereo_branch: str,
    cfg,
):
    right_pose = get_stream_world_pose_from_anchor(rs, "IR_RIGHT", "IR_LEFT", left_pose)
    mount_world_pose = get_mount_world_pose(
        rs,
        anchor_stream="IR_LEFT",
        secondary_stream="IR_RIGHT",
        anchor_world_pose=left_pose,
    )

    mount = projector_util.get_or_create_mount_empty(f"rs_projector_mount_{stereo_branch}")

    if stereo_branch == "effective":
        projector_cfg = resolve_projector_runtime_config(rs, overrides, fallback_left_stream="IR_LEFT").to_dict()
        sampled_pattern = {
            "branch": "effective",
            "pattern_width": int(projector_cfg["pattern_width"]),
            "pattern_height": int(projector_cfg["pattern_height"]),
            "fov_h_deg": float(projector_cfg["fov_h_deg"]),
            "fov_v_deg": float(projector_cfg["fov_v_deg"]),
            "pattern_path": projector_cfg.get("pattern_path"),
        }
    elif stereo_branch == "random_pattern":
        projector_cfg, sampled_pattern = _build_random_projector_config(rs, overrides, cfg)
        sampled_pattern["branch"] = "random_pattern"
    else:
        raise ValueError(f"Unsupported stereo_branch: {stereo_branch}")

    left_render = render_single_stream_with_projector(
        rs,
        "IR_LEFT",
        left_pose,
        mount,
        mount_world_pose,
        projector_cfg,
        socket_name=f"rs_projector_socket_{stereo_branch}",
    )
    right_render = render_single_stream_with_projector(
        rs,
        "IR_RIGHT",
        right_pose,
        mount,
        mount_world_pose,
        projector_cfg,
        socket_name=f"rs_projector_socket_{stereo_branch}",
    )

    return {
        "left_colors": left_render["colors"],
        "right_colors": right_render["colors"],
        "left_depth_m": left_render["depth"],
        "projector_cfg": projector_cfg,
        "projector_pattern_rgba": left_render["pattern_rgba"],
        "left_world_pose": left_pose,
        "right_world_pose": right_pose,
        "mount_world_pose": mount_world_pose,
        "sampled_pattern": sampled_pattern,
    }


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

    stereo_branch = str(args.stereo_branch or getattr(cfg, "stereo_branch", "random_pattern")).strip().lower()

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
        output_dir = str(getattr(cfg, "output_dir", str(TOOLS_DIR / "output" / "random_projector_matcher_research_fixed")))
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
            "[Random Projector Matcher Research] "
            f"seed={scene_seed} | "
            f"texture={texture_name} | "
            f"stereo_branch={stereo_branch} | "
            f"projector_energy={float(energy_cfg['projector_energy']):.3f} | "
            f"rgb_light_energy={float(energy_cfg['rgb_light_energy']):.3f} | "
            f"ir_light_energy={float(energy_cfg['ir_light_energy']):.3f}"
        )

        _apply_lighting_energy(lighting_base_state, float(energy_cfg["ir_light_energy"]))
        try:
            render_data = _render_stereo_branch_pair(
                rs,
                left_pose=left_pose,
                overrides=overrides,
                stereo_branch=stereo_branch,
                cfg=cfg,
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
        projector_pattern = np.asarray(render_data["projector_pattern_rgba"], dtype=np.uint8)
        _save_png(out_dir / "projector_pattern_raw.png", projector_pattern)
        if projector_pattern.ndim == 3 and projector_pattern.shape[2] >= 3:
            projector_gray = projector_pattern[:, :, 0]
        else:
            projector_gray = np.asarray(projector_pattern)
        projector_preview = _disp_preview(projector_gray.astype(np.float32))
        _save_png(out_dir / "projector_pattern.png", projector_preview)
        _save_png(out_dir / "projector_pattern_preview.png", projector_preview)

        manifest = {
            "camera_profile_json": str(camera_profile_json),
            "stereo_branch": stereo_branch,
            "texture_root": str(texture_root),
            "texture_name": texture_name,
            "scene_seed": scene_seed,
            "max_amount_of_samples": max_spp,
            "rgb_to_intensity_mode": rgb_to_intensity_mode,
            "plane_distance_m": plane_distance_m,
            "fixed_scene": _jsonify(getattr(cfg, "fixed_scene", {})),
            "effective_projector_render": _jsonify(energy_cfg),
            "matcher_data": _jsonify(_build_matcher_kwargs(cfg, rs)),
            "random_projector_pattern": _jsonify(getattr(cfg, "random_projector_pattern", {})),
            "sampled_pattern": _jsonify(render_data["sampled_pattern"]),
            "depth_stats": _jsonify(stereo["depth_stats"]),
            "projector_cfg": _jsonify(render_data["projector_cfg"]),
            "left_world_pose": _jsonify(render_data["left_world_pose"]),
            "right_world_pose": _jsonify(render_data["right_world_pose"]),
            "mount_world_pose": _jsonify(render_data["mount_world_pose"]),
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        print(
            "[Random Projector Matcher Research] "
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
