#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import blenderproc as bproc
import argparse
import json
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
    render_active_stereo_pair,
)
from blendforge.blender_runtime.stereo.ActiveStereoIRUtility import (
    rgb_to_intensity_u8,
    stereo_from_ir_pair,
)
from blendforge.blender_runtime.stereo.DepthAlignment import align_depth_series_ir_left_to_color
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
        description="Fixed-scene alignment research for transferring depth into the RGB camera grid."
    )
    p.add_argument("--config_file", type=str, required=True)
    p.add_argument("--camera_profile_json", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--texture_name", type=str, default=None)
    return p.parse_args(argv)


def _save_mask_png(path: Path, mask: np.ndarray) -> None:
    arr = np.asarray(mask, dtype=bool)
    _save_png(path, (arr.astype(np.uint8) * 255))


def _align_one(rs, depth_m, *, depth_value_mode: str, rectify_mode: str, splat_2x2: bool, rectify_use_distortion: bool):
    depth_list, mask_list = align_depth_series_ir_left_to_color(
        rs,
        [np.asarray(depth_m, dtype=np.float32)],
        depth_value_mode=depth_value_mode,
        splat_2x2=bool(splat_2x2),
        rectify_mode=str(rectify_mode),
        rectify_use_distortion=bool(rectify_use_distortion),
    )
    return (
        np.asarray(depth_list[0], dtype=np.float32),
        np.asarray(mask_list[0], dtype=bool),
    )


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

    max_spp = int(getattr(cfg, "max_amount_of_samples", 50))
    bproc.renderer.set_max_amount_of_samples(max_spp)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)

    room_planes, light_point, light_plane_material = _build_fixed_room_and_lights(cfg)
    lighting_base_state = None

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(getattr(cfg, "output_dir", str(TOOLS_DIR / "output" / "alignment_research_fixed")))
    out_dir = _ensure_output_dir(output_dir)

    texture_section = getattr(cfg, "fixed_texture", None)
    texture_name = str(args.texture_name or _cfg_get(texture_section, "texture_name", "Tiles074"))
    texture_root = Path(_cfg_get(texture_section, "texture_root", REPO_ROOT / "resources" / "textures_2k_plate_valid")).resolve()
    cc_material = _load_cc_material(texture_root, texture_name)

    energy_cfg = _sample_effective_energy_config(cfg)
    overrides = ProjectorOverrides(projector_energy=float(energy_cfg["projector_energy"]))

    matcher_section = _get_matcher_section(cfg)
    rgb_to_intensity_mode = str(_cfg_get(matcher_section, "rgb_to_intensity_mode", "lcn"))
    plane_distance_raw = _cfg_get(matcher_section, "plane_distance_m", None)
    plane_distance_m = None if plane_distance_raw is None else float(_sample_scalar_or_range(plane_distance_raw, cast=float))
    matcher_kwargs = _build_matcher_kwargs(cfg, rs)

    align_section = getattr(cfg, "alignment_research", None)
    pred_rectify_mode = str(_cfg_get(align_section, "pred_rectify_mode", "on"))
    gt_rectify_mode = str(_cfg_get(align_section, "gt_rectify_mode", "off"))
    splat_2x2 = bool(_cfg_get(align_section, "splat_2x2", True))
    rectify_use_distortion = bool(_cfg_get(align_section, "rectify_use_distortion", False))
    save_source_z = bool(_cfg_get(align_section, "save_source_z", True))
    save_target_z = bool(_cfg_get(align_section, "save_target_z", True))

    try:
        _prepare_objects(cfg)
        lighting_base_state = _capture_render_lighting_state(light_point, light_plane_material)
        left_pose, _right_pose, color_pose = _fixed_camera_poses(rs, cfg)

        for plane in room_planes:
            plane.replace_materials(cc_material)

        print(
            "[Alignment Research] "
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
                mount_name="rs_projector_mount_alignment_research",
                socket_name="rs_projector_socket_alignment_research",
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
        depth_color_direct_m = np.asarray(data_rgb["depth"][0], dtype=np.float32)

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

        aligned_outputs = {}
        if save_source_z:
            aligned_outputs["pred_rgb_source_z"] = _align_one(
                rs,
                depth_rect_m,
                depth_value_mode="source_z",
                rectify_mode=pred_rectify_mode,
                splat_2x2=splat_2x2,
                rectify_use_distortion=rectify_use_distortion,
            )
            aligned_outputs["gt_rgb_source_z"] = _align_one(
                rs,
                depth_gt_left_m,
                depth_value_mode="source_z",
                rectify_mode=gt_rectify_mode,
                splat_2x2=splat_2x2,
                rectify_use_distortion=rectify_use_distortion,
            )
        if save_target_z:
            aligned_outputs["pred_rgb_target_z"] = _align_one(
                rs,
                depth_rect_m,
                depth_value_mode="target_z",
                rectify_mode=pred_rectify_mode,
                splat_2x2=splat_2x2,
                rectify_use_distortion=rectify_use_distortion,
            )
            aligned_outputs["gt_rgb_target_z"] = _align_one(
                rs,
                depth_gt_left_m,
                depth_value_mode="target_z",
                rectify_mode=gt_rectify_mode,
                splat_2x2=splat_2x2,
                rectify_use_distortion=rectify_use_distortion,
            )

        stem = "000000"
        _save_png(out_dir / f"{stem}_rgb.png", rgb)
        _save_png(out_dir / f"{stem}_left_ir.png", left_ir)
        _save_png(out_dir / f"{stem}_right_ir.png", right_ir)
        _save_png(out_dir / f"{stem}_left_ir_rect.png", left_rect_u8)
        _save_png(out_dir / f"{stem}_right_ir_rect.png", right_rect_u8)
        _save_png(out_dir / f"{stem}_depth_gt_left_preview.png", _depth_preview(depth_gt_left_m))
        _save_png(out_dir / f"{stem}_depth_gt_rect_preview.png", _depth_preview(depth_gt_rect_m))
        _save_png(out_dir / f"{stem}_depth_rect_preview.png", _depth_preview(depth_rect_m))
        _save_png(out_dir / f"{stem}_depth_color_direct_preview.png", _depth_preview(depth_color_direct_m))
        _save_png(out_dir / f"{stem}_disp_rect_preview.png", _disp_preview(disp_rect_px))

        np.save(out_dir / f"{stem}_depth_gt_left.npy", depth_gt_left_m)
        np.save(out_dir / f"{stem}_depth_gt_rect.npy", depth_gt_rect_m)
        np.save(out_dir / f"{stem}_depth_rect.npy", depth_rect_m)
        np.save(out_dir / f"{stem}_disp_rect.npy", disp_rect_px)
        np.save(out_dir / f"{stem}_depth_color_direct.npy", depth_color_direct_m)
        _save_png(out_dir / "projector_pattern.png", np.asarray(render_data["projector_pattern_rgba"], dtype=np.uint8))

        for name, (depth_map, valid_mask) in aligned_outputs.items():
            np.save(out_dir / f"{stem}_{name}.npy", np.asarray(depth_map, dtype=np.float32))
            np.save(out_dir / f"{stem}_{name}_mask.npy", np.asarray(valid_mask, dtype=bool))
            _save_png(out_dir / f"{stem}_{name}_preview.png", _depth_preview(depth_map))
            _save_mask_png(out_dir / f"{stem}_{name}_mask.png", valid_mask)

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
            "alignment_research": _jsonify(getattr(cfg, "alignment_research", {})),
            "depth_stats": _jsonify(stereo["depth_stats"]),
            "projector_cfg": _jsonify(render_data["projector_cfg"]),
            "left_world_pose": _jsonify(render_data["left_world_pose"]),
            "right_world_pose": _jsonify(render_data["right_world_pose"]),
            "mount_world_pose": _jsonify(render_data["mount_world_pose"]),
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"[Alignment Research] saved={out_dir}")

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
