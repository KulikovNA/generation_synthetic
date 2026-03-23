#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import blenderproc as bproc

import argparse
import sys
from pathlib import Path

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
from blendforge.blender_runtime.stereo.DepthAlignment import align_depth_series_ir_left_to_color

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
    p = argparse.ArgumentParser(description="Fixed-scene texture sweep for active stereo projector rendering.")
    p.add_argument("--config_file", type=str, required=True)
    p.add_argument("--camera_profile_json", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start_from", type=str, default=None)
    p.add_argument("--only_texture", type=str, default=None)
    return p.parse_args(argv)


def _cfg_get(section, key: str, default):
    if section is None:
        return default
    if isinstance(section, dict):
        return section.get(key, default)
    return getattr(section, key, default)


def _iter_texture_names(texture_root: Path):
    return sorted([p.name for p in texture_root.iterdir() if p.is_dir()])


def _load_cc_material(texture_root: Path, texture_name: str):
    texture_dir = texture_root / texture_name
    has_opacity = any(texture_dir.glob("*_Opacity.*"))
    mats = bproc.loader.load_ccmaterials(
        str(texture_root),
        used_assets=[texture_name],
        use_all_materials=True,
        skip_transparent_materials=True,
    )
    if not mats:
        reason = "transparent_opacity_map" if has_opacity else "unsupported_or_incomplete_texture_folder"
        return None, reason
    return mats[0], None


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
    cam_loc = np.asarray(_cfg_get(fixed_scene, "camera_location", [0.95, -1.2, 0.95]), dtype=np.float64)
    poi = np.asarray(_cfg_get(fixed_scene, "point_of_interest", [0.0, 0.0, 0.1]), dtype=np.float64)
    rotation_factor = float(_cfg_get(fixed_scene, "rotation_factor", 9.0))

    T_w_irL = build_lookat_pose_cam(cam_loc, poi, rotation_factor=rotation_factor)

    T_color_from_ir_left = rs.get_T_blender("IR_LEFT", "COLOR")
    T_color_from_ir_right = rs.get_T_blender("IR_RIGHT", "COLOR")
    T_ir_left_from_color = np.linalg.inv(T_color_from_ir_left)
    T_w_color = T_w_irL @ T_ir_left_from_color
    T_w_irR = T_w_color @ T_color_from_ir_right

    return np.asarray(T_w_irL, dtype=np.float64), np.asarray(T_w_irR, dtype=np.float64), np.asarray(T_w_color, dtype=np.float64)


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
        output_dir = str(getattr(cfg, "output_dir", str(TOOLS_DIR / "output" / "texture_sweep_fixed")))
    out_root = _ensure_output_dir(output_dir)

    energy_cfg = _sample_effective_energy_config(cfg)
    overrides = ProjectorOverrides(projector_energy=float(energy_cfg["projector_energy"]))

    texture_root = Path(cfg.cc_textures.cc_textures_path).resolve()
    texture_names = _iter_texture_names(texture_root)
    if args.only_texture is not None:
        texture_names = [args.only_texture]
    elif args.start_from is not None:
        if args.start_from not in texture_names:
            raise ValueError(f"start_from texture '{args.start_from}' was not found in {texture_root}")
        start_idx = texture_names.index(args.start_from)
        texture_names = texture_names[start_idx:]
    limit = args.limit if args.limit is not None else _cfg_get(getattr(cfg, "texture_sweep", None), "limit", None)
    if limit is not None:
        texture_names = texture_names[: int(limit)]

    try:
        _prepare_objects(cfg)
        lighting_base_state = _capture_render_lighting_state(light_point, light_plane_material)
        left_pose, _right_pose, color_pose = _fixed_camera_poses(rs, cfg)
        skipped = []

        print(
            "[Fixed Scene] "
            f"seed={scene_seed} | "
            f"projector_energy={float(energy_cfg['projector_energy']):.3f} | "
            f"rgb_light_energy={float(energy_cfg['rgb_light_energy']):.3f} | "
            f"ir_light_energy={float(energy_cfg['ir_light_energy']):.3f} | "
            f"textures={len(texture_names)}"
        )

        for texture_idx, texture_name in enumerate(texture_names):
            tex_out_dir = _ensure_output_dir(str(out_root / texture_name))
            cc_material, skip_reason = _load_cc_material(texture_root, texture_name)
            if cc_material is None:
                skipped.append((texture_name, skip_reason))
                print(f"[{texture_idx + 1}/{len(texture_names)}] {texture_name} | skipped={skip_reason}")
                continue

            for plane in room_planes:
                plane.replace_materials(cc_material)

            _apply_lighting_energy(lighting_base_state, float(energy_cfg["ir_light_energy"]))

            try:
                render_data = render_active_stereo_pair(
                    rs,
                    left_stream="IR_LEFT",
                    right_stream="IR_RIGHT",
                    overrides=overrides,
                    left_world_pose=left_pose,
                    mount_name="rs_projector_mount_texture_sweep",
                    socket_name="rs_projector_socket_texture_sweep",
                )
            finally:
                _apply_lighting_energy(lighting_base_state, float(energy_cfg["rgb_light_energy"]))

            left_ir = rgb_to_intensity_u8(render_data["left_colors"], mode="lcn")
            right_ir = rgb_to_intensity_u8(render_data["right_colors"], mode="lcn")

            stereo = stereo_from_ir_pair(
                rs,
                left_stream="IR_LEFT",
                right_stream="IR_RIGHT",
                left_ir_u8=left_ir,
                right_ir_u8=right_ir,
                left_depth_gt_m=render_data["left_depth_m"],
                plane_distance_m=None,
            )

            bproc.utility.reset_keyframes()
            rs.set_bproc_intrinsics("COLOR")
            bproc.camera.add_camera_pose(color_pose, frame=0)
            data_rgb = bproc.renderer.render()
            rgb = np.asarray(data_rgb["colors"][0])

            depth_rect_rgb_list, _ = align_depth_series_ir_left_to_color(
                rs,
                [np.asarray(stereo["depth_rect_m"], dtype=np.float32)],
                depth_value_mode="source_z",
                splat_2x2=True,
                rectify_mode="on",
            )
            depth_gt_rgb_list, _ = align_depth_series_ir_left_to_color(
                rs,
                [np.asarray(render_data["left_depth_m"], dtype=np.float32)],
                depth_value_mode="source_z",
                splat_2x2=True,
                rectify_mode="off",
            )

            depth_gt_ir_m = np.asarray(render_data["left_depth_m"], dtype=np.float32)
            depth_rect_m = np.asarray(stereo["depth_rect_m"], dtype=np.float32)
            disp_rect_px = np.asarray(stereo["disp_rect_px"], dtype=np.float32)
            depth_rgb_m = np.asarray(depth_rect_rgb_list[0], dtype=np.float32)
            depth_gt_rgb_m = np.asarray(depth_gt_rgb_list[0], dtype=np.float32)

            stem = "000000"
            _save_png(tex_out_dir / f"{stem}_rgb.png", rgb)
            _save_png(tex_out_dir / f"{stem}_left_ir.png", left_ir)
            _save_png(tex_out_dir / f"{stem}_right_ir.png", right_ir)
            _save_png(tex_out_dir / f"{stem}_depth_gt_ir_preview.png", _depth_preview(depth_gt_ir_m))
            _save_png(tex_out_dir / f"{stem}_depth_rect_preview.png", _depth_preview(depth_rect_m))
            _save_png(tex_out_dir / f"{stem}_depth_rgb_preview.png", _depth_preview(depth_rgb_m))
            _save_png(tex_out_dir / f"{stem}_depth_gt_rgb_preview.png", _depth_preview(depth_gt_rgb_m))
            _save_png(tex_out_dir / f"{stem}_disp_rect_preview.png", _disp_preview(disp_rect_px))

            np.save(tex_out_dir / f"{stem}_depth_gt_ir.npy", depth_gt_ir_m)
            np.save(tex_out_dir / f"{stem}_depth_rect.npy", depth_rect_m)
            np.save(tex_out_dir / f"{stem}_depth_rgb.npy", depth_rgb_m)
            np.save(tex_out_dir / f"{stem}_depth_gt_rgb.npy", depth_gt_rgb_m)
            np.save(tex_out_dir / f"{stem}_disp_rect.npy", disp_rect_px)
            _save_png(tex_out_dir / "projector_pattern.png", np.asarray(render_data["projector_pattern_rgba"], dtype=np.uint8))

            print(
                f"[{texture_idx + 1}/{len(texture_names)}] {texture_name} | "
                f"valid_fraction={stereo['depth_stats']['valid_fraction']:.4f}"
            )

        if skipped:
            skipped_path = out_root / "_skipped_textures.txt"
            skipped_path.write_text(
                "".join(f"{name}\t{reason}\n" for name, reason in skipped),
                encoding="utf-8",
            )
            print(f"[Texture Sweep] skipped={len(skipped)} | log={skipped_path}")

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

    print(f"Saved outputs to: {out_root}")


if __name__ == "__main__":
    main()
