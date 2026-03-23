#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import blenderproc as bproc

import os
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import bpy

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
from blendforge.blender_runtime.CustomMaterial import make_random_material
from blendforge.blender_runtime.CustomLoadMesh import load_objs
from blendforge.blender_runtime.utils import (
    sample_pose_func_drop,
    sample_pose_func,
    build_lookat_pose_cam,
)
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

DEFAULT_PROFILE_JSON = REPO_ROOT / "prepared" / "differBig" / "d435_effective_projector_640x480.json"


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Active stereo depth generator using the effective projector profile and the new runtime API."
    )
    p.add_argument("--config_file", type=str, required=True)
    p.add_argument("--camera_profile_json", type=str, default=str(DEFAULT_PROFILE_JSON))
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--frame_idx", type=int, default=0)
    p.add_argument("--energy", type=float, default=None)
    return p.parse_args(argv)


def _ensure_output_dir(path: str) -> Path:
    out_dir = Path(path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _to_u8_preview(img: np.ndarray, *, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    lo = float(np.percentile(x, low))
    hi = float(np.percentile(x, high))
    if hi <= lo + 1e-6:
        hi = lo + 1e-6
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)


def _save_png(path: Path, img: np.ndarray) -> None:
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            scale = 255.0 if float(arr.max(initial=0.0)) <= 1.5 else 1.0
            arr = np.clip(arr * scale, 0.0, 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        data = arr
    elif arr.ndim == 3 and arr.shape[2] == 3:
        data = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        data = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    else:
        raise ValueError(f"Unsupported image shape for save_png: {arr.shape}")
    cv2.imwrite(str(path), data)


def _depth_preview(depth_m: np.ndarray) -> np.ndarray:
    d = np.asarray(depth_m, dtype=np.float32)
    inv = np.where(d > 0.0, 1.0 / np.maximum(d, 1e-6), 0.0)
    return _to_u8_preview(inv, low=1.0, high=99.0)


def _disp_preview(disp_px: np.ndarray) -> np.ndarray:
    d = np.asarray(disp_px, dtype=np.float32)
    return _to_u8_preview(d, low=1.0, high=99.0)


def _build_room_and_lights(cfg):
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

    light_plane = bproc.object.create_primitive("PLANE", scale=[1, 1, 1], location=[0, 0, 5])
    light_plane.set_name("light_plane")
    light_plane_material = bproc.material.create("light_material")
    light_plane_material.make_emissive(
        emission_strength=np.random.uniform(3, 6),
        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]),
    )
    light_plane.replace_materials(light_plane_material)

    cc_textures = bproc.loader.load_ccmaterials(cfg.cc_textures.cc_textures_path)
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)

    diap_tem = [5500, 6500]
    colour = [TemperatureToRGBConverter(tem) for tem in diap_tem]
    light_energy = float(np.random.uniform(150, 1000))

    light_point = bproc.types.Light()
    light_point.set_energy(light_energy)
    light_point.set_color(np.random.uniform(colour[0], colour[1]))
    light_point.set_location(
        bproc.sampler.shell(
            center=[0, 0, 4],
            radius_min=0.05,
            radius_max=1.0,
            elevation_min=-1,
            elevation_max=1,
            uniform_volume=True,
        )
    )
    return light_point, light_plane_material


def _prepare_objects(cfg):
    sampled_objs = load_objs(
        bop_dataset_path=os.path.join(cfg.dataset_parent_path, cfg.bop_dataset_name),
        mm2m=None,
        sample_objects=False,
        num_of_objs_to_sample=9,
        additional_scale=None,
        manifold=True,
        object_model_unit="mm",
    )

    chosen_pose_func = sample_pose_func_drop if np.random.rand() < cfg.probability_drop else sample_pose_func

    bproc.object.sample_poses(
        objects_to_sample=sampled_objs,
        sample_pose_func=chosen_pose_func,
        max_tries=1000,
    )

    for j, obj in enumerate(sampled_objs):
        obj.enable_rigidbody(True, friction=100.0, linear_damping=0.99, angular_damping=0.99)
        obj.set_shading_mode("auto")

        mat, _style = make_random_material(allowed=["plastic_new"], name_prefix=f"obj_{j:06d}")
        mats = obj.get_materials()
        if not mats:
            obj.set_material(0, mat)
        else:
            for i in range(len(mats)):
                obj.set_material(i, mat)

    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=3,
        max_simulation_time=35,
        check_object_interval=2,
        substeps_per_frame=30,
        solver_iters=30,
    )


def _build_rig_poses(cfg, rs: RealSenseProfile):
    rig_poses_ir_left: list[np.ndarray] = []
    rotation_factor = 9.0

    for _ in range(int(cfg.poses_cam)):
        radius_max = round(float(np.random.uniform(0.75, 2.0)), 2)
        cam_loc = bproc.sampler.shell(
            center=[0, 0, 0.1],
            radius_min=0.50,
            radius_max=radius_max,
            elevation_min=5,
            elevation_max=89,
            uniform_volume=False,
        )
        poi = np.array([0, 0, 0], dtype=np.float32)
        cam2world = build_lookat_pose_cam(cam_loc, poi, rotation_factor=rotation_factor)
        rig_poses_ir_left.append(cam2world)

    T_color_from_ir_left = rs.get_T_blender("IR_LEFT", "COLOR")
    T_color_from_ir_right = rs.get_T_blender("IR_RIGHT", "COLOR")
    T_ir_left_from_color = np.linalg.inv(T_color_from_ir_left)

    rig_poses_color = [T_w_irL @ T_ir_left_from_color for T_w_irL in rig_poses_ir_left]
    rig_poses_ir_right = [T_w_c @ T_color_from_ir_right for T_w_c in rig_poses_color]

    return rig_poses_ir_left, rig_poses_ir_right, rig_poses_color


def _section_get(section, key: str, default):
    if section is None:
        return default
    if isinstance(section, dict):
        return section.get(key, default)
    return getattr(section, key, default)


def _sample_uniform_from_range(value, default):
    if value is None:
        value = default

    if isinstance(value, (int, float)):
        return float(value)

    vals = list(value)
    if len(vals) == 0:
        vals = list(default)
    if len(vals) == 1:
        return float(vals[0])

    lo = float(vals[0])
    hi = float(vals[1])
    if hi < lo:
        lo, hi = hi, lo
    return float(np.random.uniform(lo, hi))


def _sample_effective_energy_config(cfg) -> dict:
    section = getattr(cfg, "effective_projector_render", None)
    rgb_section = getattr(cfg, "rgb_render", None)
    return {
        "projector_energy": _sample_uniform_from_range(
            _section_get(section, "projector_energy_range", [50.0, 150.0]),
            [50.0, 150.0],
        ),
        "rgb_light_energy": _sample_uniform_from_range(
            _section_get(
                rgb_section,
                "light_energy_range",
                _section_get(section, "rgb_light_energy_range", [150.0, 400.0]),
            ),
            [150.0, 400.0],
        ),
        "ir_light_energy": _sample_uniform_from_range(
            _section_get(section, "ir_light_energy_range", [10.0, 60.0]),
            [10.0, 60.0],
        ),
    }


def _find_node_input(node, socket_name: str):
    for sock in node.inputs:
        if sock.name == socket_name:
            return sock
    return None


def _iter_world_background_nodes():
    world = getattr(bpy.context.scene, "world", None)
    if world is None or not getattr(world, "use_nodes", False) or world.node_tree is None:
        return []
    return [node for node in world.node_tree.nodes if node.type == "BACKGROUND"]


def _iter_material_emission_nodes(material):
    mat = getattr(material, "blender_obj", material)
    if mat is None or not getattr(mat, "use_nodes", False) or mat.node_tree is None:
        return []
    return [node for node in mat.node_tree.nodes if node.type == "EMISSION"]


def _capture_render_lighting_state(light_point, emissive_material) -> dict:
    point_light_obj = getattr(light_point, "blender_obj", None)
    point_energy = None if point_light_obj is None else float(point_light_obj.data.energy)

    world_strengths = []
    for node in _iter_world_background_nodes():
        sock = _find_node_input(node, "Strength")
        if sock is not None:
            world_strengths.append((sock, float(sock.default_value)))

    emissive_strengths = []
    for node in _iter_material_emission_nodes(emissive_material):
        sock = _find_node_input(node, "Strength")
        if sock is not None:
            emissive_strengths.append((sock, float(sock.default_value)))

    return {
        "point_light_obj": point_light_obj,
        "point_energy": point_energy,
        "world_strengths": world_strengths,
        "emissive_strengths": emissive_strengths,
    }


def _apply_lighting_energy(state: dict, light_energy: float) -> None:
    target_energy = max(0.0, float(light_energy))
    point_light_obj = state.get("point_light_obj")
    base_energy = state.get("point_energy")

    scale = 1.0
    if base_energy is not None:
        base_energy = float(base_energy)
        if base_energy > 1e-8:
            scale = target_energy / base_energy
        else:
            scale = 0.0

    if point_light_obj is not None:
        point_light_obj.data.energy = target_energy

    for sock, base_value in state.get("world_strengths", []):
        sock.default_value = float(base_value) * scale

    for sock, base_value in state.get("emissive_strengths", []):
        sock.default_value = float(base_value) * scale


def _restore_render_lighting_state(state: dict):
    point_light_obj = state.get("point_light_obj")
    point_energy = state.get("point_energy")
    if point_light_obj is not None and point_energy is not None:
        point_light_obj.data.energy = float(point_energy)

    for sock, base_value in state.get("world_strengths", []):
        sock.default_value = float(base_value)

    for sock, base_value in state.get("emissive_strengths", []):
        sock.default_value = float(base_value)


def main(argv=None):
    bproc.init()
    bproc.utility.reset_keyframes()

    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    cfg = Config(args.config_file)

    rs = RealSenseProfile.from_json(args.camera_profile_json)

    max_spp = int(np.random.randint(300, 1000)) if getattr(cfg, "max_amount_of_samples", None) is None else int(cfg.max_amount_of_samples)
    bproc.renderer.set_max_amount_of_samples(max_spp)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)

    light_point, light_plane_material = _build_room_and_lights(cfg)
    energy_cfg = _sample_effective_energy_config(cfg)
    lighting_base_state = None

    output_dir = args.output_dir
    if output_dir is None:
        base_output_dir = getattr(cfg, "output_dir", "test_effective_projector")
        output_dir = str(Path(base_output_dir) / "effective_projector_depth")
    out_dir = _ensure_output_dir(output_dir)

    sampled_projector_energy = float(energy_cfg["projector_energy"] if args.energy is None else args.energy)
    overrides = ProjectorOverrides(projector_energy=sampled_projector_energy)

    try:
        _prepare_objects(cfg)
        lighting_base_state = _capture_render_lighting_state(light_point, light_plane_material)
        _apply_lighting_energy(lighting_base_state, float(energy_cfg["rgb_light_energy"]))
        rig_poses_ir_left, _rig_poses_ir_right, rig_poses_color = _build_rig_poses(cfg, rs)

        print(
            "[Energy] "
            f"projector={sampled_projector_energy:.3f} | "
            f"rgb_light_energy={float(energy_cfg['rgb_light_energy']):.3f} | "
            f"ir_light_energy={float(energy_cfg['ir_light_energy']):.3f}"
        )

        frame_idx = int(args.frame_idx)
        if frame_idx < 0 or frame_idx >= len(rig_poses_ir_left):
            raise IndexError(
                f"frame_idx={frame_idx} is out of range for {len(rig_poses_ir_left)} available poses"
            )

        left_pose = np.asarray(rig_poses_ir_left[frame_idx], dtype=np.float64)
        color_pose = np.asarray(rig_poses_color[frame_idx], dtype=np.float64)

        _apply_lighting_energy(lighting_base_state, float(energy_cfg["ir_light_energy"]))

        try:
            render_data = render_active_stereo_pair(
                rs,
                left_stream="IR_LEFT",
                right_stream="IR_RIGHT",
                overrides=overrides,
                left_world_pose=left_pose,
                mount_name="rs_projector_mount_effective_depth",
                socket_name="rs_projector_socket_effective_depth",
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

        stem = f"{frame_idx:06d}"
        _save_png(out_dir / f"{stem}_rgb.png", rgb)
        _save_png(out_dir / f"{stem}_left_ir.png", left_ir)
        _save_png(out_dir / f"{stem}_right_ir.png", right_ir)
        _save_png(out_dir / f"{stem}_depth_gt_ir_preview.png", _depth_preview(depth_gt_ir_m))
        _save_png(out_dir / f"{stem}_depth_rect_preview.png", _depth_preview(depth_rect_m))
        _save_png(out_dir / f"{stem}_depth_rgb_preview.png", _depth_preview(depth_rgb_m))
        _save_png(out_dir / f"{stem}_depth_gt_rgb_preview.png", _depth_preview(depth_gt_rgb_m))
        _save_png(out_dir / f"{stem}_disp_rect_preview.png", _disp_preview(disp_rect_px))

        np.save(out_dir / f"{stem}_depth_gt_ir.npy", depth_gt_ir_m)
        np.save(out_dir / f"{stem}_depth_rect.npy", depth_rect_m)
        np.save(out_dir / f"{stem}_depth_rgb.npy", depth_rgb_m)
        np.save(out_dir / f"{stem}_depth_gt_rgb.npy", depth_gt_rgb_m)
        np.save(out_dir / f"{stem}_disp_rect.npy", disp_rect_px)

        _save_png(out_dir / "projector_pattern.png", np.asarray(render_data["projector_pattern_rgba"], dtype=np.uint8))

        print(
            f"[Frame {frame_idx}] "
            f"valid_fraction={stereo['depth_stats']['valid_fraction']:.4f} | "
            f"mean_depth_m={stereo['depth_stats']['mean_depth_m']:.4f} | "
            f"median_depth_m={stereo['depth_stats']['median_depth_m']:.4f}"
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

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
