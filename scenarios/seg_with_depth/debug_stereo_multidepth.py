import blenderproc as bproc

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from addon_utils import enable
from filelock import FileLock

from blendforge.host.FiletoDict import Config
from blendforge.blender_runtime.ActiveStereoSceneUtility import (
    apply_lighting_energy,
    build_room_and_lights,
    capture_render_lighting_state,
    restore_render_lighting_state,
    sample_effective_energy_config,
    sample_render_sample_config,
)
from blendforge.blender_runtime.CustomFractureUtills import fracture_object_with_cell
from blendforge.blender_runtime.CustomLoadMesh import load_objs
from blendforge.blender_runtime.CustomMaterial import create_mat, make_random_material
from blendforge.blender_runtime.camera.ActiveStereoBranchRuntime import render_stereo_branch_pair
from blendforge.blender_runtime.camera.ActiveStereoProjectorRuntime import ProjectorOverrides
from blendforge.blender_runtime.camera.RealsenseProfileLoader import RealSenseProfile
from blendforge.blender_runtime.stereo.ActiveStereoIRUtility import rgb_to_intensity_u8, stereo_from_ir_pair
from blendforge.blender_runtime.stereo.DepthAlignment import align_depth_series_ir_left_to_color
from blendforge.blender_runtime.stereo.MatcherConfigUtility import (
    build_matcher_kwargs,
    cfg_get,
    sample_scalar_or_range,
)
from blendforge.blender_runtime.utils import build_lookat_pose_cam, sample_pose_func, sample_pose_func_drop
from blendforge.blender_runtime.writer.RGBDCOCOWriter import (
    _ensure_dir,
    _save_png_u16,
    _save_rgb,
    meters_to_depth_u16,
)
from blendforge.blender_runtime.writer.RGBDStereoCOCOWriter import (
    write_coco_with_stereo_depth_annotations,
)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Debug RGB-D COCO scenario with stereo multidepth and extra raw/matched outputs."
    )
    parser.add_argument("--config_file", type=str, required=True)
    return parser.parse_args(argv)


def _get_branch_matcher_section(cfg, branch_name: str):
    specific = getattr(cfg, f"matcher_{branch_name}", None)
    if specific is not None:
        return specific
    return getattr(cfg, "matcher_data", None)


def _sample_choice_or_range(value, *, cast):
    if isinstance(value, (int, float)):
        return cast(value)

    vals = list(value)
    if len(vals) == 0:
        raise ValueError("Range must not be empty")
    if len(vals) == 1:
        return cast(vals[0])
    if len(vals) > 2:
        return cast(random.choice(vals))
    return sample_scalar_or_range(vals, cast=cast)


def _sample_plane_distance(section):
    raw = cfg_get(section, "plane_distance_m", None)
    if raw is None:
        return None
    return float(sample_scalar_or_range(raw, cast=float))


def _prepare_fragments_for_scene(objects, cfg):
    chosen_pose_func = sample_pose_func_drop if np.random.rand() < float(cfg.probability_drop) else sample_pose_func
    material_randomization = getattr(cfg, "material_randomization", None)
    prob_make_random_material = float(cfg_get(material_randomization, "prob_make_random_material", 0.5))
    prob_make_random_material = float(np.clip(prob_make_random_material, 0.0, 1.0))
    allowed_random_materials = cfg_get(
        material_randomization,
        "make_random_material_allowed",
        [
            "metal",
            "dirty_metal",
            "cast_iron",
            "steel",
            "brushed_steel",
            "galvanized_steel",
            "blackened_steel",
            "plastic_new",
            "plastic_old",
        ],
    )

    for idx, obj in enumerate(objects):
        obj.enable_rigidbody(True, friction=100.0, linear_damping=0.99, angular_damping=0.99)
        obj.set_shading_mode("auto")

        if np.random.rand() < prob_make_random_material:
            mat, _style = make_random_material(
                allowed=allowed_random_materials,
                name_prefix=f"obj_{idx:06d}",
            )
        else:
            mat = create_mat(f"obj_{idx:06d}")

        materials = obj.get_materials()
        if not materials:
            obj.set_material(0, mat)
        else:
            for material_index in range(len(materials)):
                obj.set_material(material_index, mat)

    bproc.object.sample_poses(
        objects_to_sample=objects,
        sample_pose_func=chosen_pose_func,
        max_tries=1000,
    )

    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=3,
        max_simulation_time=35,
        check_object_interval=2,
        substeps_per_frame=30,
        solver_iters=30,
    )


def _fracture_objects(sampled_objs, cfg):
    fracture_cfg = getattr(cfg, "fracture", None)
    source_limit_and_count_choices = cfg_get(fracture_cfg, "source_limit_and_count_choices", [2, 3, 4, 5])
    source_noise_range = cfg_get(fracture_cfg, "source_noise_range", [0.0, 0.007])
    cell_scale_range = cfg_get(fracture_cfg, "cell_scale_range", [0.75, 1.5])
    fracture_scale = float(cfg_get(fracture_cfg, "fracture_scale", 0.05))
    seed_range = cfg_get(fracture_cfg, "seed_range", [2, 40])

    fractured_objs = []
    remaining = list(sampled_objs)
    while remaining:
        obj = remaining.pop(0)
        shards = fracture_object_with_cell(
            bpy_obj=obj.blender_obj,
            source_limit_and_count=int(_sample_choice_or_range(source_limit_and_count_choices, cast=int)),
            source_noise=float(_sample_choice_or_range(source_noise_range, cast=float)),
            cell_scale=(
                float(_sample_choice_or_range(cell_scale_range, cast=float)),
                float(_sample_choice_or_range(cell_scale_range, cast=float)),
                float(_sample_choice_or_range(cell_scale_range, cast=float)),
            ),
            scale=fracture_scale,
            seed=int(_sample_choice_or_range(seed_range, cast=int)),
        )
        fractured_objs.extend(shards)

    return fractured_objs


def _render_color_frames(rs: RealSenseProfile, color_poses, *, output_dir=None, file_prefix=None):
    rs.set_bproc_intrinsics("COLOR")
    for pose in color_poses:
        bproc.camera.add_camera_pose(np.asarray(pose, dtype=np.float64))
    if output_dir is None and file_prefix is None:
        data = bproc.renderer.render()
    else:
        render_kwargs = {}
        if output_dir is not None:
            render_kwargs["output_dir"] = output_dir
        if file_prefix is not None:
            render_kwargs["file_prefix"] = file_prefix
        data = bproc.renderer.render(**render_kwargs)
    safe = dict(data)
    if "colors" in data:
        safe["colors"] = [np.asarray(frame).copy() for frame in data["colors"]]
    if "depth" in data:
        safe["depth"] = [np.asarray(frame, dtype=np.float32).copy() for frame in data["depth"]]
    if "instance_segmaps" in data:
        safe["instance_segmaps"] = [np.asarray(frame).copy() for frame in data["instance_segmaps"]]
    if "instance_attribute_maps" in data:
        safe["instance_attribute_maps"] = copy.deepcopy(data["instance_attribute_maps"])
    return safe


def _build_rig_poses_for_fragments(cfg, rs: RealSenseProfile):
    rig_poses_ir_left = []
    rotation_factor = float(getattr(cfg, "camera_rotation_factor", 11.0))
    radius_min = float(getattr(cfg, "camera_radius_min", 0.40))
    radius_max_range = getattr(cfg, "camera_radius_max_range", [0.5, 1.0])
    shell_center = np.array(getattr(cfg, "camera_shell_center", [0.0, 0.0, 0.1]), dtype=np.float32)
    poi = np.array(getattr(cfg, "camera_poi", [0.0, 0.0, 0.0]), dtype=np.float32)
    elevation_min = float(getattr(cfg, "camera_elevation_min", 5.0))
    elevation_max = float(getattr(cfg, "camera_elevation_max", 89.0))

    for _ in range(int(cfg.poses_cam)):
        sampled_radius_max = sample_scalar_or_range(radius_max_range, cast=float)
        cam_loc = bproc.sampler.shell(
            center=shell_center.tolist(),
            radius_min=radius_min,
            radius_max=float(sampled_radius_max),
            elevation_min=elevation_min,
            elevation_max=elevation_max,
            uniform_volume=False,
        )
        rig_poses_ir_left.append(build_lookat_pose_cam(cam_loc, poi, rotation_factor=rotation_factor))

    T_color_from_ir_left = rs.get_T_blender("IR_LEFT", "COLOR")
    T_color_from_ir_right = rs.get_T_blender("IR_RIGHT", "COLOR")
    T_ir_left_from_color = np.linalg.inv(T_color_from_ir_left)

    rig_poses_color = [T_w_ir_left @ T_ir_left_from_color for T_w_ir_left in rig_poses_ir_left]
    rig_poses_ir_right = [T_w_color @ T_color_from_ir_right for T_w_color in rig_poses_color]
    return rig_poses_ir_left, rig_poses_ir_right, rig_poses_color


def _align_depth_to_color(rs: RealSenseProfile, depth_rect_m: np.ndarray, *, depth_value_mode: str, splat_2x2: bool):
    depth_list, _mask_list = align_depth_series_ir_left_to_color(
        rs,
        [np.asarray(depth_rect_m, dtype=np.float32)],
        depth_value_mode=depth_value_mode,
        splat_2x2=bool(splat_2x2),
        rectify_mode="on",
    )
    return np.asarray(depth_list[0], dtype=np.float32)


def _normalize_stereo_branch_mode(value):
    if isinstance(value, (list, tuple, set)):
        tokens = [str(item).strip().lower() for item in value]
    else:
        raw = str(value if value is not None else "both").strip().lower()
        if "," in raw:
            tokens = [part.strip() for part in raw.split(",") if part.strip()]
        else:
            tokens = [raw]

    enabled = set()
    for token in tokens:
        if token in {"both", "all"}:
            enabled.update({"effective", "random"})
        elif token in {"effective", "effective_only"}:
            enabled.add("effective")
        elif token in {"random", "random_only", "random_pattern"}:
            enabled.add("random")
        else:
            raise ValueError(
                "Unsupported stereo_branch_mode="
                f"{value!r}. Expected one of: 'effective', 'random', 'both'."
            )

    ordered = [name for name in ["effective", "random"] if name in enabled]
    if not ordered:
        raise ValueError("stereo_branch_mode must enable at least one branch")
    normalized = "both" if len(ordered) == 2 else ordered[0]
    return ordered, normalized


def _color_extension(color_file_format: str) -> str:
    return "jpg" if str(color_file_format).upper() in {"JPG", "JPEG"} else "png"


def _to_u8_preview(img: np.ndarray, *, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    valid = np.isfinite(x) & (x > 0.0)
    if x.size == 0 or not np.any(valid):
        return np.zeros((1, 1), dtype=np.uint8)
    xv = x[valid]
    lo = float(np.percentile(xv, low))
    hi = float(np.percentile(xv, high))
    if hi <= lo + 1e-6:
        hi = lo + 1e-6
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    out = (y * 255.0 + 0.5).astype(np.uint8)
    out[~valid] = 0
    return out


def _depth_preview_gray(depth_m: np.ndarray) -> np.ndarray:
    d = np.asarray(depth_m, dtype=np.float32)
    inv = np.where(d > 0.0, 1.0 / np.maximum(d, 1e-6), 0.0)
    return _to_u8_preview(inv, low=1.0, high=99.0)


def _save_gray_png(path: Path, img_u8: np.ndarray) -> None:
    _ensure_dir(str(path.parent))
    cv2.imwrite(str(path), np.asarray(img_u8, dtype=np.uint8))


def _peek_next_coco_image_id(coco_out_dir: Path) -> int:
    coco_path = coco_out_dir / "coco_annotations.json"
    if not coco_path.exists():
        return 0

    with coco_path.open("r", encoding="utf-8") as file:
        coco_data = json.load(file)

    images = coco_data.get("images") or []
    if not images:
        return 0

    return max(int(image.get("id", 0)) for image in images) + 1


def _build_debug_dirs(coco_out_dir: Path) -> dict[str, Path]:
    return {
        "raw_left_effective": coco_out_dir / "raw_left_effective",
        "raw_right_effective": coco_out_dir / "raw_right_effective",
        "raw_left_random": coco_out_dir / "raw_left_random",
        "raw_right_random": coco_out_dir / "raw_right_random",
        "depth_left_rect_effective": coco_out_dir / "depth_left_rect_effective",
        "depth_left_rect_random": coco_out_dir / "depth_left_rect_random",
    }


def _save_debug_branch_artifacts(
    *,
    debug_dirs: dict[str, Path],
    image_id: int,
    render_outputs: dict,
    stereo_outputs: dict,
    raw_color_file_format: str,
    raw_jpg_quality: int,
    depth_scale_mm: float,
    depth_save_mode: str,
) -> None:
    stem = f"{int(image_id):06d}"
    raw_ext = _color_extension(raw_color_file_format)

    for path in debug_dirs.values():
        _ensure_dir(str(path))

    for branch_name in render_outputs.keys():
        branch_render = render_outputs[branch_name]
        branch_stereo = stereo_outputs[branch_name]

        left_raw_path = debug_dirs[f"raw_left_{branch_name}"] / f"{stem}.{raw_ext}"
        right_raw_path = debug_dirs[f"raw_right_{branch_name}"] / f"{stem}.{raw_ext}"
        depth_rect_path = debug_dirs[f"depth_left_rect_{branch_name}"] / f"{stem}.png"

        _save_rgb(
            str(left_raw_path),
            np.asarray(branch_render["left_colors"]),
            fmt=raw_color_file_format,
            jpg_quality=raw_jpg_quality,
        )
        _save_rgb(
            str(right_raw_path),
            np.asarray(branch_render["right_colors"]),
            fmt=raw_color_file_format,
            jpg_quality=raw_jpg_quality,
        )

        depth_rect = np.asarray(branch_stereo["depth_rect_m"], dtype=np.float32)
        if depth_save_mode == "u16":
            _save_png_u16(
                str(depth_rect_path),
                meters_to_depth_u16(
                    depth_rect,
                    depth_scale_mm=float(depth_scale_mm),
                ),
            )
        elif depth_save_mode == "preview_gray":
            _save_gray_png(depth_rect_path, _depth_preview_gray(depth_rect))
        else:
            raise ValueError(
                f"Unsupported debug_output.depth_save_mode={depth_save_mode!r}. "
                "Expected one of: 'u16', 'preview_gray'."
            )


def _overwrite_coco_matched_depth_files(
    *,
    coco_out_dir: Path,
    image_id: int,
    depth_gt_rgb: np.ndarray,
    aligned_depths: dict,
    depth_scale_mm: float,
    gt_save_mode: str,
    save_mode: str,
) -> None:
    stem = f"{int(image_id):06d}"

    depth_gt_path = coco_out_dir / "depth" / f"{stem}.png"
    depth_gt = np.asarray(depth_gt_rgb, dtype=np.float32)

    if gt_save_mode == "preview_gray":
        _save_gray_png(depth_gt_path, _depth_preview_gray(depth_gt))
    elif gt_save_mode != "u16":
        raise ValueError(
            f"Unsupported debug_output.coco_gt_depth_save_mode={gt_save_mode!r}. "
            "Expected one of: 'u16', 'preview_gray'."
        )

    if save_mode == "u16":
        return

    for branch_name in ("effective", "random"):
        if branch_name not in aligned_depths:
            continue

        depth_path = coco_out_dir / f"depth_{branch_name}" / f"{stem}.png"
        depth = np.asarray(aligned_depths[branch_name], dtype=np.float32)

        if save_mode == "preview_gray":
            _save_gray_png(depth_path, _depth_preview_gray(depth))
        elif save_mode == "u16":
            _save_png_u16(
                str(depth_path),
                meters_to_depth_u16(depth, depth_scale_mm=float(depth_scale_mm)),
            )
        else:
            raise ValueError(
                f"Unsupported debug_output.coco_matched_depth_save_mode={save_mode!r}. "
                "Expected one of: 'u16', 'preview_gray'."
            )


def main(argv=None):
    bproc.init()
    enable("object_fracture_cell")
    bproc.utility.reset_keyframes()

    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    cfg = Config(args.config_file)
    file_prefix_rgb = str(getattr(cfg, "file_prefix_rgb", f"rgb_id{cfg.index_device}ip{cfg.process_id}_"))
    file_prefix_segmap = str(getattr(cfg, "file_prefix_segmap", f"segmap_id{cfg.index_device}ip{cfg.process_id}_"))

    camera_profile_json = str(getattr(cfg, "camera_profile_json", "")).strip()
    if not camera_profile_json:
        raise ValueError("camera_profile_json must be provided in the config for debug_stereo_multidepth.")
    rs = RealSenseProfile.from_json(camera_profile_json)

    seed = int(getattr(cfg, "scene_seed", random.SystemRandom().randrange(0, 2**31 - 1)))
    np.random.seed(seed)
    random.seed(seed)

    render_sample_cfg = sample_render_sample_config(cfg)
    bproc.renderer.set_max_amount_of_samples(int(render_sample_cfg["stereo_max_amount_of_samples"]))
    bproc.renderer.enable_depth_output(activate_antialiasing=False)

    sampled_objs = load_objs(
        bop_dataset_path=os.path.join(cfg.dataset_parent_path, cfg.bop_dataset_name),
        mm2m=None,
        sample_objects=False,
        num_of_objs_to_sample=9,
        additional_scale=None,
        manifold=True,
        object_model_unit=str(getattr(cfg, "object_model_unit", "cm")),
    )
    fractured_objs = _fracture_objects(sampled_objs, cfg)

    room_planes, light_point, light_plane_material = build_room_and_lights(cfg)
    cc_textures = bproc.loader.load_ccmaterials(cfg.cc_textures.cc_textures_path)

    split_name = str(getattr(cfg, "dataset_type", "train"))
    output_root = Path(cfg.output_dir).resolve()
    coco_out_dir = output_root / "coco_data" / split_name
    coco_out_dir.mkdir(parents=True, exist_ok=True)
    debug_dirs = _build_debug_dirs(coco_out_dir)
    lock = FileLock(str(output_root / ".lock"))

    save_ir_pairs = bool(cfg_get(getattr(cfg, "stereo_output", None), "save_ir_pairs", True))
    depth_value_mode = str(cfg_get(getattr(cfg, "stereo_output", None), "depth_value_mode", "target_z"))
    splat_2x2 = bool(cfg_get(getattr(cfg, "stereo_output", None), "splat_2x2", True))
    color_file_format = str(getattr(cfg, "color_file_format", "JPEG"))
    depth_scale = float(getattr(cfg, "depth_scale_mm", 1.0))
    coco_json_indent = getattr(cfg, "coco_json_indent", None)
    num_runs = int(getattr(cfg, "runs_one_fracture", getattr(cfg, "runs", 1)))
    enabled_branches, stereo_branch_mode = _normalize_stereo_branch_mode(
        getattr(cfg, "stereo_branch_mode", "both")
    )

    debug_output_cfg = getattr(cfg, "debug_output", None)
    raw_color_file_format = str(cfg_get(debug_output_cfg, "raw_color_file_format", "PNG"))
    raw_jpg_quality = int(cfg_get(debug_output_cfg, "raw_jpg_quality", getattr(cfg, "jpg_quality", 95)))
    depth_save_mode = str(cfg_get(debug_output_cfg, "depth_save_mode", "preview_gray")).strip().lower()
    coco_gt_depth_save_mode = str(
        cfg_get(debug_output_cfg, "coco_gt_depth_save_mode", "preview_gray")
    ).strip().lower()
    coco_matched_depth_save_mode = str(
        cfg_get(debug_output_cfg, "coco_matched_depth_save_mode", "preview_gray")
    ).strip().lower()

    lighting_base_state = None
    try:
        lighting_base_state = capture_render_lighting_state(light_point, light_plane_material)

        for run_idx in range(num_runs):
            bproc.utility.reset_keyframes()
            random_cc_texture = np.random.choice(cc_textures)
            for plane in room_planes:
                plane.replace_materials(random_cc_texture)

            _prepare_fragments_for_scene(fractured_objs, cfg)
            rig_poses_ir_left, _rig_poses_ir_right, rig_poses_color = _build_rig_poses_for_fragments(cfg, rs)

            rgb_render_cfg = sample_render_sample_config(cfg)
            rgb_energy_cfg = sample_effective_energy_config(cfg)
            bproc.renderer.set_max_amount_of_samples(int(rgb_render_cfg["rgb_max_amount_of_samples"]))
            apply_lighting_energy(lighting_base_state, float(rgb_energy_cfg["rgb_light_energy"]))
            bproc.renderer.enable_segmentation_output(
                map_by=["category_id", "instance", "fragment_id", "fracture_uid", "fracture_seed", "fracture_method"],
                default_values={
                    "category_id": 0,
                    "fragment_id": 0,
                    "fracture_uid": "",
                    "fracture_seed": 0,
                    "fracture_method": "",
                },
                output_dir=cfg.temp_dir_segmap,
                file_prefix=file_prefix_segmap,
            )
            data_rgb = _render_color_frames(
                rs,
                rig_poses_color,
                output_dir=cfg.temp_dir_rgb,
                file_prefix=file_prefix_rgb,
            )

            print(
                "[Debug Seg Stereo MultiDepth] "
                f"run={run_idx + 1}/{num_runs} | poses={len(rig_poses_ir_left)} | "
                f"seed={seed} | branches={stereo_branch_mode}"
            )

            for pose_idx, (left_pose, _color_pose) in enumerate(zip(rig_poses_ir_left, rig_poses_color)):
                energy_cfg = sample_effective_energy_config(cfg)
                render_sample_cfg = sample_render_sample_config(cfg)
                overrides = ProjectorOverrides(
                    projector_energy=float(energy_cfg["projector_energy"]),
                    projector_color_rgb=tuple(float(v) for v in energy_cfg["projector_color_rgb"]),
                )

                branch_cfg = {}
                for branch_name in enabled_branches:
                    section = _get_branch_matcher_section(cfg, branch_name)
                    branch_cfg[branch_name] = {
                        "runtime_branch": "effective" if branch_name == "effective" else "random_pattern",
                        "matcher_kwargs": build_matcher_kwargs(section, rs, stream="IR_LEFT"),
                        "intensity_mode": str(cfg_get(section, "rgb_to_intensity_mode", "lcn")),
                        "plane_distance": _sample_plane_distance(section),
                    }

                bproc.renderer.set_max_amount_of_samples(int(render_sample_cfg["stereo_max_amount_of_samples"]))
                apply_lighting_energy(lighting_base_state, float(energy_cfg["ir_light_energy"]))
                render_outputs = {}
                ir_pairs_saved = {}
                try:
                    for branch_name in enabled_branches:
                        render_branch = render_stereo_branch_pair(
                            rs,
                            left_pose=np.asarray(left_pose, dtype=np.float64),
                            overrides=overrides,
                            stereo_branch=branch_cfg[branch_name]["runtime_branch"],
                            cfg=cfg,
                            output_dir=cfg.temp_dir_rgb,
                            file_prefix_base=file_prefix_rgb,
                        )
                        render_outputs[branch_name] = render_branch
                        left_ir = rgb_to_intensity_u8(
                            render_branch["left_colors"],
                            mode=branch_cfg[branch_name]["intensity_mode"],
                        )
                        right_ir = rgb_to_intensity_u8(
                            render_branch["right_colors"],
                            mode=branch_cfg[branch_name]["intensity_mode"],
                        )
                        ir_pairs_saved[branch_name] = (left_ir.copy(), right_ir.copy())
                finally:
                    apply_lighting_energy(lighting_base_state, float(energy_cfg["rgb_light_energy"]))

                stereo_outputs = {}
                aligned_depths = {}
                for branch_name in enabled_branches:
                    left_ir_saved, right_ir_saved = ir_pairs_saved[branch_name]
                    render_branch = render_outputs[branch_name]
                    stereo_outputs[branch_name] = stereo_from_ir_pair(
                        rs,
                        left_stream="IR_LEFT",
                        right_stream="IR_RIGHT",
                        left_ir_u8=left_ir_saved.copy(),
                        right_ir_u8=right_ir_saved.copy(),
                        left_depth_gt_m=render_branch["left_depth_m"],
                        plane_distance_m=branch_cfg[branch_name]["plane_distance"],
                        **branch_cfg[branch_name]["matcher_kwargs"],
                    )
                    aligned_depths[branch_name] = _align_depth_to_color(
                        rs,
                        np.asarray(stereo_outputs[branch_name]["depth_rect_m"], dtype=np.float32),
                        depth_value_mode=depth_value_mode,
                        splat_2x2=splat_2x2,
                    )

                rgb = np.asarray(data_rgb["colors"][pose_idx])
                depth_gt_rgb = np.asarray(data_rgb["depth"][pose_idx], dtype=np.float32)

                with lock:
                    next_image_id = _peek_next_coco_image_id(coco_out_dir)
                    _save_debug_branch_artifacts(
                        debug_dirs=debug_dirs,
                        image_id=next_image_id,
                        render_outputs=render_outputs,
                        stereo_outputs=stereo_outputs,
                        raw_color_file_format=raw_color_file_format,
                        raw_jpg_quality=raw_jpg_quality,
                        depth_scale_mm=depth_scale,
                        depth_save_mode=depth_save_mode,
                    )
                    write_coco_with_stereo_depth_annotations(
                        output_dir=str(coco_out_dir),
                        instance_segmaps=[data_rgb["instance_segmaps"][pose_idx]],
                        instance_attribute_maps=[data_rgb["instance_attribute_maps"][pose_idx]],
                        colors=[rgb],
                        depths_m=[depth_gt_rgb],
                        depth_effective_m=(
                            [aligned_depths["effective"]] if "effective" in aligned_depths else None
                        ),
                        depth_random_m=(
                            [aligned_depths["random"]] if "random" in aligned_depths else None
                        ),
                        ir_left_effective=(
                            [ir_pairs_saved["effective"][0]]
                            if save_ir_pairs and "effective" in ir_pairs_saved
                            else None
                        ),
                        ir_right_effective=(
                            [ir_pairs_saved["effective"][1]]
                            if save_ir_pairs and "effective" in ir_pairs_saved
                            else None
                        ),
                        ir_left_random=(
                            [ir_pairs_saved["random"][0]]
                            if save_ir_pairs and "random" in ir_pairs_saved
                            else None
                        ),
                        ir_right_random=(
                            [ir_pairs_saved["random"][1]]
                            if save_ir_pairs and "random" in ir_pairs_saved
                            else None
                        ),
                        color_file_format=color_file_format,
                        append_to_existing_output=True,
                        jpg_quality=int(getattr(cfg, "jpg_quality", 95)),
                        indent=coco_json_indent,
                        depth_unit="mm",
                        depth_scale_mm=depth_scale,
                    )
                    _overwrite_coco_matched_depth_files(
                        coco_out_dir=coco_out_dir,
                        image_id=next_image_id,
                        depth_gt_rgb=depth_gt_rgb,
                        aligned_depths=aligned_depths,
                        depth_scale_mm=depth_scale,
                        gt_save_mode=coco_gt_depth_save_mode,
                        save_mode=coco_matched_depth_save_mode,
                    )

                print(
                    "[Debug Seg Stereo MultiDepth] "
                    f"saved pose={pose_idx + 1}/{len(rig_poses_ir_left)}"
                )
    finally:
        try:
            restore_render_lighting_state(lighting_base_state)
        except Exception:
            pass
        try:
            light_point.delete()
        except Exception:
            pass


if __name__ == "__main__":
    main()
