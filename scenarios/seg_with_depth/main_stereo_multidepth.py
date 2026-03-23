import blenderproc as bproc

import argparse
import os
import random
import sys
from pathlib import Path
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
from blendforge.blender_runtime.writer.RGBDStereoCOCOWriter import (
    write_coco_with_stereo_depth_annotations,
)
from blendforge.blender_runtime.writer.YoloWriterUtility import write_yolo_annotations


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Fractured RGB-D COCO scenario with stereo multidepth outputs.")
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


def _render_color_frames(rs: RealSenseProfile, color_poses):
    rs.set_bproc_intrinsics("COLOR")
    for pose in color_poses:
        bproc.camera.add_camera_pose(np.asarray(pose, dtype=np.float64))
    return bproc.renderer.render()


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


def main(argv=None):
    bproc.init()
    enable("object_fracture_cell")
    bproc.utility.reset_keyframes()

    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    cfg = Config(args.config_file)

    camera_profile_json = str(getattr(cfg, "camera_profile_json", "")).strip()
    if not camera_profile_json:
        raise ValueError("camera_profile_json must be provided in the config for seg_with_depth_stereo_multidepth.")
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
    yolo_out_dir = output_root / "yolo_data"
    coco_out_dir.mkdir(parents=True, exist_ok=True)
    yolo_out_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(output_root / ".lock"))

    save_ir_pairs = bool(cfg_get(getattr(cfg, "stereo_output", None), "save_ir_pairs", True))
    depth_value_mode = str(cfg_get(getattr(cfg, "stereo_output", None), "depth_value_mode", "target_z"))
    splat_2x2 = bool(cfg_get(getattr(cfg, "stereo_output", None), "splat_2x2", True))
    color_file_format = str(getattr(cfg, "color_file_format", "JPEG"))
    depth_scale = float(getattr(cfg, "depth_scale_mm", 1.0))
    num_runs = int(getattr(cfg, "runs_one_fracture", getattr(cfg, "runs", 1)))

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

            # COCO-ветки в репозитории работают именно так: сначала batch render всех поз
            # с segmentation output, затем writer берет instance maps из общего render data.
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
                file_prefix=f"segmap_id{cfg.index_device}ip{cfg.process_id}_",
            )
            data_rgb = _render_color_frames(rs, rig_poses_color)

            print(
                "[Seg Stereo MultiDepth] "
                f"run={run_idx + 1}/{num_runs} | poses={len(rig_poses_ir_left)} | seed={seed}"
            )

            for pose_idx, (left_pose, _color_pose) in enumerate(zip(rig_poses_ir_left, rig_poses_color)):
                energy_cfg = sample_effective_energy_config(cfg)
                render_sample_cfg = sample_render_sample_config(cfg)
                overrides = ProjectorOverrides(projector_energy=float(energy_cfg["projector_energy"]))

                effective_section = _get_branch_matcher_section(cfg, "effective")
                random_section = _get_branch_matcher_section(cfg, "random")
                effective_matcher_kwargs = build_matcher_kwargs(effective_section, rs, stream="IR_LEFT")
                random_matcher_kwargs = build_matcher_kwargs(random_section, rs, stream="IR_LEFT")

                effective_mode = str(cfg_get(effective_section, "rgb_to_intensity_mode", "lcn"))
                random_mode = str(cfg_get(random_section, "rgb_to_intensity_mode", "lcn"))
                effective_plane_distance = _sample_plane_distance(effective_section)
                random_plane_distance = _sample_plane_distance(random_section)

                bproc.renderer.set_max_amount_of_samples(int(render_sample_cfg["stereo_max_amount_of_samples"]))
                apply_lighting_energy(lighting_base_state, float(energy_cfg["ir_light_energy"]))
                try:
                    render_effective = render_stereo_branch_pair(
                        rs,
                        left_pose=np.asarray(left_pose, dtype=np.float64),
                        overrides=overrides,
                        stereo_branch="effective",
                        cfg=cfg,
                    )
                    render_random = render_stereo_branch_pair(
                        rs,
                        left_pose=np.asarray(left_pose, dtype=np.float64),
                        overrides=overrides,
                        stereo_branch="random_pattern",
                        cfg=cfg,
                    )
                finally:
                    apply_lighting_energy(lighting_base_state, float(energy_cfg["rgb_light_energy"]))

                left_ir_effective = rgb_to_intensity_u8(render_effective["left_colors"], mode=effective_mode)
                right_ir_effective = rgb_to_intensity_u8(render_effective["right_colors"], mode=effective_mode)
                left_ir_random = rgb_to_intensity_u8(render_random["left_colors"], mode=random_mode)
                right_ir_random = rgb_to_intensity_u8(render_random["right_colors"], mode=random_mode)

                stereo_effective = stereo_from_ir_pair(
                    rs,
                    left_stream="IR_LEFT",
                    right_stream="IR_RIGHT",
                    left_ir_u8=left_ir_effective,
                    right_ir_u8=right_ir_effective,
                    left_depth_gt_m=render_effective["left_depth_m"],
                    plane_distance_m=effective_plane_distance,
                    **effective_matcher_kwargs,
                )
                stereo_random = stereo_from_ir_pair(
                    rs,
                    left_stream="IR_LEFT",
                    right_stream="IR_RIGHT",
                    left_ir_u8=left_ir_random,
                    right_ir_u8=right_ir_random,
                    left_depth_gt_m=render_random["left_depth_m"],
                    plane_distance_m=random_plane_distance,
                    **random_matcher_kwargs,
                )

                rgb = np.asarray(data_rgb["colors"][pose_idx])
                depth_gt_rgb = np.asarray(data_rgb["depth"][pose_idx], dtype=np.float32)
                depth_effective_rgb = _align_depth_to_color(
                    rs,
                    np.asarray(stereo_effective["depth_rect_m"], dtype=np.float32),
                    depth_value_mode=depth_value_mode,
                    splat_2x2=splat_2x2,
                )
                depth_random_rgb = _align_depth_to_color(
                    rs,
                    np.asarray(stereo_random["depth_rect_m"], dtype=np.float32),
                    depth_value_mode=depth_value_mode,
                    splat_2x2=splat_2x2,
                )

                with lock:
                    write_coco_with_stereo_depth_annotations(
                        output_dir=str(coco_out_dir),
                        instance_segmaps=[data_rgb["instance_segmaps"][pose_idx]],
                        instance_attribute_maps=[data_rgb["instance_attribute_maps"][pose_idx]],
                        colors=[rgb],
                        depths_m=[depth_gt_rgb],
                        depth_effective_m=[depth_effective_rgb],
                        depth_random_m=[depth_random_rgb],
                        ir_left_effective=[left_ir_effective] if save_ir_pairs else None,
                        ir_right_effective=[right_ir_effective] if save_ir_pairs else None,
                        ir_left_random=[left_ir_random] if save_ir_pairs else None,
                        ir_right_random=[right_ir_random] if save_ir_pairs else None,
                        color_file_format=color_file_format,
                        append_to_existing_output=True,
                        jpg_quality=int(getattr(cfg, "jpg_quality", 95)),
                        depth_unit="mm",
                        depth_scale_mm=depth_scale,
                    )
                    write_yolo_annotations(
                        output_root=str(yolo_out_dir),
                        split=split_name,
                        instance_segmaps=[data_rgb["instance_segmaps"][pose_idx]],
                        instance_attribute_maps=[data_rgb["instance_attribute_maps"][pose_idx]],
                        colors=[rgb],
                        color_file_format=color_file_format,
                        append_to_existing_output=True,
                        jpg_quality=int(getattr(cfg, "jpg_quality", 95)),
                        polygon_tolerance_px=2.0,
                        min_area_px=10,
                        one_contour_per_instance=True,
                    )

                print(
                    "[Seg Stereo MultiDepth] "
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
