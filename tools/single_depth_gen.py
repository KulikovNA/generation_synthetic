#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import blenderproc as bproc  # <-- ВАЖНО: первый импорт

import os
import sys
import argparse
import numpy as np
import cv2
import bpy

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
from blendforge.blender_runtime.camera.RealsenseProjectorUtility import (
    get_or_create_mount_empty,
    animate_mount_from_cam2world,
    create_realsense_ir_projector,
    resolve_projector_target,
)

# Stereo matching (RECTIFY ALWAYS) - фасад, внутри уже новый модульный pipeline
from blendforge.blender_runtime.stereo.StereoMatching import stereo_global_matching_rectified
from blendforge.blender_runtime.stereo.ActiveStereoIRUtility import (
    build_rectify_from_rs as _shared_build_rectify_from_rs,
    convert_ir_frames_to_intensity as _shared_convert_ir_frames_to_intensity,
    rgb_to_intensity_u8 as _shared_rgb_to_intensity_u8,
)

# (необязательно) если хочешь явно создавать серии с метаданными сетки до матчинга
# from blendforge.blender_runtime.stereo.FrameSeries import FrameSeries, FrameGrid

# Align depth(IR_LEFT or IR_LEFT_RECT grid) -> COLOR grid
from blendforge.blender_runtime.stereo.DepthAlignment import align_depth_series_ir_left_to_color

from blendforge.debug_blender_runtime.ImagesWriterUtility import save_rgb_ir_stereo_rectified


# -------------------- args --------------------

def parse_args(argv):
    p = argparse.ArgumentParser(description="RGB + IR stereo (D435-like) generator (RECTIFY ONLY)")
    p.add_argument("--config_file", type=str, required=True)
    return p.parse_args(argv)


# -------------------- small utils --------------------

def _has_ximgproc() -> bool:
    return hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createDisparityWLSFilter")


def _np5(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    out = np.zeros(5, dtype=np.float64)
    if a.size:
        out[: min(5, a.size)] = a[: min(5, a.size)]
    return out


def _build_rectify_from_rs(
    rs: RealSenseProfile,
    left: str = "IR_LEFT",
    right: str = "IR_RIGHT",
    *,
    use_distortion: bool = False,
):
    return _shared_build_rectify_from_rs(rs, left=left, right=right, use_distortion=use_distortion)


# -------------------- scene helpers --------------------

def _build_room_and_lights(cfg):
    room_planes = [
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1]),
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[0, -3, 3], rotation=[-1.570796, 0, 0]),
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[0,  3, 3], rotation=[ 1.570796, 0, 0]),
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[3,  0, 3], rotation=[0, -1.570796, 0]),
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[-3, 0, 3], rotation=[0,  1.570796, 0]),
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


def _get_ir_render_value(cfg, key: str, default):
    ir_cfg = getattr(cfg, "ir_render", None)
    if ir_cfg is None:
        return default
    return getattr(ir_cfg, key, default)


def _get_ir_render_config(cfg) -> dict:
    return {
        "enabled": bool(_get_ir_render_value(cfg, "enabled", True)),
        "environment_scale": float(_get_ir_render_value(cfg, "environment_scale", 0.15)),
        "point_light_scale": float(_get_ir_render_value(cfg, "point_light_scale", 0.15)),
        "emissive_scale": float(_get_ir_render_value(cfg, "emissive_scale", 0.15)),
        "rgb_to_intensity_mode": str(_get_ir_render_value(cfg, "rgb_to_intensity_mode", "bt601")),
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


def _apply_ir_lighting_state(state: dict, ir_cfg: dict):
    point_scale = max(0.0, float(ir_cfg["point_light_scale"]))
    env_scale = max(0.0, float(ir_cfg["environment_scale"]))
    emissive_scale = max(0.0, float(ir_cfg["emissive_scale"]))

    point_light_obj = state.get("point_light_obj")
    point_energy = state.get("point_energy")
    if point_light_obj is not None and point_energy is not None:
        point_light_obj.data.energy = float(point_energy) * point_scale

    for sock, base_value in state.get("world_strengths", []):
        sock.default_value = float(base_value) * env_scale

    for sock, base_value in state.get("emissive_strengths", []):
        sock.default_value = float(base_value) * emissive_scale


def _restore_render_lighting_state(state: dict):
    point_light_obj = state.get("point_light_obj")
    point_energy = state.get("point_energy")
    if point_light_obj is not None and point_energy is not None:
        point_light_obj.data.energy = float(point_energy)

    for sock, base_value in state.get("world_strengths", []):
        sock.default_value = float(base_value)

    for sock, base_value in state.get("emissive_strengths", []):
        sock.default_value = float(base_value)


def _rgb_to_intensity_u8(img: np.ndarray, mode: str = "bt601") -> np.ndarray:
    return _shared_rgb_to_intensity_u8(img, mode=mode)


def _convert_ir_frames_to_intensity(frames, mode: str):
    return _shared_convert_ir_frames_to_intensity(frames, mode=mode)


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

    # Blender-конвенция (позы камер в сцене)
    T_color_from_ir_left = rs.get_T_blender("IR_LEFT", "COLOR")
    T_color_from_ir_right = rs.get_T_blender("IR_RIGHT", "COLOR")
    T_ir_left_from_color = np.linalg.inv(T_color_from_ir_left)

    rig_poses_color = [T_w_irL @ T_ir_left_from_color for T_w_irL in rig_poses_ir_left]
    rig_poses_ir_right = [T_w_c @ T_color_from_ir_right for T_w_c in rig_poses_color]

    return rig_poses_ir_left, rig_poses_ir_right, rig_poses_color


def _pick_mount_poses(rs: RealSenseProfile, rig_poses_ir_left, rig_poses_ir_right, rig_poses_color):
    mount_frame = rs.get_projector_mount_frame() if rs.has_projector() else "IR_LEFT"

    if mount_frame in ("IR_LEFT", "DEPTH"):
        return rig_poses_ir_left
    if mount_frame == "IR_RIGHT":
        return rig_poses_ir_right
    if mount_frame == "COLOR":
        return rig_poses_color
    return rig_poses_ir_left


def _projector_pattern_source(rs: RealSenseProfile) -> str:
    if not rs.has_projector():
        return "fallback:generated(seed=0)"

    pr = rs.get_projector()
    if getattr(pr, "pattern_path", None):
        return f"path:{pr.pattern_path}"

    seed = getattr(pr, "pattern_seed", None)
    min_sep_px = getattr(pr, "pattern_min_sep_px", None)
    dot_radius_px = getattr(pr, "pattern_dot_radius_px", None)
    return (
        "generated:"
        f"seed={seed},"
        f"dot_count={pr.dot_count},"
        f"min_sep_px={min_sep_px},"
        f"dot_radius_px={dot_radius_px}"
    )


def _log_projector_config(rs: RealSenseProfile, mount_obj):
    _target_obj, binding = resolve_projector_target(rs, mount_obj)
    local_translation = np.array2string(binding["local_translation"], precision=6, suppress_small=False)
    local_rotation_deg = np.array2string(
        np.rad2deg(binding["local_rotation_euler_rad"]),
        precision=6,
        suppress_small=False,
    )

    if not rs.has_projector():
        s = rs.get_stream("IR_LEFT")
        print(
            "[Projector] source=fallback:generated(seed=0) | "
            f"pattern_size={s.width}x{s.height} | "
            f"projector_fov_deg=legacy_from_ir_left | "
            "mount_mode=legacy_frame_lock | mount_frame=IR_LEFT | "
            f"local_transform_mode={binding['local_transform_mode']} | "
            f"local_transform_applied={binding['local_transform_applied']} | "
            f"projector_socket={binding['projector_socket_name']} | "
            f"local_translation={local_translation} | "
            f"local_rotation_euler_deg={local_rotation_deg} | "
            "local_transform_blender=identity"
        )
        return

    pr = rs.get_projector()
    fov_h_rad, fov_v_rad = rs.get_projector_fov_rad()
    local_T = rs.get_projector_local_transform_blender()
    local_T_str = np.array2string(local_T, precision=6, suppress_small=False)
    print(
        f"[Projector] source={_projector_pattern_source(rs)} | "
        f"pattern_size={pr.pattern_w}x{pr.pattern_h} | "
        f"projector_fov_deg=({np.rad2deg(fov_h_rad):.3f}, {np.rad2deg(fov_v_rad):.3f}) | "
        f"mount_mode={rs.get_projector_mount_mode()} | "
        f"mount_frame={rs.get_projector_mount_frame()} | "
        f"local_transform_mode={binding['local_transform_mode']} | "
        f"local_transform_applied={binding['local_transform_applied']} | "
        f"projector_socket={binding['projector_socket_name']} | "
        f"local_translation={local_translation} | "
        f"local_rotation_euler_deg={local_rotation_deg} | "
        f"local_transform_blender={local_T_str}"
    )


# -------------------- main --------------------

def main(argv=None):
    bproc.init()
    bproc.utility.reset_keyframes()

    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    cfg = Config(args.config_file)

    rs = RealSenseProfile.from_json(cfg.camera_path)

    # -------------------- renderer --------------------
    max_spp = int(np.random.randint(300, 1000)) if getattr(cfg, "max_amount_of_samples", None) is None else int(cfg.max_amount_of_samples)
    bproc.renderer.set_max_amount_of_samples(max_spp)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_normals_output()

    # -------------------- scene --------------------
    light_point, light_plane_material = _build_room_and_lights(cfg)
    proj = None
    mount = None
    ir_lighting_state = None

    try:
        _prepare_objects(cfg)

        # -------------------- poses --------------------
        rig_poses_ir_left, rig_poses_ir_right, rig_poses_color = _build_rig_poses(cfg, rs)

        # -------------------- output --------------------
        output_dir = getattr(cfg, "output_dir", "test")
        jpg_quality = int(getattr(cfg, "jpg_quality", 95))
        depth_scale_mm = float(getattr(cfg, "depth_scale_mm", 1.0))
        ir_render_cfg = _get_ir_render_config(cfg)
        ir_lighting_state = _capture_render_lighting_state(light_point, light_plane_material)

        # -------------------- stereo params --------------------
        depth_min_m = float(rs.depth_min_m)
        depth_max_m = float(rs.depth_max_m)

        block_size = int(getattr(cfg, "sgm_block_size", getattr(cfg, "sgm_window_size", 7)))
        min_disparity = int(getattr(cfg, "sgm_min_disparity", 0))
        preprocess = str(getattr(cfg, "stereo_preprocess", "clahe"))

        lr_check = bool(getattr(cfg, "lr_check", True))
        lr_thresh_px = float(getattr(cfg, "lr_thresh_px", 1.0))
        lr_min_keep_ratio = float(getattr(cfg, "lr_min_keep_ratio", 0.02))

        speckle_filter = bool(getattr(cfg, "speckle_filter", True))
        fill_mode = str(getattr(cfg, "fill_mode", "none"))
        fill_iters = int(getattr(cfg, "fill_iters", 0))
        depth_completion = bool(getattr(cfg, "depth_completion", False))

        want_wls = bool(getattr(cfg, "use_wls", True))
        use_wls = want_wls and _has_ximgproc()
        if want_wls and not use_wls:
            print("[Stereo] cv2.ximgproc not found -> WLS disabled (install opencv-contrib-python).")

        use_geom_mask_from_gt = bool(getattr(cfg, "use_geom_mask_from_gt", True))

        # --- rectify params ---
        rectify_use_distortion = False
        rectify_dict = _build_rectify_from_rs(
            rs, left="IR_LEFT", right="IR_RIGHT", use_distortion=rectify_use_distortion
        )

        # -------------------- mount/projector --------------------
        mount = get_or_create_mount_empty("rs_projector_mount")
        mount_poses = _pick_mount_poses(rs, rig_poses_ir_left, rig_poses_ir_right, rig_poses_color)
        _log_projector_config(rs, mount)
        print(
            "[IR Render] "
            f"enabled={ir_render_cfg['enabled']} | "
            f"environment_scale={ir_render_cfg['environment_scale']:.3f} | "
            f"point_light_scale={ir_render_cfg['point_light_scale']:.3f} | "
            f"emissive_scale={ir_render_cfg['emissive_scale']:.3f} | "
            f"rgb_to_intensity_mode={ir_render_cfg['rgb_to_intensity_mode']}"
        )

        # ---------------------------------------------------------------------
        # PASS 1: IR_LEFT (with IR projector)
        # ---------------------------------------------------------------------
        bproc.utility.reset_keyframes()
        rs.set_bproc_intrinsics("IR_LEFT")
        for i, T in enumerate(rig_poses_ir_left):
            bproc.camera.add_camera_pose(T, frame=i)

        animate_mount_from_cam2world(mount, mount_poses)
        proj = create_realsense_ir_projector(rs, mount, dots=None, energy=None)
        if ir_render_cfg["enabled"]:
            _apply_ir_lighting_state(ir_lighting_state, ir_render_cfg)
        data_ir_left = bproc.renderer.render()

        # ---------------------------------------------------------------------
        # PASS 2: IR_RIGHT (same projector, D435-like)
        # ---------------------------------------------------------------------
        bproc.utility.reset_keyframes()
        rs.set_bproc_intrinsics("IR_RIGHT")
        for i, T in enumerate(rig_poses_ir_right):
            bproc.camera.add_camera_pose(T, frame=i)

        animate_mount_from_cam2world(mount, mount_poses)
        data_ir_right = bproc.renderer.render()
        if ir_render_cfg["enabled"]:
            _restore_render_lighting_state(ir_lighting_state)

        # проектор больше не нужен
        proj.delete()
        proj = None

        # ---------------------------------------------------------------------
        # Stereo params in bproc (optional, for scene consistency)
        # ---------------------------------------------------------------------
        if hasattr(rs, "set_bproc_stereo_from_ir"):
            rs.set_bproc_stereo_from_ir(convergence_distance=1.0)
        else:
            bproc.camera.set_stereo_parameters("PARALLEL", 1.0, float(rs.baseline_m))

        # ---------------------------------------------------------------------
        # Build stereo frames
        # ---------------------------------------------------------------------
        irL = _convert_ir_frames_to_intensity(
            data_ir_left["colors"],
            mode=ir_render_cfg["rgb_to_intensity_mode"],
        )
        irR = _convert_ir_frames_to_intensity(
            data_ir_right["colors"],
            mode=ir_render_cfg["rgb_to_intensity_mode"],
        )

        # Можно оставить обычный list; pipeline внутри сам вернет FrameSeries с metadata.
        stereo_frames = [np.stack([a, b], axis=0) for a, b in zip(irL, irR)]

        # Если хочешь, можно так (необязательно):
        # stereo_frames = FrameSeries(
        #     [np.stack([a, b], axis=0) for a, b in zip(irL, irR)],
        #     grid=FrameGrid(name="IR_PAIR_ORIG")
        # )

        # Optional GT depth from IR_LEFT render (ORIGINAL IR_LEFT grid, not rectified)
        depth_gt_ir_m = data_ir_left.get("depth", None)

        # num_disparities recommendation (по профилю)
        num_disparities = int(getattr(
            cfg, "sgm_num_disparities",
            rs.recommend_num_disparities(
                z_min_m=rs.depth_min_m,
                min_disparity=min_disparity,
                stream="IR_LEFT",
            )
        ))

        print(
            "[Stereo RECTIFY] depth_range_m:",
            f"min={depth_min_m:.3f}, max={depth_max_m:.3f} | "
            f"num_disparities={num_disparities}, block_size={block_size}, min_disp={min_disparity} | "
            f"use_wls={use_wls} | use_geom_mask_from_gt={use_geom_mask_from_gt} | "
            f"rectify_use_distortion={rectify_use_distortion}"
        )

        # --------------------- ------------------------------------------------
        # MATCH: RECTIFY ONLY -> depth_rect_m + disp_rect_px (IR_LEFT_RECT grid)
        # ---------------------------------------------------------------------
        depth_rect_m, disp_rect_px = stereo_global_matching_rectified(
            stereo_frames,
            rectify=rectify_dict,   # always ON
            depth_min=depth_min_m,
            depth_max=depth_max_m,
            depth_range_policy="zero",
            block_size=block_size,
            num_disparities=num_disparities,
            min_disparity=min_disparity,
            preprocess=preprocess,
            use_wls=use_wls,
            lr_check=lr_check,
            lr_thresh_px=lr_thresh_px,
            lr_min_keep_ratio=lr_min_keep_ratio,
            speckle_filter=speckle_filter,
            fill_mode=fill_mode,
            fill_iters=fill_iters,
            depth_completion=depth_completion,
            depth_gt_frames=depth_gt_ir_m if (use_geom_mask_from_gt and depth_gt_ir_m is not None) else None,
            use_geom_mask_from_gt=use_geom_mask_from_gt,
            border_pad=True,
            pad_left=None,
        )

        # depth_rect_m / disp_rect_px теперь обычно FrameSeries (list-compatible)
        if hasattr(depth_rect_m, "grid_name"):
            print(f"[Stereo OUT] depth grid = {depth_rect_m.grid_name}")
        if hasattr(disp_rect_px, "grid_name"):
            print(f"[Stereo OUT] disp grid = {disp_rect_px.grid_name}")

        # ---------------------------------------------------------------------
        # PASS 3: RGB (COLOR)
        # ---------------------------------------------------------------------
        bproc.utility.reset_keyframes()
        rs.set_bproc_intrinsics("COLOR")
        for i, T in enumerate(rig_poses_color):
            bproc.camera.add_camera_pose(T, frame=i)

        data_rgb = bproc.renderer.render()

        # ---------------------------------------------------------------------
        # ALIGN -> COLOR grid
        # ---------------------------------------------------------------------
        # computed rectified depth -> COLOR (source is rectified IR_LEFT grid)
        depth_rect_align_rgb_m, _ = align_depth_series_ir_left_to_color(
            rs,
            depth_rect_m,
            depth_value_mode="source_z",
            splat_2x2=True,
            rectify_mode="on",
        )

        # GT depth from IR_LEFT render -> COLOR (GT is NOT rectified)
        depth_gt_rgb_m = None
        if depth_gt_ir_m is not None:
            depth_gt_rgb_m, _ = align_depth_series_ir_left_to_color(
                rs,
                depth_gt_ir_m,
                depth_value_mode="source_z",
                splat_2x2=True,
                rectify_mode="off",
            )

        # ---------------------------------------------------------------------
        # SAVE
        # ---------------------------------------------------------------------
        save_rgb_ir_stereo_rectified(
            output_dir=output_dir,
            rgb_ext="jpg",
            color_file_format="JPEG",
            jpg_quality=jpg_quality,
            data_rgbs=data_rgb["colors"],
            data_ir_lefts=irL,
            data_ir_rights=irR,
            depth_scale_mm=depth_scale_mm,
            depth_ir_left_rect_m=depth_rect_m,             # IR_LEFT_RECT grid
            depth_color_from_ir_rect_m=depth_rect_align_rgb_m,  # COLOR grid
            depth_gt_rgb_m=depth_gt_rgb_m,                 # COLOR grid
            disp_rect_px=disp_rect_px,                     # IR_LEFT_RECT grid
            save_disp_png=False,
            depth_save_mode="heatmap"
        )

    finally:
        # cleanup
        if ir_lighting_state is not None:
            try:
                _restore_render_lighting_state(ir_lighting_state)
            except Exception:
                pass
        if proj is not None:
            try:
                proj.delete()
            except Exception:
                pass
        if light_point is not None:
            try:
                light_point.delete()
            except Exception:
                pass


if __name__ == "__main__":
    main()
