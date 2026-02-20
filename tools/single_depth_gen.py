#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import blenderproc as bproc  # <-- ВАЖНО: первый импорт

import os
import sys
import argparse
import numpy as np
import cv2

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
)

# Stereo matching (RECTIFY ALWAYS) - фасад, внутри уже новый модульный pipeline
from blendforge.blender_runtime.stereo.StereoMatching import stereo_global_matching_rectified

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
    """
    Build OpenCV stereoRectify inputs (CV optical convention).

    В Blender/Cycles линзовая дисторсия обычно НЕ симулируется -> use_distortion=False.
    """
    sL = rs.get_stream(left)
    sR = rs.get_stream(right)

    K1 = np.asarray(sL.K, dtype=np.float64)
    K2 = np.asarray(sR.K, dtype=np.float64)

    if use_distortion:
        D1 = _np5(getattr(sL, "distortion_coeffs", [0, 0, 0, 0, 0]))
        D2 = _np5(getattr(sR, "distortion_coeffs", [0, 0, 0, 0, 0]))
    else:
        D1 = np.zeros(5, dtype=np.float64)
        D2 = np.zeros(5, dtype=np.float64)

    # T_color_from_left / T_color_from_right in CV convention
    T_c_l = np.asarray(rs.get_T_cv(left, "COLOR"), dtype=np.float64)
    T_c_r = np.asarray(rs.get_T_cv(right, "COLOR"), dtype=np.float64)

    # Right-from-left (CV): T_r_l = inv(T_c_r) @ T_c_l
    T_r_l = np.linalg.inv(T_c_r) @ T_c_l
    R = T_r_l[:3, :3].copy()
    t = T_r_l[:3, 3].copy()

    return {
        "K_left": K1,
        "D_left": D1,
        "K_right": K2,
        "D_right": D2,
        "R": R,
        "t": t,
        "alpha": 0.0,
        "image_size": (int(sL.width), int(sL.height)),  # (W,H)
    }


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
    return light_point


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
    mount_frame = "IR_LEFT"
    if hasattr(rs, "has_projector") and rs.has_projector():
        if hasattr(rs, "get_projector_mount_frame"):
            mount_frame = rs.get_projector_mount_frame()

    if mount_frame in ("IR_LEFT", "DEPTH"):
        return rig_poses_ir_left
    if mount_frame == "IR_RIGHT":
        return rig_poses_ir_right
    if mount_frame == "COLOR":
        return rig_poses_color
    return rig_poses_ir_left


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
    light_point = _build_room_and_lights(cfg)
    proj = None
    mount = None

    try:
        _prepare_objects(cfg)

        # -------------------- poses --------------------
        rig_poses_ir_left, rig_poses_ir_right, rig_poses_color = _build_rig_poses(cfg, rs)

        # -------------------- output --------------------
        output_dir = getattr(cfg, "output_dir", "test")
        jpg_quality = int(getattr(cfg, "jpg_quality", 95))
        depth_scale_mm = float(getattr(cfg, "depth_scale_mm", 1.0))

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

        # ---------------------------------------------------------------------
        # PASS 1: IR_LEFT (with IR projector)
        # ---------------------------------------------------------------------
        bproc.utility.reset_keyframes()
        rs.set_bproc_intrinsics("IR_LEFT")
        for i, T in enumerate(rig_poses_ir_left):
            bproc.camera.add_camera_pose(T, frame=i)

        animate_mount_from_cam2world(mount, mount_poses)
        proj = create_realsense_ir_projector(rs, mount, dots=None, energy=None)
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
        irL = data_ir_left["colors"]
        irR = data_ir_right["colors"]

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

        # ---------------------------------------------------------------------
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
            data_ir_lefts=data_ir_left["colors"],
            data_ir_rights=data_ir_right["colors"],
            depth_scale_mm=depth_scale_mm,
            depth_ir_left_rect_m=depth_rect_m,             # IR_LEFT_RECT grid
            depth_color_from_ir_rect_m=depth_rect_align_rgb_m,  # COLOR grid
            depth_gt_rgb_m=depth_gt_rgb_m,                 # COLOR grid
            disp_rect_px=disp_rect_px,                     # IR_LEFT_RECT grid
            save_disp_png=False,
        )

    finally:
        # cleanup
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