# main_rgbd_coco.py

import blenderproc as bproc
import numpy as np
import math
import sys
import argparse
import os

from addon_utils import enable
from filelock import FileLock
from mathutils import Vector, Euler

from blendforge.host.FiletoDict import Config

from blendforge.blender_runtime.CustomLightSetting import TemperatureToRGBConverter
from blendforge.blender_runtime.DepthNoiseModel import DepthNoiseModel
from blendforge.blender_runtime.CustomFractureUtills import fracture_object_with_cell
from blendforge.blender_runtime.CustomMaterial import make_random_material
from blendforge.blender_runtime.CustomLoadMesh import load_objs
from blendforge.blender_runtime.utils import (
    update_data,              # чистит name, на логику не влияет
    sample_pose_func_drop,
    sample_pose_func,
)
from blendforge.blender_runtime.writer.RGBDCOCOWriter import write_coco_with_depth_annotations


def parse_args(args):
    parser = argparse.ArgumentParser(description="RGB-D COCO fragment generator")
    parser.add_argument("--config_file", type=str, required=True)
    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)


def main(args=None):
    # -------------------------------------------------------------------------
    # Init + config
    # -------------------------------------------------------------------------
    bproc.init()
    enable("object_fracture_cell")
    bproc.utility.reset_keyframes()

    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    cfg = Config(args.config_file)

    # -------------------------------------------------------------------------
    # Output split dir (структура: train/ coco_annotations.json + folders)
    # -------------------------------------------------------------------------
    split_out_dir = os.path.join(cfg.output_dir, cfg.dataset_type)  # например ".../train"
    os.makedirs(split_out_dir, exist_ok=True)
    lock_path = os.path.join(split_out_dir, ".lock")

    # Префиксы
    file_prefix_rgb = f"rgb_id{cfg.index_device}ip{cfg.process_id}_"
    file_prefix_segmap = f"segmap_id{cfg.index_device}ip{cfg.process_id}_"

    # -------------------------------------------------------------------------
    # Load objects + intrinsics, set camera intrinsics
    # -------------------------------------------------------------------------
    sampled_objs = load_objs(
        bop_dataset_path=os.path.join(cfg.dataset_parent_path, cfg.bop_dataset_name),
        mm2m=None,
        sample_objects=False,
        num_of_objs_to_sample=9,
        additional_scale=None,
        manifold=True,
        object_model_unit="cm",
    )

    bproc.loader.load_bop_intrinsics(
        bop_dataset_path=os.path.join(cfg.dataset_parent_path, cfg.bop_dataset_name)
    )

    # Samples
    if cfg.max_amount_of_samples is None:
        bproc.renderer.set_max_amount_of_samples(int(np.random.randint(300, 1000)))
    else:
        if not isinstance(cfg.max_amount_of_samples, int):
            raise ValueError("max_amount_of_samples must be an integer")
        bproc.renderer.set_max_amount_of_samples(cfg.max_amount_of_samples)
    
    # Renderer outputs
    bproc.renderer.enable_depth_output(activate_antialiasing=False)

    # -------------------------------------------------------------------------
    # Scene: room + ceiling emissive
    # -------------------------------------------------------------------------
    room_planes = [
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1]),
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[0, -3, 3], rotation=[-1.570796, 0, 0]),
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[0, 3, 3], rotation=[1.570796, 0, 0]),
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[3, 0, 3], rotation=[0, -1.570796, 0]),
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[-3, 0, 3], rotation=[0, 1.570796, 0]),
    ]
    for plane in room_planes:
        plane.enable_rigidbody(False, collision_shape="BOX", friction=100.0, linear_damping=0.99, angular_damping=0.99)

    light_plane = bproc.object.create_primitive("PLANE", scale=[1, 1, 1], location=[0, 0, 5])
    light_plane.set_name("light_plane")
    light_plane_material = bproc.material.create("light_material")
    light_plane_material.make_emissive(
        emission_strength=np.random.uniform(3, 6),
        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]),
    )
    light_plane.replace_materials(light_plane_material)

    cc_textures = bproc.loader.load_ccmaterials(cfg.cc_textures.cc_textures_path)

    # -------------------------------------------------------------------------
    # Fracture once (canonical), then reuse fragments across many scenes
    # IMPORTANT: fracture_object_with_cell должен выставлять CP:
    #   category_id, fragment_id, fracture_uid, fracture_seed, fracture_method
    # -------------------------------------------------------------------------
    new_sampled_objs = []
    while sampled_objs:
        obj = sampled_objs.pop(0)
        obj_bpy = obj.blender_obj

        source_limit_and_count = int(np.random.choice([2, 3, 4, 5]))
        source_noise = float(np.random.uniform(0, 0.007))
        cell_scale = (
            float(np.random.uniform(0.75, 1.5)),
            float(np.random.uniform(0.75, 1.5)),
            float(np.random.uniform(0.75, 1.5)),
        )
        seed = int(np.random.randint(2, 40))

        shards = fracture_object_with_cell(
            bpy_obj=obj_bpy,
            source_limit_and_count=source_limit_and_count,
            source_noise=source_noise,
            cell_scale=cell_scale,
            scale=0.05,
            seed=seed,
        )
        new_sampled_objs.extend(shards)

    # -------------------------------------------------------------------------
    # Main generation loop: many scenes, many camera poses
    # -------------------------------------------------------------------------
    depth_noise = DepthNoiseModel(cfg)

    r = 0
    while r < int(cfg.runs_one_fracture):
        bproc.utility.reset_keyframes()

        # point light
        diap_tem = [5500, 6500]
        colour = [TemperatureToRGBConverter(tem) for tem in diap_tem]
        light_energy = np.random.uniform(150, 1000)

        light_point = bproc.types.Light()
        light_point.set_energy(float(light_energy))
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

        # room texture
        random_cc_texture = np.random.choice(cc_textures)
        for plane in room_planes:
            plane.replace_materials(random_cc_texture)

        # spawn mode
        if np.random.rand() < cfg.probability_drop:
            chosen_pose_func = sample_pose_func_drop
        else:
            chosen_pose_func = sample_pose_func

        # sample poses
        bproc.object.sample_poses(
            objects_to_sample=new_sampled_objs,
            sample_pose_func=chosen_pose_func,
            max_tries=1000,
        )

        # materials + rigidbodies
        for j, obj in enumerate(new_sampled_objs):
            obj.enable_rigidbody(True, friction=100.0, linear_damping=0.99, angular_damping=0.99)
            obj.set_shading_mode("auto")

            mat, _style = make_random_material(
                allowed=["plastic_new"],
                name_prefix=f"obj_{j:06d}",
            )
            mats = obj.get_materials()
            if not mats:
                obj.set_material(0, mat)
            else:
                for i in range(len(mats)):
                    obj.set_material(i, mat)

        # physics settle
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=3,
            max_simulation_time=35,
            check_object_interval=2,
            substeps_per_frame=30,
            solver_iters=30,
        )

        # camera poses
        rotation_factor = 11.0
        poses = 0
        while poses < int(cfg.poses_cam):
            radius_max = round(float(np.random.uniform(0.5, 1.0)), 2)
            cam_loc = bproc.sampler.shell(
                center=[0, 0, 0.1],
                radius_min=0.40,
                radius_max=radius_max,
                elevation_min=5,
                elevation_max=89,
                uniform_volume=False,
            )

            poi = np.array([0, 0, 0], dtype=np.float32)
            forward = poi - cam_loc
            dist = float(np.linalg.norm(forward))

            look_quat = Vector(forward).to_track_quat("-Z", "Y")
            R = look_quat.to_matrix()

            max_angle_deg = rotation_factor * dist
            rx = math.radians(float(np.random.uniform(-max_angle_deg, max_angle_deg)))
            ry = math.radians(float(np.random.uniform(-max_angle_deg, max_angle_deg)))
            rz = math.radians(float(np.random.uniform(-max_angle_deg, max_angle_deg)))
            R @= Euler((rx, ry, rz), "XYZ").to_matrix()

            cam2world = bproc.math.build_transformation_mat(cam_loc, np.array(R))
            bproc.camera.add_camera_pose(cam2world)
            poses += 1

        # Renderer outputs we need
        bproc.renderer.enable_normals_output()
        bproc.renderer.enable_segmentation_output(
                map_by=["category_id", "instance", "fragment_id", "fracture_uid", "fracture_seed", "fracture_method"],
                default_values={
                    "category_id": 0,
                    "fragment_id": 0,
                    "fracture_uid": "",        # или "00000000"
                    "fracture_seed": 0,
                    "fracture_method": "",
                },
                output_dir=cfg.temp_dir_segmap,
                file_prefix=file_prefix_segmap,
            )

        # render
        data = bproc.renderer.render(output_dir=cfg.temp_dir_rgb, file_prefix=file_prefix_rgb)

        # (опционально) стабилизируем name (не обязателен)
        #update_data(data)

        # synthesize raw depth + valid mask
        depth_raw_m_list, valid_u8_list = depth_noise.apply(
            depth_clean_m_list=data["depth"],                 # meters
            normals_list=data.get("normals"),                 # (H,W,3)
            instances_mask_list=None,                         # packed masks writer сделает сам
            instance_attr_maps=data["instance_attribute_maps"],
        )

        # write RGBD-COCO (single entry point)
        with FileLock(lock_path):
            write_coco_with_depth_annotations(
                output_dir=split_out_dir,
                instance_segmaps=data["instance_segmaps"],
                instance_attribute_maps=data["instance_attribute_maps"],
                colors=data["colors"],
                depths_clean_m=data["depth"],
                depths_raw_m=depth_raw_m_list,
                valid_depth_mask_u8=valid_u8_list,
                color_file_format="JPEG",
                jpg_quality=95,
                indent=2,
                supercategory="coco_annotations",
                depth_unit="mm",
                depth_scale_mm=1.0,  
            )

        light_point.delete()
        r += 1


if __name__ == "__main__":
    main()
