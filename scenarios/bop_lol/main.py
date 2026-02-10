import blenderproc as bproc
import argparse, os, sys, math, json
import numpy as np
from datetime import datetime
import bpy


from blendforge.blender_runtime.CustomLightSetting import TemperatureToRGBConverter
from blendforge.blender_runtime.CustomMaterial import make_random_material
from blendforge.blender_runtime.utils import (sample_pose_func_drop, sample_pose_func,)
from blendforge.host.FiletoDict import Config
from blendforge.blender_runtime.LolWriterUtility import write_lol_annotations

from filelock import FileLock
from mathutils import Vector, Euler

# ---------- ARGS ----------
def parse_args(args):
    p = argparse.ArgumentParser(description='LOL-style generator via double render (target/input)')
    p.add_argument('--config_file', type=str, required=True)
    return p.parse_args(args)

# ---------- MAIN ----------
def main(args=None):
    bproc.init()
    bproc.utility.reset_keyframes()

    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    cfg = Config(args.config_file)

    if cfg.max_amount_of_samples is None:
        m_a_o_s = np.random.randint(300, 1024)
        bproc.renderer.set_max_amount_of_samples(m_a_o_s)
    else:
        if not isinstance(cfg.max_amount_of_samples, int):
            raise ValueError("max_amount_of_samples must be an integer.")
        bproc.renderer.set_max_amount_of_samples(cfg.max_amount_of_samples)           

    # ---------- загрузка объектов (BOP) ----------
    sampled_objs = bproc.loader.load_bop_objs(
        bop_dataset_path=os.path.join(cfg.dataset_parent_path, cfg.bop_dataset_name),
        mm2m=True, sample_objects=False, num_of_objs_to_sample=9
    )

    # ---------- интринcики ----------
    K, width, height = bproc.loader.load_bop_intrinsics(
        bop_dataset_path=os.path.join(cfg.dataset_parent_path, cfg.bop_dataset_name)
    )
    fx = float(K[0, 0]); fy = float(K[1, 1])

    # ---------- комната ----------
    room_planes = [
        bproc.object.create_primitive('PLANE', scale=[3, 3, 1]),
        bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, -3, 3], rotation=[-1.570796, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0,  3, 3], rotation=[ 1.570796, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[3,  0, 3], rotation=[0, -1.570796, 0]),
        bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[-3, 0, 3], rotation=[0,  1.570796, 0]),
    ]
    for plane in room_planes:
        plane.enable_rigidbody(False, collision_shape='BOX', friction=100.0, linear_damping=0.99, angular_damping=0.99)

    # потолочная панель (мягкий свет)
    light_plane = bproc.object.create_primitive('PLANE', scale=[1, 1, 1], location=[0, 0, 6.5])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')
    light_plane_material.make_emissive(
        emission_strength= np.random.uniform(0, 1),  # 3, 6
        emission_color=np.random.uniform([0.6, 0.6, 0.6, 1.0], [1.0, 1.0, 1.0, 1.0])
    )
    light_plane.replace_materials(light_plane_material)

    # материалы окружения
    cc_textures_bg = bproc.loader.load_ccmaterials(cfg.cc_textures.cc_textures_path)

    # ---------- структура вывода ----------
    split = getattr(cfg, "split", "train")
    out_root = getattr(cfg, "output_dir_lol", os.path.join(getattr(cfg, "output_dir", "."), "lol"))
    input_dir  = os.path.join(out_root, split, "input")
    target_dir = os.path.join(out_root, split, "target")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    pairs_path = os.path.join(out_root, split, "pairs.txt")
    meta_path  = os.path.join(out_root, split, "meta.jsonl")
    pairs_f = open(pairs_path, "a", encoding="utf-8")
    meta_f  = open(meta_path,  "a", encoding="utf-8")

    # ---------- параметры пар ----------
    EV_MODE   = str(getattr(cfg, "ev_mode", "camera")).lower()  # "camera" | "light"
    JPEG_Q    = getattr(cfg, "jpeg_input_quality", None)        # если None — PNG, иначе JPEG quality

    # ---------- основной цикл ----------
    r = 0
    while r < int(getattr(cfg, "runs", 1)):
        bproc.utility.reset_keyframes()

        # точечный свет
        diap_tem = [3500, 6500]
        colour = [TemperatureToRGBConverter(tem) for tem in diap_tem]
        #base_energy = float(np.random.uniform(650, 1000))
        light_point = bproc.types.Light()
        #light_point.set_energy(base_energy)
        light_point.set_color(np.random.uniform(colour[0], colour[1]))
        location = bproc.sampler.shell(center=[0, 0, 5], radius_min=0.05, radius_max=1.0,
                                       elevation_min=-1, elevation_max=1, uniform_volume=True)
        light_point.set_location(location)

        # текстуры стен/пола
        random_cc_texture = np.random.choice(cc_textures_bg)
        for plane in room_planes:
            plane.replace_materials(random_cc_texture)

        # позы объектов
        chosen_pose_func = sample_pose_func_drop if np.random.rand() < float(getattr(cfg, "probability_drop", 0.3)) else sample_pose_func
        bproc.object.sample_poses(objects_to_sample=sampled_objs, sample_pose_func=chosen_pose_func, max_tries=1000)

        # физика + материалы
        for j, obj in enumerate(sampled_objs):
            obj.enable_rigidbody(True, friction=100.0, linear_damping=0.99, angular_damping=0.99)
            obj.set_shading_mode('auto')


            mat, style = make_random_material(
                allowed=["metal", 
                         "dirty_metal", 
                         "cast_iron", 
                         "steel", 
                         "brushed_steel", 
                         "galvanized_steel", 
                         "blackened_steel",],   
                name_prefix=f"obj_{j:06d}"
            )

            # назначаем материал на все слоты объекта (или создаём слот, если пусто)
            mats = obj.get_materials()
            if not mats:
                obj.set_material(0, mat)
            else:
                for i in range(len(mats)):
                    obj.set_material(i, mat)


        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=3, max_simulation_time=35,
            check_object_interval=2, substeps_per_frame=30, solver_iters=30
        )

        # позы камеры
        poses = 0
        cam_poses = []
        rotation_factor = 11.0
        while poses < int(getattr(cfg, "poses_cam", 25)):
            radius_max = round(np.random.uniform(1.0, 2.3), 2)
            loc = bproc.sampler.shell(center=[0, 0, 0.1], radius_min=0.40, radius_max=radius_max,
                                      elevation_min=5, elevation_max=89, uniform_volume=False)

            poi = np.array([0, 0, 0], dtype=np.float32)
            fwd = poi - loc
            dist = float(np.linalg.norm(fwd))

            up_axis = 'Y'
            R = Vector(fwd).to_track_quat('-Z', up_axis).to_matrix()

            max_angle_deg = rotation_factor * dist
            rx = np.random.uniform(-max_angle_deg, max_angle_deg)
            ry = np.random.uniform(-max_angle_deg, max_angle_deg)
            rz_max = max_angle_deg/2 if (ry > max_angle_deg/2 or rx > max_angle_deg/2) else max_angle_deg
            rz = np.random.uniform(-max_angle_deg, rz_max)

            R @= Euler((math.radians(rx), math.radians(ry), math.radians(rz)), 'XYZ').to_matrix()
            cam2world = bproc.math.build_transformation_mat(loc, np.array(R))

            cam_poses.append(cam2world)
            poses += 1

        # добавляем все позы
        for cpose in cam_poses:
            bproc.camera.add_camera_pose(cpose)

        # ----------- РЕНДЕР ПАРЫ -----------
        hight_energy = float(np.random.uniform(cfg.dip_energy_hight[0], cfg.dip_energy_hight[1]))
        low_energy =  float(np.random.uniform(cfg.dip_energy_low[0], cfg.dip_energy_low[1]))
        
        # TARGET: базовая экспозиция/энергия
        light_point.set_energy(hight_energy)
        data_target = bproc.renderer.render()   # список кадров

        # INPUT: недоэкспозиция
        if EV_MODE == "camera":
            light_point.set_energy(low_energy)

        data_input = bproc.renderer.render()

        # ----------- СОХРАНЕНИЕ -----------
        ts = datetime.utcnow().isoformat()
        with FileLock(os.path.join(out_root, '.lock')):
            # обе выборки одной длины
            write_lol_annotations(
                output_root=out_root,
                split=split,
                input_colors=data_input["colors"],
                target_colors=data_target["colors"],
                jpeg_input_quality=JPEG_Q,                 # None => PNG, иначе JPEG
                target_format="PNG",
                append_to_existing_output=True,
                file_prefix="",                            # глобальный счётчик
                lock_path=None, # важнейшее для мультипроцесса
                timestamp_utc=ts,
                ev_mode=EV_MODE,
                camera={"fx": fx, "fy": fy, "width": int(width), "height": int(height)},
                render={"engine": "BlenderProc/Cycles", "samples": int(bpy.context.scene.cycles.samples)},
            )


        light_point.delete()
        r += 1

    pairs_f.close()
    meta_f.close()
    print(f"[DONE] LOL-style dataset saved to: {os.path.join(out_root, split)}")

if __name__ == "__main__":
    main()