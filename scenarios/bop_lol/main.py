import blenderproc as bproc
import argparse, os, sys, math, json
import numpy as np
from datetime import datetime



from blendforge.blender_runtime.CustomLightSetting import TemperatureToRGBConverter
from blendforge.blender_runtime.CustomMaterial import (create_mat, custom_load_CCmaterials,)
from blendforge.host.FiletoDict import Config

# гарантируем видимость custom site-packages для Blender Python
import site
site.addsitedir('/home/nikita/blender/blender-3.5.1-linux-x64/custom-python-packages')

from filelock import FileLock
from mathutils import Vector, Euler

# ---------- ARGS ----------
def parse_args(args):
    p = argparse.ArgumentParser(description='LOL-style generator via double render (target/input)')
    p.add_argument('--config_file', type=str, required=True)
    return p.parse_args(args)

# ---------- позы ----------
def sample_pose_func_drop(obj: bproc.types.MeshObject):
    """Сваливаем объекты в небольшую «кучу» (цилиндр над полом)."""
    radius = 0.12
    z_min, z_max = 0.5, 1.0
    theta = np.random.uniform(0, 2*np.pi)
    r = np.random.uniform(0, radius)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.random.uniform(z_min, z_max)
    obj.set_location([x, y, z])
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

def sample_pose_func(obj: bproc.types.MeshObject):
    """Обычное разбрасывание в ограниченном объёме."""
    mn = np.array([-0.3, -0.3, 0.1], np.float32)
    mx = np.array([ 0.3,  0.3, 0.6], np.float32)
    obj.set_location(np.random.uniform(mn, mx))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

# ---------- сохранение ----------
def save_png(path, arr):
    from PIL import Image
    Image.init()
    arr = np.ascontiguousarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode='RGB').save(path, format='PNG', compress_level=4)

def save_jpeg(path, arr, quality=85):
    from PIL import Image
    Image.init()
    arr = np.ascontiguousarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode='RGB').save(path, format='JPEG', quality=int(quality), subsampling=1, optimize=True)

# ---------- MAIN ----------
def main(args=None):
    bproc.init()
    bproc.utility.reset_keyframes()

    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    cfg = Config(args.config_file)


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

    # ---------- Cycles детерминизм ----------
    import bpy
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.use_adaptive_sampling = False
    bpy.context.scene.cycles.samples = int(getattr(cfg, "max_amount_of_samples", 512) or 512)
    bpy.context.scene.cycles.seed = 12345

    # Color Management (фиксируем трансформ)
    vs = bpy.context.scene.view_settings
    vs.view_transform = str(getattr(cfg, "color_view_transform", "Filmic"))
    vs.look = 'None'
    vs.gamma = 1.0
    vs.exposure = 0.0  # будем менять далее

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
    light_plane = bproc.object.create_primitive('PLANE', scale=[1, 1, 1], location=[0, 0, 5])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')
    light_plane_material.make_emissive(
        emission_strength=np.random.uniform(3, 6),
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
    EV_LL_rng = getattr(cfg, "ev_low_range", [-4.0, -2.0])      # диапазон недоэкспозиции для input
    EV_TARGET = 0.0                                             # таргет — базовая экспозиция
    EV_MODE   = str(getattr(cfg, "ev_mode", "camera")).lower()  # "camera" | "light"
    JPEG_Q    = getattr(cfg, "jpeg_input_quality", None)        # если None — PNG, иначе JPEG quality

    # ---------- основной цикл ----------
    r = 0
    while r < int(getattr(cfg, "runs", 1)):
        bproc.utility.reset_keyframes()

        # точечный свет
        diap_tem = [3500, 6500]
        colour = [TemperatureToRGBConverter(tem) for tem in diap_tem]
        base_energy = float(np.random.uniform(150, 1000))
        light_point = bproc.types.Light()
        light_point.set_energy(base_energy)
        light_point.set_color(np.random.uniform(colour[0], colour[1]))
        location = bproc.sampler.shell(center=[0, 0, 4], radius_min=0.05, radius_max=1.0,
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
            mat = create_mat(f"obj_{j:06d}")
            for mi in range(len(obj.get_materials())):
                obj.set_material(mi, mat)

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
        # TARGET: базовая экспозиция/энергия
        vs.exposure = EV_TARGET
        light_point.set_energy(base_energy)
        data_target = bproc.renderer.render()   # список кадров

        # INPUT: недоэкспозиция
        ev_ll = float(np.random.uniform(EV_LL_rng[0], EV_LL_rng[1]))
        if EV_MODE == "camera":
            vs.exposure = ev_ll            # камерная недоэкспозиция
            light_point.set_energy(base_energy)
        else:
            vs.exposure = EV_TARGET
            light_point.set_energy(base_energy * (2.0 ** ev_ll))  # сценически темнее

        data_input = bproc.renderer.render()

        # ----------- СОХРАНЕНИЕ -----------
        ts = datetime.utcnow().isoformat()
        with FileLock(os.path.join(out_root, '.lock')):
            # обе выборки одной длины
            for k, (img_in, img_tg) in enumerate(zip(data_input["colors"], data_target["colors"])):
                stem = f"{r:04d}_{k:04d}"
                # если хотим JPEG для input
                if JPEG_Q is not None:
                    in_name = f"{stem}.jpg"
                    tg_name = f"{stem}.png"
                else:
                    in_name = f"{stem}.png"
                    tg_name = f"{stem}.png"

                in_path = os.path.join(input_dir,  in_name)
                tg_path = os.path.join(target_dir, tg_name)

                if JPEG_Q is not None:
                    save_jpeg(in_path, img_in, quality=JPEG_Q)
                else:
                    save_png(in_path, img_in)
                save_png(tg_path, img_tg)

                # pairs.txt
                pairs_f.write(f"input/{in_name} target/{tg_name}\n")

                # meta.jsonl
                meta = {
                    "id": stem,
                    "timestamp_utc": ts,
                    "domain": "sRGB",
                    "EV_mode": EV_MODE,
                    "EV_input": ev_ll,
                    "EV_target": EV_TARGET,
                    "camera": {"fx": fx, "fy": fy, "width": int(width), "height": int(height)},
                    "render": {
                        "engine": "BlenderProc/Cycles",
                        "samples": int(bpy.context.scene.cycles.samples),
                        "view_transform": vs.view_transform
                    }
                }
                meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        light_point.delete()
        r += 1

    pairs_f.close()
    meta_f.close()
    print(f"[DONE] LOL-style dataset saved to: {os.path.join(out_root, split)}")

if __name__ == "__main__":
    main()