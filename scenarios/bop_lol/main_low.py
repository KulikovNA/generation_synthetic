import blenderproc as bproc
import argparse, os, sys, math, json
import numpy as np
from datetime import datetime


# --- пути проекта (как у вас) ---
current_path = os.path.abspath(__file__)
openbox_path = os.path.dirname(current_path)
config_path = os.path.dirname(openbox_path)
project_root_path = os.path.dirname(config_path)
sys.path.append(project_root_path)

from blendforge.blender_runtime.CustomLightSetting import TemperatureToRGBConverter
from blendforge.blender_runtime.CustomMaterial import (create_mat, custom_load_CCmaterials,)
from blendforge.host.FiletoDict import Config
from blendforge.blender_runtime.CustomLoadMesh import load_objs

from filelock import FileLock

from mathutils import Vector, Euler

# ---------------- sRGB <-> Linear и экспозиция ----------------
def _srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def _linear_to_srgb(y):
    y = np.clip(y, 0.0, 1.0)
    a = 0.055
    return np.where(y <= 0.0031308, 12.92 * y, (1 + a) * (y ** (1/2.4)) - a)

def apply_exposure_srgb(img_srgb_uint8, ev_delta):
    """
    img_srgb_uint8: HxWx3 uint8 (sRGB)
    ev_delta: отрицательный для затемнения (например, -3.0)
    """
    img = img_srgb_uint8.astype(np.float32) / 255.0
    lin = _srgb_to_linear(img)
    k = 2.0 ** ev_delta
    lin2 = lin * k
    out = _linear_to_srgb(lin2)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)

# ---------------- аргументы ----------------
def parse_args(args):
    parser = argparse.ArgumentParser(description='LOL-style generator (no fracture)')
    parser.add_argument('--config_file', type=str, required=True)
    return parser.parse_args(args)

# ---------------- позы объектов ----------------
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

# ---------------- main ----------------
def main(args=None):
    bproc.init()
    bproc.utility.reset_keyframes()

    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    cfg = Config(args.config_file)

    # bop toolkit path
    bop_toolkit_path = cfg.bop_toolkit_path
    if not os.path.isabs(bop_toolkit_path):
        bop_toolkit_path = os.path.abspath(bop_toolkit_path)
    if os.path.exists(bop_toolkit_path):
        sys.path.append(bop_toolkit_path)
        print(f"[INFO] Added {bop_toolkit_path} to sys.path")
    else:
        print(f"[WARN] The path {bop_toolkit_path} does not exist.")
    sys.path.insert(0, bop_toolkit_path)

    # ---------- загрузка объектов ----------
    sampled_objs =bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(cfg.bop_parent_path, cfg.bop_dataset_name),
                                    mm2m = True,
                                    sample_objects = False,
                                    num_of_objs_to_sample = 9)

    # ---------- интринcики ----------
    K, width, height = bproc.loader.load_bop_intrinsics(
        bop_dataset_path=os.path.join(cfg.dataset_parent_path, cfg.bop_dataset_name)
    )
    fx = K[0, 0]; fy = K[1, 1]
    fov_horizontal = 2 * np.arctan(width  / (2.0 * fx))
    fov_vertical   = 2 * np.arctan(height / (2.0 * fy))

    # ---------- сэмплы/рендер ----------
    if getattr(cfg, "max_amount_of_samples", None) is None:
        bproc.renderer.set_max_amount_of_samples(np.random.randint(300, 1000))
    else:
        if not isinstance(cfg.max_amount_of_samples, int):
            raise ValueError("max_amount_of_samples must be an integer.")
        bproc.renderer.set_max_amount_of_samples(cfg.max_amount_of_samples)

    # комната
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

    # материалы окружения/объектов
    cc_textures_bg = bproc.loader.load_ccmaterials(cfg.cc_textures.cc_textures_path)

    # ---------- выходная структура LOL ----------
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

    # параметры EV
    EV_LL_rng = getattr(cfg, "ev_low_range", [-4.0, -2.0])  # диапазон затемнения
    EV_TARGET = 0.0

    # счётчики
    r = 0
    global_idx = 0

    # ---------- основной цикл по сценам ----------
    while r < int(getattr(cfg, "runs", 1)):
        bproc.utility.reset_keyframes()

        # доп. точечный свет (рандом)
        diap_tem = [3500, 6500]
        colour = [TemperatureToRGBConverter(tem) for tem in diap_tem]
        light_energy = np.random.uniform(150, 1000)
        light_point = bproc.types.Light()
        light_point.set_energy(light_energy)
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

        # физика + материалы объектов
        for j, obj in enumerate(sampled_objs):
            obj.enable_rigidbody(True, friction=100.0, linear_damping=0.99, angular_damping=0.99)
            obj.set_shading_mode('auto')
            mat = create_mat(f"obj_{j:06d}")
            for mi in range(len(obj.get_materials())):
                obj.set_material(mi, mat)

        bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3, max_simulation_time=35,
                                                          check_object_interval=2, substeps_per_frame=30, solver_iters=30)

        # BVH для проверок (если понадобится obstacle check)
        _bvh = bproc.object.create_bvh_tree_multi_objects(sampled_objs)

        # выбор поз камеры
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

            # при желании можно вернуть obstacle_in_view_check:
            # if bproc.camera.perform_obstacle_in_view_check(cam2world, {"min": 0.3}, _bvh):
            #     cam_poses.append(cam2world); poses += 1
            cam_poses.append(cam2world); poses += 1

        # рендер всех поз (получаем sRGB кадры из BlenderProc)
        for cpose in cam_poses:
            bproc.camera.add_camera_pose(cpose)
        data = bproc.renderer.render()  # data["colors"] — список HxWx3 uint8

        # сохраняем пары
        ts = datetime.utcnow().isoformat()
        with FileLock(os.path.join(out_root, '.lock')):
            for k, img_srgb in enumerate(data["colors"]):
                # target — как есть
                nl = img_srgb
                # input — затемнение в линейном пространстве
                ev_ll = float(np.random.uniform(EV_LL_rng[0], EV_LL_rng[1]))
                ll = apply_exposure_srgb(nl, ev_ll)

                stem = f"{r:04d}_{k:04d}"
                in_name = f"{stem}.png"
                tg_name = f"{stem}.png"
                in_path = os.path.join(input_dir,  in_name)
                tg_path = os.path.join(target_dir, tg_name)

                save_png(in_path, ll)
                save_png(tg_path, nl)

                # pairs.txt
                pairs_f.write(f"input/{in_name} target/{tg_name}\n")

                # meta.jsonl
                meta = {
                    "id": stem,
                    "timestamp_utc": ts,
                    "domain": "sRGB",
                    "EV_input": ev_ll,
                    "EV_target": EV_TARGET,
                    "camera": {"fx": float(fx), "fy": float(fy), "width": int(width), "height": int(height)},
                    "render": {"engine": "BlenderProc", "samples": int(getattr(cfg, 'max_amount_of_samples', 0) or 0)}
                }
                meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")

                global_idx += 1

        light_point.delete()
        r += 1

    pairs_f.close()
    meta_f.close()
    print(f"[DONE] LOL-style dataset saved to: {os.path.join(out_root, split)}")

def save_png(path, arr):
    from PIL import Image, PngImagePlugin  # гарантируем регистрацию PNG
    Image.init()
    arr = np.ascontiguousarray(arr)
    if arr.dtype == np.uint8:
        img = Image.fromarray(arr, mode='RGB')
        img.save(path, format='PNG', compress_level=4)  # compress_level 0..9
    elif arr.dtype == np.uint16:
        # Если вдруг начнёшь сохранять 16-бит рендеры
        if arr.ndim == 2:
            Image.fromarray(arr, mode='I;16').save(path, format='PNG', compress_level=4)
        else:
            # Для 16-бит RGB Pillow прямого режима нет — обычно хранят как 16-бит каналов по отдельности
            # или конвертируют в 8-бит:
            img8 = (arr / 257).astype(np.uint8)  # 16->8
            Image.fromarray(img8, mode='RGB').save(path, format='PNG', compress_level=4)
    else:
        # безопасный fallback на 8-бит
        img8 = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(img8, mode='RGB').save(path, format='PNG', compress_level=4)


if __name__ == "__main__":
    main()
