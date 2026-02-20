# module with software-implemented blender functions
import blenderproc as bproc
import argparse
import os
import numpy as np
import sys
import math
from collections import defaultdict

from blendforge.blender_runtime.utils import update_data, collect_fragments_canonical_gt
from blendforge.blender_runtime.CustomFractureUtills import fracture_object_with_cell
from blendforge.blender_runtime.CustomLightSetting import TemperatureToRGBConverter
from blendforge.blender_runtime.CustomMaterial import make_random_material
from blendforge.blender_runtime.CustomLoadMesh import load_objs
from blendforge.host.FiletoDict import Config

from filelock import FileLock
from mathutils import Vector, Euler
from addon_utils import enable

# наш кастомный BOP writer
from blendforge.blender_runtime.writer.CustomBopShapeWriterUtility import (
    write_bop as write_bop_shape,
    _BopWriterUtility,   # используем его статические методы для json
)


def parse_args(args):
    """
    Парсим аргументы командной строки.
    """
    parser = argparse.ArgumentParser(description='3D puzzle BOP dataset generator.')
    parser.add_argument(
        '--config_file',
        type=str,
        help='Path to YAML/JSON config file with generation params.'
    )
    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)


def sample_pose_func_drop(obj: bproc.types.MeshObject):
    """
    Спавним фрагменты в небольшом цилиндре над полом (куча).
    """
    radius = 0.1
    z_min, z_max = 0.5, 1.0

    theta = np.random.uniform(0, 2 * np.pi)
    r = np.random.uniform(0, radius)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.random.uniform(z_min, z_max)

    obj.set_location([x, y, z])
    obj.set_rotation_euler(bproc.sampler.uniformSO3())


def sample_pose_func(obj: bproc.types.MeshObject):
    """
    Более «рассеянный» спавн фрагментов в объёме.
    """
    id_obj = obj.get_cp("category_id") if obj.has_cp("category_id") else None

    if id_obj == 4:
        loc_min = np.random.uniform([-0.3, -0.3, 0.1], [-0.1, -0.1, 0.2])
        loc_max = np.random.uniform([0.1, 0.1, 0.3], [0.3, 0.3, 0.6])
    else:
        loc_min = np.random.uniform([-0.3, -0.3, 0.1], [-0.1, -0.1, 0.2])
        loc_max = np.random.uniform([0.1, 0.1, 0.3], [0.3, 0.3, 0.6])

    obj.set_location(np.random.uniform(loc_min, loc_max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())


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

    # BOP depth обязателен
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    # bproc.renderer.enable_normals_output()  # опционально

    # -------------------------------------------------------------------------
    # Пути датасета
    # -------------------------------------------------------------------------
    dataset_root = cfg.output_dir                     # корень всех датасетов
    dataset_name = cfg.bop_dataset_name               # имя датасета (папка внутри output_dir)
    split_name = cfg.dataset_type                     # "train"/"val"/"test"

    dataset_dir = os.path.join(dataset_root, dataset_name)
    split_dir = os.path.join(dataset_dir, split_name)
    fragments_dir = os.path.join(split_dir, "fragments")
    os.makedirs(fragments_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Загрузка объектов BOP и intrinsics
    # -------------------------------------------------------------------------
    sampled_objs = load_objs(
        bop_dataset_path=os.path.join(cfg.dataset_parent_path, cfg.bop_dataset_name),
        mm2m=None,
        sample_objects=False,
        num_of_objs_to_sample=9,
        additional_scale=None,
        manifold=True,
        object_model_unit="cm"
    )

    intrinsics, width, height = bproc.loader.load_bop_intrinsics(
        bop_dataset_path=os.path.join(cfg.dataset_parent_path, cfg.bop_dataset_name)
    )
    K = intrinsics
    fx, fy = K[0, 0], K[1, 1]
    fov_horizontal = 2 * np.arctan(width / (2.0 * fx))
    fov_vertical = 2 * np.arctan(height / (2.0 * fy))
    print(f"[INFO] FOV (h={np.degrees(fov_horizontal):.1f}°, v={np.degrees(fov_vertical):.1f}°)")

    # Сэмплирование количества path-tracing samples
    if cfg.max_amount_of_samples is None:
        m_a_o_s = np.random.randint(10, 20)
        bproc.renderer.set_max_amount_of_samples(m_a_o_s)
    else:
        if not isinstance(cfg.max_amount_of_samples, int):
            raise ValueError("max_amount_of_samples must be an integer.")
        bproc.renderer.set_max_amount_of_samples(cfg.max_amount_of_samples)

    # -------------------------------------------------------------------------
    # Сцена: комната + свет
    # -------------------------------------------------------------------------
    room_planes = [
        bproc.object.create_primitive('PLANE', scale=[3, 3, 1]),
        bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, -3, 3], rotation=[-1.570796, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 3, 3], rotation=[1.570796, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[3, 0, 3], rotation=[0, -1.570796, 0]),
        bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[-3, 0, 3], rotation=[0, 1.570796, 0]),
    ]
    for plane in room_planes:
        plane.enable_rigidbody(
            False,
            collision_shape='BOX',
            friction=100.0,
            linear_damping=0.99,
            angular_damping=0.99
        )

    light_plane = bproc.object.create_primitive('PLANE', scale=[1, 1, 1], location=[0, 0, 5])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')
    light_plane_material.make_emissive(
        emission_strength=np.random.uniform(3, 6),
        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0])
    )
    light_plane.replace_materials(light_plane_material)

    cc_textures = bproc.loader.load_ccmaterials(cfg.cc_textures.cc_textures_path)

    # -------------------------------------------------------------------------
    # Фрактурирование: целый объект → фрагменты + каноническая геометрия
    # -------------------------------------------------------------------------
    new_sampled_objs: list[bproc.types.MeshObject] = []
    nums_objects = []

    # сюда собираем GT по фрагментам (каноническая форма / пазл) в рамках ЭТОГО запуска
    fragments_gt = {
        "frag_scene_id": [],     # scene_id (пазл) для каждого фрагмента
        "frag_local_idx": [],    # локальный индекс фрагмента в пазле
        "com_F": [],             # центр масс в локальной СК фрагмента F (3,)
        "R_O_from_F": [],        # матрица 3x3, разложенная в список длины 9
        "t_O_from_F": [],        # вектор (3,) – сдвиг фрагмента в кан. СК объекта
        "mesh_filenames": [],    # путь до PLY меша внутри split_dir ("fragments/obj_...ply")
    }

    while sampled_objs:
        obj = sampled_objs.pop(0)
        obj_bpy = obj.blender_obj

        # начинаем с 2 фрагментов
        source_limit_and_count = np.random.choice([2, 3, 4, 5])
        source_noise = np.random.uniform(0, 0.007)
        cell_scale = (
            np.random.uniform(0.75, 1.5),
            np.random.uniform(0.75, 1.5),
            np.random.uniform(0.75, 1.5),
        )
        seed = np.random.randint(2, 40)

        # Фрактурируем в канонической позе (до sample_poses!)
        new_mesh_objects = fracture_object_with_cell(
            bpy_obj=obj_bpy,
            source_limit_and_count=source_limit_and_count,
            source_noise=source_noise,
            cell_scale=cell_scale,
            scale=0.05,
            seed=seed
        )

        nums_objects.append(len(new_mesh_objects))
        new_sampled_objs.extend(new_mesh_objects)

        # --- Сбор GT по фрагментам (канон) + экспорт мешей ---
        collect_fragments_canonical_gt(
            new_mesh_objects=new_mesh_objects,
            split_dir=split_dir,
            fragments_gt=fragments_gt,
        )

    print(f"[INFO] Total fractured objects: {len(new_sampled_objs)}, per source: {nums_objects}")

    # -------------------------------------------------------------------------
    # Основной цикл генерации (runs_one_fracture) — разные сцены / ракурсы
    # -------------------------------------------------------------------------
    file_prefix_rgb = f'rgb_id{cfg.index_device}ip{cfg.process_id}_'

    r = 0
    while r < int(cfg.runs_one_fracture):
        print(f"[SCENE] Run {r + 1}/{cfg.runs_one_fracture}")
        bproc.utility.reset_keyframes()

        # Свет (точечный)
        diap_tem = [5500, 6500]
        colour = [TemperatureToRGBConverter(tem) for tem in diap_tem]
        light_energy = np.random.uniform(150, 1000)

        light_point = bproc.types.Light()
        light_point.set_energy(light_energy)
        light_point.set_color(np.random.uniform(colour[0], colour[1]))
        location = bproc.sampler.shell(
            center=[0, 0, 4],
            radius_min=0.05,
            radius_max=1.0,
            elevation_min=-1,
            elevation_max=1,
            uniform_volume=True
        )
        light_point.set_location(location)

        # Текстуры комнаты
        random_cc_texture = np.random.choice(cc_textures)
        for plane in room_planes:
            plane.replace_materials(random_cc_texture)

        # Выбор стратегии спавна фрагментов
        rand_value = np.random.rand()
        if rand_value < cfg.probability_drop:
            print("  -> Using sample_pose_func_drop (pile)")
            chosen_pose_func = sample_pose_func_drop
        else:
            print("  -> Using sample_pose_func (scatter)")
            chosen_pose_func = sample_pose_func

        # Позиции фрагментов
        bproc.object.sample_poses(
            objects_to_sample=new_sampled_objs,
            sample_pose_func=chosen_pose_func,
            max_tries=1000
        )

        # Физика + материалы
        for j, obj in enumerate(new_sampled_objs):
            obj.enable_rigidbody(True, friction=100.0, linear_damping=0.99, angular_damping=0.99)
            obj.set_shading_mode('auto')

            mat, style = make_random_material(
                allowed=["plastic_new"],
                name_prefix=f"obj_{j:06d}"
            )
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
            solver_iters=30
        )

        # BVH для obstacle check (если захочешь включить)
        bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(new_sampled_objs)

        # Камеры
        rotation_factor = 11.0
        poses = 0
        while poses < cfg.poses_cam:
            radius_max = round(np.random.uniform(0.5, 1.0), 2)
            location = bproc.sampler.shell(
                center=[0, 0, 0.1],
                radius_min=0.40,
                radius_max=radius_max,
                elevation_min=5,
                elevation_max=89,
                uniform_volume=False
            )

            poi = np.array([0, 0, 0], dtype=np.float32)
            forward_vector = poi - location
            distance_to_center = np.linalg.norm(forward_vector)

            up_axis = 'Y'
            look_quat = Vector(forward_vector).to_track_quat('-Z', up_axis)
            rotation_matrix = look_quat.to_matrix()

            max_angle_deg = rotation_factor * distance_to_center
            random_x_deg = np.random.uniform(-max_angle_deg, max_angle_deg)
            random_y_deg = np.random.uniform(-max_angle_deg, max_angle_deg)
            if (random_y_deg > max_angle_deg / 2
                    or random_x_deg > max_angle_deg / 2
                    or (random_y_deg > max_angle_deg / 2 and random_x_deg > max_angle_deg / 2)):
                random_z_deg = np.random.uniform(-max_angle_deg, max_angle_deg / 2)
            else:
                random_z_deg = np.random.uniform(-max_angle_deg, max_angle_deg)

            random_x = math.radians(random_x_deg)
            random_y = math.radians(random_y_deg)
            random_z = math.radians(random_z_deg)
            random_euler = Euler((random_x, random_y, random_z), 'XYZ')
            rotation_matrix @= random_euler.to_matrix()

            cam2world_matrix = bproc.math.build_transformation_mat(
                location,
                np.array(rotation_matrix)
            )

            # Можно включить obstacle_in_view_check, если нужно
            # if bproc.camera.perform_obstacle_in_view_check(
            #         cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            #     bproc.camera.add_camera_pose(cam2world_matrix)
            #     poses += 1
            #     continue

            bproc.camera.add_camera_pose(cam2world_matrix)
            poses += 1

        # Рендерим
        data = bproc.renderer.render(output_dir=cfg.temp_dir_rgb, file_prefix=file_prefix_rgb)
        # update_data(data)  # сейчас не нужен, оставил на будущее

        # dataset_info — только при первом проходе
        dataset_info_path = os.path.join(dataset_dir, "dataset_info.json")
        if not os.path.exists(dataset_info_path):
            dataset_info = {
                "name": dataset_name,
                "source_bop": cfg.bop_dataset_name,
                "description": "3D fragment puzzle dataset (BOP-style)",
                "camera": {
                    "width": int(width),
                    "height": int(height),
                    "fx": float(fx),
                    "fy": float(fy),
                    "cx": float(K[0, 2]),
                    "cy": float(K[1, 2]),
                },
                "note": "dataset_info is written by CustomBopShapeWriterUtility",
            }
        else:
            dataset_info = None

        # Lock на уровне dataset/split (если будешь параллелить)
        lock_path = os.path.join(dataset_dir, split_name, ".lock")
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)

        calc_mask_info_coco = getattr(cfg, "calc_mask_info_coco", False)
        num_worker = getattr(cfg, "num_worker", None)

        with FileLock(lock_path):
            write_bop_shape(
                output_dir=dataset_root,
                target_objects=new_sampled_objs,
                depths=data["depth"],
                colors=data["colors"],
                color_file_format="PNG",
                dataset=dataset_name,
                append_to_existing_output=True,
                depth_scale=1.0,
                jpg_quality=95,
                save_world2cam=True,
                ignore_dist_thres=100.0,
                m2mm=None,
                annotation_unit='mm',
                frames_per_chunk=1000,  # игнорируется
                calc_mask_info_coco=calc_mask_info_coco,
                delta=0.015,
                num_worker=num_worker,
                split=split_name,
                fragments_gt=None,      # пишем в конце, один раз
                scenes_meta=None,       # тоже в конце
                dataset_info=dataset_info,
            )

        light_point.delete()
        r += 1

    # -------------------------------------------------------------------------
    # После всех сцен: достраиваем fragments_gt.json + scenes.json
    # -------------------------------------------------------------------------
    scene_gt_path = os.path.join(split_dir, "scene_gt.json")
    if not os.path.exists(scene_gt_path):
        print(f"[WARN] scene_gt.json not found at {scene_gt_path}, skip fragments_gt/scenes.json writing.")
        return

    scene_gt = _BopWriterUtility.load_json(scene_gt_path, keys_to_int=True)

    lock_path = os.path.join(dataset_dir, split_name, ".lock")
    with FileLock(lock_path):
        # 1) дописываем новые фрагменты в общий fragments_gt.json (merge внутри write_fragments_gt_json)
        _BopWriterUtility.write_fragments_gt_json(
            split_dir=split_dir,
            fragments_gt=fragments_gt,
            filename="fragments_gt.json",
        )

        # 2) читаем объединённую структуру по всем запускам
        fragments_gt_all = _BopWriterUtility.load_json(
            os.path.join(split_dir, "fragments_gt.json")
        )

        # --- строим scenes_meta по ВСЕМ фрагментам и ВСЕМ кадрам ---
        # scene_id -> множество im_id, где этот пазл виден
        scene_to_image_ids = defaultdict(set)
        for im_id, inst_list in scene_gt.items():
            for inst in inst_list:
                # scene_id совпадает с obj_id (parent_category_id)
                s_id = inst.get("scene_id", inst.get("obj_id"))
                if s_id is None:
                    continue
                scene_to_image_ids[int(s_id)].add(int(im_id))

        # scene_id -> локальные индексы фрагментов
        scene_to_frag_local_indices = defaultdict(set)
        for s_id, f_idx in zip(
            fragments_gt_all["frag_scene_id"],
            fragments_gt_all["frag_local_idx"],
        ):
            scene_to_frag_local_indices[int(s_id)].add(int(f_idx))

        scenes_meta = {
            "scene_ids": fragments_gt_all.get("scene_ids", []),
            "scenes": {}
        }

        for sid in fragments_gt_all.get("scene_ids", []):
            sid_int = int(sid)
            frag_local_indices = sorted(scene_to_frag_local_indices.get(sid_int, set()))
            frag_count = len(frag_local_indices)
            image_ids = sorted(scene_to_image_ids.get(sid_int, set()))
            model_file = f"models_priori_objects/obj_{sid_int:06d}.ply"

            scenes_meta["scenes"][str(sid_int)] = {
                "scene_id": sid_int,
                "obj_id": sid_int,
                "model_file": model_file,
                "frag_count": int(frag_count),
                "frag_local_indices": frag_local_indices,
                "image_ids": image_ids,
            }

        _BopWriterUtility.write_scenes_json(
            split_dir=split_dir,
            scenes_meta=scenes_meta,
            filename="scenes.json",
        )


if __name__ == "__main__":
    main()
