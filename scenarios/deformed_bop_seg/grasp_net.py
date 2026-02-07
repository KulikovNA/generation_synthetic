# module with software-implemented blender functions
import blenderproc as bproc
import argparse
import os
import numpy as np
import sys
import math

import bpy  # как в BOP writer'е
from mathutils import Matrix, Vector, Euler

# BlenderProc internal writer utility (как в BOP writer'е)
from blenderproc.python.writer.WriterUtility import _WriterUtility

# Получаем абсолютный путь к main.py
current_path = os.path.abspath(__file__)
openbox_path = os.path.dirname(current_path)
config_path = os.path.dirname(openbox_path)
project_root_path = os.path.dirname(config_path)
sys.path.append(project_root_path)

from utils.blend.utils import (
    TemperatureToRGBConverter,
    update_data,
    fracture_object_with_cell,
)
from utils.blend.CustomMaterial import make_random_material
from utils.blend.CustomLoadMesh import load_objs
from utils.config.FiletoDict import Config
from utils.blend.CustomGraspNetWriterUtility import write_graspnet_scene

import site
site.addsitedir('/home/nikita/blender/blender-3.5.1-linux-x64/custom-python-packages')
from filelock import FileLock
from addon_utils import enable


def parse_args(args):
    parser = argparse.ArgumentParser(description='Starting script.')
    parser.add_argument(
        '--config_file',
        help="Config file path",
        type=str
    )
    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)


def sample_pose_func_drop(obj: bproc.types.MeshObject):
    """Спавним объект в небольшом цилиндре над полом (куча фрагментов)."""
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
    """Общий сэмплер поз (рассыпанные фрагменты)."""
    id_obj = obj.get_cp("category_id")
    if id_obj == 4:
        min_vec = np.random.uniform([-0.3, -0.3, 0.1], [-0.1, -0.1, 0.2])
        max_vec = np.random.uniform([0.1, 0.1, 0.3], [0.3, 0.3, 0.6])
    else:
        min_vec = np.random.uniform([-0.3, -0.3, 0.1], [-0.1, -0.1, 0.2])
        max_vec = np.random.uniform([0.1, 0.1, 0.3], [0.3, 0.3, 0.6])

    obj.set_location(np.random.uniform(min_vec, max_vec))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())


def main(args=None):

    # --- Инициализация BlenderProc и аддонов ---
    bproc.init()
    enable("object_fracture_cell")
    bproc.utility.reset_keyframes()


    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    
    # --- Парсим аргументы и конфиг ---
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    cfg = Config(args.config_file)

    # --- Подключаем bop_toolkit (как в BOP writer'е) ---
    bop_toolkit_path = cfg.bop_toolkit_path
    if not os.path.isabs(bop_toolkit_path):
        bop_toolkit_path = os.path.abspath(bop_toolkit_path)
    if os.path.exists(bop_toolkit_path):
        sys.path.append(bop_toolkit_path)
        print(f"Added {bop_toolkit_path} to sys.path")
    else:
        print(f"Error: The path {bop_toolkit_path} does not exist.")
    sys.path.insert(0, bop_toolkit_path)

    # префикс для рендеринга 
    file_prefix_rgb = f'rgb_id{cfg.index_device}ip{cfg.process_id}_'
    file_prefix_segmap = f'segmap_id{cfg.index_device}ip{cfg.process_id}_'

    # --- Загружаем BOP-объекты ---
    sampled_objs = load_objs(
        bop_dataset_path=os.path.join(cfg.dataset_parent_path, cfg.bop_dataset_name),
        mm2m=None,
        sample_objects=False,
        num_of_objs_to_sample=9,
        additional_scale=None,
        manifold=True,
        object_model_unit="cm"
    )

    # --- Intrinsics BOP ---
    intrinsics, width, height = bproc.loader.load_bop_intrinsics(
        bop_dataset_path=os.path.join(cfg.dataset_parent_path, cfg.bop_dataset_name)
    )
    K = intrinsics
    fx = K[0, 0]
    fy = K[1, 1]
    fov_horizontal = 2 * np.arctan(width / (2.0 * fx))
    fov_vertical = 2 * np.arctan(height / (2.0 * fy))
    print(f"FOV horiz={np.degrees(fov_horizontal):.2f} deg, vert={np.degrees(fov_vertical):.2f} deg")

    # --- Сэмплы (noise) рендера ---
    if cfg.max_amount_of_samples is None:
        m_a_o_s = np.random.randint(50, 60)
        bproc.renderer.set_max_amount_of_samples(m_a_o_s)
    else:
        if not isinstance(cfg.max_amount_of_samples, int):
            raise ValueError("max_amount_of_samples must be an integer.")
        bproc.renderer.set_max_amount_of_samples(cfg.max_amount_of_samples)

    # --- Комната ---
    room_planes = [
        bproc.object.create_primitive('PLANE', scale=[3, 3, 1]),
        bproc.object.create_primitive(
            'PLANE', scale=[3, 3, 1],
            location=[0, -3, 3], rotation=[-1.570796, 0, 0]
        ),
        bproc.object.create_primitive(
            'PLANE', scale=[3, 3, 1],
            location=[0, 3, 3], rotation=[1.570796, 0, 0]
        ),
        bproc.object.create_primitive(
            'PLANE', scale=[3, 3, 1],
            location=[3, 0, 3], rotation=[0, -1.570796, 0]
        ),
        bproc.object.create_primitive(
            'PLANE', scale=[3, 3, 1],
            location=[-3, 0, 3], rotation=[0, 1.570796, 0]
        ),
    ]
    for plane in room_planes:
        plane.enable_rigidbody(
            False,
            collision_shape='BOX',
            friction=100.0,
            linear_damping=0.99,
            angular_damping=0.99
        )

    # --- Плоскость света ---
    light_plane = bproc.object.create_primitive('PLANE', scale=[1, 1, 1], location=[0, 0, 5])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')
    light_plane_material.make_emissive(
        emission_strength=np.random.uniform(3, 6),
        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0])
    )
    light_plane.replace_materials(light_plane_material)

    # --- CC textures для комнатных плоскостей ---
    cc_textures = bproc.loader.load_ccmaterials(cfg.cc_textures.cc_textures_path)

    # --- Фрактурирование BOP-объектов в фрагменты ---
    new_sampled_objs = []
    nums_objects = []
    while sampled_objs:
        obj = sampled_objs.pop(0)
        obj_bpy = obj.blender_obj

        source_limit_and_count = np.random.choice([1, 2, 3, 4, 5])
        source_noise = np.random.uniform(0, 0.007)
        cell_scale = (
            np.random.uniform(0.75, 1.5),
            np.random.uniform(0.75, 1.5),
            np.random.uniform(0.75, 1.5),
        )
        seed = np.random.randint(2, 40)

        new_mesh_objects = fracture_object_with_cell(
            bpy_obj=obj_bpy,
            source_limit_and_count=source_limit_and_count,
            source_noise=source_noise,
            cell_scale=cell_scale,
            scale=0.05,
            seed=seed,
        )
        nums_objects.append(len(new_mesh_objects))
        new_sampled_objs.extend(new_mesh_objects)

    print('AAAAAAAAAAAAAAAAAAAAAAAAAAAA')
    for obj in new_sampled_objs:
        print(obj.get_name(), "cat_id =", obj.get_cp("category_id"))
    print('AAAAAAAAAAAAAAAAAAAAAAAAAAAA')

    # --- Корень датасета (train/val/test) ---
    dataset_root = os.path.join(cfg.output_dir, cfg.dataset_type)
    os.makedirs(dataset_root, exist_ok=True)

    # === Основной цикл: один run = одна GraspNet-сцена ===
    r = 0
    while r < int(cfg.runs_one_fracture):
        print(f"\n===== RUN {r} / {int(cfg.runs_one_fracture)-1} =====")
        bproc.utility.reset_keyframes()

        # --- Случайный свет ---
        diap_tem = [5500, 6500]
        colour = [TemperatureToRGBConverter(tem) for tem in diap_tem]
        light_energy = np.random.uniform(150, 1000)

        light_point = bproc.types.Light()
        light_point.set_energy(light_energy)
        light_point.set_color(np.random.uniform(colour[0], colour[1]))
        location = bproc.sampler.shell(
            center=[0, 0, 4],
            radius_min=0.05,
            radius_max=1,
            elevation_min=-1,
            elevation_max=1,
            uniform_volume=True,
        )
        light_point.set_location(location)

        # --- Материал для комнаты ---
        random_cc_texture = np.random.choice(cc_textures)
        for plane in room_planes:
            plane.replace_materials(random_cc_texture)

        # --- Сэмплинг поз объектов (куча или рассыпано) ---
        rand_value = np.random.rand()
        if rand_value < cfg.probability_drop:
            print("Using sample_pose_func_drop (pile) for this scene")
            chosen_pose_func = sample_pose_func_drop
        else:
            print("Using sample_pose_func (normal scatter) for this scene")
            chosen_pose_func = sample_pose_func

        bproc.object.sample_poses(
            objects_to_sample=new_sampled_objs,
            sample_pose_func=chosen_pose_func,
            max_tries=1000,
        )

        # --- Физика + материалы на фрагменты ---
        for j, obj in enumerate(new_sampled_objs):
            obj.enable_rigidbody(True, friction=100.0,
                                 linear_damping=0.99, angular_damping=0.99)
            obj.set_shading_mode('auto')

            mat, style = make_random_material(
                allowed=["plastic_new"],
                name_prefix=f"obj_{j:06d}",
            )

            mats = obj.get_materials()
            if not mats:
                obj.set_material(0, mat)
            else:
                for idx_mat in range(len(mats)):
                    obj.set_material(idx_mat, mat)

        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=3,
            max_simulation_time=35,
            check_object_interval=2,
            substeps_per_frame=30,
            solver_iters=30,
        )

        # --- Камеры ---
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
                uniform_volume=False,
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
            if (
                random_y_deg > max_angle_deg / 2
                or random_x_deg > max_angle_deg / 2
                or (random_y_deg > max_angle_deg / 2 and random_x_deg > max_angle_deg / 2)
            ):
                random_z_deg = np.random.uniform(-max_angle_deg, max_angle_deg / 2)
            else:
                random_z_deg = np.random.uniform(-max_angle_deg, max_angle_deg)

            random_x = math.radians(random_x_deg)
            random_y = math.radians(random_y_deg)
            random_z = math.radians(random_z_deg)

            random_euler = Euler((random_x, random_y, random_z), 'XYZ')
            rotation_matrix @= random_euler.to_matrix()

            cam2world_matrix = bproc.math.build_transformation_mat(
                location, np.array(rotation_matrix)
            )
            bproc.camera.add_camera_pose(cam2world_matrix)
            poses += 1

        # --- Рендер под GraspNet: depth + сегментация по category_id/instance ---
        bproc.renderer.enable_normals_output()
        bproc.renderer.enable_segmentation_output(
            map_by=["category_id", "instance", "name"],
            default_values={"category_id": 0},
            output_dir=cfg.temp_dir_segmap, file_prefix=file_prefix_segmap
        )

        # --- Рендер всех поз в память ---
        data = bproc.renderer.render(output_dir=cfg.temp_dir_rgb, file_prefix = file_prefix_rgb)
        update_data(data)  # правим имена в instance_attribute_maps (remove_postfix)

        colors = data["colors"]
        depths = data["depth"]
        instance_segmaps = data["instance_segmaps"]
        instance_attribute_maps = data["instance_attribute_maps"]

        # --- КАМЕРНЫЕ МАТРИЦЫ В СТИЛЕ BOP (cam->world) ---
        num_views = len(colors)
        frame_start = bpy.context.scene.frame_start

        cam_poses_world = []
        for i in range(num_views):
            frame_id = frame_start + i
            bpy.context.scene.frame_set(frame_id)
            cam2world = _WriterUtility.get_cam_attribute(
                bpy.context.scene.camera,
                'cam2world_matrix'
            )
            cam_poses_world.append(np.array(cam2world, dtype=np.float32))

        cam_poses_world = np.stack(cam_poses_world, axis=0)

        # --- Запись сцены в формате GraspNet, камера по умолчанию realsense ---
        lock = FileLock(os.path.join(dataset_root, ".lock"))
        with lock:
            write_graspnet_scene(
                output_dir=dataset_root,
                scene_id=None,  # автоинкремент scene_XXXX
                colors=colors,
                depths=depths,
                cam_poses_world=cam_poses_world,
                K=K,
                instance_segmaps=instance_segmaps,
                instance_attribute_maps=instance_attribute_maps,
                camera_name="realsense",
                label_offset=0,  # category_id == object_id (1..9)
            )

        light_point.delete()
        r += 1


if __name__ == "__main__":
    main()
