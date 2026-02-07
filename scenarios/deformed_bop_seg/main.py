import blenderproc as bproc 

import numpy as np
import math

import sys
import argparse
import os

from blendforge.blender_runtime.CustomLightSetting import TemperatureToRGBConverter
from blendforge.blender_runtime.utils import update_data
from blendforge.blender_runtime.CustomFractureUtills import fracture_object_with_cell
from blendforge.blender_runtime.CustomMaterial import make_random_material
from blendforge.blender_runtime.CustomLoadMesh import load_objs
from blendforge.host.FiletoDict import Config

from filelock import FileLock
from mathutils import Matrix, Vector, Euler
from addon_utils import enable

def parse_args(args):
    """
    Parse the arguments.
    """
    
    parser = argparse.ArgumentParser(description = 'Starting script.')
    parser.add_argument('--config_file', 
                        help="Number of times the scene is created. 25 shots are taken for each scene",
                        type = str)
    print(vars(parser.parse_args(args)))
    
    return parser.parse_args(args)

def sample_pose_func_drop(obj: bproc.types.MeshObject):
    """
    Спавним объект в небольшом цилиндре над полом, чтобы все упали примерно в одну область.
    """
    # Пусть центр будет (0, 0), а высота ~ 0.5..1.0
    # Радиус цилиндра
    radius = 0.1  # чем меньше, тем плотнее куча
    # Высота
    z_min, z_max = 0.5, 1.0

    # Случайный угол (0..2π)
    theta = np.random.uniform(0, 2*np.pi)
    r = np.random.uniform(0, radius)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.random.uniform(z_min, z_max)

    obj.set_location([x, y, z])
    obj.set_rotation_euler(bproc.sampler.uniformSO3())


# Define a function that samples 6-DoF poses (recomendations setting)
def sample_pose_func(obj: bproc.types.MeshObject):
    
    id_obj = obj.get_cp("category_id")
    if id_obj == 4: 
        min = np.random.uniform([-0.3, -0.3, 0.1], [-0.1, -0.1, 0.2])
        max = np.random.uniform([0.1, 0.1, 0.3], [0.3, 0.3, 0.6])
    else: 
        min = np.random.uniform([-0.3, -0.3, 0.1], [-0.1, -0.1, 0.2])
        max = np.random.uniform([0.1, 0.1, 0.3], [0.3, 0.3, 0.6])
    
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

def main(args = None):

    bproc.init()
    # Включаем необходимые аддоны
    enable("object_fracture_cell")
    bproc.utility.reset_keyframes()
    # parse argument
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    cfg = Config(args.config_file)
    
    # префикс для рендеринга 
    file_prefix_rgb = f'rgb_id{cfg.index_device}ip{cfg.process_id}_'
    file_prefix_segmap = f'segmap_id{cfg.index_device}ip{cfg.process_id}_'

    #bproc.renderer.enable_depth_output(activate_antialiasing=False)
    

    sampled_objs = load_objs(bop_dataset_path = os.path.join(cfg.dataset_parent_path, cfg.bop_dataset_name),
                            mm2m = None,
                            sample_objects = False,
                            num_of_objs_to_sample = 9,
                            additional_scale = None, 
                            manifold = True,
                            object_model_unit = "cm")

    # load BOP datset intrinsics
    intrinsics, width, height = bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(cfg.dataset_parent_path, cfg.bop_dataset_name))
    K = intrinsics  # Матрица камеры
    fx = K[0, 0]
    fy = K[1, 1]

    # Вычисляем FOV (радианы) по ширине/высоте и фокусным расстояниям
    fov_horizontal = 2 * np.arctan(width / (2.0 * fx))
    fov_vertical   = 2 * np.arctan(height / (2.0 * fy))

    if cfg.max_amount_of_samples is None:
        m_a_o_s = np.random.randint(300, 1000)
        bproc.renderer.set_max_amount_of_samples(m_a_o_s)
    else:
        if not isinstance(cfg.max_amount_of_samples, int):
            raise ValueError("max_amount_of_samples must be an integer.")
        bproc.renderer.set_max_amount_of_samples(cfg.max_amount_of_samples)          
    
    # create room
    room_planes = [bproc.object.create_primitive('PLANE', scale=[3, 3, 1]),
                bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, -3, 3], rotation=[-1.570796, 0, 0]),
                bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 3, 3], rotation=[1.570796, 0, 0]),
                bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[3, 0, 3], rotation=[0, -1.570796, 0]),
                bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[-3, 0, 3], rotation=[0, 1.570796, 0])]
    for plane in room_planes:
        plane.enable_rigidbody(False, collision_shape='BOX', friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)

    # sample light color and strenght from ceiling
    light_plane = bproc.object.create_primitive('PLANE', scale=[1, 1, 1], location=[0, 0, 5])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')
    light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                    emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))    
    light_plane.replace_materials(light_plane_material)

    # 
    #cc_textures_obj = custom_load_CCmaterials(cfg.cc_textures.cc_textures_object_path)

    # sample CC Texture and assign to room planes f
    cc_textures = bproc.loader.load_ccmaterials(cfg.cc_textures.cc_textures_path)

    # фрактурируем
    new_sampled_objs = []
    nums_objects = []
    
    while sampled_objs: 
        obj = sampled_objs.pop(0)
        obj_bpy = obj.blender_obj  
        
        # рандомизация параметров
        source_limit_and_count = np.random.choice([1, 2, 3, 4, 5])
        source_noise = np.random.uniform(0, 0.007)
        cell_scale = (
                np.random.uniform(0.75, 1.5),
                np.random.uniform(0.75, 1.5),
                np.random.uniform(0.75, 1.5)
            )
        #seed = np.random.randint(2, 10000)
        seed = np.random.randint(2, 40)

        new_mesh_objects = fracture_object_with_cell(
                                                    bpy_obj = obj_bpy,
                                                    source_limit_and_count = source_limit_and_count,
                                                    source_noise = source_noise,
                                                    cell_scale = cell_scale,
                                                    scale = 0.05, 
                                                    seed = seed
                                                )
        nums_objects.append(len(new_mesh_objects))
        new_sampled_objs.extend(new_mesh_objects)
    
    #generation object on scen and generation camera position
    r = 0
    
    while r < int(cfg.runs_one_fracture):
        # Clear all key frames from the previous run
        bproc.utility.reset_keyframes()

        # options light parametrs
        diap_tem = [5500, 6500]
        colour = [TemperatureToRGBConverter(tem) for tem in diap_tem]
        
        light_energy = np.random.uniform(150, 1000)
        
        # sample point light on shell (defolt: recomendation)
        light_point = bproc.types.Light()
        light_point.set_energy(light_energy)
        light_point.set_color(np.random.uniform(colour[0], colour[1]))
        location = bproc.sampler.shell(center = [0, 0, 4], radius_min = 0.05, radius_max = 1,
                                    elevation_min = -1, elevation_max = 1, uniform_volume = True)
        light_point.set_location(location)

        # set shading and physics properties and randomize PBR materials
        #for j, obj in enumerate(sampled_bop_objs + distractor_bop_objs): 

        # sample CC Texture and assign to room planes
        random_cc_texture = np.random.choice(cc_textures)
        for plane in room_planes:
            plane.replace_materials(random_cc_texture)
        
            
        # Sample object poses and check collisions
        #bproc.object.sample_poses(objects_to_sample = sampled_bop_objs + distractor_bop_objs,   
        
        rand_value = np.random.rand()
        if rand_value < cfg.probability_drop:
            print("Using sample_pose_func_drop (pile) for this scene")
            chosen_pose_func = sample_pose_func_drop
        else:
            print("Using sample_pose_func (normal scatter) for this scene")
            chosen_pose_func = sample_pose_func

        bproc.object.sample_poses(objects_to_sample = new_sampled_objs,
                                sample_pose_func = chosen_pose_func,
                                max_tries = 1000) 
        
        for j, obj in enumerate(new_sampled_objs):
            obj.enable_rigidbody(True, friction=100.0, linear_damping=0.99, angular_damping=0.99)
            obj.set_shading_mode('auto')

            # пример: равновероятно из пула стилей
            mat, style = make_random_material(
                allowed=["plastic_new"],   
                name_prefix=f"obj_{j:06d}"
            )

            # назначаем материал на все слоты объекта (или создаём слот, если пусто)
            mats = obj.get_materials()
            if not mats:
                obj.set_material(0, mat)
            else:
                for i in range(len(mats)):
                    obj.set_material(i, mat)

        # Physics Positioning
        #print("LOOOOOOOOL")
        bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                max_simulation_time=35,
                                                check_object_interval=2,
                                                substeps_per_frame = 30,
                                                solver_iters=30)
        
        # BVH tree used for camera obstacle checks
        #bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_bop_objs + distractor_bop_objs)
        bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(new_sampled_objs)
        rotation_factor = 11.0
        poses = 0
        while poses < cfg.poses_cam:
            # Sample location
            radius_max = round(np.random.uniform(0.5, 1.0), 2)
            location = bproc.sampler.shell(center = [0, 0, 0.1],
                                    radius_min = 0.40,
                                    radius_max = radius_max,
                                    elevation_min = 5,
                                    elevation_max = 89,
                                    uniform_volume = False)
            # Точка интереса (центр сцены)
            poi = np.array([0, 0, 0], dtype=np.float32)
            
            # Вектор взгляда (камера -> центр)
            forward_vector = poi - location
            
            # Расстояние от камеры до центра
            distance_to_center = np.linalg.norm(forward_vector)
            
            # «Базовая» ориентация: камера смотрит из location на [0, 0, 0]
            # '-Z' – направление взгляда, 'Y' – какая ось считается «вверх»
            up_axis = 'Y'
            look_quat = Vector(forward_vector).to_track_quat('-Z', up_axis)
            
            # Преобразуем quaternion в матрицу 3x3
            rotation_matrix = look_quat.to_matrix()
            
            # ----------------------------
            # Рандомизация поворота вокруг всех осей
            # ----------------------------
            # Чем дальше камера, тем больше максимально допустимый угол.  
            # rotation_factor градусов на единицу расстояния:
            max_angle_deg = rotation_factor * distance_to_center
            
            # Генерируем случайные углы (в градусах) вокруг X, Y, Z
            # Можно ограничить угол, чтобы не «уходить» слишком далеко за пределы поля зрения
            random_x_deg = np.random.uniform(-max_angle_deg, max_angle_deg)
            random_y_deg = np.random.uniform(-max_angle_deg, max_angle_deg)
            if random_y_deg > max_angle_deg/2 or random_x_deg > max_angle_deg/2 or (random_y_deg > max_angle_deg/2 and random_x_deg > max_angle_deg/2):
                random_z_deg = np.random.uniform(-max_angle_deg, max_angle_deg/2)
            else: 
                random_z_deg = np.random.uniform(-max_angle_deg, max_angle_deg)
            
            # Переводим градусы в радианы
            random_x = math.radians(random_x_deg)
            random_y = math.radians(random_y_deg)
            random_z = math.radians(random_z_deg)
            
            # Собираем случайный «доворот» (порядок осей – XYZ по умолчанию)
            random_euler = Euler((random_x, random_y, random_z), 'XYZ')
            
            # Умножаем исходную матрицу ориентации на «доворот»
            rotation_matrix @= random_euler.to_matrix()
            
            # Формируем итоговую cam2world_matrix
            cam2world_matrix = bproc.math.build_transformation_mat(location, np.array(rotation_matrix))
            
            # Проверяем, нет ли препятствий и т.д. (если используете obstacle_in_view_check)
            # Здесь bop_bvh_tree и параметры зависят от вашего контекста
            # if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            #     bproc.camera.add_camera_pose(cam2world_matrix)
            #     poses += 1
            
            # Для примера просто добавим позу без obstacle_in_view_check:
            bproc.camera.add_camera_pose(cam2world_matrix)
            poses += 1


        #activate normal rendering
        bproc.renderer.enable_normals_output()
        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"], default_values={"category_id": 0}, output_dir=cfg.temp_dir_segmap, file_prefix=file_prefix_segmap)
        # render the whole pipeline
        data = bproc.renderer.render(output_dir=cfg.temp_dir_rgb, file_prefix = file_prefix_rgb)
        #update_data(data)
        update_data(data)

        lock = FileLock(os.path.join(cfg.output_dir, cfg.dataset_type, '.lock'))
        with lock:
            bproc.writer.write_coco_annotations(os.path.join(cfg.output_dir, cfg.dataset_type),
                                                    instance_segmaps=data["instance_segmaps"],
                                                    instance_attribute_maps=data["instance_attribute_maps"],
                                                    colors=data["colors"],
                                                    color_file_format="JPEG")
    

        #del light_point
        light_point.delete()
        #next
        r += 1




if __name__ == "__main__":
    main()
