# module with software-implemented blender functions
import blenderproc as bproc 
# This module makes it easy to write user-friendly command-line interfaces.
import argparse
# This module provides a portable way of using operating system dependent functionality
import os
# library designed to work with multidimensional arrays
import numpy as np
import sys

from blendforge.blender_runtime.CustomLightSetting import TemperatureToRGBConverter
from blendforge.blender_runtime.utils import update_data
from blendforge.blender_runtime.CustomMaterial import (custom_load_CCmaterials, create_mat)
from blendforge.blender_runtime.CustomFractureUtills import fracture_object_with_cell
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


def main(args = None):

    bproc.init()
    # Включаем необходимые аддоны
    enable("object_fracture_cell")
    
    # parse argument
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    cfg = Config(args.config_file)

    # директории для сохранения (train/val/test)
    #types_data = cfg.process_config.types_data
    # количество кадров 
    #frame_iterations = cfg.process_config.frame_iterations
    # количество сцен
    #scene_iterations = cfg.process_config.scene_iterations
    
    # префикс для рендеринга 
    #file_prefix_rgb = f'rgb_id{cfg.index_device}ip{cfg.process_id}_'
    #file_prefix_segmap = f'segmap_id{cfg.index_device}ip{cfg.process_id}_'

    intervals = [(1.25, 1.5), (6, 6.25), (6.25, 6.5), (6.5, 6.75), (6.75, 6.9),
                (1.5, 1.75), (1.75, 2), (2, 2.25), (2.25, 2.5), (2.5, 2.75), 
                (2.75, 3), (3, 3.25),(3.25, 3.5), (3.5, 3.75), (3.75, 4), 
                (4, 4.25), (4.25, 4.5),(4.5, 4.75), (4.75, 5), (5, 5.15), 
                (5.15, 5.25), (5.25, 5.45), (5.45, 5,65), (5.65, 5,75), (5.75, 6)]

    #fov_horizontal = np.radians(69.4)
    #fov_vertical = np.radians(42.5)
    #fov_horizontal = np.radians(65)
    #fov_vertical = np.radians(47)

    sampled_objs = load_objs(bop_dataset_path = os.path.join(cfg.dataset_parent_path, cfg.bop_dataset_name),
                            mm2m = None,
                            sample_objects = False,
                            num_of_objs_to_sample = 9,
                            additional_scale = None, 
                            manifold = True,
                            object_model_unit = "cm")

    # load BOP datset intrinsics
    bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(cfg.dataset_parent_path, cfg.bop_dataset_name))

    bproc.renderer.enable_depth_output(activate_antialiasing=False)

    if cfg.max_amount_of_samples is None:
        bproc.renderer.set_max_amount_of_samples(1024)
    else:
        if not isinstance(cfg.max_amount_of_samples, int):
            raise ValueError("max_amount_of_samples must be an integer.")
        bproc.renderer.set_max_amount_of_samples(cfg.max_amount_of_samples)          
    
    # create room
    room_planes = [bproc.object.create_primitive('PLANE', scale=[7, 7, 1]),
                bproc.object.create_primitive('PLANE', scale=[7, 7, 1], location=[0, -7, 7], rotation=[-1.570796, 0, 0]),
                bproc.object.create_primitive('PLANE', scale=[7, 7, 1], location=[0, 7, 7], rotation=[1.570796, 0, 0]),
                bproc.object.create_primitive('PLANE', scale=[7, 7, 1], location=[7, 0, 7], rotation=[0, -1.570796, 0]),
                bproc.object.create_primitive('PLANE', scale=[7, 7, 1], location=[-7, 0, 7], rotation=[0, 1.570796, 0])]
    for plane in room_planes:
        plane.enable_rigidbody(True, collision_shape='BOX', friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)

    # sample light color and strenght from ceiling
    light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')
    light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                    emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))    
    light_plane.replace_materials(light_plane_material)

    # 
    cc_textures_obj = custom_load_CCmaterials(cfg.cc_textures.cc_textures_object_path)

    # sample CC Texture and assign to room planes f
    cc_textures = bproc.loader.load_ccmaterials(cfg.cc_textures.cc_textures_path)

    # Clear all key frames from the previous run
    bproc.utility.reset_keyframes()

    # options light parametrs
    diap_tem = [5500, 6500]
    colour = [TemperatureToRGBConverter(tem) for tem in diap_tem]
    
    light_energy = np.random.uniform(200, 1600)
    
    # sample point light on shell (defolt: recomendation)
    light_point = bproc.types.Light()
    light_point.set_energy(light_energy)
    light_point.set_color(np.random.uniform(colour[0], colour[1]))
    location = bproc.sampler.shell(center = [0, 0, 4], radius_min = 0.05, radius_max = 0.06,
                                elevation_min = -1, elevation_max = 1, uniform_volume = True)
    light_point.set_location(location)

    # set shading and physics properties and randomize PBR materials
    #for j, obj in enumerate(sampled_bop_objs + distractor_bop_objs): 

    # sample CC Texture and assign to room planes
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)
    
    # Define a function that samples 6-DoF poses (recomendations setting)
    def sample_pose_func(obj: bproc.types.MeshObject):
        
        id_obj = obj.get_cp("category_id")
        if id_obj == 4: 
            min = np.random.uniform([0.1, 0.0, 0.0], [-0.2, -0.2, 0.0])
            max = np.random.uniform([0.4, 0.4, 0.4], [0.3, 0.3, 0.6])
        else: 
            min = np.random.uniform([-0.4, -0.4, 0.0], [-0.2, -0.2, 0.0])
            max = np.random.uniform([0.1, 0.0, 0.4], [0.3, 0.3, 0.6])
        
        obj.set_location(np.random.uniform(min, max))
        obj.set_rotation_euler(bproc.sampler.uniformSO3())
        
    # Sample object poses and check collisions
    #bproc.object.sample_poses(objects_to_sample = sampled_bop_objs + distractor_bop_objs,   
    
    # фрактурируем
    new_sampled_objs = []
    nums_objects = []
    while sampled_objs: 
        obj = sampled_objs.pop(0)
        obj_bpy = obj.blender_obj  
        source_limit_and_count = np.random.choice([2, 3, 4, 5])
        new_mesh_objects = fracture_object_with_cell(obj_bpy, source_limit_and_count = source_limit_and_count, scale = 0.05)
        nums_objects.append(len(new_mesh_objects))
        new_sampled_objs.extend(new_mesh_objects)

    bproc.object.sample_poses(objects_to_sample = new_sampled_objs,
                            sample_pose_func = sample_pose_func, 
                            max_tries = 1000) 
    
    for j, obj in enumerate(new_sampled_objs):
        obj.enable_rigidbody(True, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
        obj.set_shading_mode('auto')
         
        cc_mat = np.random.choice(cc_textures_obj)
        custom_random_mat = create_mat("obj_{:06d}".format(j))
        mat = np.random.choice([cc_mat, custom_random_mat])
        
        for i in range(len(obj.get_materials())):
            obj.set_material(i, mat)

    
    # Physics Positioning
    #print("LOOOOOOOOL")
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                            max_simulation_time=30,
                                            check_object_interval=2,
                                            substeps_per_frame = 25,
                                            solver_iters=25)
    
    # BVH tree used for camera obstacle checks
    #bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_bop_objs + distractor_bop_objs)
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(new_sampled_objs)

    # Параметры камеры (из документации)
    #fov_horizontal = np.radians(65)  # Горизонтальное поле зрения в радианах
    #fov_vertical = np.radians(47)    # Вертикальное поле зрения в радианах
    
    intrinsics = {
        "fx": 608.074476453993,
        "fy": 608.1943718149462,
        "width": 640,
        "height": 480
    }
    # запас 10 граудусов
    zapas = 0.174533

    fov_horizontal = (2 * np.arctan(intrinsics["width"] / (2 * intrinsics["fx"]))) - zapas
    fov_vertical = (2 * np.arctan(intrinsics["height"] / (2 * intrinsics["fy"]))) - zapas
    
    # Размер объекта (в метрах)
    object_size = 214.41196119269955 / 1000  # Преобразуем в метры

    # Цикл для генерации 25 позиций камеры
    poses = 0
    while poses < 1:
        # Получаем позицию камеры (выборка по сфере)
        location = bproc.sampler.shell(center=[0, 0, 0.005],
                                    radius_min=intervals[poses][0],
                                    radius_max=intervals[poses][1],
                                    elevation_min=5,
                                    elevation_max=89,
                                    uniform_volume=False)
    
        # Определяем точку интереса (POI)
        poi = _get_poi(new_sampled_objs, cfg.id_target_obj, nums_objects)
        
        # Вектор направления от камеры к объекту
        forward_vector = poi - location

        # Вычисляем расстояние до объекта
        distance_to_object = np.linalg.norm(forward_vector)
        
        # Поле зрения, которое занимает объект (одинаково по осям)
        alpha = 2 * np.arctan(object_size / (2 * distance_to_object))

        # Вычисляем допустимый угол поворота  
        delta_max_horizontal = np.random.uniform(-((fov_horizontal / 2) - alpha), ((fov_horizontal / 2) - alpha))  
        delta_max_vertical = np.random.uniform(-((fov_vertical / 2) - alpha), ((fov_vertical / 2) - alpha)) 

        up_axis = 'Y'

        # Создаем матрицу поворота, направляя камеру к объекту
        rotation_matrix = Vector(forward_vector).to_track_quat('-Z', up_axis).to_matrix()

        # Добавляем углы поворота
        rotation_matrix @= Euler((delta_max_vertical, delta_max_horizontal, 0)).to_matrix()
        np_rotation_matrix = np.array(rotation_matrix)
        
        # Создаем cam2world_matrix
        cam2world_matrix = bproc.math.build_transformation_mat(location, np_rotation_matrix)

        # Проверяем и добавляем позицию камеры
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            bproc.camera.add_camera_pose(cam2world_matrix)
            poses += 1

    #activate normal rendering
    bproc.renderer.enable_normals_output()
    #bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"], default_values={"category_id": 0}, output_dir=cfg.temp_dir_segmap, file_prefix=file_prefix_segmap)
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"], default_values={"category_id": 0})
    # render the whole pipeline
    data = bproc.renderer.render()
    update_data(data)
             
def _get_poi(sampled_objs, id_target_object, list_source_limit):
    if id_target_object is not None:
        first_skeep_items = sum(list_source_limit[:id_target_object])
        second_skeep_items = sum(list_source_limit[id_target_object+1:])
        poi = bproc.object.compute_poi(np.array(sampled_objs[first_skeep_items:-second_skeep_items]))
    else: 
        poi = bproc.object.compute_poi(np.random.choice(sampled_objs, size=10))
    
    return poi

        

if __name__ == "__main__":
    main()
