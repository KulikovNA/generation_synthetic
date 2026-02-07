# module with software-implemented blender functions
import blenderproc as bproc 
import argparse, os, sys, math
import numpy as np

from blendforge.blender_runtime.CustomLightSetting import TemperatureToRGBConverter
from blendforge.blender_runtime.CustomMaterial import (create_mat, custom_load_CCmaterials,)
from blendforge.host.FiletoDict import Config


from filelock import FileLock
from mathutils import Vector, Euler

def parse_args(args):
    parser = argparse.ArgumentParser(description='Starting script.')
    parser.add_argument('--config_file',
                        help="Path to config file",
                        type=str)
    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)

# ---------- ТВОИ МЕТОДЫ РАЗМЕЩЕНИЯ ОБЪЕКТОВ ----------
def sample_pose_func_drop(obj: bproc.types.MeshObject):
    """Спавним объект в небольшом цилиндре над полом — все падают в одну область."""
    radius = 0.1
    z_min, z_max = 0.5, 1.0
    theta = np.random.uniform(0, 2*np.pi)
    r = np.random.uniform(0, radius)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.random.uniform(z_min, z_max)
    obj.set_location([x, y, z])
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

def sample_pose_func(obj: bproc.types.MeshObject):
    """Обычное разбрасывание в ограниченном объёме (с веткой по category_id как у тебя)."""
    id_obj = obj.get_cp("category_id")
    if id_obj == 4:
        mn = np.random.uniform([-0.3, -0.3, 0.1], [-0.1, -0.1, 0.2])
        mx = np.random.uniform([ 0.1,  0.1, 0.3], [ 0.3,  0.3, 0.6])
    else:
        mn = np.random.uniform([-0.3, -0.3, 0.1], [-0.1, -0.1, 0.2])
        mx = np.random.uniform([ 0.1,  0.1, 0.3], [ 0.3,  0.3, 0.6])
    obj.set_location(np.random.uniform(mn, mx))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())
# ------------------------------------------------------

def main(args=None):
    bproc.init()

    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    cfg = Config(args.config_file)

    if cfg.max_amount_of_samples is None:
        m_a_o_s = np.random.randint(300, 1000)
        bproc.renderer.set_max_amount_of_samples(m_a_o_s)
    else:
        if not isinstance(cfg.max_amount_of_samples, int):
            raise ValueError("max_amount_of_samples must be an integer.")
        bproc.renderer.set_max_amount_of_samples(cfg.max_amount_of_samples)   

    # объекты (как было у тебя в этом скрипте)
    sampled_bop_objs = bproc.loader.load_bop_objs(
        bop_dataset_path=os.path.join(cfg.bop_parent_path, cfg.bop_dataset_name),
        mm2m=True,
        sample_objects=False,
        num_of_objs_to_sample=9
    )

    # интринсики грузим (для консистентности пайплайна)
    bproc.loader.load_bop_intrinsics(
        bop_dataset_path=os.path.join(cfg.bop_parent_path, cfg.bop_dataset_name)
    )

    bproc.renderer.enable_depth_output(activate_antialiasing=False)

    # комната
    room_planes = [
        bproc.object.create_primitive('PLANE', scale=[7, 7, 1]),
        bproc.object.create_primitive('PLANE', scale=[7, 7, 1], location=[0, -7, 7], rotation=[-1.570796, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[7, 7, 1], location=[0,  7, 7], rotation=[ 1.570796, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[7, 7, 1], location=[7,  0, 7], rotation=[0, -1.570796, 0]),
        bproc.object.create_primitive('PLANE', scale=[7, 7, 1], location=[-7, 0, 7], rotation=[0,  1.570796, 0]),
    ]
    for plane in room_planes:
        plane.enable_rigidbody(False, collision_shape='BOX', friction=100.0, linear_damping=0.99, angular_damping=0.99)

    # мягкий потолочный свет
    light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')
    light_plane_material.make_emissive(
        emission_strength=np.random.uniform(3, 6),
        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0])
    )
    light_plane.replace_materials(light_plane_material)

    # материалы
    cc_textures_obj = custom_load_CCmaterials(cfg.cc_textures.cc_textures_object_path)
    cc_textures = bproc.loader.load_ccmaterials(cfg.cc_textures.cc_textures_path)

    r = 0
    while r < int(cfg.runs):
        bproc.utility.reset_keyframes()

        # точечный свет
        diap_tem = [5500, 6500]
        colour = [TemperatureToRGBConverter(tem) for tem in diap_tem]
        light_energy = np.random.uniform(200, 1600)
        light_point = bproc.types.Light()
        light_point.set_energy(light_energy)
        light_point.set_color(np.random.uniform(colour[0], colour[1]))
        location = bproc.sampler.shell(center=[0, 0, 4], radius_min=0.05, radius_max=0.06,
                                       elevation_min=-1, elevation_max=1, uniform_volume=True)
        light_point.set_location(location)

        # материалы объектов
        for j, obj in enumerate(sampled_bop_objs):
            obj.enable_rigidbody(True, friction=100.0, linear_damping=0.99, angular_damping=0.99)
            obj.set_shading_mode('auto')
            cc_mat = np.random.choice(cc_textures_obj)
            custom_random_mat = create_mat(f"obj_{j:06d}")
            mat = np.random.choice([cc_mat, custom_random_mat])
            for i in range(len(obj.get_materials())):
                obj.set_material(i, mat)

        # материалы стен/пола
        random_cc_texture = np.random.choice(cc_textures)
        for plane in room_planes:
            plane.replace_materials(random_cc_texture)

        # ----------- РАЗМЕЩЕНИЕ ОБЪЕКТОВ (твои методы) -----------
        if np.random.rand() < float(getattr(cfg, "probability_drop", 0.7)):
            chosen_pose_func = sample_pose_func_drop
        else:
            chosen_pose_func = sample_pose_func

        bproc.object.sample_poses(
            objects_to_sample=sampled_bop_objs,
            sample_pose_func=chosen_pose_func,
            max_tries=1000
        )
        # ---------------------------------------------------------

        # физика
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=3,
            max_simulation_time=30,
            check_object_interval=2,
            substeps_per_frame=25,
            solver_iters=25
        )

        # BVH для obstacle check
        bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_bop_objs)

        # ----------- ПОЗЫ КАМЕРЫ (твоя логика) -----------
        rotation_factor = 11.0
        poses = 0
        max_poses = int(getattr(cfg, 'poses_cam', 25))
        while poses < max_poses:
            radius_max = round(np.random.uniform(1.0, 2.3), 2)
            cam_loc = bproc.sampler.shell(center=[0, 0, 0.1],
                                          radius_min=0.40,
                                          radius_max=radius_max,
                                          elevation_min=5,
                                          elevation_max=89,
                                          uniform_volume=False)
            poi = np.array([0, 0, 0], dtype=np.float32)
            fwd = poi - cam_loc
            dist = float(np.linalg.norm(fwd))

            up_axis = 'Y'
            rot_m = Vector(fwd).to_track_quat('-Z', up_axis).to_matrix()

            max_angle_deg = rotation_factor * dist
            rx_deg = np.random.uniform(-max_angle_deg, max_angle_deg)
            ry_deg = np.random.uniform(-max_angle_deg, max_angle_deg)
            if (ry_deg > max_angle_deg / 2) or (rx_deg > max_angle_deg / 2):
                rz_deg = np.random.uniform(-max_angle_deg, max_angle_deg / 2)
            else:
                rz_deg = np.random.uniform(-max_angle_deg, max_angle_deg)

            rot_m @= Euler((math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)), 'XYZ').to_matrix()
            cam2world = bproc.math.build_transformation_mat(cam_loc, np.array(rot_m))

            if bproc.camera.perform_obstacle_in_view_check(cam2world, {"min": 0.3}, bop_bvh_tree):
                bproc.camera.add_camera_pose(cam2world)
                poses += 1
        # -------------------------------------------------

        # рендер
        data = bproc.renderer.render(output_dir=cfg.temp_dir_rgb, file_prefix=cfg.file_prefix_rgb)

        # запись
        lock = FileLock(os.path.join(cfg.output_dir, '.lock'))
        with lock:
            bproc.writer.write_bop(
                cfg.output_dir,
                dataset=cfg.bop_dataset_name,
                depths=data["depth"],
                colors=data["colors"],
                color_file_format="JPEG",
                ignore_dist_thres=10
            )

        light_point.delete()
        r += 1

if __name__ == "__main__":
    main()
