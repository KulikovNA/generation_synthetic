import blenderproc as bproc   # ОБЯЗАТЕЛЬНО ПЕРВЫМ!
import os, sys, math, numpy as np, imageio
from mathutils import Euler, Vector

# ─── ПАРАМЕТРЫ ────────────────────────────────────────────────────────
BOP_PARENT_PATH  = "prepared/"
BOP_DATASET_NAME = "differBig"
BOP_TOOLKIT_PATH = "/home/nikita/data_generator/generation_dataset/bop_toolkit"

OBJECT_TEXTURES_DIR = "/home/nikita/data_generator/generation_dataset/syntetic_datasets/resources/textures_1k_modules"   # текстуры для объектов
WALL_TEXTURES_DIR   = "/home/nikita/data_generator/generation_dataset/syntetic_datasets/resources/textures_2k_plate"     # текстуры для фоновой стенки

OUTPUT_DIR = "output"
PNG_NAME   = "scene.png"

FOV_PADDING = 1.05   # 5 % запаса рамки
# ──────────────────────────────────────────────────────────────────────


def safe_load_ccmats(folder):
    return bproc.loader.load_ccmaterials(folder) if os.path.isdir(folder) else []


def make_plain_mat(name, color):
    m = bproc.material.create(name)
    m.set_principled_shader_value("Base Color", color)
    return m


def main():
    # bop-toolkit в PYTHONPATH
    if os.path.isdir(BOP_TOOLKIT_PATH) and BOP_TOOLKIT_PATH not in sys.path:
        sys.path.insert(0, BOP_TOOLKIT_PATH)

    bproc.init()

    # материалы
    obj_mats  = safe_load_ccmats(OBJECT_TEXTURES_DIR)
    wall_mats = safe_load_ccmats(WALL_TEXTURES_DIR)
    if not obj_mats:
        obj_mats  = [make_plain_mat("obj_fallback",  [0.8, 0.2, 0.2, 1])]
    if not wall_mats:
        wall_mats = [make_plain_mat("wall_fallback", [0.6, 0.6, 0.6, 1])]

    # объекты
    objs = bproc.loader.load_bop_objs(
        bop_dataset_path=os.path.join(BOP_PARENT_PATH, BOP_DATASET_NAME),
        mm2m=True, sample_objects=False)

    for o in objs:
        mat = np.random.choice(obj_mats)
        for i in range(len(o.get_materials())):
            o.set_material(i, mat)
        o.set_shading_mode("auto")
        o.set_location(np.random.uniform([-0.15, -0.15, 0.4],
                                         [ 0.15,  0.15, 0.6]))
        o.set_rotation_euler(bproc.sampler.uniformSO3())
        o.enable_rigidbody(True)

    # пол и стенка
    ground = bproc.object.create_primitive("PLANE", scale=[5, 5, 1])
    ground.replace_materials(make_plain_mat("ground", [0.9, 0.9, 0.9, 1]))

    wall = bproc.object.create_primitive(
        "PLANE", scale=[5, 5, 1],
        location=[0, 2.5, 2.5], rotation=[math.radians(90), 0, 0])
    wall.replace_materials(np.random.choice(wall_mats))

    # свет
    light = bproc.types.Light()
    light.set_type("AREA")
    light.set_location([0, 0, 3])
    light.set_energy(1500)

    # физика (роняем, фиксируем)
    bproc.object.simulate_physics_and_fix_final_poses(1, 4)

    # ---------- камера сверху ----------
    # собираем ВСЕ вершины bbox в мировых координатах
    world_coords = []
    for obj in objs:
        bb_local = obj.get_bound_box()           # (8,3) в системе объекта
        M = obj.get_local2world_mat()            # ← здесь нужный метод
        bb_local_h = np.hstack([bb_local, np.ones((8, 1))])  # (8,4)
        bb_world  = (M @ bb_local_h.T).T[:, :3]               # (8,3) world
        world_coords.append(bb_world)

    coords = np.vstack(world_coords)            # (N,3) всех объектов
    bbox_min, bbox_max = coords.min(axis=0), coords.max(axis=0)
    center_xy = 0.5 * (bbox_min[:2] + bbox_max[:2])
    size_xy   = FOV_PADDING * max(bbox_max[0]-bbox_min[0],
                                bbox_max[1]-bbox_min[1]) * 0.5

    fov_x, fov_y = bproc.camera.get_fov()        # радианы
    fov = min(fov_x, fov_y)
    cam_height = size_xy / math.tan(fov/2) + bbox_max[2] + 0.05

    cam_loc = [center_xy[0], center_xy[1], cam_height]
    rot = Euler((math.radians(-90), 0, 0), 'XYZ').to_matrix()
    bproc.camera.add_camera_pose(
    bproc.math.build_transformation_mat(cam_loc, rot))
# ------------------------------------

    rot = Euler((math.radians(-90), 0, 0), 'XYZ').to_matrix()
    bproc.camera.add_camera_pose(
        bproc.math.build_transformation_mat(cam_loc, rot))

    bproc.camera.set_resolution(640, 480)
    bproc.camera.set_intrinsics_from_blender_params(lens=35)
    # ------------------------------------

    # рендер
    rgb = (bproc.renderer.render()["colors"][0] * 255).astype(np.uint8)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    imageio.imwrite(os.path.join(OUTPUT_DIR, PNG_NAME), rgb)


if __name__ == "__main__":
    main()
