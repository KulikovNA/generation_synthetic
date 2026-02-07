import numpy as np
import bpy
import os
from typing import List, Dict, Tuple

from blenderproc.python.types.MeshObjectUtility import MeshObject

import bmesh

def collect_fragments_canonical_gt(
        new_mesh_objects: List[MeshObject],
        split_dir: str,
        fragments_gt: Dict[str, list],
) -> None:
    """
    Собирает канонический GT по фрагментам:
    - scene_id (parent_category_id),
    - fragment_local_idx,
    - com_F,
    - R_O_from_F, t_O_from_F,
    - относительный путь к PLY мешу.

    Всё добавляет в словарь fragments_gt *in-place*.
    """

    import bpy as _bpy  # локальный импорт, чтобы сценарий можно было reuse вне Blender

    for frag in new_mesh_objects:
        # 1) Идентификация пазла и фрагмента
        if not frag.has_cp("parent_category_id"):
            raise RuntimeError(
                f"Fragment {frag.get_name()} has no 'parent_category_id' "
                f"(проверь fracture_object_with_cell)."
            )
        scene_id = int(frag.get_cp("parent_category_id"))

        frag_idx = int(frag.get_cp("fragment_local_idx")) if frag.has_cp("fragment_local_idx") else 0

        # 2) Канонический T_{O<-F}: считаем, что O = world в момент фрактурирования
        H_F2w = np.array(frag.get_local2world_mat(), dtype=np.float32)  # 4x4
        R = H_F2w[:3, :3]
        t = H_F2w[:3, 3]

        # 3) Центр масс в локальной СК фрагмента F
        bpy_mesh = frag.blender_obj.data
        if len(bpy_mesh.vertices) > 0:
            verts_local = np.array([v.co[:] for v in bpy_mesh.vertices], dtype=np.float32)
            com = verts_local.mean(axis=0)
        else:
            com = np.zeros(3, dtype=np.float32)

        # 4) Экспорт меша фрагмента (один раз, если файла ещё нет)
        mesh_name = frag.get_name()  # например, obj_000001_frag_000
        mesh_rel = os.path.join("fragments", f"{mesh_name}.ply")
        mesh_abs = os.path.join(split_dir, mesh_rel)

        if not os.path.exists(mesh_abs):
            os.makedirs(os.path.dirname(mesh_abs), exist_ok=True)
            _bpy.ops.object.select_all(action='DESELECT')
            frag.blender_obj.select_set(True)
            _bpy.context.view_layer.objects.active = frag.blender_obj
            _bpy.ops.export_mesh.ply(
                filepath=mesh_abs,
                use_selection=True,
                use_normals=True,
            )

        # 5) Записываем в fragments_gt
        fragments_gt["frag_scene_id"].append(scene_id)
        fragments_gt["frag_local_idx"].append(frag_idx)
        fragments_gt["com_F"].append(com.tolist())
        fragments_gt["R_O_from_F"].append(R.reshape(-1).tolist())
        fragments_gt["t_O_from_F"].append(t.tolist())
        fragments_gt["mesh_filenames"].append(mesh_rel)


def compute_poi_and_bbox(objects: List[MeshObject]) -> Tuple[np.ndarray, np.ndarray]:
    """ 
    Computes a point of interest (POI) and the global bounding box for all objects.
    
    :param objects: The list of mesh objects that should be considered.
    :return: A tuple (poi, global_bbox), where:
        - poi: Point of interest in the scene.
        - global_bbox: Bounding box enclosing all objects (min point, max point).
    """
    # List to store mean bounding box points
    mean_bb_points = []
    # List to store all bounding box points
    all_bb_points = []

    for obj in objects:
        # Get bounding box corners
        bb_points = obj.get_bound_box()
        all_bb_points.extend(bb_points)  # Collect all bounding box points
        # Compute mean coords of bounding box
        mean_bb_points.append(np.mean(bb_points, axis=0))

    # Query point - mean of means
    mean_bb_point = np.mean(mean_bb_points, axis=0)
    # Closest point (from means) to query point (mean of means)
    poi = mean_bb_points[np.argmin(np.linalg.norm(mean_bb_points - mean_bb_point, axis=1))]

    # Compute global bounding box
    all_bb_points = np.array(all_bb_points)
    global_bbox_min = np.min(all_bb_points, axis=0)
    global_bbox_max = np.max(all_bb_points, axis=0)
    global_bbox = np.array([global_bbox_min, global_bbox_max])

    return poi, global_bbox

def remove_postfix(name: str) -> str:
    """
    Удаляет числовой постфикс из имени объекта, если он присутствует.

    Args:
        name (str): Имя объекта, возможно содержащее числовой постфикс.

    Returns:
        str: Имя объекта без числового постфикса.
    """
    parts = name.split('_')
    if len(parts) > 1:
        return '_'.join(parts[:-1])
    return name

def update_data(data: Dict) -> None:
    """
    Обновляет имена объектов в данных, удаляя числовые постфиксы.

    Args:
        data (Dict): Данные, содержащие имена объектов для обновления.
    """
    for frame in data["instance_attribute_maps"]:
        for obj in frame:
            obj["name"] = remove_postfix(obj["name"])


# --------------------- ФУНКЦИИ ДЛЯ ПРОВЕРКИ / ПОЧИНКИ МЕША ---------------------

def merge_by_distance(obj, distance=0.0001):
    """Сливает близко расположенные вершины (Remove Doubles / Merge by Distance)."""
    
    # Сделаем объект активным и перейдём в Edit Mode
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Убедимся, что все вершины выбраны
    bpy.ops.mesh.select_all(action='SELECT')

    # Попробуем вызвать merge_by_distance
    #try:
    #    # Современный оператор (в большинстве актуальных версий)
    #    bpy.ops.mesh.merge_by_distance(threshold=distance)
    #except RuntimeError:
    #    # Если он не найден, попробуем старый remove_doubles
    bpy.ops.mesh.remove_doubles(threshold=distance)

    # Возвращаемся в Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')

def select_and_fill_non_manifold(obj):
    """
    1) Выделяем все не-манифольдные рёбра (Select Non-Manifold).
    2) Если есть выделенные рёбра, пытаемся заполнить (Fill).
    """
    
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')   
    # Сбрасываем выделение, выбираем не-манифольдные грани/рёбра
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold()

    # Проверяем, есть ли что-то выделенное
    mesh = bmesh.from_edit_mesh(bpy.context.object.data)
    selected_edges = [e for e in mesh.edges if e.select]
    
    if not selected_edges:
        print("Нет не-манифольдных краёв – fill() пропущен.")
    else:
        # Пытаемся заполнить
        try:
            bpy.ops.mesh.fill()
        except RuntimeError as ex:
            print(f"Fill не сработал: {ex}")
            # При желании можно сделать fallback-действие или просто пропустить

    bpy.ops.object.mode_set(mode='OBJECT')


def remove_interior_faces(obj):
    """Выделяем внутренние грани и удаляем их."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    # Выбираем внутренние грани
    bpy.ops.mesh.select_interior_faces()
    # Удаляем выбранные грани
    bpy.ops.mesh.delete(type='FACE')
    bpy.ops.object.mode_set(mode='OBJECT')

def recalc_normals(obj):
    """Перерасчёт нормалей вовне."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

def voxel_remesh(obj, voxel_size=0.01, smooth_shade=True):
    """
    Применяем Voxel Remesh, чтобы «лечить» любые оставшиеся дыры/самопересечения.
    Потеряем исходную топологию, но получим манифольдный объект.
    """
    bpy.context.view_layer.objects.active = obj
    mod = obj.modifiers.new("RemeshFix", 'REMESH')
    mod.mode = 'VOXEL'
    mod.voxel_size = voxel_size
    mod.use_smooth_shade = smooth_shade
    bpy.ops.object.modifier_apply(modifier=mod.name)

def is_non_manifold(obj):
    """
    Проверка, остались ли не-манифольдные элементы. Возвращает True, если да.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold()

    # Проверим, выделилось ли что-то
    bm = bmesh.from_edit_mesh(bpy.context.object.data)
    selected_verts = [v for v in bm.verts if v.select]
    # Возвращаемся в Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')
    return len(selected_verts) > 0