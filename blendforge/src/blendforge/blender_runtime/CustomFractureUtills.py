import bpy
import numpy as np
import random 
import blenderproc as bproc

from typing import List, Optional, Tuple
import bmesh
import uuid
from mathutils import Vector
from blenderproc.python.types.MeshObjectUtility import MeshObject, convert_to_meshes


def fracture_object_with_cell(
        bpy_obj,
        source_limit_and_count: int = 3,
        source_noise: float = 0.001,
        cell_scale: Optional[Tuple[float]] = None,
        margin: float = 0.001,
        fracture_settings=None,
        scale=None,
        seed: int = 1,
        max_attempts: int = 4
) -> List[MeshObject]:
    if cell_scale is None:
        cell_scale = (1.0, 1.0, 1.0)

    obj_name = bpy_obj.name  # имя исходного объекта, типа "obj_000001"
    success = False
    attempt = 0
    final_shards: List[MeshObject] = []

    while attempt < max_attempts and not success:
        attempt += 1
        print(f"\n--- Попытка фрактурирования {attempt}/{max_attempts} для объекта {obj_name} ---")

        # 1. Запоминаем имена объектов, которые уже были в сцене
        existing_objects = {o.name for o in bpy.context.scene.objects}

        # 2. Снимаем выделение со всех объектов и выбираем исходный объект
        bpy.ops.object.select_all(action='DESELECT')
        bpy_obj.select_set(True)
        bpy.context.view_layer.objects.active = bpy_obj

        # 3. Particle System для точек разлома
        ps_mod = bpy_obj.modifiers.new(name="FractureSeeds", type='PARTICLE_SYSTEM')
        ps = ps_mod.particle_system
        ps.seed = seed
        ps.settings.count = source_limit_and_count
        ps.settings.frame_end = 1
        ps.settings.lifetime = 1
        ps.settings.emit_from = 'FACE'
        ps.settings.use_emit_random = True

        bpy.context.view_layer.update()

        # 4. Запускаем Fracture Cell
        bpy.ops.object.add_fracture_cell_objects(
            source={'PARTICLE_OWN'},
            source_limit=source_limit_and_count,
            source_noise=source_noise,
            cell_scale=cell_scale,
            use_smooth_faces=False,
            use_sharp_edges=True,
            use_sharp_edges_apply=True,
            use_data_match=True,
            use_island_split=True,
            margin=margin,
            material_index=0,
            use_interior_vgroup=False,
            mass_mode='VOLUME',
            mass=1,
            recursion=0,
            recursion_source_limit=0,
            recursion_clamp=0,
            recursion_chance=0,
            recursion_chance_select='SIZE_MIN',
            use_recenter=True,
            use_remove_original=False,
            use_debug_points=False,
            use_debug_redraw=True,
            use_debug_bool=False
        )

        bpy.context.view_layer.update()

        # 5. Находим новые объекты
        new_objects = [obj for obj in bpy.context.scene.objects if obj.name not in existing_objects]
        mesh_objects = convert_to_meshes(new_objects)

        # 6. Проверяем геометрию фрагментов
        degenerate_found = False
        for shard in mesh_objects:
            bpy_mesh = shard.blender_obj.data
            verts = bpy_mesh.vertices
            faces = bpy_mesh.polygons
            if len(verts) == 0 or len(faces) == 0:
                degenerate_found = True
                break

        if degenerate_found:
            print("Найдены вырожденные фрагменты, повторное фрактурирование с изменёнными параметрами.")
            # Удаляем все созданные осколки данной итерации
            for obj in new_objects:
                if obj.name in bpy.context.scene.objects:
                    bpy.data.objects.remove(bpy.data.objects[obj.name], do_unlink=True)
            # Корректируем параметры
            source_noise *= 0.5
            cell_scale = (1.0, 1.0, 1.0)
            if source_limit_and_count > 2:
                source_limit_and_count -= 1
            seed += 1
        else:
            success = True
            final_shards = mesh_objects

        # Удаляем Particle System с исходного объекта
        mods_to_remove = [mod for mod in bpy_obj.modifiers if mod.type == 'PARTICLE_SYSTEM']
        for mod in mods_to_remove:
            bpy_obj.modifiers.remove(mod)

    if success:
        # category_id исходного объекта (из имени вида obj_000001)
        obj_id = int(obj_name.split("_")[-1])

        # Уникальный тег фрактурирования для ЭТОГО объекта в ЭТОМ запуске
        fracture_tag = uuid.uuid4().hex[:8]  # например "3a9b7c2f"

        for local_idx, cur_obj in enumerate(final_shards):
            # 1) Имя фрагмента: включает исходный obj_name + fracture_tag + локальный индекс
            new_name = f"{obj_name}_fx{fracture_tag}_frag_{local_idx:03d}"
            cur_obj.blender_obj.name = new_name

            # 2) Материал
            mats = cur_obj.get_materials()
            if mats:
                mats[-1].set_name("vertex_col_material")

            # 3) Масштаб
            if scale is not None:
                cur_obj.set_scale(Vector((scale, scale, scale)))

            # 4) category_id для BOP
            cur_obj.set_cp("category_id", obj_id)

            # 5) Доп. CP под пазл
            cur_obj.set_cp("fragment_local_idx", local_idx)
            cur_obj.set_cp("parent_name", obj_name)
            cur_obj.set_cp("parent_category_id", obj_id)
            cur_obj.set_cp("fracture_tag", fracture_tag)

        # Удаляем исходный объект
        if obj_name in bpy.context.scene.objects:
            print(f"Удаление исходного объекта: {obj_name}")
            bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)

        print(f"Фрактурирование объекта {obj_name} завершено, fracture_tag={fracture_tag}.")
    else:
        print(f"Не удалось адекватно сфрактурировать {obj_name} после {max_attempts} попыток.")
        for obj in new_objects:
            if obj.name in bpy.context.scene.objects:
                bpy.data.objects.remove(bpy.data.objects[obj.name], do_unlink=True)
        final_shards = []

    return final_shards


