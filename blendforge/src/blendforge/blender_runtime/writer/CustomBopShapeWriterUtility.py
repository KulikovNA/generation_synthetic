"""
Custom BOP writer with extra shape/puzzle annotations (без чанков).

Формат директорий (для каждого датасета):

dataset_root/
├── camera.json
├── dataset_info.json
├── models_priori_objects/
│   └── ...
├── train/
│   ├── rgb/
│   ├── depth/
│   ├── mask/
│   ├── mask_visib/
│   ├── scene_camera.json
│   ├── scene_gt.json
│   ├── scene_gt_info.json
│   ├── scene_gt_coco.json
│   ├── fragments_gt.json           # GT по фрагментам и пазлу
│   ├── fragments/                  # меши фрагментов (PLY/STL и т.п.)
│   └── scenes.json                 # meta: scene_id -> image_ids, frag_count, ...
├── val/
└── test/
"""

from functools import partial
import json
from multiprocessing import Pool
import os
import glob
import trimesh
from typing import List, Optional, Dict, Tuple, Any
import warnings
import datetime

import numpy as np
import png
import cv2
import bpy
from mathutils import Matrix
import sys

from blenderproc.python.types.MeshObjectUtility import MeshObject, get_all_mesh_objects
from blenderproc.python.writer.WriterUtility import _WriterUtility
from blenderproc.python.types.LinkUtility import Link
from blenderproc.python.utility.SetupUtility import SetupUtility
from blenderproc.python.utility.MathUtility import change_target_coordinate_frame_of_transformation_matrix

# EGL is not available under windows
if sys.platform in ["linux", "linux2"]:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'


def write_bop(
        output_dir: str,
        target_objects: Optional[List[MeshObject]] = None,
        depths: List[np.ndarray] = None,
        colors: List[np.ndarray] = None,
        color_file_format: str = "PNG",
        dataset: str = "",
        append_to_existing_output: bool = True,
        depth_scale: float = 1.0,
        jpg_quality: int = 95,
        save_world2cam: bool = True,
        ignore_dist_thres: float = 100.,
        m2mm: Optional[bool] = None,
        annotation_unit: str = 'mm',
        frames_per_chunk: int = 1000,  # теперь игнорируется (оставлен для совместимости сигнатуры)
        calc_mask_info_coco: bool = True,
        delta: float = 0.015,
        num_worker: Optional[int] = None,
        # --- новое: явно задаём сплит ---
        split: str = "train",
        # --- дополнительные структуры под 3D-пазл ---
        fragments_gt: Optional[Dict[str, Any]] = None,
        scenes_meta: Optional[Dict[str, Any]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
):
    """Write the BOP-style data + дополнительные файлы под 3D-пазл фрагментов.

    ВАЖНО: Чанки (000000, 000001, ...) не используются. Для каждого сплита
    (train/val/test) всё лежит в одном каталоге split_dir.

    :param output_dir: Корень датасета (dataset_root).
    :param target_objects: Объекты, для которых сохраняются GT-позы.
                           Ожидается, что фрагменты имеют CP:
                           - category_id
                           - fragment_local_idx (локальный индекс 1..K в пазле)
                           - scene_id (или parent_category_id / category_id как fallback)
    :param depths: Список depth-изображений в метрах.
    :param colors: Список RGB-изображений.
    :param color_file_format: "PNG" или "JPEG".
    :param jpg_quality: Качество JPEG (если выбран JPEG).
    :param dataset: Имя подпапки датасета внутри output_dir.
    :param append_to_existing_output: Если False и split_dir уже существует → ошибка.
    :param depth_scale: Масштаб к uint16 глубине (как в BOP).
    :param save_world2cam: Сохранять ли cam_R_w2c, cam_t_w2c в scene_camera.json.
    :param ignore_dist_thres: Порог по расстоянию до объекта (отбрасывание "улетевших").
    :param m2mm: Устаревший флаг, используй annotation_unit='mm'.
    :param annotation_unit: 'm', 'dm', 'cm', 'mm' — единицы в scene_gt.
    :param frames_per_chunk: НЕ используется (сохранён для совместимости).
    :param calc_mask_info_coco: Считать ли mask, gt_info, coco.
    :param delta: Толеранс для visibility.
    :param num_worker: Число процессов для расчёта mask/info.
    :param split: Имя сплита ('train', 'val', 'test', ...).
    :param fragments_gt: dict → fragments_gt.json в split_dir.
                         Ожидаемые ключи (рекомендуется):
                             "scene_ids": [...]
                             "frag_counts": [...]
                             "total_frags": int
                             "frag_scene_id": [...]
                             "frag_local_idx": [...]
                             "com_F": [...]
                             "R_O_from_F": [...]
                             "t_O_from_F": [...]
                             "mesh_filenames": [...]
    :param scenes_meta: dict → scenes.json в split_dir.
                        Рекомендуемый формат:
                        {
                          "scene_ids": [...],
                          "scenes": {
                            "1": {
                              "scene_id": 1,
                              "obj_id": 1,
                              "model_file": "...",
                              "frag_count": 4,
                              "frag_local_indices": [1,2,3,4],
                              "image_ids": [0,3,7,9]
                            },
                            ...
                          }
                        }
    :param dataset_info: dict → dataset_info.json (в корне датасета).
    """

    # Путь до корня датасета и сплита
    dataset_dir = os.path.join(output_dir, dataset) if dataset else output_dir
    split_dir = os.path.join(dataset_dir, split)
    camera_path = os.path.join(dataset_dir, 'camera.json')

    # Создаём корневую папку датасета/сплита
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    if os.path.exists(split_dir):
        if not append_to_existing_output:
            raise FileExistsError(f"The split folder already exists: {split_dir}")
    else:
        os.makedirs(split_dir)

    # dataset_info.json (только пишем/обновляем, если передан словарь)
    if dataset_info is not None:
        info_path = os.path.join(dataset_dir, "dataset_info.json")
        _BopWriterUtility.save_json(info_path, dataset_info)
        print(f"[CustomBopShapeWriter] Saved dataset_info.json to: {info_path}")

    # Выбор объектов
    if target_objects is not None:
        dataset_objects = target_objects
        for obj in dataset_objects:
            if obj.is_hidden():
                print(
                    f"WARNING: The given object {obj.get_name()} is hidden. "
                    "However, the writer will still add coco annotations for it."
                )
    elif dataset:
        dataset_objects = []
        for obj in get_all_mesh_objects():
            if "bop_dataset_name" in obj.blender_obj and not obj.is_hidden():
                if obj.blender_obj["bop_dataset_name"] == dataset:
                    dataset_objects.append(obj)
    else:
        dataset_objects = [obj for obj in get_all_mesh_objects() if not obj.is_hidden()]

    if not dataset_objects:
        raise RuntimeError(
            f"The scene does not contain any visible object "
            f"(or from the specified dataset '{dataset}')."
        )

    # Сохраняем глобальный camera.json (один на весь датасет)
    _BopWriterUtility.write_camera(camera_path, depth_scale=depth_scale)

    # Масштаб для аннотаций поз
    assert annotation_unit in ['m', 'dm', 'cm', 'mm'], (
        f"Invalid annotation unit: `{annotation_unit}`. "
        f"Supported are 'm', 'dm', 'cm', 'mm'"
    )
    annotation_scale = {'m': 1., 'dm': 10., 'cm': 100., 'mm': 1000.}[annotation_unit]
    if m2mm is not None:
        warnings.warn("`m2mm` is deprecated, use `annotation_unit='mm'` instead!")
        annotation_scale = 1000.

    # Пишем все кадры флэтом в split_dir (без чанков)
    _BopWriterUtility.write_frames_flat(
        split_dir=split_dir,
        dataset_objects=dataset_objects,
        depths=depths,
        colors=colors,
        color_file_format=color_file_format,
        annotation_scale=annotation_scale,
        ignore_dist_thres=ignore_dist_thres,
        save_world2cam=save_world2cam,
        depth_scale=depth_scale,
        jpg_quality=jpg_quality,
    )

    # fragments_gt.json
    if fragments_gt is not None:
        _BopWriterUtility.write_fragments_gt_json(
            split_dir=split_dir,
            fragments_gt=fragments_gt,
            filename="fragments_gt.json",
        )

    # scenes.json
    if scenes_meta is not None:
        _BopWriterUtility.write_scenes_json(
            split_dir=split_dir,
            scenes_meta=scenes_meta,
            filename="scenes.json",
        )

    # Маски / gt_info / COCO (работаем по одному split_dir как "сцене")
    if calc_mask_info_coco:
        # Устанавливаем bop_toolkit и PyOpenGL
        SetupUtility.setup_pip(["git+https://github.com/thodan/bop_toolkit", "PyOpenGL==3.1.0"])
        # numpy>=1.20: np.float deprecated
        np.float = float  # type: ignore[attr-defined]

        # Конвертим объекты в trimesh
        trimesh_objects = {}
        for obj in dataset_objects:
            if isinstance(obj, Link):
                if not obj.visuals:
                    continue
                if len(obj.visuals) > 1:
                    warnings.warn('Writer only supports one visual mesh per Link.')
                mesh_obj = obj.visuals[0]
            else:
                mesh_obj = obj

            cat_id = mesh_obj.get_cp('category_id') if mesh_obj.has_cp('category_id') else None
            if cat_id is None:
                continue
            if cat_id in trimesh_objects:
                continue

            trimesh_obj = mesh_obj.mesh_as_trimesh()
            if not np.all(np.isclose(np.array(mesh_obj.blender_obj.scale), mesh_obj.blender_obj.scale[0])):
                print(
                    "WARNING: non-uniform scale on object; BOP annotations may be inconsistent with pyrender."
                )
            trimesh_objects[cat_id] = trimesh_obj

        # Пул для pyrender
        width = bpy.context.scene.render.resolution_x
        height = bpy.context.scene.render.resolution_y
        pool = Pool(
            num_worker,
            initializer=_BopWriterUtility._pyrender_init,
            initargs=[width, height, trimesh_objects],
        )

        # Один "chunk_dir" = split_dir
        chunk_dirs = [split_dir]
        starting_frame_id = 0

        _BopWriterUtility.calc_gt_masks(
            pool=pool,
            chunk_dirs=chunk_dirs,
            starting_frame_id=starting_frame_id,
            annotation_scale=annotation_scale,
            delta=delta,
        )

        _BopWriterUtility.calc_gt_info(
            pool=pool,
            chunk_dirs=chunk_dirs,
            starting_frame_id=starting_frame_id,
            annotation_scale=annotation_scale,
            delta=delta,
        )

        _BopWriterUtility.calc_gt_coco(
            chunk_dirs=chunk_dirs,
            dataset_objects=dataset_objects,
            starting_frame_id=starting_frame_id,
        )

        pool.close()
        pool.join()


def bop_pose_to_pyrender_coordinate_system(cam_R_m2c: np.ndarray, cam_t_m2c: np.ndarray) -> np.ndarray:
    """Converts an object pose in BOP format to pyrender camera coordinate system.

    :param cam_R_m2c: 3x3 rotation matrix.
    :param cam_t_m2c: 3-dim translation vector.
    :return: 4x4 pose in pyrender coordinate system.
    """
    bop_pose = np.eye(4)
    bop_pose[:3, :3] = cam_R_m2c
    bop_pose[:3, 3] = cam_t_m2c
    return change_target_coordinate_frame_of_transformation_matrix(bop_pose, ["X", "-Y", "-Z"])


class _BopWriterUtility:
    """Утилиты для записи BOP-подобного датасета в плоском формате (без чанков)."""

    # -------------------------------------------------------------------------
    # Базовые I/O функции
    # -------------------------------------------------------------------------
    @staticmethod
    def load_json(path, keys_to_int: bool = False):
        """Loads content of a JSON file.

        :param path: Path to the JSON file.
        :param keys_to_int: Convert digit dict keys to integers.
        """
        def convert_keys_to_int(x):
            return {int(k) if k.lstrip('-').isdigit() else k: v for k, v in x.items()}

        with open(path, 'r', encoding="utf-8") as f:
            if keys_to_int:
                return json.load(f, object_hook=convert_keys_to_int)
            return json.load(f)

    @staticmethod
    def save_json(path, content):
        """Saves the content to a JSON file in a human-friendly format."""
        text = ""
        with open(path, 'w', encoding="utf-8") as file:
            if isinstance(content, dict):
                text += '{\n'
                content_sorted = sorted(content.items(), key=lambda x: x[0])
                for elem_id, (k, v) in enumerate(content_sorted):
                    text += f'  "{k}": {json.dumps(v, sort_keys=True)}'
                    if elem_id != len(content) - 1:
                        text += ','
                    text += '\n'
                text += '}'
                file.write(text)
            elif isinstance(content, list):
                text += '[\n'
                for elem_id, elem in enumerate(content):
                    text += f'  {json.dumps(elem, sort_keys=True)}'
                    if elem_id != len(content) - 1:
                        text += ','
                    text += '\n'
                text += ']'
                file.write(text)
            else:
                json.dump(content, file, sort_keys=True)

    @staticmethod
    def save_depth(path: str, im: np.ndarray):
        """Saves a depth image (16-bit PNG)."""
        if not path.endswith(".png"):
            raise ValueError('Only PNG format is currently supported.')

        im = im.copy()
        im[im > 65535] = 65535
        im_uint16 = np.round(im).astype(np.uint16)

        w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
        with open(path, 'wb') as f:
            w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))

    @staticmethod
    def write_camera(camera_path: str, depth_scale: float = 1.0):
        """Writes global camera.json in dataset root."""
        bpy.context.scene.frame_set(1)
        cam_K = _WriterUtility.get_cam_attribute(bpy.context.scene.camera, 'cam_K')
        camera = {
            'cx': cam_K[0][2],
            'cy': cam_K[1][2],
            'depth_scale': depth_scale,
            'fx': cam_K[0][0],
            'fy': cam_K[1][1],
            'height': bpy.context.scene.render.resolution_y,
            'width': bpy.context.scene.render.resolution_x,
        }
        _BopWriterUtility.save_json(camera_path, camera)

    # -------------------------------------------------------------------------
    # Новые методы: fragments_gt.json + scenes.json
    # -------------------------------------------------------------------------
    @staticmethod
    def write_fragments_gt_json(
            split_dir: str,
            fragments_gt: Dict,
            filename: str = "fragments_gt.json",
    ) -> None:
        """
        JSON-описание канонических фрагментов.

        Формат (после merge'а всех запусков):

        {
          "frag_scene_id":   [ ... ],  # длина = total_frags
          "frag_local_idx":  [ ... ],
          "com_F":           [ [x,y,z], ... ],
          "R_O_from_F":      [ [9 значений], ... ],
          "t_O_from_F":      [ [3 знач.], ... ],
          "mesh_filenames":  [ "fragments/obj_...ply", ... ],

          "scene_ids":   [ ... уникальные значения frag_scene_id ... ],
          "frag_counts": [ ... кол-во фрагментов на каждый scene_id ... ],
          "total_frags": int
        }
        """
        os.makedirs(split_dir, exist_ok=True)
        json_path = os.path.join(split_dir, filename)

        # какие поля считаем "пер-фрагментными" (их надо конкатенировать)
        per_frag_keys = [
            "frag_scene_id",
            "frag_local_idx",
            "com_F",
            "R_O_from_F",
            "t_O_from_F",
            "mesh_filenames",
        ]

        # инициализация "старых" данных
        if os.path.exists(json_path):
            old = _BopWriterUtility.load_json(json_path)
        else:
            old = {}

        # соберём объединённый словарь
        merged = {k: [] for k in per_frag_keys}

        # 1) старые значения
        for k in per_frag_keys:
            if k in old and isinstance(old[k], list):
                merged[k].extend(old[k])

        # 2) новые значения из текущего запуска
        for k in per_frag_keys:
            if k in fragments_gt and isinstance(fragments_gt[k], list):
                merged[k].extend(fragments_gt[k])

        # базовая проверка длины
        lengths = [len(merged[k]) for k in per_frag_keys]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"[write_fragments_gt_json] Inconsistent list lengths: {dict(zip(per_frag_keys, lengths))}"
            )

        # пересчёт агрегированных полей
        if merged["frag_scene_id"]:
            frag_scene_id_arr = np.array(merged["frag_scene_id"], dtype=int)
            scene_ids_unique, counts = np.unique(frag_scene_id_arr, return_counts=True)

            merged["scene_ids"] = scene_ids_unique.tolist()
            merged["frag_counts"] = counts.tolist()
            merged["total_frags"] = int(frag_scene_id_arr.size)
        else:
            merged["scene_ids"] = []
            merged["frag_counts"] = []
            merged["total_frags"] = 0

        _BopWriterUtility.save_json(json_path, merged)
        print(f"[CustomBopShapeWriter] Saved merged fragments_gt.json to: {json_path}")


    @staticmethod
    def write_scenes_json(
            split_dir: str,
            scenes_meta: Dict[str, Any],
            filename: str = "scenes.json",
    ) -> None:
        """Сохраняет scenes.json в split_dir.

        Ожидаемый формат scenes_meta:
        {
          "scene_ids": [...],
          "scenes": {
            "1": { ... },
            ...
          }
        }
        """
        os.makedirs(split_dir, exist_ok=True)
        scenes_path = os.path.join(split_dir, filename)
        _BopWriterUtility.save_json(scenes_path, scenes_meta)
        print(f"[CustomBopShapeWriter] Saved scenes.json to: {scenes_path}")

    # -------------------------------------------------------------------------
    # GT для кадра / камеры
    # -------------------------------------------------------------------------
    @staticmethod
    def get_frame_gt(
            dataset_objects: List[MeshObject],
            unit_scaling: float,
            ignore_dist_thres: float,
            destination_frame: Optional[List[str]] = None,
    ):
        """Returns GT pose annotations between active camera and objects.

        Дополнительно:
        - если у объекта есть CP "fragment_local_idx", то он пишется как frag_local_idx
        - scene_id берётся из CP "scene_id", иначе "parent_category_id", иначе category_id
        """
        if destination_frame is None:
            destination_frame = ["X", "-Y", "-Z"]

        H_c2w_opencv = Matrix(
            _WriterUtility.get_cam_attribute(
                bpy.context.scene.camera,
                'cam2world_matrix',
                local_frame_change=destination_frame,
            )
        )

        frame_gt = []
        for obj in dataset_objects:
            # базовый объект, у которого лежат CP (для Link берём visuals[0])
            if isinstance(obj, Link):
                if not obj.visuals:
                    continue
                if len(obj.visuals) > 1:
                    warnings.warn('BOP Writer only supports poses of one visual mesh per Link.')
                H_m2w = Matrix(obj.get_visual_local2world_mats()[0])
                base_obj = obj.visuals[0]
            else:
                H_m2w = Matrix(obj.get_local2world_mat())
                base_obj = obj

            assert base_obj.has_cp("category_id"), (
                f"{base_obj.get_name()} has no custom property 'category_id'"
            )
            cat_id = base_obj.get_cp("category_id")

            # fragment_local_idx (локальный индекс фрагмента в пазле)
            frag_local_idx = None
            if base_obj.has_cp("fragment_local_idx"):
                try:
                    frag_local_idx = int(base_obj.get_cp("fragment_local_idx"))
                except Exception:
                    frag_local_idx = None

            # scene_id (ID пазла / канонического объекта)
            scene_id = None
            if base_obj.has_cp("scene_id"):
                try:
                    scene_id = int(base_obj.get_cp("scene_id"))
                except Exception:
                    scene_id = None
            if scene_id is None and base_obj.has_cp("parent_category_id"):
                try:
                    scene_id = int(base_obj.get_cp("parent_category_id"))
                except Exception:
                    scene_id = None
            if scene_id is None:
                scene_id = int(cat_id)

            cam_H_m2c = H_c2w_opencv.inverted() @ H_m2w
            cam_R_m2c = cam_H_m2c.to_quaternion().to_matrix()
            cam_t_m2c = cam_H_m2c.to_translation()

            if not np.linalg.norm(list(cam_t_m2c)) > ignore_dist_thres:
                cam_t_m2c_scaled = list(cam_t_m2c * unit_scaling)
                gt_entry: Dict[str, Any] = {
                    'cam_R_m2c': list(cam_R_m2c[0]) + list(cam_R_m2c[1]) + list(cam_R_m2c[2]),
                    'cam_t_m2c': cam_t_m2c_scaled,
                    'obj_id': int(cat_id),
                    'scene_id': int(scene_id),
                }
                if frag_local_idx is not None:
                    gt_entry['frag_local_idx'] = int(frag_local_idx)

                frame_gt.append(gt_entry)
            else:
                print(
                    f'Ignored obj {cat_id}: distance > ignore_dist_thres ({ignore_dist_thres}) '
                    f'or pose not in meters.'
                )

        return frame_gt

    @staticmethod
    def get_frame_camera(
            save_world2cam: bool,
            depth_scale: float = 1.0,
            unit_scaling: float = 1000.,
            destination_frame: Optional[List[str]] = None,
    ):
        """Returns camera parameters for the active camera (scene_camera.json)."""
        if destination_frame is None:
            destination_frame = ["X", "-Y", "-Z"]

        cam_K = _WriterUtility.get_cam_attribute(bpy.context.scene.camera, 'cam_K')
        frame_camera_dict: Dict[str, Any] = {
            'cam_K': cam_K[0] + cam_K[1] + cam_K[2],
            'depth_scale': depth_scale,
        }

        if save_world2cam:
            H_c2w_opencv = Matrix(
                _WriterUtility.get_cam_attribute(
                    bpy.context.scene.camera,
                    'cam2world_matrix',
                    local_frame_change=destination_frame,
                )
            )
            H_w2c_opencv = H_c2w_opencv.inverted()
            R_w2c_opencv = H_w2c_opencv.to_quaternion().to_matrix()
            t_w2c_opencv = H_w2c_opencv.to_translation() * unit_scaling

            frame_camera_dict['cam_R_w2c'] = \
                list(R_w2c_opencv[0]) + list(R_w2c_opencv[1]) + list(R_w2c_opencv[2])
            frame_camera_dict['cam_t_w2c'] = list(t_w2c_opencv)

        return frame_camera_dict

    # -------------------------------------------------------------------------
    # Плоская запись кадров без чанков
    # -------------------------------------------------------------------------
    @staticmethod
    def write_frames_flat(
            split_dir: str,
            dataset_objects: list,
            depths: List[np.ndarray],
            colors: List[np.ndarray],
            color_file_format: str = "PNG",
            depth_scale: float = 1.0,
            annotation_scale: float = 1000.,
            ignore_dist_thres: float = 100.,
            save_world2cam: bool = True,
            jpg_quality: int = 95,
    ):
        """Записывает все кадры сплита в одну папку (без чанков).

        Форматы:
        split_dir/
            rgb/{im_id:06d}.png or .jpg
            depth/{im_id:06d}.png
            scene_gt.json
            scene_camera.json
        """

        # Папки для изображений
        rgb_dir = os.path.join(split_dir, 'rgb')
        depth_dir = os.path.join(split_dir, 'depth')
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        scene_gt_path = os.path.join(split_dir, 'scene_gt.json')
        scene_camera_path = os.path.join(split_dir, 'scene_camera.json')

        # Загружаем уже существующие GT (если есть) → будем аппендить
        if os.path.exists(scene_gt_path):
            scene_gt = _BopWriterUtility.load_json(scene_gt_path, keys_to_int=True)
        else:
            scene_gt = {}
        if os.path.exists(scene_camera_path):
            scene_camera = _BopWriterUtility.load_json(scene_camera_path, keys_to_int=True)
        else:
            scene_camera = {}

        # Определяем стартовый im_id
        if scene_gt:
            curr_im_id = max(scene_gt.keys()) + 1
        else:
            curr_im_id = 0

        num_new_frames = bpy.context.scene.frame_end - bpy.context.scene.frame_start
        if len(depths) != len(colors) or len(depths) != num_new_frames:
            raise Exception(
                "The amount of images in depths/colors does not match "
                "frame_end - frame_start."
            )

        # Проходим по кадрам Blender-сцены
        for frame_id in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):
            bpy.context.scene.frame_set(frame_id)

            # GT поз для объектов и камеры
            scene_gt[curr_im_id] = _BopWriterUtility.get_frame_gt(
                dataset_objects=dataset_objects,
                unit_scaling=annotation_scale,
                ignore_dist_thres=ignore_dist_thres,
            )
            scene_camera[curr_im_id] = _BopWriterUtility.get_frame_camera(
                save_world2cam=save_world2cam,
                depth_scale=depth_scale,
                unit_scaling=annotation_scale,
            )

            # Индекс в списке depths/colors (как в оригинальном BOP writer — используем frame_id)
            color_rgb = colors[frame_id]
            color_bgr = color_rgb.copy()
            color_bgr[..., :3] = color_bgr[..., :3][..., ::-1]

            if color_file_format.upper() == 'PNG':
                rgb_path = os.path.join(rgb_dir, f"{curr_im_id:06d}.png")
                cv2.imwrite(rgb_path, color_bgr)
            elif color_file_format.upper() == 'JPEG':
                rgb_path = os.path.join(rgb_dir, f"{curr_im_id:06d}.jpg")
                cv2.imwrite(
                    rgb_path,
                    color_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality],
                )
            else:
                raise ValueError(f"Unsupported color_file_format: {color_file_format}")

            depth = depths[frame_id]
            depth_mm = 1000.0 * depth  # [m] -> [mm]
            depth_mm_scaled = depth_mm / float(depth_scale)
            depth_path = os.path.join(depth_dir, f"{curr_im_id:06d}.png")
            _BopWriterUtility.save_depth(depth_path, depth_mm_scaled)

            curr_im_id += 1

        # Сохраняем обновлённые scene_gt / scene_camera
        _BopWriterUtility.save_json(scene_gt_path, scene_gt)
        _BopWriterUtility.save_json(scene_camera_path, scene_camera)

    # -------------------------------------------------------------------------
    # pyrender init + calc_gt_masks / calc_gt_info / calc_gt_coco
    # -------------------------------------------------------------------------
    @staticmethod
    def _pyrender_init(
            ren_width: int,
            ren_height: int,
            trimesh_objects: Dict[int, trimesh.Trimesh],
    ):
        """Initializes a worker process for calc_gt_masks and calc_gt_info."""
        # pylint: disable=import-outside-toplevel
        import pyrender
        # pylint: enable=import-outside-toplevel

        global renderer, renderer_large, dataset_objects

        dataset_objects = {}
        renderer = pyrender.OffscreenRenderer(
            viewport_width=ren_width,
            viewport_height=ren_height,
        )
        renderer_large = pyrender.OffscreenRenderer(
            viewport_width=ren_width * 3,
            viewport_height=ren_height * 3,
        )
        for key, mesh in trimesh_objects.items():
            material = pyrender.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                metallicFactor=0.2,
                roughnessFactor=0.8,
                doubleSided=True,
            )
            dataset_objects[key] = pyrender.Mesh.from_trimesh(mesh=mesh, material=material)

    @staticmethod
    def _calc_gt_masks_iteration(
            annotation_scale: float,
            K: np.ndarray,
            delta: float,
            dist_im: np.ndarray,
            chunk_dir: str,
            im_id: int,
            gt_data: Tuple[int, Dict[str, Any]],
    ):
        """One iteration of calc_gt_masks(), executed inside a worker process."""
        # pylint: disable=import-outside-toplevel
        import pyrender
        from bop_toolkit_lib import inout, misc, visibility
        # pylint: enable=import-outside-toplevel

        global renderer, dataset_objects

        gt_id, gt = gt_data

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        camera = pyrender.IntrinsicsCamera(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            znear=0.1,
            zfar=100000,
        )

        scene = pyrender.Scene()
        scene.add(camera)

        t = np.array(gt['cam_t_m2c'])
        t /= annotation_scale

        pose = bop_pose_to_pyrender_coordinate_system(
            cam_R_m2c=np.array(gt['cam_R_m2c']).reshape(3, 3),
            cam_t_m2c=t,
        )
        scene.add(dataset_objects[gt['obj_id']], pose=pose)

        _, depth_gt = renderer.render(scene=scene)
        dist_gt = misc.depth_im_to_dist_im_fast(depth_gt, K)

        mask = dist_gt > 0
        mask_visib = visibility.estimate_visib_mask_gt(
            dist_im, dist_gt, delta, visib_mode='bop19')

        mask_dir = os.path.join(chunk_dir, 'mask')
        mask_visib_dir = os.path.join(chunk_dir, 'mask_visib')
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(mask_visib_dir, exist_ok=True)

        mask_path = os.path.join(mask_dir, f'{im_id:06d}_{gt_id:06d}.png')
        mask_visib_path = os.path.join(mask_visib_dir, f'{im_id:06d}_{gt_id:06d}.png')

        inout.save_im(mask_path, 255 * mask.astype(np.uint8))
        inout.save_im(mask_visib_path, 255 * mask_visib.astype(np.uint8))

    @staticmethod
    def calc_gt_masks(
            pool: Pool,
            chunk_dirs: List[str],
            starting_frame_id: int = 0,
            annotation_scale: float = 1000.,
            delta: float = 0.015,
    ):
        """Calculates GT masks for all images in each chunk_dir (split_dir)."""
        # pylint: disable=import-outside-toplevel
        from bop_toolkit_lib import inout, misc
        # pylint: enable=import-outside-toplevel

        for dir_counter, chunk_dir in enumerate(chunk_dirs):
            gt_path = os.path.join(chunk_dir, 'scene_gt.json')
            cam_path = os.path.join(chunk_dir, 'scene_camera.json')
            scene_gt = _BopWriterUtility.load_json(gt_path, keys_to_int=True)
            scene_camera = _BopWriterUtility.load_json(cam_path, keys_to_int=True)

            im_ids = sorted(scene_gt.keys())
            if dir_counter == 0:
                im_ids = im_ids[starting_frame_id:]

            for im_counter, im_id in enumerate(im_ids):
                if im_counter % 100 == 0:
                    misc.log(f'Calculating GT masks - {chunk_dir}, {im_counter}')

                K = np.array(scene_camera[im_id]['cam_K']).reshape(3, 3)

                depth_path = os.path.join(chunk_dir, 'depth', f'{im_id:06d}.png')
                depth_im = inout.load_depth(depth_path)
                depth_im *= scene_camera[im_id]['depth_scale']  # to [mm]
                depth_im /= 1000.  # to [m]
                dist_im = misc.depth_im_to_dist_im_fast(depth_im, K)

                pool.map(
                    partial(
                        _BopWriterUtility._calc_gt_masks_iteration,
                        annotation_scale,
                        K,
                        delta,
                        dist_im,
                        chunk_dir,
                        im_id,
                    ),
                    enumerate(scene_gt[im_id]),
                )

    @staticmethod
    def _calc_gt_info_iteration(
            annotation_scale: float,
            ren_cy_offset: int,
            ren_cx_offset: int,
            im_height: int,
            im_width: int,
            K: np.ndarray,
            delta: float,
            depth: np.ndarray,
            gt: Dict[str, Any],
    ):
        """One iteration of calc_gt_info(), executed inside a worker process."""
        # pylint: disable=import-outside-toplevel
        import pyrender
        from bop_toolkit_lib import misc, visibility
        # pylint: enable=import-outside-toplevel

        global renderer_large, dataset_objects, renderer

        if renderer._renderer is not None:
            renderer._renderer.delete()
            renderer._renderer = None

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        im_size = (depth.shape[1], depth.shape[0])
        camera = pyrender.IntrinsicsCamera(
            fx=fx,
            fy=fy,
            cx=cx + ren_cx_offset,
            cy=cy + ren_cy_offset,
            znear=0.1,
            zfar=100000,
        )

        scene = pyrender.Scene()
        scene.add(camera)

        t = np.array(gt['cam_t_m2c'])
        t /= annotation_scale
        pose = bop_pose_to_pyrender_coordinate_system(
            cam_R_m2c=np.array(gt['cam_R_m2c']).reshape(3, 3),
            cam_t_m2c=t,
        )
        scene.add(dataset_objects[gt['obj_id']], pose=pose)

        _, depth_gt_large = renderer_large.render(scene=scene)
        depth_gt = depth_gt_large[
                   ren_cy_offset:(ren_cy_offset + im_height),
                   ren_cx_offset:(ren_cx_offset + im_width),
                   ]

        dist_gt = misc.depth_im_to_dist_im_fast(depth_gt, K)
        dist_im = misc.depth_im_to_dist_im_fast(depth, K)

        visib_gt = visibility.estimate_visib_mask_gt(
            dist_im, dist_gt, delta, visib_mode='bop19')

        obj_mask_gt_large = depth_gt_large > 0
        obj_mask_gt = dist_gt > 0

        px_count_all = int(np.sum(obj_mask_gt_large))
        px_count_valid = int(np.sum(dist_im[obj_mask_gt] > 0))
        px_count_visib = int(visib_gt.sum())

        visib_fract = float(px_count_visib / float(px_count_all)) if px_count_all > 0 else 0.0

        bbox = [-1, -1, -1, -1]
        if px_count_visib > 0:
            ys, xs = obj_mask_gt_large.nonzero()
            ys -= ren_cy_offset
            xs -= ren_cx_offset
            bbox = misc.calc_2d_bbox(xs, ys, im_size)

        bbox_visib = [-1, -1, -1, -1]
        if px_count_visib > 0:
            ys, xs = visib_gt.nonzero()
            bbox_visib = misc.calc_2d_bbox(xs, ys, im_size)

        return {
            'px_count_all': px_count_all,
            'px_count_valid': px_count_valid,
            'px_count_visib': px_count_visib,
            'visib_fract': visib_fract,
            'bbox_obj': [int(e) for e in bbox],
            'bbox_visib': [int(e) for e in bbox_visib],
        }

    @staticmethod
    def calc_gt_info(
            pool: Pool,
            chunk_dirs: List[str],
            starting_frame_id: int = 0,
            annotation_scale: float = 1000.,
            delta: float = 0.015,
    ):
        """Calculates GT info for all images in each chunk_dir (split_dir)."""
        # pylint: disable=import-outside-toplevel
        from bop_toolkit_lib import inout, misc
        # pylint: enable=import-outside-toplevel

        im_width = bpy.context.scene.render.resolution_x
        im_height = bpy.context.scene.render.resolution_y
        ren_cx_offset, ren_cy_offset = im_width, im_height

        for dir_counter, chunk_dir in enumerate(chunk_dirs):
            gt_path = os.path.join(chunk_dir, 'scene_gt.json')
            cam_path = os.path.join(chunk_dir, 'scene_camera.json')
            scene_gt = _BopWriterUtility.load_json(gt_path, keys_to_int=True)
            scene_camera = _BopWriterUtility.load_json(cam_path, keys_to_int=True)

            # Загружаем существующий gt_info при аппенде
            info_path = os.path.join(chunk_dir, 'scene_gt_info.json')
            if dir_counter == 0 and starting_frame_id > 0 and os.path.isfile(info_path):
                misc.log(f"Loading existing gt info from {info_path}")
                scene_gt_info = _BopWriterUtility.load_json(info_path, keys_to_int=True)
            else:
                scene_gt_info = {}

            im_ids = sorted(scene_gt.keys())
            if dir_counter == 0:
                im_ids = im_ids[starting_frame_id:]

            for im_counter, im_id in enumerate(im_ids):
                if im_counter % 100 == 0:
                    misc.log(f'Calculating GT info - {chunk_dir}, {im_counter}')

                depth_path = os.path.join(chunk_dir, 'depth', f'{im_id:06d}.png')
                assert os.path.isfile(depth_path)
                depth = inout.load_depth(depth_path)
                depth *= scene_camera[im_id]['depth_scale']
                depth /= 1000.  # to [m]

                K = np.array(scene_camera[im_id]['cam_K']).reshape(3, 3)

                scene_gt_info[im_id] = pool.map(
                    partial(
                        _BopWriterUtility._calc_gt_info_iteration,
                        annotation_scale,
                        ren_cy_offset,
                        ren_cx_offset,
                        im_height,
                        im_width,
                        K,
                        delta,
                        depth,
                    ),
                    scene_gt[im_id],
                )

            misc.ensure_dir(os.path.dirname(info_path))
            inout.save_json(info_path, scene_gt_info)

    @staticmethod
    def calc_gt_coco(
            chunk_dirs: List[str],
            dataset_objects: List[MeshObject],
            starting_frame_id: int = 0,
    ):
        """Calculates COCO annotations for each split_dir."""
        # pylint: disable=import-outside-toplevel
        from bop_toolkit_lib import inout, misc, pycoco_utils
        # pylint: enable=import-outside-toplevel

        for dir_counter, chunk_dir in enumerate(chunk_dirs):
            # dataset_name = имя папки над split_dir
            dataset_name = os.path.basename(os.path.dirname(chunk_dir))

            CATEGORIES = [{
                'id': obj.get_cp('category_id'),
                'name': str(obj.get_cp('category_id')),
                'supercategory': dataset_name,
            } for obj in dataset_objects]

            CATEGORIES = list({frozenset(item.items()): item for item in CATEGORIES}.values())

            INFO = {
                "description": dataset_name + f'_{os.path.basename(chunk_dir)}',
                "url": "https://github.com/thodan/bop_toolkit",
                "version": "0.1.0",
                "year": datetime.date.today().year,
                "contributor": "",
                "date_created": datetime.datetime.utcnow().isoformat(' '),
            }

            coco_path = os.path.join(chunk_dir, 'scene_gt_coco.json')

            if dir_counter == 0 and starting_frame_id > 0 and os.path.isfile(coco_path):
                misc.log(f"Loading existing COCO annotations from {coco_path}")
                coco_scene_output = _BopWriterUtility.load_json(coco_path)
                if coco_scene_output["annotations"]:
                    segmentation_id = coco_scene_output["annotations"][-1]['id'] + 1
                else:
                    segmentation_id = 1
            else:
                coco_scene_output = {
                    "info": INFO,
                    "licenses": [],
                    "categories": CATEGORIES,
                    "images": [],
                    "annotations": [],
                }
                segmentation_id = 1

            gt_path = os.path.join(chunk_dir, 'scene_gt.json')
            info_path = os.path.join(chunk_dir, 'scene_gt_info.json')
            scene_gt = _BopWriterUtility.load_json(gt_path, keys_to_int=True)
            scene_gt_info = inout.load_json(info_path, keys_to_int=True)

            misc.log(f'Calculating COCO annotations - {chunk_dir}')

            for scene_view, inst_list in scene_gt.items():
                im_id = int(scene_view)
                if dir_counter == 0 and im_id < starting_frame_id:
                    continue

                img_path_jpg = os.path.join(chunk_dir, 'rgb', f'{im_id:06d}.jpg')
                img_path_png = os.path.join(chunk_dir, 'rgb', f'{im_id:06d}.png')
                if os.path.isfile(img_path_jpg):
                    img_path = img_path_jpg
                else:
                    img_path = img_path_png

                relative_img_path = os.path.relpath(img_path, os.path.dirname(coco_path))
                im_size = (
                    bpy.context.scene.render.resolution_x,
                    bpy.context.scene.render.resolution_y,
                )
                image_info = pycoco_utils.create_image_info(im_id, relative_img_path, im_size)
                coco_scene_output["images"].append(image_info)
                gt_info = scene_gt_info[scene_view]

                for idx, inst in enumerate(inst_list):
                    category_info = inst['obj_id']
                    visibility = gt_info[idx]['visib_fract']
                    ignore_gt = visibility < 0.1

                    mask_visib_p = os.path.join(
                        chunk_dir,
                        'mask_visib',
                        f'{im_id:06d}_{idx:06d}.png',
                    )
                    mask_full_p = os.path.join(
                        chunk_dir,
                        'mask',
                        f'{im_id:06d}_{idx:06d}.png',
                    )

                    binary_inst_mask_visib = inout.load_depth(mask_visib_p).astype(bool)
                    if binary_inst_mask_visib.sum() < 1:
                        continue

                    binary_inst_mask_full = inout.load_depth(mask_full_p).astype(bool)
                    if binary_inst_mask_full.sum() < 1:
                        continue

                    bounding_box = pycoco_utils.bbox_from_binary_mask(binary_inst_mask_full)

                    annotation_info = pycoco_utils.create_annotation_info(
                        segmentation_id,
                        im_id,
                        category_info,
                        binary_inst_mask_visib,
                        bounding_box,
                        tolerance=2,
                        ignore=ignore_gt,
                    )

                    if annotation_info is not None:
                        coco_scene_output["annotations"].append(annotation_info)

                    segmentation_id += 1

            with open(coco_path, 'w', encoding='utf-8') as output_json_file:
                json.dump(coco_scene_output, output_json_file)
