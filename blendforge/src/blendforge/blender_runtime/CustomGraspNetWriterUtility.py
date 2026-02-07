"""
Custom GraspNet-style writer for BlenderProc.

Генерит структуру:

scenes/
  scene_0000/
    object_id_list.txt
    rs_wrt_kn.npy
    kinect/ или realsense/
      rgb/0000.png ...
      depth/0000.png ...
      label/0000.png ...
      camK.npy
      camera_poses.npy
      cam0_wrt_table.npy
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import cv2

from blenderproc.python.types.MeshObjectUtility import MeshObject  # пока не используем, но пусть будет


# ---------------------------------------------------------------------
# 1. Строим карты классов (category_id) из instance_segmaps + attribute_maps
# ---------------------------------------------------------------------
def build_category_segmaps(
    instance_segmaps: List[np.ndarray],
    instance_attribute_maps: List[List[dict]],
    category_key: str = "category_id",
) -> List[np.ndarray]:
    """
    Преобразует instance-карты и атрибуты в карты классов (category_id).

    Parameters
    ----------
    instance_segmaps : list[np.ndarray]
        Для каждого кадра карта инстансов [H, W], где значение = idx объекта (как в data["instance_segmaps"]).
    instance_attribute_maps : list[list[dict]]
        Для каждого кадра список словарей:
            {"idx": int, "category_id": int, "name": ..., ...}
        Это как data["instance_attribute_maps"] из BlenderProc.
    category_key : str
        Ключ, из которого брать класс — по умолчанию "category_id".

    Returns
    -------
    list[np.ndarray]
        Список карт классов [H, W], dtype uint16:
        0 — фон, >0 — category_id (у тебя это 1..9).
    """
    if len(instance_segmaps) != len(instance_attribute_maps):
        raise ValueError(
            f"len(instance_segmaps)={len(instance_segmaps)} "
            f"!= len(instance_attribute_maps)={len(instance_attribute_maps)}"
        )

    cat_segmaps: List[np.ndarray] = []

    print("\n[DEBUG][build_category_segmaps] === START ===")
    print(f"[DEBUG] Кол-во кадров: {len(instance_segmaps)}")

    for frame_id, (seg_inst, attr_list) in enumerate(zip(instance_segmaps, instance_attribute_maps)):
        seg_inst = np.asarray(seg_inst)
        if seg_inst.ndim != 2:
            raise ValueError(
                f"Ожидается 2D instance_segmap [H,W], а получено shape={seg_inst.shape}"
            )

        h, w = seg_inst.shape
        uniq_inst = np.unique(seg_inst)
        print(f"\n[DEBUG][frame {frame_id}] seg_inst shape={seg_inst.shape}, "
              f"уникальные instance-метки (первые 20): {uniq_inst[:20]}")

        print(f"[DEBUG][frame {frame_id}] len(attr_list)={len(attr_list)}")
        if frame_id == 0:
            # Печатаем атрибуты для первого кадра целиком (обычно их немного)
            for inst in attr_list:
                print(f"[DEBUG][frame {frame_id}] inst attr: {inst}")

        # карта классов для кадра
        cat_map = np.zeros_like(seg_inst, dtype=np.uint16)

        # attr_list — список инстансов для кадра
        for inst in attr_list:
            idx = int(inst["idx"])
            cat_id = int(inst.get(category_key, 0))

            if cat_id <= 0:
                print(f"[DEBUG][frame {frame_id}] inst idx={idx}: category_id={cat_id} -> ПРОПУСКАЕМ (фон/мусор)")
                continue

            # Проверим, вообще есть ли такой idx в маске
            if idx not in uniq_inst:
                print(f"[WARN][frame {frame_id}] inst idx={idx} НЕТ в seg_inst (uniq_inst) — метка в карте не встречается")
            else:
                print(f"[DEBUG][frame {frame_id}] inst idx={idx}, category_id={cat_id} -> "
                      f"проставляем в cat_map[seg_inst == {idx}]")

            cat_map[seg_inst == idx] = cat_id

        uniq_cat = np.unique(cat_map)
        print(f"[DEBUG][frame {frame_id}] Готовый cat_map: shape={cat_map.shape}, "
              f"уникальные category_id (первые 20): {uniq_cat[:20]}")

        # Небольшая сводка по числу пикселей на класс для первых кадров
        if frame_id < 3:
            for cid in uniq_cat:
                count = int((cat_map == cid).sum())
                print(f"[DEBUG][frame {frame_id}] category_id={cid}, пикселей={count}")

        cat_segmaps.append(cat_map)

    print("[DEBUG][build_category_segmaps] === END ===\n")
    return cat_segmaps


# ---------------------------------------------------------------------
# 2. GraspNet writer
# ---------------------------------------------------------------------
def write_graspnet_scene(
    output_dir: str,
    scene_id: Optional[int],
    colors: List[np.ndarray],
    depths: List[np.ndarray],
    cam_poses_world: np.ndarray,
    K: np.ndarray,
    *,
    # Вариант 1: напрямую подать карты классов (segmaps)
    segmaps: Optional[List[np.ndarray]] = None,
    # Вариант 2: подать instance_segmaps + attribute_maps, а мы сами посчитаем карты классов
    instance_segmaps: Optional[List[np.ndarray]] = None,
    instance_attribute_maps: Optional[List[List[dict]]] = None,
    camera_name: str = "kinect",
    table_T_world: Optional[np.ndarray] = None,
    label_offset: int = 0,
    append_to_existing_output: bool = True,
) -> int:
    """
    Пишет ОДНУ сцену в формате GraspNet (только часть scenes/*).

    Обязательные параметры:
    -----------------------
    output_dir : str
        Корень набора данных. Внутри будет создано: output_dir/scenes/scene_xxxx/...
    scene_id : Optional[int]
        Номер сцены (0-based). Если None и append_to_existing_output=True,
        будет взят max существующий + 1. Фактический id возвращается из функции.
    colors : list[np.ndarray]
        Список RGB(A) изображений, как в data["colors"]: [N,H,W,3/4], float в [0,1] или uint8.
    depths : list[np.ndarray]
        Список карт глубины в метрах: [N,H,W], float32.
    cam_poses_world : np.ndarray
        Пози/ориентации камер в мировых координатах: [N,4,4], cam->world (cam2world).
        Можно получить из bproc.camera.get_camera_poses().
    K : np.ndarray
        Матрица intrinsics 3x3: [[fx, 0, cx],[0, fy, cy],[0,0,1]].

    Сегментация (выбрать один вариант):
    -----------------------------------
    segmaps : list[np.ndarray], optional
        Уже готовые карты классов [N,H,W] с category_id (0 — фон, >0 — класс).
    instance_segmaps : list[np.ndarray], optional
        Карты инстансов [N,H,W], как data["instance_segmaps"].
    instance_attribute_maps : list[list[dict]], optional
        Атрибуты инстансов, как data["instance_attribute_maps"].
        Если segmaps не заданы, и заданы оба instance_*,
        будет построен segmaps = build_category_segmaps(...).

    Прочие параметры:
    -----------------
    camera_name : str
        Имя подкаталога камеры: "kinect" или "realsense".
    table_T_world : np.ndarray, optional
        4x4 матрица table->world. Если None, считаем table=world, берём identity.
    label_offset : int
        Сколько нужно вычесть из значения в маске, чтобы получить obj_id в object_id_list.txt.
        Если category_id == object_id (как у тебя 1..9), ставь 0.
    append_to_existing_output : bool
        Если scene_id=None и этот флаг True, доклеиваем сцену после существующих.

    Returns
    -------
    int
        Фактический scene_id, по которому записана сцена.
    """

    print("\n[DEBUG][write_graspnet_scene] === START ===")
    print(f"[DEBUG] camera_name={camera_name}, label_offset={label_offset}")
    print(f"[DEBUG] colors={len(colors)}, depths={len(depths)}, "
          f"segmaps={'None' if segmaps is None else len(segmaps)}, "
          f"instance_segmaps={'None' if instance_segmaps is None else len(instance_segmaps)}, "
          f"instance_attribute_maps={'None' if instance_attribute_maps is None else len(instance_attribute_maps)}")

    # --- готовим segmaps, если они не переданы явно ---
    if segmaps is None:
        if instance_segmaps is None or instance_attribute_maps is None:
            raise ValueError(
                "Нужно либо передать segmaps, либо (instance_segmaps + instance_attribute_maps)."
            )
        print("[DEBUG] segmaps=None, строим segmaps через build_category_segmaps(...)")
        segmaps = build_category_segmaps(instance_segmaps, instance_attribute_maps)
    else:
        print("[DEBUG] segmaps переданы напрямую, build_category_segmaps НЕ вызываем")

    # --- базовые проверки ---
    num_views = len(colors)
    if not (len(depths) == len(segmaps) == num_views):
        raise ValueError(
            f"colors: {len(colors)}, depths: {len(depths)}, segmaps: {len(segmaps)} — должны быть одинаковой длины"
        )

    cam_poses_world = np.asarray(cam_poses_world)
    if cam_poses_world.shape[0] != num_views:
        raise ValueError(
            f"cam_poses_world.shape[0] = {cam_poses_world.shape[0]}, а кадров {num_views}"
        )

    K = np.asarray(K, dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"Ожидается K формы (3,3), получено {K.shape}")

    if table_T_world is None:
        table_T_world = np.eye(4, dtype=np.float32)
    else:
        table_T_world = np.asarray(table_T_world, dtype=np.float32)
        if table_T_world.shape != (4, 4):
            raise ValueError(f"Ожидается table_T_world формы (4,4), получено {table_T_world.shape}")

    # --- глобальная сводка по segmaps ---
    all_labels_global = set()
    for i, seg in enumerate(segmaps):
        seg = np.asarray(seg)
        uniq = np.unique(seg)
        all_labels_global.update(int(v) for v in uniq)

        if i < 3:  # первые несколько кадров детальнее
            print(f"[DEBUG][segmap frame {i}] shape={seg.shape}, "
                  f"min={int(seg.min())}, max={int(seg.max())}, uniq (первые 20)={uniq[:20]}")

    print(f"[DEBUG] Глобальные уникальные значения по всем segmaps: "
          f"{sorted(all_labels_global) if all_labels_global else 'ПУСТО'}")

    # --- определяем scene_id, если он не задан ---
    scenes_root = Path(output_dir) / "scenes"
    scenes_root.mkdir(parents=True, exist_ok=True)

    if scene_id is None:
        if append_to_existing_output:
            existing = []
            for d in scenes_root.glob("scene_*"):
                if d.is_dir():
                    try:
                        sid = int(str(d.name).split("_")[-1])
                        existing.append(sid)
                    except ValueError:
                        pass
            scene_id = max(existing) + 1 if existing else 0
        else:
            if any(scenes_root.iterdir()):
                raise RuntimeError(
                    f"{scenes_root} не пустой, а scene_id=None и append_to_existing_output=False"
                )
            scene_id = 0

    print(f"[DEBUG] Итоговый scene_id={scene_id}")

    # --- пути ---
    scene_root = scenes_root / f"scene_{scene_id:04d}"
    cam_root = scene_root / camera_name

    rgb_dir = cam_root / "rgb"
    depth_dir = cam_root / "depth"
    label_dir = cam_root / "label"

    for d in (rgb_dir, depth_dir, label_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- camK.npy ---
    np.save(cam_root / "camK.npy", K.astype(np.float32))
    print(f"[DEBUG] camK.npy сохранён в {cam_root / 'camK.npy'}")

    # --- camera_poses.npy: позы всех кадров относительно первого ---
    T0 = cam_poses_world[0]              # cam_0 -> world
    T0_inv = np.linalg.inv(T0)
    cam_poses_rel = np.einsum("ij,njk->nik", T0_inv, cam_poses_world)
    np.save(cam_root / "camera_poses.npy", cam_poses_rel.astype(np.float32))
    print(f"[DEBUG] camera_poses.npy сохранён в {cam_root / 'camera_poses.npy'}")

    # --- cam0_wrt_table.npy ---
    T_table_inv = np.linalg.inv(table_T_world)
    cam0_wrt_table = T_table_inv @ T0
    np.save(cam_root / "cam0_wrt_table.npy", cam0_wrt_table.astype(np.float32))
    print(f"[DEBUG] cam0_wrt_table.npy сохранён в {cam_root / 'cam0_wrt_table.npy'}")

    # --- rs_wrt_kn.npy ---
    rs_wrt_kn = np.repeat(np.eye(4, dtype=np.float32)[None, ...], num_views, axis=0)
    np.save(scene_root / "rs_wrt_kn.npy", rs_wrt_kn)
    print(f"[DEBUG] rs_wrt_kn.npy сохранён в {scene_root / 'rs_wrt_kn.npy'}")

    # --- object_id_list.txt ---
    all_labels = set()
    for seg in segmaps:
        seg = np.asarray(seg)
        uniq = np.unique(seg)
        for v in uniq:
            v_int = int(v)
            if v_int > 0:
                all_labels.add(v_int)

    cat_ids = sorted(all_labels)  # это "label" в маске (= category_id)
    obj_ids = [cid - label_offset for cid in cat_ids]

    print(f"[DEBUG] cat_ids (из segmaps >0)={cat_ids}")
    print(f"[DEBUG] obj_ids (cat_id - label_offset)={obj_ids}")

    scene_root.mkdir(parents=True, exist_ok=True)
    object_id_list_path = scene_root / "object_id_list.txt"
    with open(object_id_list_path, "w") as f:
        for oid in obj_ids:
            f.write(f"{oid}\n")
    print(f"[DEBUG] object_id_list.txt сохранён в {object_id_list_path}")

    # Если список пустой — явно подсветим
    if not obj_ids:
        print("[WARN] object_id_list.txt ПУСТОЙ — значит во всех segmaps нет меток >0")

    # --- запись rgb / depth / label ---
    for i in range(num_views):
        # --- RGB ---
        rgb = np.asarray(colors[i])

        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0.0, 1.0) * 255.0
            rgb = rgb.astype(np.uint8)

        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]

        rgb_bgr = rgb[..., ::-1].copy()
        rgb_path = rgb_dir / f"{i:04d}.png"
        cv2.imwrite(str(rgb_path), rgb_bgr)

        # --- depth ---
        depth = np.asarray(depths[i], dtype=np.float32)
        depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
        depth_path = depth_dir / f"{i:04d}.png"
        cv2.imwrite(str(depth_path), depth_mm)

        # --- label ---
        label = np.asarray(segmaps[i]).astype(np.uint16)
        label_path = label_dir / f"{i:04d}.png"
        cv2.imwrite(str(label_path), label)

        if i < 3:
            uniq_label = np.unique(label)
            print(f"[DEBUG][label frame {i}] path={label_path}, "
                  f"min={int(label.min())}, max={int(label.max())}, uniq={uniq_label[:20]}")

    print("[DEBUG][write_graspnet_scene] === END ===\n")
    return scene_id
