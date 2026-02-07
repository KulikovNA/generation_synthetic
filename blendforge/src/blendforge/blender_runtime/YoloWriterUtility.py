import os
import json
from pathlib import Path
from typing import Optional, Dict, Union, List

import numpy as np
import cv2

from blenderproc.python.utility.LabelIdMapping import LabelIdMapping


def write_yolo_annotations(
    output_root: str,                          # <-- корень датасета (где images/, labels/)
    split: str,                                # <-- "train" | "val" | "test"
    instance_segmaps: List[np.ndarray],
    instance_attribute_maps: List[dict],
    colors: List[np.ndarray],
    color_file_format: str = "JPEG",
    supercategory: str = "coco_annotations",
    append_to_existing_output: bool = True,
    jpg_quality: int = 95,
    label_mapping: Optional[LabelIdMapping] = None,
    file_prefix: str = "",
    polygon_tolerance_px: float = 2.0,         # approxPolyDP epsilon в px (0 => без упрощения)
    min_area_px: int = 10,
    one_contour_per_instance: bool = True,
    write_ultralytics_yaml: bool = True,
):
    """
    Пишет:
      output_root/images/<split>/<prefix><id>.jpg|png
      output_root/labels/<split>/<prefix><id>.txt

    Формат строки Ultralytics YOLO-seg:
      <cls> x1 y1 x2 y2 ... (все координаты нормированы [0..1])
    """

    split = str(split)
    output_root = str(output_root)

    # ---- dirs (как у тебя в merged_yolo_dataset) ----
    images_dir = os.path.join(output_root, "images", split)
    labels_dir = os.path.join(output_root, "labels", split)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    ext = "png" if color_file_format.upper() == "PNG" else "jpg"

    # ---- image_offset (аналог COCO) ----
    image_offset = _YoloWriterUtility.compute_image_offset(images_dir, ext, file_prefix) \
        if append_to_existing_output else 0

    # ---- categories + per-frame instance->category map (как в COCO utility) ----
    categories, instance_2_category_maps = _YoloWriterUtility.collect_categories_and_instance_maps(
        instance_attribute_maps=instance_attribute_maps,
        supercategory=supercategory,
        label_mapping=label_mapping
    )

    # ---- стабильный mapping category_id -> yolo_id в корне датасета ----
    cat2yolo, names = _YoloWriterUtility.load_or_create_class_map(
        dataset_root=output_root,
        categories=categories
    )

    # ---- sanity ----
    if not (len(colors) == len(instance_segmaps) == len(instance_2_category_maps)):
        raise RuntimeError(
            f"Length mismatch: colors={len(colors)}, segmaps={len(instance_segmaps)}, attrmaps={len(instance_2_category_maps)}"
        )

    # ---- write frames ----
    for i in range(len(colors)):
        stem = f"{file_prefix}{i + image_offset:06d}"

        # image
        rgb = colors[i]
        bgr = rgb.copy()
        bgr[..., :3] = bgr[..., :3][..., ::-1]
        if color_file_format.upper() == "PNG":
            cv2.imwrite(os.path.join(images_dir, f"{stem}.png"), bgr)
        elif color_file_format.upper() == "JPEG":
            cv2.imwrite(os.path.join(images_dir, f"{stem}.jpg"), bgr,
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
        else:
            raise RuntimeError(f'Unknown color_file_format={color_file_format}. Try "PNG" or "JPEG"')

        # labels
        inst_segmap = instance_segmaps[i]
        h, w = inst_segmap.shape[:2]
        inst2cat = instance_2_category_maps[i]

        instances = np.unique(inst_segmap)
        instances = instances[instances != 0]

        lines: List[str] = []

        for inst_id in instances:
            inst_id = int(inst_id)
            if inst_id not in inst2cat:
                continue

            cat_id = int(inst2cat[inst_id])
            if cat_id not in cat2yolo:
                continue

            yolo_cls = int(cat2yolo[cat_id])

            mask = (inst_segmap == inst_id).astype(np.uint8)
            if int(mask.sum()) < int(min_area_px):
                continue

            polys = _YoloWriterUtility.mask_to_yolo_polygons(
                mask=mask,
                w=w, h=h,
                tolerance_px=polygon_tolerance_px,
                one_contour=one_contour_per_instance
            )
            if not polys:
                continue

            for poly in polys:
                coords = " ".join(f"{v:.6f}" for v in poly)
                lines.append(f"{yolo_cls} {coords}")

        with open(os.path.join(labels_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    # ---- write data.yaml/classes.txt once in root ----
    if write_ultralytics_yaml:
        _YoloWriterUtility.write_ultralytics_files(output_root=output_root, names=names)


class _YoloWriterUtility:
    @staticmethod
    def compute_image_offset(images_dir: str, ext: str, file_prefix: str) -> int:
        max_id = -1
        for p in Path(images_dir).glob(f"{file_prefix}*.{ext}"):
            stem = p.stem
            if not stem.startswith(file_prefix):
                continue
            num = stem[len(file_prefix):]
            if num.isdigit():
                max_id = max(max_id, int(num))
        return max_id + 1

    @staticmethod
    def collect_categories_and_instance_maps(instance_attribute_maps: List[dict],
                                            supercategory: str,
                                            label_mapping: Optional[LabelIdMapping]):
        categories = []
        visited = set()
        instance_2_category_maps = []

        for inst_attribute_map in instance_attribute_maps:
            inst2cat: Dict[int, int] = {}
            for inst in inst_attribute_map:
                if int(inst.get("category_id", 0)) == 0:
                    continue

                inst_supercategory = "coco_annotations"
                if "bop_dataset_name" in inst:
                    inst_supercategory = inst["bop_dataset_name"]
                elif "supercategory" in inst:
                    inst_supercategory = inst["supercategory"]

                if supercategory in [inst_supercategory, "coco_annotations"]:
                    cat_id = int(inst["category_id"])
                    if cat_id not in visited:
                        cat = {"id": cat_id, "supercategory": inst_supercategory}
                        if label_mapping is not None:
                            cat["name"] = label_mapping.label_from_id(cat_id)
                        elif "name" in inst:
                            cat["name"] = inst["name"]
                        else:
                            cat["name"] = str(cat_id)
                        categories.append(cat)
                        visited.add(cat_id)

                    inst2cat[int(inst["idx"])] = cat_id
            instance_2_category_maps.append(inst2cat)

        categories = sorted(categories, key=lambda x: int(x["id"]))
        return categories, instance_2_category_maps

    @staticmethod
    def load_or_create_class_map(dataset_root: str, categories):
        map_path = Path(dataset_root) / "yolo_classes.json"
        if map_path.exists():
            data = json.loads(map_path.read_text(encoding="utf-8"))
            cat2yolo = {int(k): int(v["yolo_id"]) for k, v in data["by_category_id"].items()}
            yolo2name = {int(v["yolo_id"]): str(v["name"]) for v in data["by_category_id"].values()}
        else:
            cat2yolo, yolo2name = {}, {}

        used_names = set(yolo2name.values())
        next_id = (max(yolo2name.keys()) + 1) if yolo2name else 0

        for c in categories:
            cat_id = int(c["id"])
            raw_name = str(c.get("name", "")).strip()

            # ---- ВОТ КЛЮЧЕВОЕ ----
            if (not raw_name) or (raw_name.lower() == "obj"):
                raw_name = f"obj_{cat_id:06d}"

            # если имя уже занято — делаем уникальным
            name = raw_name
            if name in used_names:
                name = f"{raw_name}_{cat_id:06d}"
            used_names.add(name)
            # ----------------------

            if cat_id not in cat2yolo:
                cat2yolo[cat_id] = next_id
                yolo2name[next_id] = name
                next_id += 1

        out = {
            "by_category_id": {
                str(cat_id): {"yolo_id": int(yolo_id), "name": yolo2name[int(yolo_id)]}
                for cat_id, yolo_id in sorted(cat2yolo.items(), key=lambda kv: kv[1])
            }
        }
        map_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

        names = [yolo2name[i] for i in range(len(yolo2name))]
        return cat2yolo, names


    @staticmethod
    def mask_to_yolo_polygons(mask: np.ndarray, w: int, h: int,
                             tolerance_px: float, one_contour: bool) -> List[List[float]]:
        m = (mask.astype(np.uint8) * 255)
        res = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = res[0] if len(res) == 2 else res[1]
        if not contours:
            return []

        if one_contour:
            contours = [max(contours, key=cv2.contourArea)]

        polys = []
        for cnt in contours:
            if cnt is None or len(cnt) < 3:
                continue

            if tolerance_px and tolerance_px > 0:
                cnt = cv2.approxPolyDP(cnt, float(tolerance_px), True)

            pts = cnt.reshape(-1, 2)
            if pts.shape[0] < 3:
                continue

            xs = np.clip(pts[:, 0].astype(np.float32) / float(w), 0.0, 1.0)
            ys = np.clip(pts[:, 1].astype(np.float32) / float(h), 0.0, 1.0)

            poly = []
            for x, y in zip(xs, ys):
                poly.extend([float(x), float(y)])

            if len(poly) >= 6:
                polys.append(poly)

        return polys

    @staticmethod
    def write_ultralytics_files(output_root: str, names: List[str]):
        output_root = Path(output_root)

        (output_root / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

        #lines = ["path: ."]

        lines = []
        for sp in ["train", "val", "test"]:
            if (output_root / "images" / sp).exists():
                lines.append(f"{sp}: images/{sp}")
        lines.append("names:")
        for i, n in enumerate(names):
            lines.append(f"  {i}: {json.dumps(n, ensure_ascii=False)}")

        (output_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
