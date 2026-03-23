from __future__ import annotations

import datetime
import os
from typing import Any, Dict, List, Optional, Union

import bpy
import cv2
import numpy as np
from blenderproc.python.writer.WriterUtility import _WriterUtility

from blendforge.blender_runtime.writer.RGBDCOCOWriter import (
    _ensure_dir,
    _load_json,
    _write_json_atomic,
    _save_rgb,
    _save_png_u8,
    _save_png_u16,
    area_from_binary_mask,
    bbox_from_binary_mask,
    binary_mask_to_rle,
    create_image_info,
    meters_to_depth_u16,
)


def write_coco_with_stereo_depth_annotations(
    output_dir: str,
    instance_segmaps: List[np.ndarray],
    instance_attribute_maps: List[List[dict]],
    colors: List[np.ndarray],
    depths_m: List[np.ndarray],
    depth_effective_m: List[np.ndarray],
    depth_random_m: List[np.ndarray],
    *,
    color_file_format: str = "JPEG",
    append_to_existing_output: bool = True,
    jpg_quality: int = 95,
    file_prefix: Optional[str] = None,
    indent: Optional[int] = 2,
    supercategory: str = "coco_annotations",
    ir_left_effective: Optional[List[np.ndarray]] = None,
    ir_right_effective: Optional[List[np.ndarray]] = None,
    ir_left_random: Optional[List[np.ndarray]] = None,
    ir_right_random: Optional[List[np.ndarray]] = None,
    depth_unit: str = "mm",
    depth_scale_mm: float = 1.0,
) -> None:
    writer = RGBDStereoCOCOWriter(
        output_dir=output_dir,
        color_file_format=color_file_format,
        jpg_quality=jpg_quality,
        append_to_existing_output=append_to_existing_output,
        indent=indent,
    )

    writer.write_camera_info_from_scene_if_missing(
        depth_unit=depth_unit,
        depth_scale_mm=float(depth_scale_mm),
        indent=(indent if indent is not None else 2),
    )

    writer.write_rgbd_coco(
        colors=colors,
        instance_segmaps=instance_segmaps,
        instance_attribute_maps=instance_attribute_maps,
        depth_m=depths_m,
        depth_effective_m=depth_effective_m,
        depth_random_m=depth_random_m,
        ir_left_effective=ir_left_effective,
        ir_right_effective=ir_right_effective,
        ir_left_random=ir_left_random,
        ir_right_random=ir_right_random,
        file_prefix=file_prefix,
        depth_unit=depth_unit,
        depth_scale_mm=float(depth_scale_mm),
        supercategory=supercategory,
    )


def _save_gray_u8(path: str, arr: np.ndarray) -> None:
    gray = np.asarray(arr)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    _ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, gray)


class RGBDStereoCOCOWriter:
    def __init__(
        self,
        output_dir: str,
        *,
        color_file_format: str = "JPEG",
        jpg_quality: int = 95,
        append_to_existing_output: bool = True,
        indent: Optional[Union[int, str]] = None,
    ) -> None:
        self.output_dir = output_dir
        self.color_file_format = color_file_format
        self.jpg_quality = int(jpg_quality)
        self.append = bool(append_to_existing_output)
        self.indent = indent

        self.dir_images = os.path.join(output_dir, "images")
        self.dir_depth = os.path.join(output_dir, "depth")
        self.dir_depth_effective = os.path.join(output_dir, "depth_effective")
        self.dir_depth_random = os.path.join(output_dir, "depth_random")
        self.dir_masks = os.path.join(output_dir, "masks")
        self.dir_ir_left_effective = os.path.join(output_dir, "ir_left_effective")
        self.dir_ir_right_effective = os.path.join(output_dir, "ir_right_effective")
        self.dir_ir_left_random = os.path.join(output_dir, "ir_left_random")
        self.dir_ir_right_random = os.path.join(output_dir, "ir_right_random")

        for path in [
            self.dir_images,
            self.dir_depth,
            self.dir_depth_effective,
            self.dir_depth_random,
            self.dir_masks,
        ]:
            _ensure_dir(path)

        self.coco_path = os.path.join(output_dir, "coco_annotations.json")
        self.camera_info_path = os.path.join(output_dir, "camera_info.json")

    def write_camera_info_from_scene_if_missing(
        self,
        *,
        depth_unit: str = "mm",
        depth_scale_mm: float = 1.0,
        indent: int = 2,
    ) -> None:
        if os.path.exists(self.camera_info_path):
            return
        if depth_scale_mm <= 0:
            raise ValueError("depth_scale_mm must be > 0")

        try:
            bpy.context.scene.frame_set(1)
        except Exception:
            pass

        cam_K = _WriterUtility.get_cam_attribute(bpy.context.scene.camera, "cam_K")
        width = int(bpy.context.scene.render.resolution_x)
        height = int(bpy.context.scene.render.resolution_y)

        _write_json_atomic(
            self.camera_info_path,
            {
                "width": width,
                "height": height,
                "fx": float(cam_K[0][0]),
                "fy": float(cam_K[1][1]),
                "cx": float(cam_K[0][2]),
                "cy": float(cam_K[1][2]),
                "K": cam_K,
                "depth_unit": str(depth_unit),
                "depth_scale": float(depth_scale_mm),
                "depth_scale_m": float(depth_scale_mm) / 1000.0,
            },
            indent=indent,
        )

    def write_rgbd_coco(
        self,
        *,
        colors: List[np.ndarray],
        instance_segmaps: List[np.ndarray],
        instance_attribute_maps: List[List[dict]],
        depth_m: List[np.ndarray],
        depth_effective_m: List[np.ndarray],
        depth_random_m: List[np.ndarray],
        ir_left_effective: Optional[List[np.ndarray]] = None,
        ir_right_effective: Optional[List[np.ndarray]] = None,
        ir_left_random: Optional[List[np.ndarray]] = None,
        ir_right_random: Optional[List[np.ndarray]] = None,
        file_prefix: Optional[str] = None,
        depth_unit: str = "mm",
        depth_scale_mm: float = 1.0,
        supercategory: str = "coco_annotations",
    ) -> None:
        if depth_scale_mm <= 0:
            raise ValueError("depth_scale_mm must be > 0")

        n = len(colors)
        expected_lengths = [
            len(instance_segmaps),
            len(instance_attribute_maps),
            len(depth_m),
            len(depth_effective_m),
            len(depth_random_m),
        ]
        if any(length != n for length in expected_lengths):
            raise ValueError("Input lists length mismatch")

        for name, seq in [
            ("ir_left_effective", ir_left_effective),
            ("ir_right_effective", ir_right_effective),
            ("ir_left_random", ir_left_random),
            ("ir_right_random", ir_right_random),
        ]:
            if seq is not None and len(seq) != n:
                raise ValueError(f"{name} length mismatch")

        if self.append and os.path.exists(self.coco_path):
            coco = _load_json(self.coco_path)
            image_offset = (max(img["id"] for img in coco["images"]) + 1) if coco.get("images") else 0
            ann_offset = max(ann["id"] for ann in coco["annotations"]) if coco.get("annotations") else 0
        else:
            coco = {
                "info": {
                    "description": supercategory,
                    "url": "",
                    "version": "1.0",
                    "year": datetime.datetime.utcnow().year,
                    "contributor": "Unknown",
                    "date_created": datetime.datetime.utcnow().isoformat(" "),
                },
                "licenses": [{
                    "id": 1,
                    "name": "Attribution-NonCommercial-ShareAlike License",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                }],
                "categories": [],
                "images": [],
                "annotations": [],
            }
            image_offset = 0
            ann_offset = 0

        visited_cat_ids = {int(c["id"]) for c in coco.get("categories", [])}
        next_ann_id = ann_offset + 1

        for frame_idx in range(n):
            img_id = frame_idx + image_offset
            stem = f"{img_id:06d}" if file_prefix is None else f"{file_prefix}{img_id:06d}"

            rgb_ext = "jpg" if self.color_file_format.upper() in ["JPG", "JPEG"] else "png"
            rgb_rel = f"images/{stem}.{rgb_ext}"
            _save_rgb(
                os.path.join(self.output_dir, rgb_rel),
                colors[frame_idx],
                fmt=self.color_file_format,
                jpg_quality=self.jpg_quality,
            )

            depth_rel = f"depth/{stem}.png"
            depth_effective_rel = f"depth_effective/{stem}.png"
            depth_random_rel = f"depth_random/{stem}.png"

            _save_png_u16(
                os.path.join(self.output_dir, depth_rel),
                meters_to_depth_u16(depth_m[frame_idx], depth_scale_mm),
            )
            _save_png_u16(
                os.path.join(self.output_dir, depth_effective_rel),
                meters_to_depth_u16(depth_effective_m[frame_idx], depth_scale_mm),
            )
            _save_png_u16(
                os.path.join(self.output_dir, depth_random_rel),
                meters_to_depth_u16(depth_random_m[frame_idx], depth_scale_mm),
            )

            ir_fields: Dict[str, str] = {}
            if ir_left_effective is not None:
                rel = f"ir_left_effective/{stem}.png"
                _save_gray_u8(os.path.join(self.output_dir, rel), ir_left_effective[frame_idx])
                ir_fields["ir_left_effective_file"] = rel
            if ir_right_effective is not None:
                rel = f"ir_right_effective/{stem}.png"
                _save_gray_u8(os.path.join(self.output_dir, rel), ir_right_effective[frame_idx])
                ir_fields["ir_right_effective_file"] = rel
            if ir_left_random is not None:
                rel = f"ir_left_random/{stem}.png"
                _save_gray_u8(os.path.join(self.output_dir, rel), ir_left_random[frame_idx])
                ir_fields["ir_left_random_file"] = rel
            if ir_right_random is not None:
                rel = f"ir_right_random/{stem}.png"
                _save_gray_u8(os.path.join(self.output_dir, rel), ir_right_random[frame_idx])
                ir_fields["ir_right_random_file"] = rel

            inst_seg = instance_segmaps[frame_idx]
            if inst_seg.ndim != 2:
                raise ValueError(f"Expected 2D instance_segmap, got {inst_seg.shape}")
            height, width = int(inst_seg.shape[0]), int(inst_seg.shape[1])

            inst_list = instance_attribute_maps[frame_idx]
            idx_to_inst: Dict[int, Dict[str, Any]] = {}

            for inst in inst_list:
                try:
                    category_id = int(inst.get("category_id", 0))
                except Exception:
                    category_id = 0
                if category_id == 0:
                    continue

                try:
                    idx = int(inst["idx"])
                except Exception:
                    continue

                idx_to_inst[idx] = inst
                if category_id not in visited_cat_ids:
                    coco["categories"].append({
                        "id": category_id,
                        "name": f"obj_{category_id:06d}",
                        "supercategory": inst.get("bop_dataset_name", inst.get("supercategory", supercategory)),
                    })
                    visited_cat_ids.add(category_id)

            instances = np.array(sorted(idx_to_inst.keys()), dtype=np.int32)
            if instances.size > 255:
                raise ValueError(f"Too many instances ({instances.size}) for uint8 packed map")

            packed = np.zeros((height, width), dtype=np.uint8)
            instid_to_index: Dict[int, int] = {}
            for local_idx, inst_id in enumerate(instances, start=1):
                iid = int(inst_id)
                instid_to_index[iid] = local_idx
                packed[inst_seg == iid] = np.uint8(local_idx)

            masks_rel = f"masks/{stem}.png"
            _save_png_u8(os.path.join(self.output_dir, masks_rel), packed)

            img_entry = create_image_info(img_id, rgb_rel, height, width)
            img_entry.update({
                "depth_file": depth_rel,
                "depth_inpainted_file": depth_rel,
                "depth_effective_file": depth_effective_rel,
                "depth_random_file": depth_random_rel,
                "depth_unit": str(depth_unit),
                "depth_scale": float(depth_scale_mm),
                "depth_scale_m": float(depth_scale_mm) / 1000.0,
                "instances_mask_file": masks_rel,
            })
            img_entry.update(ir_fields)
            coco["images"].append(img_entry)

            for inst_id in instances:
                iid = int(inst_id)
                inst = idx_to_inst.get(iid)
                if inst is None:
                    continue

                category_id = int(inst.get("category_id", 0))
                if category_id == 0:
                    continue

                instance_index = int(instid_to_index[iid])
                binary = (packed == instance_index).astype(np.uint8)
                area = area_from_binary_mask(binary)
                if area < 1:
                    continue

                ann: Dict[str, Any] = {
                    "id": int(next_ann_id),
                    "image_id": int(img_id),
                    "category_id": int(category_id),
                    "iscrowd": 0,
                    "area": int(area),
                    "bbox": bbox_from_binary_mask(binary),
                    "segmentation": binary_mask_to_rle(binary),
                    "width": int(width),
                    "height": int(height),
                    "instance_index": int(instance_index),
                }
                next_ann_id += 1

                if "fragment_id" in inst:
                    try:
                        ann["fragment_id"] = int(inst["fragment_id"])
                    except Exception:
                        ann["fragment_id"] = inst["fragment_id"]
                if "fracture_uid" in inst:
                    ann["fracture_uid"] = str(inst["fracture_uid"])
                elif "fracture_tag" in inst:
                    ann["fracture_uid"] = str(inst["fracture_tag"])
                if "fracture_seed" in inst:
                    try:
                        ann["fracture_seed"] = int(inst["fracture_seed"])
                    except Exception:
                        ann["fracture_seed"] = inst["fracture_seed"]
                if "fracture_method" in inst:
                    ann["fracture_method"] = str(inst["fracture_method"])

                coco["annotations"].append(ann)

        coco["categories"] = sorted(coco["categories"], key=lambda c: int(c["id"]))
        _write_json_atomic(self.coco_path, coco, indent=self.indent)
