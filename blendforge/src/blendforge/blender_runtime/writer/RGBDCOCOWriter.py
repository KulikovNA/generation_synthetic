# RGBDCOCOWriter.py
from __future__ import annotations

import datetime
import json
import os
from itertools import groupby
from typing import Any, Dict, List, Optional, Union

import bpy  # BlenderProc runtime
import cv2
import numpy as np
from blenderproc.python.writer.WriterUtility import _WriterUtility


# -----------------------------------------------------------------------------
# Public one-call API
# -----------------------------------------------------------------------------
def write_coco_with_depth_annotations(
    output_dir: str,
    instance_segmaps: List[np.ndarray],
    instance_attribute_maps: List[List[dict]],
    colors: List[np.ndarray],
    depths_clean_m: List[np.ndarray],
    *,
    # --- RGB options (like BlenderProc coco writer) ---
    color_file_format: str = "JPEG",
    append_to_existing_output: bool = True,
    jpg_quality: int = 95,
    file_prefix: Optional[str] = None,        # <-- CHANGED: Optional[str], can be None
    indent: Optional[int] = 2,
    supercategory: str = "coco_annotations",
    # --- Depth extras ---
    depths_raw_m: Optional[List[np.ndarray]] = None,          # if None -> raw=clean
    valid_depth_mask_u8: Optional[List[np.ndarray]] = None,   # if None -> computed from raw depth
    depth_unit: str = "mm",
    depth_scale_mm: float = 1.0,   # BOP-compatible: depth_mm = depth_u16 * depth_scale_mm
    z_min_m: float = 0.0,
    z_max_m: float = 10.0,
) -> None:
    """
    If file_prefix is None:
        images/000000.jpg
        depth_raw/000000.png
        depth_inpainted/000000.png
        masks/000000.png
        mask_valid/000000.png
    Else:
        images/{file_prefix}000000.jpg
        ...
    """
    writer = RGBDCOCOWriter(
        output_dir=output_dir,
        color_file_format=color_file_format,
        jpg_quality=jpg_quality,
        append_to_existing_output=append_to_existing_output,
        indent=indent,
        z_min_m=z_min_m,
        z_max_m=z_max_m,
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
        depth_clean_m=depths_clean_m,
        depth_raw_m=depths_raw_m,
        valid_mask_u8=valid_depth_mask_u8,
        file_prefix=file_prefix,                  # <-- now can pass None
        depth_unit=depth_unit,
        depth_scale_mm=float(depth_scale_mm),
        supercategory=supercategory,
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(path: str, obj: Any, indent: Optional[Union[int, str]] = None) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


def _as_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Accepts float [0..1] or uint8; returns uint8 RGB (3ch)."""
    if img.dtype == np.uint8:
        out = img
    else:
        out = np.clip(img, 0.0, 1.0)
        out = (out * 255.0 + 0.5).astype(np.uint8)
    if out.ndim == 3 and out.shape[2] == 4:
        out = out[:, :, :3]
    return out


def _save_rgb(path: str, rgb: np.ndarray, fmt: str = "JPEG", jpg_quality: int = 95) -> None:
    rgb8 = _as_uint8_rgb(rgb)
    bgr = rgb8[:, :, ::-1]
    _ensure_dir(os.path.dirname(path))
    if fmt.upper() == "PNG":
        cv2.imwrite(path, bgr)
    elif fmt.upper() in ["JPG", "JPEG"]:
        cv2.imwrite(path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
    else:
        raise ValueError(f"Unknown color format: {fmt}")


def _save_png_u16(path: str, arr: np.ndarray) -> None:
    if arr.dtype != np.uint16:
        raise ValueError(f"_save_png_u16 expects uint16, got {arr.dtype}")
    _ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, arr)


def _save_png_u8(path: str, arr: np.ndarray) -> None:
    if arr.dtype != np.uint8:
        raise ValueError(f"_save_png_u8 expects uint8, got {arr.dtype}")
    _ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, arr)


def meters_to_depth_u16(depth_m: np.ndarray, depth_scale_mm: float = 1.0) -> np.ndarray:
    """
    BOP-compatible:
      depth_u16 = round((depth_m * 1000) / depth_scale_mm)
      0 means invalid
    """
    if depth_scale_mm <= 0:
        raise ValueError("depth_scale_mm must be > 0")

    d = depth_m.astype(np.float32, copy=False)
    d = np.where(np.isfinite(d), d, 0.0)
    d = np.where(d > 0.0, d, 0.0)

    depth_mm = d * 1000.0
    scaled = depth_mm / float(depth_scale_mm)
    scaled = np.clip(scaled, 0.0, 65535.0)
    return (scaled + 0.5).astype(np.uint16)


def depth_u16_to_meters(depth_u16: np.ndarray, depth_scale_mm: float = 1.0) -> np.ndarray:
    """meters = u16 * depth_scale_mm / 1000."""
    return depth_u16.astype(np.float32) * (float(depth_scale_mm) / 1000.0)


def binary_mask_to_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
    """Uncompressed COCO RLE with counts list, Fortran order (same style as BlenderProc)."""
    rle: Dict[str, Any] = {"counts": [], "size": list(binary_mask.shape)}
    counts: List[int] = rle["counts"]
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order="F"))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def bbox_from_binary_mask(binary_mask: np.ndarray) -> List[int]:
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h = int(rmax - rmin + 1)
    w = int(cmax - cmin + 1)
    return [int(cmin), int(rmin), w, h]


def area_from_binary_mask(binary_mask: np.ndarray) -> int:
    return int(binary_mask.sum().tolist())


def create_image_info(image_id: int, file_name: str, H: int, W: int) -> Dict[str, Any]:
    return {
        "id": int(image_id),
        "file_name": file_name,
        "width": int(W),
        "height": int(H),
        "date_captured": datetime.datetime.utcnow().isoformat(" "),
        "license": 1,
        "coco_url": "",
        "flickr_url": "",
    }


# -----------------------------------------------------------------------------
# Writer
# -----------------------------------------------------------------------------
class RGBDCOCOWriter:
    """
    Writes split/
      coco_annotations.json
      camera_info.json
      images/
      depth_raw/
      depth_inpainted/
      mask_valid/
      masks/              (packed instance map: pixel = instance_index, 0 background)
    """

    def __init__(
        self,
        output_dir: str,
        *,
        color_file_format: str = "JPEG",
        jpg_quality: int = 95,
        append_to_existing_output: bool = True,
        indent: Optional[Union[int, str]] = None,
        z_min_m: float = 0.0,
        z_max_m: float = 10.0,
    ) -> None:
        self.output_dir = output_dir
        self.color_file_format = color_file_format
        self.jpg_quality = int(jpg_quality)
        self.append = bool(append_to_existing_output)
        self.indent = indent
        self.z_min_m = float(z_min_m)
        self.z_max_m = float(z_max_m)

        self.dir_images = os.path.join(output_dir, "images")
        self.dir_depth_raw = os.path.join(output_dir, "depth_raw")
        self.dir_depth_clean = os.path.join(output_dir, "depth_inpainted")
        self.dir_valid = os.path.join(output_dir, "mask_valid")
        self.dir_masks = os.path.join(output_dir, "masks")

        for d in [self.dir_images, self.dir_depth_raw, self.dir_depth_clean, self.dir_valid, self.dir_masks]:
            _ensure_dir(d)

        self.coco_path = os.path.join(output_dir, "coco_annotations.json")
        self.camera_info_path = os.path.join(output_dir, "camera_info.json")

    # -------------------------------------------------------------------------
    # Camera info (BOP-like, from scene)
    # -------------------------------------------------------------------------
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
        fx = float(cam_K[0][0])
        fy = float(cam_K[1][1])
        cx = float(cam_K[0][2])
        cy = float(cam_K[1][2])

        width = int(bpy.context.scene.render.resolution_x)
        height = int(bpy.context.scene.render.resolution_y)

        obj = {
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "K": cam_K,
            "depth_unit": str(depth_unit),
            "depth_scale": float(depth_scale_mm),
            "depth_scale_m": float(depth_scale_mm) / 1000.0,
        }

        _write_json_atomic(self.camera_info_path, obj, indent=indent)

    # -------------------------------------------------------------------------
    # Main writer
    # -------------------------------------------------------------------------
    def write_rgbd_coco(
        self,
        *,
        colors: List[np.ndarray],
        instance_segmaps: List[np.ndarray],
        instance_attribute_maps: List[List[dict]],
        depth_clean_m: List[np.ndarray],
        depth_raw_m: Optional[List[np.ndarray]] = None,
        valid_mask_u8: Optional[List[np.ndarray]] = None,
        file_prefix: Optional[str] = "rgb_",     # <-- CHANGED: Optional[str]
        depth_unit: str = "mm",
        depth_scale_mm: float = 1.0,
        supercategory: str = "coco_annotations",
    ) -> None:
        if depth_scale_mm <= 0:
            raise ValueError("depth_scale_mm must be > 0")

        if self.append and os.path.exists(self.coco_path):
            coco = _load_json(self.coco_path)
            image_offset = (max(img["id"] for img in coco["images"]) + 1) if coco.get("images") else 0
            ann_offset = (max(ann["id"] for ann in coco["annotations"])) if coco.get("annotations") else 0
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
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }],
                "categories": [],
                "images": [],
                "annotations": [],
            }
            image_offset = 0
            ann_offset = 0

        visited_cat_ids = {int(c["id"]) for c in coco.get("categories", [])}

        if not (len(colors) == len(instance_segmaps) == len(instance_attribute_maps) == len(depth_clean_m)):
            raise ValueError("Input lists length mismatch")
        if depth_raw_m is not None and len(depth_raw_m) != len(colors):
            raise ValueError("depth_raw_m length mismatch")
        if valid_mask_u8 is not None and len(valid_mask_u8) != len(colors):
            raise ValueError("valid_mask_u8 length mismatch")

        next_ann_id = ann_offset + 1

        for frame_idx in range(len(colors)):
            img_id = frame_idx + image_offset

            # ------------------ STEM (new rule) ------------------
            # if file_prefix is None -> "000000"
            # else -> "{file_prefix}000000"
            stem = f"{img_id:06d}" if file_prefix is None else f"{file_prefix}{img_id:06d}"
            # -----------------------------------------------------

            # ---------- RGB ----------
            rgb_ext = "jpg" if self.color_file_format.upper() in ["JPG", "JPEG"] else "png"
            rgb_rel = f"images/{stem}.{rgb_ext}"
            _save_rgb(
                os.path.join(self.output_dir, rgb_rel),
                colors[frame_idx],
                fmt=self.color_file_format,
                jpg_quality=self.jpg_quality,
            )

            # ---------- Depth clean ----------
            d_clean_u16 = meters_to_depth_u16(depth_clean_m[frame_idx], depth_scale_mm)
            d_clean_rel = f"depth_inpainted/{stem}.png"
            _save_png_u16(os.path.join(self.output_dir, d_clean_rel), d_clean_u16)

            # ---------- Depth raw ----------
            if depth_raw_m is None or depth_raw_m[frame_idx] is None:
                d_raw_u16 = d_clean_u16
            else:
                d_raw_u16 = meters_to_depth_u16(depth_raw_m[frame_idx], depth_scale_mm)
            d_raw_rel = f"depth_raw/{stem}.png"
            _save_png_u16(os.path.join(self.output_dir, d_raw_rel), d_raw_u16)

            # ---------- Valid depth mask ----------
            if valid_mask_u8 is None or valid_mask_u8[frame_idx] is None:
                z_m = depth_u16_to_meters(d_raw_u16, depth_scale_mm)
                valid = (d_raw_u16 > 0) & (z_m >= self.z_min_m) & (z_m <= self.z_max_m)
                vm = (valid.astype(np.uint8) * 255)
            else:
                vm = valid_mask_u8[frame_idx].astype(np.uint8)
                if vm.max() <= 1:
                    vm = vm * 255
            vm_rel = f"mask_valid/{stem}.png"
            _save_png_u8(os.path.join(self.output_dir, vm_rel), vm)

            # ---------- Instance segmap ----------
            inst_seg = instance_segmaps[frame_idx]
            if inst_seg.ndim != 2:
                raise ValueError(f"Expected 2D instance_segmap, got {inst_seg.shape}")
            H, W = int(inst_seg.shape[0]), int(inst_seg.shape[1])

            inst_list = instance_attribute_maps[frame_idx]
            idx_to_inst: Dict[int, Dict[str, Any]] = {}

            for inst in inst_list:
                try:
                    cid = int(inst.get("category_id", 0))
                except Exception:
                    cid = 0
                if cid == 0:
                    continue

                try:
                    idx = int(inst["idx"])
                except Exception:
                    continue

                idx_to_inst[idx] = inst

                if cid not in visited_cat_ids:
                    cname = f"obj_{cid:06d}"
                    coco["categories"].append({
                        "id": cid,
                        "name": cname,
                        "supercategory": inst.get("bop_dataset_name", inst.get("supercategory", supercategory)),
                    })
                    visited_cat_ids.add(cid)

            # ---------- Packed instance map ----------
            instances = np.array(sorted(idx_to_inst.keys()), dtype=np.int32)

            if instances.size > 255:
                raise ValueError(f"Too many instances ({instances.size}) for uint8 packed map")

            packed = np.zeros((H, W), dtype=np.uint8)
            instid_to_index: Dict[int, int] = {}
            for k, inst_id in enumerate(instances, start=1):
                iid = int(inst_id)
                instid_to_index[iid] = k
                packed[inst_seg == iid] = np.uint8(k)

            masks_rel = f"masks/{stem}.png"
            _save_png_u8(os.path.join(self.output_dir, masks_rel), packed)

            # ---------- COCO image entry ----------
            img_entry = create_image_info(img_id, rgb_rel, H, W)
            img_entry.update({
                "depth_raw_file": d_raw_rel,
                "depth_inpainted_file": d_clean_rel,
                "depth_unit": str(depth_unit),
                "depth_scale": float(depth_scale_mm),
                "depth_scale_m": float(depth_scale_mm) / 1000.0,
                "instances_mask_file": masks_rel,
                "valid_depth_mask_file": vm_rel,
            })
            coco["images"].append(img_entry)

            # ---------- COCO annotations ----------
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

                bbox = bbox_from_binary_mask(binary)
                segm = binary_mask_to_rle(binary)

                ann: Dict[str, Any] = {
                    "id": int(next_ann_id),
                    "image_id": int(img_id),
                    "category_id": int(category_id),
                    "iscrowd": 0,
                    "area": int(area),
                    "bbox": bbox,
                    "segmentation": segm,
                    "width": int(W),
                    "height": int(H),
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
