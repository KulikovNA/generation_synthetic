from __future__ import annotations

import os
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import bpy
import cv2
import numpy as np

from blenderproc.python.types.LinkUtility import Link
from blenderproc.python.types.MeshObjectUtility import MeshObject, get_all_mesh_objects
from blenderproc.python.utility.SetupUtility import SetupUtility

from blendforge.blender_runtime.writer.CustomBopShapeWriterUtility import _BopWriterUtility


def write_bop_multidepth(
    output_dir: str,
    *,
    target_objects: Optional[List[MeshObject]] = None,
    colors: Sequence[np.ndarray],
    depth_gt: Sequence[np.ndarray],
    depth_effective: Sequence[np.ndarray],
    depth_random: Sequence[np.ndarray],
    ir_left_effective: Optional[Sequence[np.ndarray]] = None,
    ir_right_effective: Optional[Sequence[np.ndarray]] = None,
    ir_left_random: Optional[Sequence[np.ndarray]] = None,
    ir_right_random: Optional[Sequence[np.ndarray]] = None,
    color_file_format: str = "JPEG",
    dataset: str = "",
    split: str = "train_pbr",
    append_to_existing_output: bool = True,
    depth_scale: float = 1.0,
    jpg_quality: int = 95,
    save_world2cam: bool = True,
    ignore_dist_thres: float = 100.0,
    m2mm: Optional[bool] = None,
    annotation_unit: str = "mm",
    frames_per_chunk: int = 1000,
    calc_mask_info_coco: bool = True,
    delta: float = 0.015,
    num_worker: Optional[int] = None,
) -> Dict[str, Any]:
    """Write a chunked BOP dataset with extra stereo depth/IR outputs.

    Output structure:

    dataset_dir/
      camera.json
      split/
        000000/
          depth/
          depth_effective/
          depth_random/
          ir_left_effective/
          ir_left_random/
          ir_right_effective/
          ir_right_random/
          mask/
          mask_visib/
          rgb/
          scene_camera.json
          scene_gt.json
          scene_gt_info.json
          scene_gt_coco.json
        000001/
          ...
    """

    dataset_dir = os.path.join(output_dir, dataset) if dataset else output_dir
    split_dir = os.path.join(dataset_dir, split)
    camera_path = os.path.join(dataset_dir, "camera.json")

    os.makedirs(dataset_dir, exist_ok=True)
    if os.path.exists(split_dir):
        if not append_to_existing_output:
            raise FileExistsError(f"The split folder already exists: {split_dir}")
    else:
        os.makedirs(split_dir)

    dataset_objects = _resolve_dataset_objects(target_objects=target_objects, dataset=dataset)
    if not dataset_objects:
        raise RuntimeError(
            f"The scene does not contain any visible object (or from the specified dataset '{dataset}')."
        )

    _BopWriterUtility.write_camera(camera_path, depth_scale=depth_scale)

    assert annotation_unit in ["m", "dm", "cm", "mm"], (
        f"Invalid annotation unit: `{annotation_unit}`. Supported are 'm', 'dm', 'cm', 'mm'"
    )
    annotation_scale = {"m": 1.0, "dm": 10.0, "cm": 100.0, "mm": 1000.0}[annotation_unit]
    if m2mm is not None:
        warnings.warn("`m2mm` is deprecated, use `annotation_unit='mm'` instead!")
        annotation_scale = 1000.0

    write_result = _write_frames_chunked_multidepth(
        split_dir=split_dir,
        dataset_objects=dataset_objects,
        colors=colors,
        depth_gt=depth_gt,
        depth_effective=depth_effective,
        depth_random=depth_random,
        ir_left_effective=ir_left_effective,
        ir_right_effective=ir_right_effective,
        ir_left_random=ir_left_random,
        ir_right_random=ir_right_random,
        color_file_format=color_file_format,
        depth_scale=depth_scale,
        annotation_scale=annotation_scale,
        ignore_dist_thres=ignore_dist_thres,
        save_world2cam=save_world2cam,
        jpg_quality=jpg_quality,
        frames_per_chunk=int(frames_per_chunk),
    )

    if calc_mask_info_coco and write_result["chunk_dirs"]:
        _calc_chunk_annotations(
            dataset_objects=dataset_objects,
            chunk_start_ids=write_result["chunk_start_ids"],
            annotation_scale=annotation_scale,
            delta=delta,
            num_worker=num_worker,
        )

    return {
        "dataset_dir": dataset_dir,
        "split_dir": split_dir,
        "image_ids": write_result["image_ids"],
        "chunk_ids": write_result["chunk_ids"],
        "chunk_dirs": write_result["chunk_dirs"],
    }


def _resolve_dataset_objects(
    *,
    target_objects: Optional[List[MeshObject]],
    dataset: str,
) -> List[MeshObject]:
    if target_objects is not None:
        dataset_objects = target_objects
        for obj in dataset_objects:
            if obj.is_hidden():
                print(
                    f"WARNING: The given object {obj.get_name()} is hidden. "
                    "The writer will still add BOP annotations for it."
                )
        return dataset_objects

    if dataset:
        dataset_objects = []
        for obj in get_all_mesh_objects():
            if "bop_dataset_name" in obj.blender_obj and not obj.is_hidden():
                if obj.blender_obj["bop_dataset_name"] == dataset:
                    dataset_objects.append(obj)
        return dataset_objects

    return [obj for obj in get_all_mesh_objects() if not obj.is_hidden()]


def _save_rgb(path: str, color_rgb: np.ndarray, *, color_file_format: str, jpg_quality: int) -> None:
    color_bgr = np.asarray(color_rgb).copy()
    color_bgr[..., :3] = color_bgr[..., :3][..., ::-1]

    fmt = color_file_format.upper()
    if fmt == "PNG":
        cv2.imwrite(path, color_bgr)
    elif fmt == "JPEG":
        cv2.imwrite(path, color_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
    else:
        raise ValueError(f"Unsupported color_file_format: {color_file_format}")


def _save_gray_u8(path: str, img: np.ndarray) -> None:
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    cv2.imwrite(path, arr)


def _validate_optional_sequence(name: str, seq: Optional[Sequence[np.ndarray]], n: int) -> None:
    if seq is not None and len(seq) != n:
        raise ValueError(f"{name} length must match number of frames ({n}), got {len(seq)}.")


def _sorted_chunk_ids(split_dir: str) -> List[int]:
    out: List[int] = []
    p = Path(split_dir)
    if not p.exists():
        return out
    for child in p.iterdir():
        if child.is_dir() and child.name.isdigit():
            out.append(int(child.name))
    return sorted(out)


def _resolve_current_chunk(split_dir: str, frames_per_chunk: int) -> tuple[int, int, Dict[int, Any], Dict[int, Any]]:
    chunk_ids = _sorted_chunk_ids(split_dir)
    if not chunk_ids:
        return 0, 0, {}, {}

    curr_chunk_id = int(chunk_ids[-1])
    chunk_dir = os.path.join(split_dir, f"{curr_chunk_id:06d}")
    scene_gt_path = os.path.join(chunk_dir, "scene_gt.json")
    scene_camera_path = os.path.join(chunk_dir, "scene_camera.json")
    if not os.path.exists(scene_gt_path):
        return curr_chunk_id, 0, {}, {}

    scene_gt = _BopWriterUtility.load_json(scene_gt_path, keys_to_int=True)
    scene_camera = _BopWriterUtility.load_json(scene_camera_path, keys_to_int=True) if os.path.exists(scene_camera_path) else {}
    curr_im_id = (max(scene_gt.keys()) + 1) if scene_gt else 0
    if curr_im_id >= frames_per_chunk:
        return curr_chunk_id + 1, 0, {}, {}
    return curr_chunk_id, curr_im_id, scene_gt, scene_camera


def _ensure_chunk_dirs(chunk_dir: str, save_ir_pairs: bool) -> None:
    required_dirs = [
        "rgb",
        "depth",
        "depth_effective",
        "depth_random",
    ]
    if save_ir_pairs:
        required_dirs.extend(
            [
                "ir_left_effective",
                "ir_left_random",
                "ir_right_effective",
                "ir_right_random",
            ]
        )
    for name in required_dirs:
        os.makedirs(os.path.join(chunk_dir, name), exist_ok=True)


def _frame_ids_for_payload(n: int) -> List[int]:
    frame_ids = list(range(bpy.context.scene.frame_start, bpy.context.scene.frame_end))
    if len(frame_ids) == n:
        return frame_ids
    if n == 1:
        return [int(bpy.context.scene.frame_current)]
    raise ValueError("The amount of images in colors/depths does not match frame_end - frame_start.")


def _write_frames_chunked_multidepth(
    *,
    split_dir: str,
    dataset_objects: List[MeshObject],
    colors: Sequence[np.ndarray],
    depth_gt: Sequence[np.ndarray],
    depth_effective: Sequence[np.ndarray],
    depth_random: Sequence[np.ndarray],
    ir_left_effective: Optional[Sequence[np.ndarray]],
    ir_right_effective: Optional[Sequence[np.ndarray]],
    ir_left_random: Optional[Sequence[np.ndarray]],
    ir_right_random: Optional[Sequence[np.ndarray]],
    color_file_format: str,
    depth_scale: float,
    annotation_scale: float,
    ignore_dist_thres: float,
    save_world2cam: bool,
    jpg_quality: int,
    frames_per_chunk: int,
) -> Dict[str, Any]:
    n = len(colors)
    if len(depth_gt) != n or len(depth_effective) != n or len(depth_random) != n:
        raise ValueError("colors, depth_gt, depth_effective and depth_random must have the same length.")

    _validate_optional_sequence("ir_left_effective", ir_left_effective, n)
    _validate_optional_sequence("ir_right_effective", ir_right_effective, n)
    _validate_optional_sequence("ir_left_random", ir_left_random, n)
    _validate_optional_sequence("ir_right_random", ir_right_random, n)

    frame_ids = _frame_ids_for_payload(n)
    save_ir_pairs = any(seq is not None for seq in [ir_left_effective, ir_right_effective, ir_left_random, ir_right_random])

    curr_chunk_id, curr_im_id, chunk_gt, chunk_camera = _resolve_current_chunk(split_dir, frames_per_chunk)
    chunk_dir = os.path.join(split_dir, f"{curr_chunk_id:06d}")
    if curr_im_id == 0:
        chunk_gt = {}
        chunk_camera = {}

    chunk_start_ids: Dict[str, int] = {}
    image_ids: List[int] = []
    chunk_ids: List[int] = []
    chunk_dirs: List[str] = []

    for local_idx, frame_id in enumerate(frame_ids):
        if curr_im_id == 0:
            chunk_dir = os.path.join(split_dir, f"{curr_chunk_id:06d}")
            _ensure_chunk_dirs(chunk_dir, save_ir_pairs)
            chunk_gt = {}
            chunk_camera = {}

        if chunk_dir not in chunk_start_ids:
            chunk_start_ids[chunk_dir] = int(curr_im_id)
            chunk_dirs.append(chunk_dir)

        bpy.context.scene.frame_set(frame_id)

        chunk_gt[curr_im_id] = _BopWriterUtility.get_frame_gt(
            dataset_objects=dataset_objects,
            unit_scaling=annotation_scale,
            ignore_dist_thres=ignore_dist_thres,
        )
        chunk_camera[curr_im_id] = _BopWriterUtility.get_frame_camera(
            save_world2cam=save_world2cam,
            depth_scale=depth_scale,
            unit_scaling=annotation_scale,
        )

        rgb_ext = "png" if color_file_format.upper() == "PNG" else "jpg"
        _save_rgb(
            os.path.join(chunk_dir, "rgb", f"{curr_im_id:06d}.{rgb_ext}"),
            colors[local_idx],
            color_file_format=color_file_format,
            jpg_quality=jpg_quality,
        )

        depth_gt_mm_scaled = 1000.0 * np.asarray(depth_gt[local_idx], dtype=np.float32) / float(depth_scale)
        depth_effective_mm_scaled = (
            1000.0 * np.asarray(depth_effective[local_idx], dtype=np.float32) / float(depth_scale)
        )
        depth_random_mm_scaled = (
            1000.0 * np.asarray(depth_random[local_idx], dtype=np.float32) / float(depth_scale)
        )

        _BopWriterUtility.save_depth(os.path.join(chunk_dir, "depth", f"{curr_im_id:06d}.png"), depth_gt_mm_scaled)
        _BopWriterUtility.save_depth(
            os.path.join(chunk_dir, "depth_effective", f"{curr_im_id:06d}.png"),
            depth_effective_mm_scaled,
        )
        _BopWriterUtility.save_depth(
            os.path.join(chunk_dir, "depth_random", f"{curr_im_id:06d}.png"),
            depth_random_mm_scaled,
        )

        if ir_left_effective is not None:
            _save_gray_u8(
                os.path.join(chunk_dir, "ir_left_effective", f"{curr_im_id:06d}.png"),
                ir_left_effective[local_idx],
            )
        if ir_right_effective is not None:
            _save_gray_u8(
                os.path.join(chunk_dir, "ir_right_effective", f"{curr_im_id:06d}.png"),
                ir_right_effective[local_idx],
            )
        if ir_left_random is not None:
            _save_gray_u8(
                os.path.join(chunk_dir, "ir_left_random", f"{curr_im_id:06d}.png"),
                ir_left_random[local_idx],
            )
        if ir_right_random is not None:
            _save_gray_u8(
                os.path.join(chunk_dir, "ir_right_random", f"{curr_im_id:06d}.png"),
                ir_right_random[local_idx],
            )

        _BopWriterUtility.save_json(os.path.join(chunk_dir, "scene_gt.json"), chunk_gt)
        _BopWriterUtility.save_json(os.path.join(chunk_dir, "scene_camera.json"), chunk_camera)

        image_ids.append(int(curr_im_id))
        chunk_ids.append(int(curr_chunk_id))

        curr_im_id += 1
        if curr_im_id >= frames_per_chunk:
            curr_chunk_id += 1
            curr_im_id = 0
            chunk_gt = {}
            chunk_camera = {}

    return {
        "image_ids": image_ids,
        "chunk_ids": chunk_ids,
        "chunk_dirs": chunk_dirs,
        "chunk_start_ids": chunk_start_ids,
    }


def _ensure_bop_toolkit_ready() -> None:
    try:
        import bop_toolkit_lib  # noqa: F401
    except ImportError:
        SetupUtility.setup_pip(["git+https://github.com/thodan/bop_toolkit", "PyOpenGL==3.1.0"])
    np.float = float  # type: ignore[attr-defined]


def _build_trimesh_objects(dataset_objects: List[MeshObject]) -> Dict[int, Any]:
    trimesh_objects: Dict[int, Any] = {}
    for obj in dataset_objects:
        if isinstance(obj, Link):
            if not obj.visuals:
                continue
            if len(obj.visuals) > 1:
                warnings.warn("Writer only supports one visual mesh per Link.")
            mesh_obj = obj.visuals[0]
        else:
            mesh_obj = obj

        cat_id = mesh_obj.get_cp("category_id") if mesh_obj.has_cp("category_id") else None
        if cat_id is None or cat_id in trimesh_objects:
            continue

        trimesh_obj = mesh_obj.mesh_as_trimesh()
        if not np.all(np.isclose(np.array(mesh_obj.blender_obj.scale), mesh_obj.blender_obj.scale[0])):
            print("WARNING: non-uniform scale on object; BOP annotations may be inconsistent with pyrender.")
        trimesh_objects[int(cat_id)] = trimesh_obj
    return trimesh_objects


def _calc_chunk_annotations(
    *,
    dataset_objects: List[MeshObject],
    chunk_start_ids: Dict[str, int],
    annotation_scale: float,
    delta: float,
    num_worker: Optional[int],
) -> None:
    if not chunk_start_ids:
        return

    _ensure_bop_toolkit_ready()
    trimesh_objects = _build_trimesh_objects(dataset_objects)
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y

    pool = Pool(
        num_worker,
        initializer=_BopWriterUtility._pyrender_init,
        initargs=[width, height, trimesh_objects],
    )
    try:
        for chunk_dir, starting_frame_id in chunk_start_ids.items():
            _BopWriterUtility.calc_gt_masks(
                pool=pool,
                chunk_dirs=[chunk_dir],
                starting_frame_id=int(starting_frame_id),
                annotation_scale=annotation_scale,
                delta=delta,
            )
            _BopWriterUtility.calc_gt_info(
                pool=pool,
                chunk_dirs=[chunk_dir],
                starting_frame_id=int(starting_frame_id),
                annotation_scale=annotation_scale,
                delta=delta,
            )
            _BopWriterUtility.calc_gt_coco(
                chunk_dirs=[chunk_dir],
                dataset_objects=dataset_objects,
                starting_frame_id=int(starting_frame_id),
            )
    finally:
        pool.close()
        pool.join()


__all__ = ["write_bop_multidepth"]
