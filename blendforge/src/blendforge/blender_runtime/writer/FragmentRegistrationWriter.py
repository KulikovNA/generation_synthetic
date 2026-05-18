from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
from itertools import groupby
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import bpy
import cv2
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from blenderproc.python.writer.WriterUtility import _WriterUtility


SURFACE_LABEL_SHELL = 0
SURFACE_LABEL_FRACTURE = 1
SURFACE_LABEL_UNKNOWN = 255

SURFACE_MASK_BACKGROUND = 0
SURFACE_MASK_SHELL = 1
SURFACE_MASK_FRACTURE = 2
SURFACE_MASK_INVALID = 255


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json_atomic(path: str, obj: Any, indent: Optional[int] = 2) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    kwargs: Dict[str, Any] = {"ensure_ascii": False}
    if indent is None:
        kwargs["separators"] = (",", ":")
    else:
        kwargs["indent"] = indent
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, **kwargs)
    os.replace(tmp, path)


def save_png_u8(path: str, arr: np.ndarray) -> None:
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image for {path}, got {arr.dtype}")
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, arr)


def save_rgb_png(path: str, rgb: np.ndarray) -> None:
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = (rgb * 255.0 + 0.5).astype(np.uint8)
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, rgb[:, :, ::-1])


def meters_to_depth_u16(depth_m: np.ndarray, depth_scale_mm: float = 1.0) -> np.ndarray:
    if depth_scale_mm <= 0:
        raise ValueError("depth_scale_mm must be > 0")
    d = depth_m.astype(np.float32, copy=False)
    d = np.where(np.isfinite(d), d, 0.0)
    d = np.where(d > 0.0, d, 0.0)
    scaled = d * 1000.0 / float(depth_scale_mm)
    scaled = np.clip(scaled, 0.0, 65535.0)
    return (scaled + 0.5).astype(np.uint16)


def save_depth_png(path: str, depth_m: np.ndarray, depth_scale_mm: float = 1.0) -> None:
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, meters_to_depth_u16(depth_m, depth_scale_mm=depth_scale_mm))


def binary_mask_to_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
    rle: Dict[str, Any] = {"counts": [], "size": list(binary_mask.shape)}
    counts: List[int] = rle["counts"]
    for idx, (value, elements) in enumerate(groupby(binary_mask.ravel(order="F"))):
        if idx == 0 and value == 1:
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
    return [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]


def matrix_to_list(mat: Any) -> List[List[float]]:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {arr.shape}")
    return [[float(v) for v in row] for row in arr.tolist()]


def transform_points(mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts_h = np.concatenate([pts.astype(np.float64), np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    out = (mat.astype(np.float64) @ pts_h.T).T[:, :3]
    return out.astype(np.float32)


def transform_normal(mat: np.ndarray, normal: np.ndarray) -> np.ndarray:
    linear = mat[:3, :3].astype(np.float64)
    n = np.linalg.inv(linear).T @ normal.astype(np.float64)
    norm = np.linalg.norm(n)
    if norm <= 1e-12:
        return np.zeros(3, dtype=np.float32)
    return (n / norm).astype(np.float32)


def normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def barycentric_for_triangle(point: np.ndarray, tri: np.ndarray) -> np.ndarray:
    a, b, c = tri.astype(np.float64)
    p = point.astype(np.float64)
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) <= 1e-18:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w], dtype=np.float32)


def stable_face_barycentric_samples(samples_per_face: int) -> np.ndarray:
    if samples_per_face <= 1:
        return np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float32)
    base = [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [0.60, 0.20, 0.20],
        [0.20, 0.60, 0.20],
        [0.20, 0.20, 0.60],
        [0.50, 0.25, 0.25],
        [0.25, 0.50, 0.25],
        [0.25, 0.25, 0.50],
    ]
    if samples_per_face <= len(base):
        return np.asarray(base[:samples_per_face], dtype=np.float32)

    out = list(base)
    rng = np.random.default_rng(17)
    while len(out) < samples_per_face:
        b = rng.dirichlet(np.ones(3, dtype=np.float32)).astype(np.float32)
        out.append(b.tolist())
    return np.asarray(out, dtype=np.float32)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scale_to_filename_token(scale: float) -> str:
    text = f"{float(scale):.10g}".replace("-", "m").replace(".", "p")
    return text


def model_filename_for_scale(object_id: str, fragment_output_scale: float) -> str:
    if np.isclose(float(fragment_output_scale), 1.0):
        return f"{object_id}.ply"
    return f"{object_id}__scale_{scale_to_filename_token(float(fragment_output_scale))}.ply"


def get_camera_info(depth_scale_mm: float) -> Dict[str, Any]:
    cam_K = _WriterUtility.get_cam_attribute(bpy.context.scene.camera, "cam_K")
    width = int(bpy.context.scene.render.resolution_x)
    height = int(bpy.context.scene.render.resolution_y)
    return {
        "width": width,
        "height": height,
        "K": cam_K,
        "fx": float(cam_K[0][0]),
        "fy": float(cam_K[1][1]),
        "cx": float(cam_K[0][2]),
        "cy": float(cam_K[1][2]),
        "depth_format": "uint16_png",
        "depth_units": "millimeter",
        "depth_scale": float(depth_scale_mm),
        "depth_scale_m": float(depth_scale_mm) / 1000.0,
        "coordinate_convention": "BOP/OpenCV: X right, Y down, Z forward",
        "extrinsics_convention": "T_C_from_W maps homogeneous world points to camera points",
    }


def get_T_W_from_C_opencv() -> np.ndarray:
    mat = _WriterUtility.get_cam_attribute(
        bpy.context.scene.camera,
        "cam2world_matrix",
        local_frame_change=["X", "-Y", "-Z"],
    )
    return np.asarray(mat, dtype=np.float64)


def get_T_C_from_W_opencv() -> np.ndarray:
    return np.linalg.inv(get_T_W_from_C_opencv())


def fragment_filter_annotation(fragment_filter_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "enabled": bool(fragment_filter_cfg.get("enabled", False)),
        "min_vertices": int(fragment_filter_cfg.get("min_vertices", 0)),
        "min_faces": int(fragment_filter_cfg.get("min_faces", 0)),
        "ignore_reason": str(fragment_filter_cfg.get("ignore_reason", "small_fragment")),
        "ignored_fragment_policy": (
            "mesh_and_pose_metadata_only; excluded from COCO, GT frame fragments, "
            "instance_masks, surface_masks and visible_points"
        ),
    }


def fragment_annotation_ignore_reason(
    mesh: bpy.types.Mesh,
    fragment_filter_cfg: Mapping[str, Any],
) -> Optional[str]:
    if not bool(fragment_filter_cfg.get("enabled", False)):
        return None

    reason = str(fragment_filter_cfg.get("ignore_reason", "small_fragment"))
    min_vertices = int(fragment_filter_cfg.get("min_vertices", 0))
    min_faces = int(fragment_filter_cfg.get("min_faces", 0))
    num_vertices = int(len(mesh.vertices))
    num_faces = int(len(mesh.polygons))

    failed = []
    if min_vertices > 0 and num_vertices < min_vertices:
        failed.append(f"num_vertices<{min_vertices}")
    if min_faces > 0 and num_faces < min_faces:
        failed.append(f"num_faces<{min_faces}")
    if not failed:
        return None
    return f"{reason}: {', '.join(failed)}"


class DigitalTwinSurface:
    def __init__(self, bpy_obj: bpy.types.Object):
        bm = None
        import bmesh

        bm = bmesh.new()
        bm.from_mesh(bpy_obj.data)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        self.bvh = BVHTree.FromBMesh(bm)
        bm.free()

    def nearest(self, point_o: Sequence[float]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int], float]:
        result = self.bvh.find_nearest(Vector(point_o))
        if result is None:
            return None, None, None, float("inf")
        loc, normal, face_idx, distance = result
        if loc is None or normal is None:
            return None, None, None, float("inf")
        return (
            np.asarray(loc[:], dtype=np.float32),
            normalize_vec(np.asarray(normal[:], dtype=np.float32)),
            int(face_idx),
            float(distance),
        )


class FragmentRegistrationWriter:
    def __init__(
        self,
        dataset_root: str,
        split: str,
        scene_id: int,
        *,
        overwrite_scene: bool = False,
        depth_scale_mm: float = 1.0,
        indent: int = 2,
    ) -> None:
        self.dataset_root = dataset_root
        self.split = split
        self.scene_id_int = int(scene_id)
        self.scene_id = f"scene_{self.scene_id_int:06d}"
        self.depth_scale_mm = float(depth_scale_mm)
        self.indent = int(indent)

        self.split_dir = os.path.join(dataset_root, split)
        self.models_dir = os.path.join(self.split_dir, "models")
        self.scene_dir = os.path.join(self.split_dir, self.scene_id)

        if os.path.exists(self.scene_dir) and overwrite_scene:
            shutil.rmtree(self.scene_dir)
        if os.path.exists(self.scene_dir) and os.listdir(self.scene_dir) and not overwrite_scene:
            raise FileExistsError(
                f"Scene directory already exists and is not empty: {self.scene_dir}. "
                "Set overwrite_scene=True for this scenario only if replacement is intended."
            )

        self.fragments_dir = os.path.join(self.scene_dir, "fragments")
        self.labels_dir = os.path.join(self.fragments_dir, "labels")
        self.samples_dir = os.path.join(self.fragments_dir, "samples")
        self.images_dir = os.path.join(self.scene_dir, "images")
        self.depth_dir = os.path.join(self.scene_dir, "depth")
        self.instance_masks_dir = os.path.join(self.scene_dir, "instance_masks")
        self.surface_masks_dir = os.path.join(self.scene_dir, "surface_masks")
        self.visible_points_dir = os.path.join(self.scene_dir, "visible_points")

        for path in [
            self.models_dir,
            self.fragments_dir,
            self.labels_dir,
            self.samples_dir,
            self.images_dir,
            self.depth_dir,
            self.instance_masks_dir,
            self.surface_masks_dir,
            self.visible_points_dir,
        ]:
            ensure_dir(path)

    def rel_to_scene(self, path: str) -> str:
        return os.path.relpath(path, self.scene_dir).replace(os.sep, "/")

    def rel_to_split(self, path: str) -> str:
        return os.path.relpath(path, self.split_dir).replace(os.sep, "/")

    def write_object_model_ply(
        self,
        object_bpy: bpy.types.Object,
        object_id: str,
        *,
        model_metadata: Optional[Mapping[str, Any]] = None,
        overwrite_model: bool = False,
    ) -> Tuple[str, str]:
        meta = dict(model_metadata or {})
        fragment_output_scale = float(meta.get("fragment_output_scale", 1.0))
        filename = model_filename_for_scale(object_id, fragment_output_scale)
        abs_path = os.path.join(self.models_dir, filename)
        meta_path = os.path.splitext(abs_path)[0] + ".meta.json"

        expected_meta = {
            "object_id": object_id,
            "mesh": os.path.basename(abs_path),
            "mesh_format": "ply",
            **meta,
        }

        if os.path.exists(abs_path):
            if not os.path.exists(meta_path):
                if not overwrite_model:
                    raise FileExistsError(
                        f"Model already exists without metadata: {abs_path}. "
                        "Remove it or enable overwrite_model for this scenario."
                    )
            else:
                with open(meta_path, "r", encoding="utf-8") as f:
                    existing_meta = json.load(f)
                comparable_existing = {k: v for k, v in existing_meta.items() if k != "mesh_sha256"}
                if comparable_existing == expected_meta and not overwrite_model:
                    current_hash = file_sha256(abs_path)
                    expected_hash = existing_meta.get("mesh_sha256")
                    if expected_hash is not None and str(expected_hash) != current_hash:
                        raise FileExistsError(
                            f"Model hash mismatch for existing file: {abs_path}. "
                            "The sidecar metadata does not match the mesh bytes."
                        )
                    return self.rel_to_split(abs_path), current_hash
                if not overwrite_model:
                    raise FileExistsError(
                        f"Model metadata mismatch for existing file: {abs_path}. "
                        "Use another scale/config or enable overwrite_model only intentionally."
                    )

        write_mesh_object_ply(abs_path, object_bpy)
        mesh_hash = file_sha256(abs_path)
        expected_meta["mesh_sha256"] = mesh_hash
        write_json_atomic(meta_path, expected_meta, indent=self.indent)
        return self.rel_to_split(abs_path), mesh_hash

    def write_fragment_assets(
        self,
        fragments: Sequence[Any],
        object_id: str,
        object_model_rel_from_split: str,
        object_category_id: int,
        digital_twin_surface: DigitalTwinSurface,
        T_O_from_F_by_name: Mapping[str, np.ndarray],
        surface_labeling_cfg: Mapping[str, Any],
        fragment_filter_cfg: Optional[Mapping[str, Any]] = None,
        fracture_metadata: Optional[Mapping[str, Any]] = None,
        geometry_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, np.ndarray], Dict[int, Dict[str, np.ndarray]]]:
        fragment_records: List[Dict[str, Any]] = []
        ignored_fragment_records: List[Dict[str, Any]] = []
        face_labels_by_fragment: Dict[int, np.ndarray] = {}
        samples_by_fragment: Dict[int, Dict[str, np.ndarray]] = {}
        fragment_filter_cfg = fragment_filter_cfg or {}

        for idx, frag in enumerate(fragments):
            frag_bpy = frag.blender_obj if hasattr(frag, "blender_obj") else frag
            fragment_id = int(frag.get_cp("fragment_id") if hasattr(frag, "get_cp") and frag.has_cp("fragment_id") else idx)
            fragment_name = f"fragment_{fragment_id:04d}"
            mesh_abs = os.path.join(self.fragments_dir, f"{fragment_name}.ply")
            face_labels_abs = os.path.join(self.labels_dir, f"{fragment_name}_face_labels.npy")
            samples_abs = os.path.join(self.samples_dir, f"{fragment_name}_samples.npz")

            write_mesh_object_ply(mesh_abs, frag_bpy)

            T_O_from_F = np.asarray(T_O_from_F_by_name[frag_bpy.name], dtype=np.float64)
            mesh = frag_bpy.data
            T_W_from_F = np.asarray(frag.get_local2world_mat() if hasattr(frag, "get_local2world_mat") else frag_bpy.matrix_world, dtype=np.float64)
            fracture_uid = str(frag.get_cp("fracture_uid")) if hasattr(frag, "has_cp") and frag.has_cp("fracture_uid") else ""
            fracture_seed = int(frag.get_cp("fracture_seed")) if hasattr(frag, "has_cp") and frag.has_cp("fracture_seed") else 0
            fracture_method = str(frag.get_cp("fracture_method")) if hasattr(frag, "has_cp") and frag.has_cp("fracture_method") else ""
            ignore_reason = fragment_annotation_ignore_reason(
                mesh,
                fragment_filter_cfg,
            )

            record = {
                "fragment_id": fragment_id,
                "name": fragment_name,
                "mesh": self.rel_to_scene(mesh_abs),
                "mesh_format": "ply",
                "mesh_sha256": file_sha256(mesh_abs),
                "category_id": int(object_category_id),
                "fracture_uid": fracture_uid,
                "fracture_seed": fracture_seed,
                "fracture_method": fracture_method,
                "T_O_from_F": matrix_to_list(T_O_from_F),
                "T_W_from_F": matrix_to_list(T_W_from_F),
                "num_vertices": int(len(mesh.vertices)),
                "num_faces": int(len(mesh.polygons)),
                "annotation_status": "ignored" if ignore_reason else "annotated",
                "ignore_reason": ignore_reason,
            }
            if ignore_reason:
                ignored_fragment_records.append(record)
                continue

            face_labels = compute_face_labels(
                frag_bpy,
                digital_twin_surface=digital_twin_surface,
                T_O_from_F=T_O_from_F,
                distance_threshold_m=float(surface_labeling_cfg.get("distance_threshold_m", 0.0005)),
                normal_cos_threshold=float(surface_labeling_cfg.get("normal_cos_threshold", 0.75)),
                samples_per_face=int(surface_labeling_cfg.get("samples_per_face", 5)),
                unknown_policy=str(surface_labeling_cfg.get("unknown_policy", "ignore")),
                cleanup_small_fracture_components=bool(
                    surface_labeling_cfg.get("cleanup_small_fracture_components", True)
                ),
                min_fracture_component_faces=int(surface_labeling_cfg.get("min_fracture_component_faces", 12)),
                min_fracture_component_area=float(surface_labeling_cfg.get("min_fracture_component_area", 0.0)),
                small_component_shell_neighbor_ratio=float(
                    surface_labeling_cfg.get("small_component_shell_neighbor_ratio", 0.6)
                ),
            )
            np.save(face_labels_abs, face_labels)

            samples = sample_fragment_surface(
                frag_bpy,
                T_O_from_F=T_O_from_F,
                face_labels=face_labels,
                samples_per_face=int(surface_labeling_cfg.get("samples_per_face", 5)),
            )
            np.savez_compressed(samples_abs, **samples)

            face_labels_by_fragment[fragment_id] = face_labels
            samples_by_fragment[fragment_id] = samples

            record.update(
                {
                    "face_labels": self.rel_to_scene(face_labels_abs),
                    "face_labels_sha256": file_sha256(face_labels_abs),
                    "samples": self.rel_to_scene(samples_abs),
                    "face_label_order": "ply_face_element_order",
                }
            )
            fragment_records.append(record)

        annotation = {
            "object_id": object_id,
            "object_model": f"../../models/{os.path.basename(object_model_rel_from_split)}",
            "coordinate_systems": {
                "object": "O",
                "world": "W",
                "camera": "C",
                "fragment_prefix": "F",
            },
            "geometry": dict(geometry_metadata or {"units": "meter"}),
            "fracture": dict(fracture_metadata or {}),
            "surface_labeling": {
                "method": str(surface_labeling_cfg.get("method", "distance_and_normal_to_digital_twin")),
                "digital_twin_space": "O",
                "normal_space": "O",
                "distance_threshold_m": float(surface_labeling_cfg.get("distance_threshold_m", 0.0005)),
                "normal_cos_threshold": float(surface_labeling_cfg.get("normal_cos_threshold", 0.75)),
                "samples_per_face": int(surface_labeling_cfg.get("samples_per_face", 5)),
                "unknown_policy": str(surface_labeling_cfg.get("unknown_policy", "ignore")),
                "cleanup_small_fracture_components": bool(
                    surface_labeling_cfg.get("cleanup_small_fracture_components", True)
                ),
                "min_fracture_component_faces": int(surface_labeling_cfg.get("min_fracture_component_faces", 12)),
                "min_fracture_component_area": float(surface_labeling_cfg.get("min_fracture_component_area", 0.0)),
                "small_component_shell_neighbor_ratio": float(
                    surface_labeling_cfg.get("small_component_shell_neighbor_ratio", 0.6)
                ),
                "label_encoding": {
                    "0": "shell",
                    "1": "fracture",
                    "255": "unknown",
                },
            },
            "surface_label_encoding": {
                "0": "shell",
                "1": "fracture",
                "255": "unknown",
            },
            "fragment_filter": fragment_filter_annotation(fragment_filter_cfg),
            "num_fragments_total": int(len(fragment_records) + len(ignored_fragment_records)),
            "num_fragments_annotated": int(len(fragment_records)),
            "num_fragments_ignored": int(len(ignored_fragment_records)),
            "fragments": sorted(fragment_records, key=lambda item: int(item["fragment_id"])),
            "ignored_fragments": sorted(ignored_fragment_records, key=lambda item: int(item["fragment_id"])),
        }
        write_json_atomic(
            os.path.join(self.fragments_dir, "fragment_annotations.json"),
            annotation,
            indent=self.indent,
        )

        return fragment_records, face_labels_by_fragment, samples_by_fragment

    def write_frames(
        self,
        *,
        colors: Sequence[np.ndarray],
        depths_m: Sequence[np.ndarray],
        instance_segmaps: Sequence[np.ndarray],
        instance_attribute_maps: Sequence[Sequence[dict]],
        fragments: Sequence[Any],
        fragment_records: Sequence[Mapping[str, Any]],
        face_labels_by_fragment: Mapping[int, np.ndarray],
        T_O_from_F_by_fragment: Mapping[int, np.ndarray],
        write_flags: Mapping[str, bool],
        visible_points_pixel_stride: int = 1,
    ) -> None:
        if not (len(colors) == len(depths_m) == len(instance_segmaps) == len(instance_attribute_maps)):
            raise ValueError("Rendered frame list length mismatch")

        fragment_by_id: Dict[int, Any] = {}
        for frag in fragments:
            fragment_id = int(frag.get_cp("fragment_id"))
            fragment_by_id[fragment_id] = frag
        annotated_fragment_ids = {int(record["fragment_id"]) for record in fragment_records}

        gt_frames: List[Dict[str, Any]] = []
        coco = new_coco_document()
        coco["categories"].append({"id": 1, "name": "fragment", "supercategory": "fragment"})

        for frame_idx in range(len(colors)):
            bpy.context.scene.frame_set(frame_idx)
            stem = f"frame_{frame_idx:06d}"

            rgb_rel = f"images/{stem}.png"
            depth_rel = f"depth/{stem}.png"
            instance_rel = f"instance_masks/{stem}.png"
            surface_rel = f"surface_masks/{stem}.png"
            visible_rel = f"visible_points/{stem}.npz"

            if write_flags.get("write_rgb", True):
                save_rgb_png(os.path.join(self.scene_dir, rgb_rel), colors[frame_idx])
            if write_flags.get("write_depth", True):
                save_depth_png(os.path.join(self.scene_dir, depth_rel), depths_m[frame_idx], self.depth_scale_mm)

            packed, fragment_id_to_mask_value = remap_instance_segmap(
                instance_segmaps[frame_idx],
                instance_attribute_maps[frame_idx],
                allowed_fragment_ids=annotated_fragment_ids,
            )
            if write_flags.get("write_instance_masks", True):
                save_png_u8(os.path.join(self.scene_dir, instance_rel), packed)

            visible = compute_visible_points_for_frame(
                packed_instance_mask=packed,
                fragment_id_to_mask_value=fragment_id_to_mask_value,
                fragment_by_id=fragment_by_id,
                face_labels_by_fragment=face_labels_by_fragment,
                T_O_from_F_by_fragment=T_O_from_F_by_fragment,
                pixel_stride=max(1, int(visible_points_pixel_stride)),
            )
            if write_flags.get("write_surface_masks", True):
                save_png_u8(os.path.join(self.scene_dir, surface_rel), visible["surface_mask"])
            if write_flags.get("write_visible_points", True):
                np.savez_compressed(
                    os.path.join(self.scene_dir, visible_rel),
                    u=visible["u"],
                    v=visible["v"],
                    fragment_id=visible["fragment_id"],
                    surface_label=visible["surface_label"],
                    points_C=visible["points_C"],
                    points_F=visible["points_F"],
                    points_O=visible["points_O"],
                    face_id=visible["face_id"],
                    barycentric=visible["barycentric"],
                    shell_indices=visible["shell_indices"],
                    fracture_indices=visible["fracture_indices"],
                )

            H, W = packed.shape
            image_entry = {
                "id": int(frame_idx),
                "file_name": rgb_rel,
                "width": int(W),
                "height": int(H),
                "date_captured": datetime.datetime.utcnow().isoformat(" "),
                "license": 1,
                "depth_file": depth_rel,
                "instance_mask_file": instance_rel,
                "surface_mask_file": surface_rel,
                "visible_points_file": visible_rel,
            }
            coco["images"].append(image_entry)

            frame_fragment_entries: List[Dict[str, Any]] = []
            T_C_from_W = get_T_C_from_W_opencv()

            for record in sorted(fragment_records, key=lambda item: int(item["fragment_id"])):
                fragment_id = int(record["fragment_id"])
                mask_value = fragment_id_to_mask_value.get(fragment_id)
                if mask_value is None:
                    continue

                binary = packed == int(mask_value)
                visible_pixels = int(binary.sum())
                if visible_pixels < 1:
                    continue

                surf = visible["surface_mask"]
                visible_shell_pixels = int(np.logical_and(binary, surf == SURFACE_MASK_SHELL).sum())
                visible_fracture_pixels = int(np.logical_and(binary, surf == SURFACE_MASK_FRACTURE).sum())

                frag = fragment_by_id[fragment_id]
                T_W_from_F = np.asarray(frag.get_local2world_mat(), dtype=np.float64)
                T_O_from_F = np.asarray(T_O_from_F_by_fragment[fragment_id], dtype=np.float64)
                T_C_from_F = T_C_from_W @ T_W_from_F
                T_C_from_O = T_C_from_F @ np.linalg.inv(T_O_from_F)

                frame_fragment_entries.append(
                    {
                        "fragment_id": int(fragment_id),
                        "instance_mask_value": int(mask_value),
                        "T_C_from_F": matrix_to_list(T_C_from_F),
                        "T_C_from_O": matrix_to_list(T_C_from_O),
                        "bbox_visib": bbox_from_binary_mask(binary),
                        "visible_pixels": visible_pixels,
                        "visible_shell_pixels": visible_shell_pixels,
                        "visible_fracture_pixels": visible_fracture_pixels,
                    }
                )

                add_coco_annotation(
                    coco,
                    frame_id=frame_idx,
                    binary_mask=binary,
                    record=record,
                )

            gt_frames.append(
                {
                    "frame_id": int(frame_idx),
                    "image": rgb_rel,
                    "depth": depth_rel,
                    "instance_mask": instance_rel,
                    "surface_mask": surface_rel,
                    "visible_points": visible_rel,
                    "T_C_from_W": matrix_to_list(T_C_from_W),
                    "fragments": frame_fragment_entries,
                }
            )

        write_json_atomic(
            os.path.join(self.scene_dir, "gt_annotations.json"),
            {"scene_id": self.scene_id, "frames": gt_frames},
            indent=self.indent,
        )
        write_json_atomic(os.path.join(self.scene_dir, "coco_annotations.json"), coco, indent=self.indent)

    def write_scene_meta(
        self,
        *,
        object_id: str,
        object_model_rel_from_split: str,
        num_fragments: int,
        num_frames: int,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        meta: Dict[str, Any] = {
            "scene_id": self.scene_id,
            "split": self.split,
            "object_id": object_id,
            "object_model": f"../models/{os.path.basename(object_model_rel_from_split)}",
            "num_fragments": int(num_fragments),
            "num_frames": int(num_frames),
            "units": "meter",
            "paths": {
                "fragments": "fragments",
                "images": "images",
                "depth": "depth",
                "instance_masks": "instance_masks",
                "surface_masks": "surface_masks",
                "visible_points": "visible_points",
            },
            "instance_id_encoding": {
                "0": "background",
                "n": "compact annotated-fragment instance id; see gt_annotations/coco_annotations for fragment_id",
            },
            "surface_label_encoding": {
                "0": "shell",
                "1": "fracture",
                "255": "unknown",
            },
            "surface_mask_encoding": {
                "0": "background",
                "1": "shell",
                "2": "fracture",
                "255": "invalid",
            },
        }
        if extra:
            meta.update(dict(extra))
        write_json_atomic(os.path.join(self.scene_dir, "scene_meta.json"), meta, indent=self.indent)

    def write_camera_info(self) -> None:
        write_json_atomic(
            os.path.join(self.scene_dir, "camera_info.json"),
            get_camera_info(depth_scale_mm=self.depth_scale_mm),
            indent=self.indent,
        )


def write_mesh_object_ply(path: str, bpy_obj: bpy.types.Object) -> None:
    mesh = bpy_obj.data
    mesh.update(calc_edges=True)
    ensure_dir(os.path.dirname(path))

    vertices = [v.co.copy() for v in mesh.vertices]
    normals = [v.normal.copy() for v in mesh.vertices]
    faces = [list(poly.vertices) for poly in mesh.polygons]
    non_tri_faces = [face for face in faces if len(face) != 3]
    if non_tri_faces:
        raise ValueError(f"PLY export expects triangulated mesh: {bpy_obj.name}")

    with open(path, "w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment object_name {bpy_obj.name}\n")
        f.write("comment vertex_coordinates local_fragment_or_object_frame\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for co, normal in zip(vertices, normals):
            f.write(
                f"{co.x:.9g} {co.y:.9g} {co.z:.9g} "
                f"{normal.x:.9g} {normal.y:.9g} {normal.z:.9g}\n"
            )
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def triangle_area(tri: np.ndarray) -> float:
    return float(0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))


def build_face_adjacency(mesh: bpy.types.Mesh) -> List[List[int]]:
    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
    for poly in mesh.polygons:
        vertices = list(poly.vertices)
        for idx, v0 in enumerate(vertices):
            v1 = vertices[(idx + 1) % len(vertices)]
            edge = (int(min(v0, v1)), int(max(v0, v1)))
            edge_to_faces.setdefault(edge, []).append(int(poly.index))

    adjacency = [set() for _ in mesh.polygons]
    for faces in edge_to_faces.values():
        if len(faces) < 2:
            continue
        for face_idx in faces:
            adjacency[face_idx].update(other for other in faces if other != face_idx)

    return [sorted(face_neighbors) for face_neighbors in adjacency]


def connected_label_components(
    labels: np.ndarray,
    *,
    target_label: int,
    adjacency: Sequence[Sequence[int]],
) -> List[List[int]]:
    visited = np.zeros((len(labels),), dtype=bool)
    components: List[List[int]] = []

    for start_idx, label in enumerate(labels):
        if visited[start_idx] or int(label) != int(target_label):
            continue

        component: List[int] = []
        stack = [int(start_idx)]
        visited[start_idx] = True
        while stack:
            face_idx = stack.pop()
            component.append(face_idx)
            for neighbor_idx in adjacency[face_idx]:
                if visited[neighbor_idx] or int(labels[neighbor_idx]) != int(target_label):
                    continue
                visited[neighbor_idx] = True
                stack.append(int(neighbor_idx))
        components.append(component)

    return components


def clean_small_fracture_components(
    mesh: bpy.types.Mesh,
    labels: np.ndarray,
    *,
    min_faces: int,
    min_area: float,
    shell_neighbor_ratio: float,
) -> np.ndarray:
    min_faces = max(0, int(min_faces))
    min_area = max(0.0, float(min_area))
    shell_neighbor_ratio = float(np.clip(shell_neighbor_ratio, 0.0, 1.0))
    if min_faces == 0 and min_area <= 0.0:
        return labels

    adjacency = build_face_adjacency(mesh)
    components = connected_label_components(
        labels,
        target_label=SURFACE_LABEL_FRACTURE,
        adjacency=adjacency,
    )
    if not components:
        return labels

    verts = np.asarray([v.co[:] for v in mesh.vertices], dtype=np.float64)
    face_areas = np.zeros((len(mesh.polygons),), dtype=np.float64)
    for poly in mesh.polygons:
        if len(poly.vertices) == 3:
            face_areas[poly.index] = triangle_area(verts[list(poly.vertices)])

    cleaned = labels.copy()
    for component in components:
        component_set = set(component)
        boundary_neighbors = set()
        for face_idx in component:
            for neighbor_idx in adjacency[face_idx]:
                if neighbor_idx not in component_set:
                    boundary_neighbors.add(neighbor_idx)

        if not boundary_neighbors:
            continue

        shell_neighbors = sum(1 for idx in boundary_neighbors if int(cleaned[idx]) == SURFACE_LABEL_SHELL)
        ratio = float(shell_neighbors) / float(len(boundary_neighbors))
        is_small_by_faces = min_faces > 0 and len(component) <= min_faces
        is_small_by_area = min_area > 0.0 and float(face_areas[component].sum()) <= min_area

        if (is_small_by_faces or is_small_by_area) and ratio >= shell_neighbor_ratio:
            cleaned[component] = SURFACE_LABEL_SHELL

    return cleaned


def compute_face_labels(
    frag_bpy: bpy.types.Object,
    *,
    digital_twin_surface: DigitalTwinSurface,
    T_O_from_F: np.ndarray,
    distance_threshold_m: float,
    normal_cos_threshold: float,
    samples_per_face: int,
    unknown_policy: str,
    cleanup_small_fracture_components: bool = False,
    min_fracture_component_faces: int = 12,
    min_fracture_component_area: float = 0.0,
    small_component_shell_neighbor_ratio: float = 0.6,
) -> np.ndarray:
    mesh = frag_bpy.data
    labels = np.full((len(mesh.polygons),), SURFACE_LABEL_UNKNOWN, dtype=np.uint8)
    bary_samples = stable_face_barycentric_samples(max(1, samples_per_face))
    verts = np.asarray([v.co[:] for v in mesh.vertices], dtype=np.float32)

    for poly in mesh.polygons:
        if len(poly.vertices) != 3:
            labels[poly.index] = SURFACE_LABEL_UNKNOWN
            continue

        tri = verts[list(poly.vertices)]
        normal_f = normalize_vec(np.asarray(poly.normal[:], dtype=np.float32))
        normal_o = transform_normal(T_O_from_F, normal_f)
        if np.linalg.norm(normal_o) <= 1e-12:
            labels[poly.index] = SURFACE_LABEL_UNKNOWN
            continue

        shell_votes = 0
        valid_votes = 0
        for bary in bary_samples:
            point_f = bary @ tri
            point_o = transform_points(T_O_from_F, point_f.reshape(1, 3))[0]
            _loc, nearest_normal_o, _face_idx, distance = digital_twin_surface.nearest(point_o)
            if nearest_normal_o is None:
                continue
            valid_votes += 1
            normal_dot = float(np.dot(normal_o, nearest_normal_o))
            if distance <= distance_threshold_m and normal_dot >= normal_cos_threshold:
                shell_votes += 1

        if valid_votes == 0:
            labels[poly.index] = SURFACE_LABEL_UNKNOWN
        elif shell_votes > valid_votes / 2.0:
            labels[poly.index] = SURFACE_LABEL_SHELL
        else:
            labels[poly.index] = SURFACE_LABEL_FRACTURE

    if unknown_policy == "fracture":
        labels[labels == SURFACE_LABEL_UNKNOWN] = SURFACE_LABEL_FRACTURE
    elif unknown_policy == "shell":
        labels[labels == SURFACE_LABEL_UNKNOWN] = SURFACE_LABEL_SHELL
    elif unknown_policy != "ignore":
        raise ValueError(f"Unsupported unknown_policy: {unknown_policy}")

    if cleanup_small_fracture_components:
        labels = clean_small_fracture_components(
            mesh,
            labels,
            min_faces=min_fracture_component_faces,
            min_area=min_fracture_component_area,
            shell_neighbor_ratio=small_component_shell_neighbor_ratio,
        )

    return labels


def sample_fragment_surface(
    frag_bpy: bpy.types.Object,
    *,
    T_O_from_F: np.ndarray,
    face_labels: np.ndarray,
    samples_per_face: int,
) -> Dict[str, np.ndarray]:
    mesh = frag_bpy.data
    bary_samples = stable_face_barycentric_samples(max(1, samples_per_face))
    verts = np.asarray([v.co[:] for v in mesh.vertices], dtype=np.float32)

    points_f: List[np.ndarray] = []
    normals_f: List[np.ndarray] = []
    face_ids: List[int] = []
    barycentrics: List[np.ndarray] = []
    labels: List[int] = []

    for poly in mesh.polygons:
        if len(poly.vertices) != 3:
            continue
        tri = verts[list(poly.vertices)]
        normal_f = normalize_vec(np.asarray(poly.normal[:], dtype=np.float32))
        for bary in bary_samples:
            point_f = bary @ tri
            points_f.append(point_f.astype(np.float32))
            normals_f.append(normal_f)
            face_ids.append(int(poly.index))
            barycentrics.append(bary.astype(np.float32))
            labels.append(int(face_labels[poly.index]))

    points_f_arr = np.asarray(points_f, dtype=np.float32).reshape(-1, 3)
    normals_f_arr = np.asarray(normals_f, dtype=np.float32).reshape(-1, 3)
    points_o_arr = transform_points(T_O_from_F, points_f_arr)

    return {
        "points_F": points_f_arr,
        "normals_F": normals_f_arr,
        "points_O": points_o_arr,
        "face_id": np.asarray(face_ids, dtype=np.int32),
        "barycentric": np.asarray(barycentrics, dtype=np.float32).reshape(-1, 3),
        "surface_label": np.asarray(labels, dtype=np.uint8),
    }


def remap_instance_segmap(
    instance_segmap: np.ndarray,
    instance_attribute_map: Sequence[dict],
    allowed_fragment_ids: Optional[Iterable[int]] = None,
) -> Tuple[np.ndarray, Dict[int, int]]:
    if instance_segmap.ndim != 2:
        raise ValueError(f"Expected 2D instance segmap, got {instance_segmap.shape}")

    allowed = None if allowed_fragment_ids is None else {int(v) for v in allowed_fragment_ids}
    idx_to_fragment: Dict[int, int] = {}
    for attr in instance_attribute_map:
        try:
            idx = int(attr["idx"])
        except Exception:
            continue
        try:
            fragment_id = int(attr.get("fragment_id", -1))
        except Exception:
            fragment_id = -1
        if fragment_id < 0:
            continue
        if allowed is not None and fragment_id not in allowed:
            continue
        idx_to_fragment[idx] = fragment_id

    fragment_ids = sorted(set(idx_to_fragment.values()))
    if len(fragment_ids) > 255:
        raise ValueError("uint8 instance masks support at most 255 fragments")

    fragment_to_mask = {fragment_id: i + 1 for i, fragment_id in enumerate(fragment_ids)}
    packed = np.zeros(instance_segmap.shape, dtype=np.uint8)
    for idx, fragment_id in idx_to_fragment.items():
        packed[instance_segmap == int(idx)] = np.uint8(fragment_to_mask[fragment_id])
    return packed, fragment_to_mask


def compute_visible_points_for_frame(
    *,
    packed_instance_mask: np.ndarray,
    fragment_id_to_mask_value: Mapping[int, int],
    fragment_by_id: Mapping[int, Any],
    face_labels_by_fragment: Mapping[int, np.ndarray],
    T_O_from_F_by_fragment: Mapping[int, np.ndarray],
    pixel_stride: int = 1,
) -> Dict[str, np.ndarray]:
    height, width = packed_instance_mask.shape
    surface_mask = np.zeros((height, width), dtype=np.uint8)

    mask_value_to_fragment = {int(v): int(k) for k, v in fragment_id_to_mask_value.items()}

    cam_info = get_camera_info(depth_scale_mm=1.0)
    fx = float(cam_info["fx"])
    fy = float(cam_info["fy"])
    cx = float(cam_info["cx"])
    cy = float(cam_info["cy"])
    T_W_from_C = get_T_W_from_C_opencv()
    T_C_from_W = np.linalg.inv(T_W_from_C)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    origin_w = T_W_from_C[:3, 3].astype(np.float64)
    R_W_from_C = T_W_from_C[:3, :3].astype(np.float64)

    us: List[int] = []
    vs: List[int] = []
    fragment_ids: List[int] = []
    surface_labels: List[int] = []
    points_c: List[np.ndarray] = []
    points_f: List[np.ndarray] = []
    points_o: List[np.ndarray] = []
    face_ids: List[int] = []
    barycentrics: List[np.ndarray] = []

    for v in range(height):
        for u in range(width):
            mask_value = int(packed_instance_mask[v, u])
            if mask_value == 0:
                continue
            fragment_id = mask_value_to_fragment.get(mask_value)
            if fragment_id is None:
                continue
            frag = fragment_by_id.get(fragment_id)
            if frag is None:
                continue

            ray_c = np.array([(u + 0.5 - cx) / fx, (v + 0.5 - cy) / fy, 1.0], dtype=np.float64)
            ray_c /= max(float(np.linalg.norm(ray_c)), 1e-12)
            ray_w = R_W_from_C @ ray_c
            ray_w /= max(float(np.linalg.norm(ray_w)), 1e-12)

            hit, loc, _normal, face_idx, hit_obj, _matrix = bpy.context.scene.ray_cast(
                depsgraph,
                Vector(origin_w),
                Vector(ray_w),
            )
            if not hit or hit_obj is None or int(face_idx) < 0:
                surface_mask[v, u] = SURFACE_MASK_INVALID
                continue
            if hit_obj.name != frag.blender_obj.name:
                continue

            labels = face_labels_by_fragment.get(fragment_id)
            if labels is None or int(face_idx) >= len(labels):
                surface_label = SURFACE_LABEL_UNKNOWN
            else:
                surface_label = int(labels[int(face_idx)])

            if surface_label == SURFACE_LABEL_SHELL:
                surface_mask[v, u] = SURFACE_MASK_SHELL
            elif surface_label == SURFACE_LABEL_FRACTURE:
                surface_mask[v, u] = SURFACE_MASK_FRACTURE
            else:
                surface_mask[v, u] = SURFACE_MASK_INVALID

            collect_visible_point = (v % pixel_stride == 0) and (u % pixel_stride == 0)
            if not collect_visible_point:
                continue

            T_W_from_F = np.asarray(frag.get_local2world_mat(), dtype=np.float64)
            T_F_from_W = np.linalg.inv(T_W_from_F)
            T_O_from_F = np.asarray(T_O_from_F_by_fragment[fragment_id], dtype=np.float64)

            point_w = np.asarray(loc[:], dtype=np.float64).reshape(1, 3)
            point_c = transform_points(T_C_from_W, point_w)[0]
            point_f = transform_points(T_F_from_W, point_w)[0]
            point_o = transform_points(T_O_from_F, point_f.reshape(1, 3))[0]

            mesh = frag.blender_obj.data
            poly = mesh.polygons[int(face_idx)]
            tri = np.asarray([mesh.vertices[i].co[:] for i in poly.vertices], dtype=np.float32)
            bary = barycentric_for_triangle(point_f, tri)

            us.append(int(u))
            vs.append(int(v))
            fragment_ids.append(int(fragment_id))
            surface_labels.append(int(surface_label))
            points_c.append(point_c)
            points_f.append(point_f)
            points_o.append(point_o)
            face_ids.append(int(face_idx))
            barycentrics.append(bary)

    surface_labels_arr = np.asarray(surface_labels, dtype=np.uint8)
    shell_indices = np.where(surface_labels_arr == SURFACE_LABEL_SHELL)[0].astype(np.int32)
    fracture_indices = np.where(surface_labels_arr == SURFACE_LABEL_FRACTURE)[0].astype(np.int32)

    return {
        "surface_mask": surface_mask,
        "u": np.asarray(us, dtype=np.int32),
        "v": np.asarray(vs, dtype=np.int32),
        "fragment_id": np.asarray(fragment_ids, dtype=np.int32),
        "surface_label": surface_labels_arr,
        "points_C": np.asarray(points_c, dtype=np.float32).reshape(-1, 3),
        "points_F": np.asarray(points_f, dtype=np.float32).reshape(-1, 3),
        "points_O": np.asarray(points_o, dtype=np.float32).reshape(-1, 3),
        "face_id": np.asarray(face_ids, dtype=np.int32),
        "barycentric": np.asarray(barycentrics, dtype=np.float32).reshape(-1, 3),
        "shell_indices": shell_indices,
        "fracture_indices": fracture_indices,
    }


def new_coco_document() -> Dict[str, Any]:
    return {
        "info": {
            "description": "Fragment registration instance segmentation",
            "url": "",
            "version": "1.0",
            "year": datetime.datetime.utcnow().year,
            "contributor": "Unknown",
            "date_created": datetime.datetime.utcnow().isoformat(" "),
        },
        "licenses": [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            }
        ],
        "categories": [],
        "images": [],
        "annotations": [],
    }


def add_coco_annotation(
    coco: Dict[str, Any],
    *,
    frame_id: int,
    binary_mask: np.ndarray,
    record: Mapping[str, Any],
) -> None:
    area = int(binary_mask.sum())
    if area < 1:
        return
    ann_id = len(coco["annotations"]) + 1
    coco["annotations"].append(
        {
            "id": int(ann_id),
            "image_id": int(frame_id),
            "category_id": 1,
            "iscrowd": 0,
            "area": area,
            "bbox": bbox_from_binary_mask(binary_mask),
            "segmentation": binary_mask_to_rle(binary_mask.astype(np.uint8)),
            "fragment_id": int(record["fragment_id"]),
            "fracture_uid": str(record.get("fracture_uid", "")),
            "fracture_seed": int(record.get("fracture_seed", 0)),
            "fracture_method": str(record.get("fracture_method", "")),
        }
    )
