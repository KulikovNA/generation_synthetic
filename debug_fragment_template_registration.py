from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Debug configuration. Edit these globals directly; this script intentionally
# does not use argparse.
# ---------------------------------------------------------------------------
DATASET_ROOT = "output/fragment_template_registration/differBig/2026-05-17"
SPLIT = "train"

# None means: use the latest scene_XXXXXX with gt_annotations.json.
SCENE_ID: Optional[str] = None

OUTPUT_ROOT = "debug_fragment_template_registration"

# None means all frames. Keep this low when visible_points are dense.
MAX_FRAMES: Optional[int] = None
FRAME_IDS: Optional[List[int]] = None

DEPTH_OVERLAY_ALPHA = 0.55
INSTANCE_OVERLAY_ALPHA = 0.55
SURFACE_OVERLAY_ALPHA = 0.58
POINT_RADIUS = 1
POINT_DRAW_STRIDE = 1

WRITE_POINT_CLOUDS = True
WRITE_FRAGMENT_SAMPLE_CLOUDS = True

# Sanity thresholds are reported; the script does not stop on failures.
TRANSFORM_TOLERANCE = 1e-4
PROJECTION_TOLERANCE_PX = 1.0

ORACLE_KABSCH_ENABLE = True
ORACLE_KABSCH_MIN_POINTS = 6
ORACLE_KABSCH_WRITE_ALIGNED_CLOUDS = True
ORACLE_KABSCH_WRITE_JSON = True
ORACLE_KABSCH_RESIDUAL_TOLERANCE = 1e-4
ORACLE_KABSCH_TRANSLATION_TOLERANCE = 1e-4
ORACLE_KABSCH_ROTATION_TOLERANCE_DEG = 0.1
# The current symmetry check uses this object-space axis exactly as requested.
ORACLE_KABSCH_SYMMETRY_AXIS_O = np.array([0.0, 1.0, 0.0], dtype=np.float64)


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


SURFACE_LABEL_SHELL = 0
SURFACE_LABEL_FRACTURE = 1
SURFACE_LABEL_UNKNOWN = 255

SURFACE_MASK_BACKGROUND = 0
SURFACE_MASK_SHELL = 1
SURFACE_MASK_FRACTURE = 2
SURFACE_MASK_INVALID = 255


def repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def scene_name_from_id(scene_id: Optional[str]) -> Optional[str]:
    if scene_id is None:
        return None
    if isinstance(scene_id, int):
        return f"scene_{scene_id:06d}"
    if re.fullmatch(r"\d+", str(scene_id)):
        return f"scene_{int(scene_id):06d}"
    return str(scene_id)


def find_latest_scene(split_dir: str) -> str:
    scenes = []
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(split_dir)
    for name in os.listdir(split_dir):
        if re.fullmatch(r"scene_\d{6}", name):
            scene_dir = os.path.join(split_dir, name)
            if os.path.isfile(os.path.join(scene_dir, "gt_annotations.json")):
                scenes.append(name)
    if not scenes:
        raise FileNotFoundError(f"No scene_XXXXXX with gt_annotations.json in {split_dir}")
    return sorted(scenes)[-1]


def read_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def read_u8(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img


def read_depth_u16(path: str) -> np.ndarray:
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(path)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth.astype(np.uint16, copy=False)


def apply_overlay(image_bgr: np.ndarray, colors_bgr: np.ndarray, valid: np.ndarray, alpha: float) -> np.ndarray:
    out = image_bgr.copy()
    if valid.any():
        blended = (
            image_bgr[valid].astype(np.float32) * (1.0 - alpha)
            + colors_bgr[valid].astype(np.float32) * alpha
        )
        out[valid] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def colorize_depth(depth_u16: np.ndarray, depth_scale_mm: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    depth_m = depth_u16.astype(np.float32) * (float(depth_scale_mm) / 1000.0)
    valid = depth_u16 > 0
    color = np.zeros((depth_u16.shape[0], depth_u16.shape[1], 3), dtype=np.uint8)
    stats: Dict[str, Any] = {
        "valid_pixels": int(valid.sum()),
        "min_m": None,
        "max_m": None,
        "p01_m": None,
        "p99_m": None,
    }
    if not valid.any():
        return color, stats

    vals = depth_m[valid]
    lo = float(np.percentile(vals, 1))
    hi = float(np.percentile(vals, 99))
    if hi <= lo:
        hi = float(vals.max())
        lo = float(vals.min())
    if hi <= lo:
        hi = lo + 1e-6

    norm = np.clip((depth_m - lo) / (hi - lo), 0.0, 1.0)
    norm_u8 = (norm * 255.0 + 0.5).astype(np.uint8)
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    color = cv2.applyColorMap(norm_u8, cmap)
    color[~valid] = 0
    stats.update(
        {
            "min_m": float(vals.min()),
            "max_m": float(vals.max()),
            "p01_m": lo,
            "p99_m": hi,
        }
    )
    return color, stats


def instance_color_bgr(instance_id: int) -> Tuple[int, int, int]:
    if instance_id == 0:
        return (0, 0, 0)
    rng = np.random.default_rng(int(instance_id) * 7919 + 17)
    color = rng.integers(50, 255, size=3, dtype=np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def colorize_instance_mask(mask: np.ndarray) -> np.ndarray:
    colors = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for instance_id in np.unique(mask):
        instance_id = int(instance_id)
        if instance_id == 0:
            continue
        colors[mask == instance_id] = instance_color_bgr(instance_id)
    return colors


def colorize_surface_mask(mask: np.ndarray) -> np.ndarray:
    colors = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colors[mask == SURFACE_MASK_SHELL] = (60, 220, 60)
    colors[mask == SURFACE_MASK_FRACTURE] = (40, 40, 255)
    colors[mask == SURFACE_MASK_INVALID] = (255, 0, 255)
    return colors


def surface_label_to_rgb(labels: np.ndarray) -> np.ndarray:
    colors = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    colors[labels == SURFACE_LABEL_SHELL] = np.array([60, 220, 60], dtype=np.uint8)
    colors[labels == SURFACE_LABEL_FRACTURE] = np.array([255, 40, 40], dtype=np.uint8)
    colors[labels == SURFACE_LABEL_UNKNOWN] = np.array([255, 0, 255], dtype=np.uint8)
    return colors


def draw_points(
    image_bgr: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    labels: np.ndarray,
    *,
    radius: int,
    stride: int,
) -> np.ndarray:
    out = image_bgr.copy()
    if u.size == 0:
        return out
    stride = max(1, int(stride))
    for idx in range(0, u.size, stride):
        label = int(labels[idx])
        if label == SURFACE_LABEL_SHELL:
            color = (60, 255, 60)
        elif label == SURFACE_LABEL_FRACTURE:
            color = (40, 40, 255)
        else:
            color = (255, 0, 255)
        cv2.circle(out, (int(u[idx]), int(v[idx])), int(radius), color, -1, lineType=cv2.LINE_AA)
    return out


def add_title(image_bgr: np.ndarray, title: str) -> np.ndarray:
    bar_h = 28
    out = np.zeros((image_bgr.shape[0] + bar_h, image_bgr.shape[1], 3), dtype=np.uint8)
    out[bar_h:] = image_bgr
    cv2.putText(out, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return out


def make_grid(images: Sequence[np.ndarray], cols: int = 2) -> np.ndarray:
    if not images:
        raise ValueError("No images for grid")
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    target_h = max(heights)
    target_w = max(widths)
    padded = []
    for img in images:
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[: img.shape[0], : img.shape[1]] = img
        padded.append(canvas)
    rows = []
    for start in range(0, len(padded), cols):
        row_imgs = list(padded[start : start + cols])
        while len(row_imgs) < cols:
            row_imgs.append(np.zeros_like(padded[0]))
        rows.append(np.concatenate(row_imgs, axis=1))
    return np.concatenate(rows, axis=0)


def save_image(path: str, image_bgr: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, image_bgr)


def transform_points(mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pts_h = np.concatenate([pts.astype(np.float64), np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return (mat.astype(np.float64) @ pts_h.T).T[:, :3]


def estimate_rigid_transform_kabsch(
    src: np.ndarray,
    dst: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    src: [N, 3], points in digital twin coordinates O.
    dst: [N, 3], corresponding points in camera coordinates C.

    Returns T_dst_from_src, i.e. T_C_from_O for this dataset.
    """
    T = np.eye(4, dtype=np.float64)
    diagnostic: Dict[str, Any] = {
        "num_points": 0,
        "rank": 0,
        "det_R": None,
        "singular_values": [],
        "success": False,
        "fail_reason": None,
    }

    try:
        src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
        dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    except Exception as exc:
        diagnostic["fail_reason"] = f"invalid_input_shape: {exc}"
        return T, diagnostic

    diagnostic["num_points"] = int(src.shape[0])
    if src.shape != dst.shape:
        diagnostic["fail_reason"] = f"src_dst_shape_mismatch: {src.shape} vs {dst.shape}"
        return T, diagnostic
    if src.shape[0] < int(ORACLE_KABSCH_MIN_POINTS):
        diagnostic["fail_reason"] = f"not_enough_points: {src.shape[0]} < {ORACLE_KABSCH_MIN_POINTS}"
        return T, diagnostic
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        diagnostic["fail_reason"] = "non_finite_points"
        return T, diagnostic

    if weights is None:
        w = np.full((src.shape[0],), 1.0 / float(src.shape[0]), dtype=np.float64)
    else:
        try:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
        except Exception as exc:
            diagnostic["fail_reason"] = f"invalid_weights_shape: {exc}"
            return T, diagnostic
        if w.shape[0] != src.shape[0]:
            diagnostic["fail_reason"] = f"weights_length_mismatch: {w.shape[0]} vs {src.shape[0]}"
            return T, diagnostic
        if not np.isfinite(w).all() or np.any(w < 0.0):
            diagnostic["fail_reason"] = "invalid_weights_values"
            return T, diagnostic
        weight_sum = float(w.sum())
        if weight_sum <= 1e-12:
            diagnostic["fail_reason"] = "zero_weight_sum"
            return T, diagnostic
        w = w / weight_sum

    src_centroid = np.sum(src * w[:, None], axis=0)
    dst_centroid = np.sum(dst * w[:, None], axis=0)
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid
    covariance = (src_centered * w[:, None]).T @ dst_centered

    try:
        U, singular_values, Vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError as exc:
        diagnostic["fail_reason"] = f"svd_failed: {exc}"
        return T, diagnostic

    rank = int(np.linalg.matrix_rank(covariance, tol=1e-12))
    diagnostic["rank"] = rank
    diagnostic["singular_values"] = [float(v) for v in singular_values.tolist()]
    if rank < 2:
        diagnostic["fail_reason"] = f"degenerate_covariance_rank: {rank}"
        return T, diagnostic

    R = Vt.T @ U.T
    if float(np.linalg.det(R)) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    det_r = float(np.linalg.det(R))
    diagnostic["det_R"] = det_r
    if not np.isfinite(R).all() or not np.isfinite(det_r):
        diagnostic["fail_reason"] = "non_finite_rotation"
        return T, diagnostic
    if det_r < 0.0:
        diagnostic["fail_reason"] = f"reflection_after_correction: det_R={det_r}"
        return T, diagnostic

    t = dst_centroid - R @ src_centroid
    T[:3, :3] = R
    T[:3, 3] = t
    diagnostic["success"] = True
    return T, diagnostic


def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    R_delta = np.asarray(R_pred, dtype=np.float64) @ np.asarray(R_gt, dtype=np.float64).T
    cos_angle = (float(np.trace(R_delta)) - 1.0) * 0.5
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def translation_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(t_pred, dtype=np.float64) - np.asarray(t_gt, dtype=np.float64)))


def transform_residual_stats(T: np.ndarray, src: np.ndarray, dst: np.ndarray) -> Dict[str, Any]:
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src.size == 0:
        return {"num_points": 0, "mean": None, "rmse": None, "max": None, "p95": None}
    residual = np.linalg.norm(transform_points(np.asarray(T, dtype=np.float64), src) - dst, axis=1)
    return {
        "num_points": int(residual.size),
        "mean": float(np.mean(residual)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "max": float(np.max(residual)),
        "p95": float(np.percentile(residual, 95)),
    }


def symmetry_axis_error_deg(R_pred: np.ndarray, R_gt: np.ndarray, axis_o: np.ndarray) -> float:
    axis_o = np.asarray(axis_o, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis_o))
    if norm <= 1e-12:
        return float("nan")
    axis_o = axis_o / norm
    axis_pred = np.asarray(R_pred, dtype=np.float64) @ axis_o
    axis_gt = np.asarray(R_gt, dtype=np.float64) @ axis_o
    axis_pred /= max(float(np.linalg.norm(axis_pred)), 1e-12)
    axis_gt /= max(float(np.linalg.norm(axis_gt)), 1e-12)
    cos_angle = abs(float(np.dot(axis_pred, axis_gt)))
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def max_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not clean:
        return None
    return float(max(clean))


def finite_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def project_points(points_c: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points_c.size == 0:
        empty = np.zeros((0,), dtype=np.float64)
        return empty, empty, empty.astype(bool)
    z = points_c[:, 2].astype(np.float64)
    valid = z > 1e-12
    u = np.full((points_c.shape[0],), np.nan, dtype=np.float64)
    v = np.full((points_c.shape[0],), np.nan, dtype=np.float64)
    u[valid] = K[0, 0] * points_c[valid, 0] / z[valid] + K[0, 2]
    v[valid] = K[1, 1] * points_c[valid, 1] / z[valid] + K[1, 2]
    return u, v, valid


def write_point_cloud_ply(path: str, points: np.ndarray, colors_rgb: Optional[np.ndarray] = None) -> None:
    ensure_dir(os.path.dirname(path))
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if colors_rgb is None:
        colors_rgb = np.full((points.shape[0], 3), 220, dtype=np.uint8)
    else:
        colors_rgb = np.asarray(colors_rgb, dtype=np.uint8).reshape(-1, 3)
        if colors_rgb.shape[0] != points.shape[0]:
            raise ValueError("points/colors length mismatch")

    with open(path, "w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors_rgb):
            f.write(f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def read_ply_counts(path: str) -> Dict[str, int]:
    counts = {"vertices": -1, "faces": -1}
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("element vertex "):
                counts["vertices"] = int(line.split()[-1])
            elif line.startswith("element face "):
                counts["faces"] = int(line.split()[-1])
            elif line == "end_header":
                break
    return counts


def count_values(arr: np.ndarray) -> Dict[str, int]:
    values, counts = np.unique(arr, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def max_or_none(values: np.ndarray) -> Optional[float]:
    if values.size == 0:
        return None
    return float(np.max(values))


def mean_or_none(values: np.ndarray) -> Optional[float]:
    if values.size == 0:
        return None
    return float(np.mean(values))


def select_frames(gt: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    frames = list(gt.get("frames", []))
    if FRAME_IDS is not None:
        allowed = {int(v) for v in FRAME_IDS}
        frames = [frame for frame in frames if int(frame["frame_id"]) in allowed]
    if MAX_FRAMES is not None:
        frames = frames[: int(MAX_FRAMES)]
    return frames


def frame_transform_map(frame: Mapping[str, Any], key: str) -> Dict[int, np.ndarray]:
    out = {}
    for item in frame.get("fragments", []):
        out[int(item["fragment_id"])] = np.asarray(item[key], dtype=np.float64)
    return out


def analyze_visible_points(
    visible: Mapping[str, np.ndarray],
    frame: Mapping[str, Any],
    K: np.ndarray,
    face_counts_by_fragment: Mapping[int, int],
) -> Dict[str, Any]:
    labels = visible["surface_label"].astype(np.uint8)
    fragment_ids = visible["fragment_id"].astype(np.int32)
    points_c = visible["points_C"].astype(np.float64)
    points_f = visible["points_F"].astype(np.float64)
    points_o = visible["points_O"].astype(np.float64)
    face_ids = visible["face_id"].astype(np.int64)
    bary = visible["barycentric"].astype(np.float64)
    shell_indices = visible["shell_indices"].astype(np.int64)

    by_fragment_o = frame_transform_map(frame, "T_C_from_O")
    by_fragment_f = frame_transform_map(frame, "T_C_from_F")

    err_o_all = []
    err_f_all = []
    err_o_shell = []
    face_id_errors = 0
    for fragment_id in np.unique(fragment_ids):
        idx = np.where(fragment_ids == int(fragment_id))[0]
        if int(fragment_id) in by_fragment_o:
            pred_c_o = transform_points(by_fragment_o[int(fragment_id)], points_o[idx])
            err = np.linalg.norm(pred_c_o - points_c[idx], axis=1)
            err_o_all.append(err)
            shell_idx = idx[labels[idx] == SURFACE_LABEL_SHELL]
            if shell_idx.size:
                pred_shell = transform_points(by_fragment_o[int(fragment_id)], points_o[shell_idx])
                err_o_shell.append(np.linalg.norm(pred_shell - points_c[shell_idx], axis=1))
        if int(fragment_id) in by_fragment_f:
            pred_c_f = transform_points(by_fragment_f[int(fragment_id)], points_f[idx])
            err_f_all.append(np.linalg.norm(pred_c_f - points_c[idx], axis=1))

        face_count = int(face_counts_by_fragment.get(int(fragment_id), -1))
        if face_count >= 0:
            face_id_errors += int(np.logical_or(face_ids[idx] < 0, face_ids[idx] >= face_count).sum())

    err_o = np.concatenate(err_o_all) if err_o_all else np.zeros((0,), dtype=np.float64)
    err_f = np.concatenate(err_f_all) if err_f_all else np.zeros((0,), dtype=np.float64)
    err_shell = np.concatenate(err_o_shell) if err_o_shell else np.zeros((0,), dtype=np.float64)

    proj_u, proj_v, proj_valid = project_points(points_c, K)
    if proj_valid.any():
        proj_err = np.sqrt((proj_u[proj_valid] - visible["u"][proj_valid]) ** 2 + (proj_v[proj_valid] - visible["v"][proj_valid]) ** 2)
    else:
        proj_err = np.zeros((0,), dtype=np.float64)

    bary_sum_err = np.abs(bary.sum(axis=1) - 1.0) if bary.size else np.zeros((0,), dtype=np.float64)
    bary_oob = np.logical_or(bary < -1e-4, bary > 1.0001).any(axis=1) if bary.size else np.zeros((0,), dtype=bool)

    return {
        "num_visible_points": int(labels.size),
        "num_shell_points": int(shell_indices.size),
        "surface_label_counts": count_values(labels),
        "fragment_id_counts": count_values(fragment_ids),
        "T_C_from_O_error_all_max": max_or_none(err_o),
        "T_C_from_O_error_all_mean": mean_or_none(err_o),
        "T_C_from_O_error_shell_max": max_or_none(err_shell),
        "T_C_from_O_error_shell_mean": mean_or_none(err_shell),
        "T_C_from_F_error_all_max": max_or_none(err_f),
        "T_C_from_F_error_all_mean": mean_or_none(err_f),
        "projection_error_px_max": max_or_none(proj_err),
        "projection_error_px_mean": mean_or_none(proj_err),
        "projection_valid_points": int(proj_valid.sum()),
        "barycentric_sum_error_max": max_or_none(bary_sum_err),
        "barycentric_out_of_bounds": int(bary_oob.sum()),
        "face_id_out_of_range": int(face_id_errors),
        "passes_transform_tolerance": bool(err_shell.size == 0 or float(err_shell.max()) <= TRANSFORM_TOLERANCE),
        "passes_projection_tolerance": bool(proj_err.size == 0 or float(proj_err.max()) <= PROJECTION_TOLERANCE_PX),
    }


def oracle_kabsch_for_frame(visible: Mapping[str, np.ndarray], frame: Mapping[str, Any]) -> Dict[str, Any]:
    if not ORACLE_KABSCH_ENABLE:
        return {
            "enabled": False,
            "num_fragments_total": 0,
            "num_fragments_success": 0,
            "num_fragments_failed": 0,
            "fragments": [],
        }

    labels = visible["surface_label"].astype(np.uint8)
    fragment_ids = visible["fragment_id"].astype(np.int32)
    points_c = visible["points_C"].astype(np.float64)
    points_o = visible["points_O"].astype(np.float64)
    by_fragment_o = frame_transform_map(frame, "T_C_from_O")

    results: List[Dict[str, Any]] = []
    visible_fragment_ids = sorted(int(fragment_id) for fragment_id in np.unique(fragment_ids))
    for fragment_id in visible_fragment_ids:
        idx = np.where(
            (fragment_ids == int(fragment_id))
            & (labels == SURFACE_LABEL_SHELL)
        )[0]
        q_o = points_o[idx]
        p_c = points_c[idx]

        result: Dict[str, Any] = {
            "fragment_id": int(fragment_id),
            "num_shell_points": int(idx.size),
            "success": False,
            "fail_reason": None,
            "residual_mean": None,
            "residual_rmse": None,
            "residual_max": None,
            "residual_p95": None,
            "gt_residual_mean": None,
            "gt_residual_rmse": None,
            "gt_residual_max": None,
            "gt_residual_p95": None,
            "oracle_residual_rmse_minus_gt": None,
            "passes_oracle_vs_gt_residual": False,
            "translation_error_m": None,
            "rotation_error_deg": None,
            "axis_error_deg": None,
            "det_R": None,
            "singular_values": [],
            "rank": 0,
            "passes_residual_tolerance": False,
            "passes_translation_tolerance": False,
            "passes_rotation_tolerance": False,
            "passes_axis_tolerance": False,
            "T_C_from_O_oracle": None,
            "T_C_from_O_gt": None,
        }

        T_gt = by_fragment_o.get(int(fragment_id))
        if T_gt is None:
            result["fail_reason"] = "missing_T_C_from_O_gt"
            results.append(result)
            continue

        result["T_C_from_O_gt"] = [[float(v) for v in row] for row in np.asarray(T_gt, dtype=np.float64).tolist()]
        T_oracle, diagnostic = estimate_rigid_transform_kabsch(q_o, p_c)
        result["success"] = bool(diagnostic.get("success", False))
        result["fail_reason"] = diagnostic.get("fail_reason")
        result["det_R"] = finite_float_or_none(diagnostic.get("det_R"))
        result["singular_values"] = [float(v) for v in diagnostic.get("singular_values", [])]
        result["rank"] = int(diagnostic.get("rank", 0))

        if not result["success"]:
            results.append(result)
            continue

        result["T_C_from_O_oracle"] = [[float(v) for v in row] for row in T_oracle.tolist()]
        oracle_residual = transform_residual_stats(T_oracle, q_o, p_c)
        gt_residual = transform_residual_stats(np.asarray(T_gt, dtype=np.float64), q_o, p_c)

        translation_err = translation_error(T_oracle[:3, 3], np.asarray(T_gt, dtype=np.float64)[:3, 3])
        rotation_err = rotation_error_deg(T_oracle[:3, :3], np.asarray(T_gt, dtype=np.float64)[:3, :3])
        axis_err = symmetry_axis_error_deg(
            T_oracle[:3, :3],
            np.asarray(T_gt, dtype=np.float64)[:3, :3],
            ORACLE_KABSCH_SYMMETRY_AXIS_O,
        )

        residual_rmse = finite_float_or_none(oracle_residual["rmse"])
        gt_residual_rmse = finite_float_or_none(gt_residual["rmse"])
        residual_delta = None
        if residual_rmse is not None and gt_residual_rmse is not None:
            residual_delta = float(residual_rmse - gt_residual_rmse)

        result.update(
            {
                "residual_mean": finite_float_or_none(oracle_residual["mean"]),
                "residual_rmse": residual_rmse,
                "residual_max": finite_float_or_none(oracle_residual["max"]),
                "residual_p95": finite_float_or_none(oracle_residual["p95"]),
                "gt_residual_mean": finite_float_or_none(gt_residual["mean"]),
                "gt_residual_rmse": gt_residual_rmse,
                "gt_residual_max": finite_float_or_none(gt_residual["max"]),
                "gt_residual_p95": finite_float_or_none(gt_residual["p95"]),
                "oracle_residual_rmse_minus_gt": finite_float_or_none(residual_delta),
                "passes_oracle_vs_gt_residual": bool(
                    residual_rmse is not None
                    and gt_residual_rmse is not None
                    and residual_rmse <= gt_residual_rmse + ORACLE_KABSCH_RESIDUAL_TOLERANCE
                ),
                "translation_error_m": finite_float_or_none(translation_err),
                "rotation_error_deg": finite_float_or_none(rotation_err),
                "axis_error_deg": finite_float_or_none(axis_err),
                "passes_residual_tolerance": bool(
                    residual_rmse is not None and residual_rmse <= ORACLE_KABSCH_RESIDUAL_TOLERANCE
                ),
                "passes_translation_tolerance": bool(translation_err <= ORACLE_KABSCH_TRANSLATION_TOLERANCE),
                "passes_rotation_tolerance": bool(rotation_err <= ORACLE_KABSCH_ROTATION_TOLERANCE_DEG),
                "passes_axis_tolerance": bool(axis_err <= ORACLE_KABSCH_ROTATION_TOLERANCE_DEG),
            }
        )
        results.append(result)

    success_results = [item for item in results if item["success"]]
    return {
        "enabled": True,
        "min_points": int(ORACLE_KABSCH_MIN_POINTS),
        "symmetry_axis_O": [float(v) for v in ORACLE_KABSCH_SYMMETRY_AXIS_O.tolist()],
        "num_fragments_total": int(len(results)),
        "num_fragments_success": int(len(success_results)),
        "num_fragments_failed": int(len(results) - len(success_results)),
        "residual_rmse_max": max_optional(item["residual_rmse"] for item in success_results),
        "residual_max_max": max_optional(item["residual_max"] for item in success_results),
        "gt_residual_rmse_max": max_optional(item["gt_residual_rmse"] for item in success_results),
        "gt_residual_max_max": max_optional(item["gt_residual_max"] for item in success_results),
        "translation_error_max_m": max_optional(item["translation_error_m"] for item in success_results),
        "rotation_error_max_deg": max_optional(item["rotation_error_deg"] for item in success_results),
        "axis_error_max_deg": max_optional(item["axis_error_deg"] for item in success_results),
        "oracle_vs_gt_residual_failures": int(
            sum(1 for item in success_results if not item["passes_oracle_vs_gt_residual"])
        ),
        "fragments": results,
    }


def write_oracle_kabsch_clouds(
    out_dir: str,
    stem: str,
    visible: Mapping[str, np.ndarray],
    oracle_stats: Mapping[str, Any],
) -> None:
    if not ORACLE_KABSCH_ENABLE or not ORACLE_KABSCH_WRITE_ALIGNED_CLOUDS:
        return
    if not oracle_stats.get("enabled", False):
        return

    labels = visible["surface_label"].astype(np.uint8)
    fragment_ids = visible["fragment_id"].astype(np.int32)
    points_c = visible["points_C"].astype(np.float64)
    points_o = visible["points_O"].astype(np.float64)

    for item in oracle_stats.get("fragments", []):
        if not item.get("success", False) or item.get("T_C_from_O_oracle") is None:
            continue
        fragment_id = int(item["fragment_id"])
        idx = np.where(
            (fragment_ids == fragment_id)
            & (labels == SURFACE_LABEL_SHELL)
        )[0]
        if idx.size == 0:
            continue

        T_oracle = np.asarray(item["T_C_from_O_oracle"], dtype=np.float64)
        aligned_c = transform_points(T_oracle, points_o[idx])

        green = np.tile(np.array([[40, 255, 40]], dtype=np.uint8), (idx.size, 1))
        white = np.tile(np.array([[255, 255, 255]], dtype=np.uint8), (idx.size, 1))
        write_point_cloud_ply(
            os.path.join(out_dir, "pointclouds", f"{stem}_fragment_{fragment_id:04d}_oracle_aligned_O_to_C.ply"),
            aligned_c,
            green,
        )
        write_point_cloud_ply(
            os.path.join(out_dir, "pointclouds", f"{stem}_fragment_{fragment_id:04d}_shell_points_C.ply"),
            points_c[idx],
            white,
        )


def debug_fragments(scene_dir: str, out_dir: str, fragment_annotations: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    fragment_stats = []
    face_counts_by_fragment: Dict[int, int] = {}
    all_samples_o = []
    all_sample_colors = []

    for frag in fragment_annotations.get("fragments", []):
        fragment_id = int(frag["fragment_id"])
        mesh_path = os.path.join(scene_dir, frag["mesh"])
        labels_path = os.path.join(scene_dir, frag["face_labels"])
        samples_path = os.path.join(scene_dir, frag["samples"])

        ply_counts = read_ply_counts(mesh_path)
        face_counts_by_fragment[fragment_id] = int(ply_counts["faces"])

        face_labels = np.load(labels_path)
        samples = np.load(samples_path)
        sample_labels = samples["surface_label"].astype(np.uint8)
        sample_colors = surface_label_to_rgb(sample_labels)

        if WRITE_FRAGMENT_SAMPLE_CLOUDS:
            write_point_cloud_ply(
                os.path.join(out_dir, "pointclouds", f"fragment_{fragment_id:04d}_samples_O.ply"),
                samples["points_O"],
                sample_colors,
            )
            write_point_cloud_ply(
                os.path.join(out_dir, "pointclouds", f"fragment_{fragment_id:04d}_samples_F.ply"),
                samples["points_F"],
                sample_colors,
            )

        all_samples_o.append(samples["points_O"])
        all_sample_colors.append(sample_colors)

        fragment_stats.append(
            {
                "fragment_id": fragment_id,
                "mesh": frag["mesh"],
                "mesh_vertices_header": int(ply_counts["vertices"]),
                "mesh_faces_header": int(ply_counts["faces"]),
                "num_vertices_annotation": int(frag.get("num_vertices", -1)),
                "num_faces_annotation": int(frag.get("num_faces", -1)),
                "face_labels_shape": list(face_labels.shape),
                "face_labels_counts": count_values(face_labels),
                "face_labels_match_ply_face_count": bool(face_labels.shape[0] == ply_counts["faces"]),
                "samples_points": int(samples["points_O"].shape[0]),
                "samples_label_counts": count_values(sample_labels),
            }
        )

    if WRITE_FRAGMENT_SAMPLE_CLOUDS and all_samples_o:
        write_point_cloud_ply(
            os.path.join(out_dir, "pointclouds", "all_fragment_samples_O.ply"),
            np.concatenate(all_samples_o, axis=0),
            np.concatenate(all_sample_colors, axis=0),
        )

    return fragment_stats, face_counts_by_fragment


def debug_ignored_fragments(scene_dir: str, fragment_annotations: Mapping[str, Any]) -> List[Dict[str, Any]]:
    ignored_stats = []
    for frag in fragment_annotations.get("ignored_fragments", []):
        fragment_id = int(frag["fragment_id"])
        mesh_path = os.path.join(scene_dir, frag["mesh"])
        ply_counts = read_ply_counts(mesh_path)
        ignored_stats.append(
            {
                "fragment_id": fragment_id,
                "mesh": frag["mesh"],
                "mesh_vertices_header": int(ply_counts["vertices"]),
                "mesh_faces_header": int(ply_counts["faces"]),
                "num_vertices_annotation": int(frag.get("num_vertices", -1)),
                "num_faces_annotation": int(frag.get("num_faces", -1)),
                "annotation_status": frag.get("annotation_status", "ignored"),
                "ignore_reason": frag.get("ignore_reason"),
            }
        )
    return ignored_stats


def debug_frame(
    scene_dir: str,
    out_dir: str,
    frame: Mapping[str, Any],
    camera_info: Mapping[str, Any],
    face_counts_by_fragment: Mapping[int, int],
) -> Dict[str, Any]:
    frame_id = int(frame["frame_id"])
    stem = f"frame_{frame_id:06d}"
    frame_dir = os.path.join(out_dir, "frames", stem)
    ensure_dir(frame_dir)

    rgb = read_image_bgr(os.path.join(scene_dir, frame["image"]))
    depth_u16 = read_depth_u16(os.path.join(scene_dir, frame["depth"]))
    instance_mask = read_u8(os.path.join(scene_dir, frame["instance_mask"]))
    surface_mask = read_u8(os.path.join(scene_dir, frame["surface_mask"]))
    visible = np.load(os.path.join(scene_dir, frame["visible_points"]))

    depth_color, depth_stats = colorize_depth(depth_u16, float(camera_info.get("depth_scale", 1.0)))
    depth_overlay = apply_overlay(rgb, depth_color, depth_u16 > 0, DEPTH_OVERLAY_ALPHA)

    instance_color = colorize_instance_mask(instance_mask)
    instance_overlay = apply_overlay(rgb, instance_color, instance_mask > 0, INSTANCE_OVERLAY_ALPHA)

    surface_color = colorize_surface_mask(surface_mask)
    surface_overlay = apply_overlay(rgb, surface_color, surface_mask > 0, SURFACE_OVERLAY_ALPHA)

    visible_overlay = draw_points(
        rgb,
        visible["u"],
        visible["v"],
        visible["surface_label"],
        radius=POINT_RADIUS,
        stride=POINT_DRAW_STRIDE,
    )

    K = np.asarray(camera_info["K"], dtype=np.float64)
    visible_stats = analyze_visible_points(visible, frame, K, face_counts_by_fragment)
    oracle_stats = oracle_kabsch_for_frame(visible, frame)

    save_image(os.path.join(frame_dir, f"{stem}_rgb.png"), rgb)
    save_image(os.path.join(frame_dir, f"{stem}_depth_colormap.png"), depth_color)
    save_image(os.path.join(frame_dir, f"{stem}_depth_overlay.png"), depth_overlay)
    save_image(os.path.join(frame_dir, f"{stem}_instance_overlay.png"), instance_overlay)
    save_image(os.path.join(frame_dir, f"{stem}_surface_overlay.png"), surface_overlay)
    save_image(os.path.join(frame_dir, f"{stem}_visible_points_overlay.png"), visible_overlay)

    sheet = make_grid(
        [
            add_title(rgb, "RGB"),
            add_title(depth_overlay, "Depth overlay"),
            add_title(instance_overlay, "Instance mask overlay"),
            add_title(surface_overlay, "Surface mask overlay"),
            add_title(visible_overlay, "Visible points: green=shell, red=fracture"),
            add_title(depth_color, "Depth colormap"),
        ],
        cols=2,
    )
    save_image(os.path.join(frame_dir, f"{stem}_contact_sheet.png"), sheet)

    if WRITE_POINT_CLOUDS:
        colors_rgb = surface_label_to_rgb(visible["surface_label"].astype(np.uint8))
        write_point_cloud_ply(
            os.path.join(out_dir, "pointclouds", f"{stem}_visible_points_C.ply"),
            visible["points_C"],
            colors_rgb,
        )
        write_point_cloud_ply(
            os.path.join(out_dir, "pointclouds", f"{stem}_visible_points_O.ply"),
            visible["points_O"],
            colors_rgb,
        )

    write_oracle_kabsch_clouds(out_dir, stem, visible, oracle_stats)
    if ORACLE_KABSCH_ENABLE and ORACLE_KABSCH_WRITE_JSON:
        save_json(os.path.join(frame_dir, f"{stem}_oracle_kabsch.json"), oracle_stats)

    surface_counts = count_values(surface_mask)
    instance_counts = count_values(instance_mask)

    frame_stats = {
        "frame_id": frame_id,
        "image": frame["image"],
        "depth": frame["depth"],
        "depth_stats": depth_stats,
        "instance_mask_counts": instance_counts,
        "surface_mask_counts": surface_counts,
        "gt_fragments": frame.get("fragments", []),
        "visible_points": visible_stats,
        "oracle_kabsch": oracle_stats,
    }
    save_json(os.path.join(frame_dir, f"{stem}_summary.json"), frame_stats)
    return frame_stats


def write_text_report(path: str, summary: Mapping[str, Any]) -> None:
    lines = []
    lines.append(f"scene: {summary['scene_id']}")
    lines.append(f"scene_dir: {summary['scene_dir']}")
    lines.append(f"output_dir: {summary['output_dir']}")
    lines.append("")
    lines.append("Fragments:")
    for frag in summary["fragments"]:
        lines.append(
            "  fragment_{fragment_id:04d}: faces={mesh_faces_header}, labels={face_labels_counts}, "
            "samples={samples_points}, labels_match={face_labels_match_ply_face_count}".format(**frag)
        )
    if summary.get("ignored_fragments"):
        lines.append("")
        lines.append("Ignored fragments:")
        for frag in summary["ignored_fragments"]:
            lines.append(
                "  fragment_{fragment_id:04d}: vertices={mesh_vertices_header}, faces={mesh_faces_header}, "
                "reason={ignore_reason}".format(**frag)
            )
    lines.append("")
    lines.append("Frames:")
    for frame in summary["frames"]:
        vp = frame["visible_points"]
        lines.append(
            "  frame_{:06d}: visible={}, shell={}, T_C_from_O_shell_max={}, proj_max_px={}, "
            "surface_mask_counts={}".format(
                int(frame["frame_id"]),
                int(vp["num_visible_points"]),
                int(vp["num_shell_points"]),
                vp["T_C_from_O_error_shell_max"],
                vp["projection_error_px_max"],
                frame["surface_mask_counts"],
            )
        )
        oracle = frame.get("oracle_kabsch", {})
        if oracle.get("enabled", False):
            lines.append(
                "    oracle_kabsch: success={}/{}, rmse_max={}, trans_max={}, "
                "rot_max_deg={}, axis_max_deg={}".format(
                    int(oracle.get("num_fragments_success", 0)),
                    int(oracle.get("num_fragments_total", 0)),
                    oracle.get("residual_rmse_max"),
                    oracle.get("translation_error_max_m"),
                    oracle.get("rotation_error_max_deg"),
                    oracle.get("axis_error_max_deg"),
                )
            )
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    dataset_root = repo_path(DATASET_ROOT)
    split_dir = os.path.join(dataset_root, SPLIT)
    selected_scene = scene_name_from_id(SCENE_ID) or find_latest_scene(split_dir)
    scene_dir = os.path.join(split_dir, selected_scene)
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(scene_dir)

    out_dir = os.path.join(repo_path(OUTPUT_ROOT), f"{SPLIT}_{selected_scene}")
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "frames"))
    ensure_dir(os.path.join(out_dir, "pointclouds"))

    camera_info = load_json(os.path.join(scene_dir, "camera_info.json"))
    gt = load_json(os.path.join(scene_dir, "gt_annotations.json"))
    fragment_annotations = load_json(os.path.join(scene_dir, "fragments", "fragment_annotations.json"))
    scene_meta_path = os.path.join(scene_dir, "scene_meta.json")
    scene_meta = load_json(scene_meta_path) if os.path.isfile(scene_meta_path) else {}

    fragment_stats, face_counts_by_fragment = debug_fragments(scene_dir, out_dir, fragment_annotations)
    ignored_fragment_stats = debug_ignored_fragments(scene_dir, fragment_annotations)

    frame_stats = []
    for frame in select_frames(gt):
        frame_stats.append(debug_frame(scene_dir, out_dir, frame, camera_info, face_counts_by_fragment))

    summary = {
        "scene_id": selected_scene,
        "scene_dir": scene_dir,
        "output_dir": out_dir,
        "camera_info": camera_info,
        "scene_meta": scene_meta,
        "fragment_annotations_geometry": fragment_annotations.get("geometry", {}),
        "surface_labeling": fragment_annotations.get("surface_labeling", {}),
        "fragment_filter": fragment_annotations.get("fragment_filter", {}),
        "fragments": fragment_stats,
        "ignored_fragments": ignored_fragment_stats,
        "frames": frame_stats,
    }
    save_json(os.path.join(out_dir, "summary.json"), summary)
    write_text_report(os.path.join(out_dir, "summary.txt"), summary)

    print(f"Debug scene: {selected_scene}")
    print(f"Debug output: {out_dir}")
    print(f"Frames processed: {len(frame_stats)}")
    print(f"Fragments processed: {len(fragment_stats)}")


if __name__ == "__main__":
    main()
