#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2: reconstruct an effective projector seed from stage1 outputs.

Current policy in this version:
  1) Projector center is fixed internally from the D435 mechanical prior
     in the left IR frame:
         projector_x_from_left_m = 0.029
         projector_y_m = 0.0
         projector_z_m = 0.0
  2) Only projector orientation is tuned via --yaw_deg / --pitch_deg / --roll_deg.
  3) FOV is auto-estimated from sparse projector-frame points by default
     with an additional margin (--fov_margin_frac).
     If --fov_x_deg and --fov_y_deg are both provided, fixed FOV is used instead.
  4) uniform_radius_percentile is user-controlled and currently defaults to 25.
  5) texture_blur_sigma_px is disabled by default (0.0).
  6) merge_radius_px is conservative by default to avoid over-merging.

Conventions:
  - Left camera frame (CV-style): +X right, +Y down, +Z forward.
  - Projector local frame (internal): +x right, +y down, +z forward.
  - Internal UV: image coordinates +u right, +v down, origin top-left.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


DEFAULT_D435_PROJECTOR_X_FROM_LEFT_M = 0.029
DEFAULT_D435_PROJECTOR_Y_M = 0.0
DEFAULT_D435_PROJECTOR_Z_M = 0.0
DEFAULT_D435_FOV_X_DEG = 92.0
DEFAULT_D435_FOV_Y_DEG = 65.0



# ---------------------------- io/utils ----------------------------

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_session_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("session_")])


def resolve_session(root: Path, session: str) -> Path:
    if session == "latest":
        sessions = list_session_dirs(root)
        if not sessions:
            raise FileNotFoundError(f"No session_* dirs found in {root}")
        return sessions[-1]
    p = root / session
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def normalize_u8(img: np.ndarray,
                 percentile_low: float = 0.0,
                 percentile_high: float = 100.0) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.percentile(x, percentile_low))
    hi = float(np.percentile(x, percentile_high))
    if hi <= lo + 1e-12:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - lo) / (hi - lo)
    return np.clip(np.round(255.0 * y), 0, 255).astype(np.uint8)


def robust_stats(x: np.ndarray) -> Dict:
    x = np.asarray(x)
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p5": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


# ---------------------------- sparse geometry ----------------------------

def triangulate_rectified_points_positive_z(left_xy: np.ndarray,
                                            disparity: np.ndarray,
                                            P1: np.ndarray,
                                            baseline_m: float) -> np.ndarray:
    if left_xy.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    fx = float(P1[0, 0])
    fy = float(P1[1, 1])
    cx = float(P1[0, 2])
    cy = float(P1[1, 2])

    d = disparity.astype(np.float64)
    valid = d > 1e-9
    xyz = np.zeros((left_xy.shape[0], 3), dtype=np.float64)

    z = np.zeros_like(d)
    z[valid] = fx * baseline_m / d[valid]
    x = ((left_xy[:, 0].astype(np.float64) - cx) * z) / fx
    y = ((left_xy[:, 1].astype(np.float64) - cy) * z) / fy

    xyz[:, 0] = x
    xyz[:, 1] = y
    xyz[:, 2] = z
    xyz[~valid] = 0.0
    return xyz.astype(np.float32)


def fit_plane_svd(xyz: np.ndarray) -> Dict[str, np.ndarray]:
    if xyz.shape[0] < 3:
        raise ValueError("Need at least 3 points to fit a plane")
    c = np.mean(xyz, axis=0)
    A = xyz - c[None, :]
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    n = vt[-1].astype(np.float64)
    n /= np.linalg.norm(n) + 1e-12
    if n[2] > 0:
        n = -n
    d = -float(n @ c.astype(np.float64))
    signed = xyz.astype(np.float64) @ n + d
    rmse = float(np.sqrt(np.mean(signed ** 2)))
    return {
        "center": c.astype(np.float64),
        "normal": n.astype(np.float64),
        "offset_d": np.array([d], dtype=np.float64),
        "signed_distances": signed.astype(np.float64),
        "rmse": np.array([rmse], dtype=np.float64),
    }


def rotx_deg(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(float(angle_deg))
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=np.float64)


def roty_deg(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(float(angle_deg))
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=np.float64)


def rotz_deg(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(float(angle_deg))
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def apply_local_euler_to_projector_basis(R_pl: np.ndarray,
                                         yaw_deg: float,
                                         pitch_deg: float,
                                         roll_deg: float) -> np.ndarray:
    """
    Apply additional LOCAL rotations to the projector basis.

    Local projector axes:
      x -> right
      y -> down
      z -> forward

    Rotations are applied in the order:
      yaw  around local y (down)
      pitch around local x (right)
      roll around local z (forward)

    Because projector coordinates are computed as p = R_pl @ (X - C),
    a local-frame correction is left-multiplied: R_new = Q @ R_pl.
    """
    Q = rotz_deg(roll_deg) @ rotx_deg(pitch_deg) @ roty_deg(yaw_deg)
    R_new = Q @ R_pl

    # Re-orthonormalize rows for numerical stability.
    right = R_new[0]
    down = R_new[1]
    forward = R_new[2]

    forward /= np.linalg.norm(forward) + 1e-12
    right = right - np.dot(right, forward) * forward
    right /= np.linalg.norm(right) + 1e-12
    down = np.cross(forward, right)
    down /= np.linalg.norm(down) + 1e-12

    return np.stack([right, down, forward], axis=0)


def make_projector_pose_seed(xyz: np.ndarray,
                             yaw_deg: float = 0.0,
                             pitch_deg: float = 0.0,
                             roll_deg: float = 0.0) -> Dict[str, np.ndarray]:
    pts = xyz.astype(np.float64)
    center = np.mean(pts, axis=0)

    C = np.array([
        DEFAULT_D435_PROJECTOR_X_FROM_LEFT_M,
        DEFAULT_D435_PROJECTOR_Y_M,
        DEFAULT_D435_PROJECTOR_Z_M,
    ], dtype=np.float64)

    forward = center - C
    forward /= np.linalg.norm(forward) + 1e-12

    # "up" in left CV world is negative Y, because +Y points down.
    up_ref = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, up_ref)
    if np.linalg.norm(right) < 1e-9:
        up_ref = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        right = np.cross(forward, up_ref)

    right /= np.linalg.norm(right) + 1e-12
    down = np.cross(forward, right)
    down /= np.linalg.norm(down) + 1e-12

    R_pl_seed = np.stack([right, down, forward], axis=0)
    R_pl = apply_local_euler_to_projector_basis(
        R_pl_seed,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
    )

    return {
        "C_left_m": C,
        "R_projector_from_left": R_pl,
        "right_axis_left": R_pl[0],
        "down_axis_left": R_pl[1],
        "forward_axis_left": R_pl[2],
        "cloud_center_left_m": center,
        "seed_forward_axis_left": forward,
        "seed_right_axis_left": right,
        "seed_down_axis_left": down,
    }


def transform_left_to_projector(xyz_left: np.ndarray,
                                C_left: np.ndarray,
                                R_pl: np.ndarray) -> np.ndarray:
    X = xyz_left.astype(np.float64) - C_left.reshape(1, 3)
    return (R_pl @ X.T).T.astype(np.float64)


def compute_fov_from_projector_points(xyz_proj: np.ndarray,
                                      margin_frac: float = 0.06) -> Tuple[float, float]:
    valid = xyz_proj[:, 2] > 1e-9
    pts = xyz_proj[valid]
    if pts.shape[0] == 0:
        raise ValueError("No positive-z points in projector frame")

    ax = np.arctan2(pts[:, 0], pts[:, 2])
    ay = np.arctan2(pts[:, 1], pts[:, 2])

    half_x = max(abs(np.min(ax)), abs(np.max(ax)))
    half_y = max(abs(np.min(ay)), abs(np.max(ay)))
    half_x *= (1.0 + margin_frac)
    half_y *= (1.0 + margin_frac)

    return float(2.0 * half_x), float(2.0 * half_y)


def intrinsics_from_fov(width: int,
                        height: int,
                        fov_x_rad: float,
                        fov_y_rad: float):
    fx = (0.5 * width) / np.tan(0.5 * fov_x_rad)
    fy = (0.5 * height) / np.tan(0.5 * fov_y_rad)
    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K, fx, fy, cx, cy


def project_projector_uv(xyz_proj: np.ndarray,
                         Kp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = xyz_proj[:, 2] > 1e-9
    pts = xyz_proj[valid]
    uv = np.zeros((xyz_proj.shape[0], 2), dtype=np.float64)
    if pts.shape[0] > 0:
        x = pts[:, 0] / pts[:, 2]
        y = pts[:, 1] / pts[:, 2]
        uv_valid = np.stack([
            Kp[0, 0] * x + Kp[0, 2],
            Kp[1, 1] * y + Kp[1, 2],
        ], axis=1)
        uv[valid] = uv_valid
    return uv, valid


def convert_uv_for_texture_export(uv: np.ndarray,
                                  width: int,
                                  height: int,
                                  flip_u: bool = False,
                                  flip_v: bool = False) -> np.ndarray:
    out = np.asarray(uv, dtype=np.float32).copy()
    if out.size == 0:
        return out
    if flip_u:
        out[:, 0] = (float(width) - 1.0) - out[:, 0]
    if flip_v:
        out[:, 1] = (float(height) - 1.0) - out[:, 1]
    return out


# ---------------------------- support-mask components ----------------------------

def load_stage1_support_masks(rect_npz: np.lib.npyio.NpzFile):
    if "rect_support_mask_left" in rect_npz and "rect_support_mask_right" in rect_npz:
        mask_l = rect_npz["rect_support_mask_left"].astype(np.uint8) > 0
        mask_r = rect_npz["rect_support_mask_right"].astype(np.uint8) > 0
    else:
        raise KeyError(
            "rect_support_mask_left/right not found in rectification_data.npz. "
            "Run the updated stage1 first."
        )
    return mask_l, mask_r


def extract_support_components(mask: np.ndarray,
                               score_img: np.ndarray,
                               min_area_px: int = 2,
                               max_area_px: int = 10_000) -> Dict[str, np.ndarray]:
    m = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    score = np.asarray(score_img, dtype=np.float32)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    centers = []
    areas = []
    radii = []
    mean_scores = []
    max_scores = []

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_area_px) or area > int(max_area_px):
            continue

        ys, xs = np.nonzero(labels == i)
        if xs.size == 0:
            continue
        w = np.clip(score[ys, xs].astype(np.float64), 1e-6, None)
        sw = float(np.sum(w))
        cx = float(np.sum(w * xs) / sw)
        cy = float(np.sum(w * ys) / sw)
        r_eq = float(np.sqrt(area / np.pi))

        centers.append([cx, cy])
        areas.append(area)
        radii.append(r_eq)
        mean_scores.append(float(np.mean(score[ys, xs])))
        max_scores.append(float(np.max(score[ys, xs])))

    if not centers:
        z2 = np.zeros((0, 2), dtype=np.float32)
        z1 = np.zeros((0,), dtype=np.float32)
        return {
            "center_xy": z2,
            "area_px": z1,
            "radius_px": z1,
            "mean_score": z1,
            "max_score": z1,
        }

    return {
        "center_xy": np.asarray(centers, dtype=np.float32),
        "area_px": np.asarray(areas, dtype=np.float32),
        "radius_px": np.asarray(radii, dtype=np.float32),
        "mean_score": np.asarray(mean_scores, dtype=np.float32),
        "max_score": np.asarray(max_scores, dtype=np.float32),
    }


# ---------------------------- plane intersection ----------------------------

def pixel_rays_to_plane(points_xy: np.ndarray,
                        P: np.ndarray,
                        origin_left: np.ndarray,
                        plane_n: np.ndarray,
                        plane_d: float) -> Tuple[np.ndarray, np.ndarray]:
    if points_xy.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=bool)

    fx = float(P[0, 0])
    fy = float(P[1, 1])
    cx = float(P[0, 2])
    cy = float(P[1, 2])

    x = (points_xy[:, 0].astype(np.float64) - cx) / fx
    y = (points_xy[:, 1].astype(np.float64) - cy) / fy
    dirs = np.stack([x, y, np.ones_like(x)], axis=1)

    n = plane_n.reshape(3).astype(np.float64)
    O = origin_left.reshape(3).astype(np.float64)

    denom = dirs @ n
    valid = np.abs(denom) > 1e-9
    if not np.any(valid):
        return np.zeros((points_xy.shape[0], 3), dtype=np.float32), valid

    pts = np.zeros((points_xy.shape[0], 3), dtype=np.float64)
    num = -(O @ n + plane_d)
    t = np.zeros((points_xy.shape[0],), dtype=np.float64)
    t[valid] = num / denom[valid]
    valid = valid & (t > 1e-9)
    pts[valid] = O.reshape(1, 3) + dirs[valid] * t[valid, None]
    return pts.astype(np.float32), valid


def estimate_projector_radius_from_component(points_xy: np.ndarray,
                                             radius_px: np.ndarray,
                                             P: np.ndarray,
                                             origin_left: np.ndarray,
                                             plane_n: np.ndarray,
                                             plane_d: float,
                                             C_proj_left: np.ndarray,
                                             R_pl: np.ndarray,
                                             Kp: np.ndarray) -> np.ndarray:
    if points_xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    pts_center, valid_c = pixel_rays_to_plane(points_xy, P, origin_left, plane_n, plane_d)
    pts_x, valid_x = pixel_rays_to_plane(
        np.stack([points_xy[:, 0] + radius_px, points_xy[:, 1]], axis=1),
        P, origin_left, plane_n, plane_d
    )
    pts_y, valid_y = pixel_rays_to_plane(
        np.stack([points_xy[:, 0], points_xy[:, 1] + radius_px], axis=1),
        P, origin_left, plane_n, plane_d
    )

    uv_c, uv_valid_c = project_projector_uv(
        transform_left_to_projector(pts_center, C_proj_left, R_pl), Kp
    )
    uv_x, uv_valid_x = project_projector_uv(
        transform_left_to_projector(pts_x, C_proj_left, R_pl), Kp
    )
    uv_y, uv_valid_y = project_projector_uv(
        transform_left_to_projector(pts_y, C_proj_left, R_pl), Kp
    )

    valid = valid_c & valid_x & valid_y & uv_valid_c & uv_valid_x & uv_valid_y
    r = np.zeros((points_xy.shape[0],), dtype=np.float32)
    if np.any(valid):
        dx = np.linalg.norm(uv_x[valid] - uv_c[valid], axis=1)
        dy = np.linalg.norm(uv_y[valid] - uv_c[valid], axis=1)
        r[valid] = (0.5 * (dx + dy)).astype(np.float32)
    return r


# ---------------------------- clustering / reconstruction ----------------------------

def cluster_uv_points(uv: np.ndarray,
                      radius_uv: np.ndarray,
                      strength: np.ndarray,
                      source_label: np.ndarray,
                      merge_radius_px: float) -> Dict[str, np.ndarray]:
    if uv.shape[0] == 0:
        z2 = np.zeros((0, 2), dtype=np.float32)
        z1 = np.zeros((0,), dtype=np.float32)
        zi = np.zeros((0,), dtype=np.uint8)
        return {
            "uv": z2,
            "radius_uv": z1,
            "strength": z1,
            "source_flags": zi,
            "member_count": z1,
        }

    order = np.argsort(-strength.astype(np.float64))
    used = np.zeros((uv.shape[0],), dtype=bool)

    out_uv = []
    out_r = []
    out_s = []
    out_flag = []
    out_count = []

    mr2 = float(merge_radius_px) ** 2

    for idx in order:
        if used[idx]:
            continue
        d2 = np.sum((uv - uv[idx:idx + 1]) ** 2, axis=1)
        members = np.where((~used) & (d2 <= mr2))[0]
        if members.size == 0:
            continue

        used[members] = True
        w = np.maximum(strength[members].astype(np.float64), 1e-6)
        uv_m = np.sum(uv[members] * w[:, None], axis=0) / np.sum(w)
        r_m = float(np.percentile(radius_uv[members], 90))
        s_m = float(np.sum(strength[members]))
        sources = source_label[members]
        flag = 0
        if np.any(sources == 0):
            flag |= 1
        if np.any(sources == 1):
            flag |= 2

        out_uv.append(uv_m)
        out_r.append(r_m)
        out_s.append(s_m)
        out_flag.append(flag)
        out_count.append(int(members.size))

    return {
        "uv": np.asarray(out_uv, dtype=np.float32),
        "radius_uv": np.asarray(out_r, dtype=np.float32),
        "strength": np.asarray(out_s, dtype=np.float32),
        "source_flags": np.asarray(out_flag, dtype=np.uint8),
        "member_count": np.asarray(out_count, dtype=np.int32),
    }


def suppress_intersections_keep_radius(uv: np.ndarray,
                                       radius_px: np.ndarray,
                                       strength: np.ndarray,
                                       source_flags: np.ndarray,
                                       member_count: np.ndarray,
                                       min_gap_px: float = 0.0) -> Dict[str, np.ndarray]:
    if uv.shape[0] == 0:
        z2 = np.zeros((0, 2), dtype=np.float32)
        z1 = np.zeros((0,), dtype=np.float32)
        zi = np.zeros((0,), dtype=np.uint8)
        return {
            "uv": z2,
            "radius_px": z1,
            "strength": z1,
            "source_flags": zi,
            "member_count": np.zeros((0,), dtype=np.int32),
            "kept_indices": np.zeros((0,), dtype=np.int32),
        }

    is_shared = (source_flags == 3).astype(np.int32)
    order = np.lexsort((
        -strength.astype(np.float64),
        -member_count.astype(np.float64),
        -is_shared.astype(np.float64),
    ))[::-1]

    keep = []
    for idx in order:
        ui = uv[idx].astype(np.float64)
        ri = float(radius_px[idx])
        ok = True
        for j in keep:
            uj = uv[j].astype(np.float64)
            rj = float(radius_px[j])
            d = np.linalg.norm(ui - uj)
            if d < (ri + rj + float(min_gap_px)):
                ok = False
                break
        if ok:
            keep.append(int(idx))

    keep = np.asarray(keep, dtype=np.int32)
    return {
        "uv": uv[keep].astype(np.float32),
        "radius_px": radius_px[keep].astype(np.float32),
        "strength": strength[keep].astype(np.float32),
        "source_flags": source_flags[keep].astype(np.uint8),
        "member_count": member_count[keep].astype(np.int32),
        "kept_indices": keep,
    }


# ---------------------------- rasterization ----------------------------

def rasterize_weighted_dot_texture(width: int,
                                   height: int,
                                   uv: np.ndarray,
                                   radius_px: np.ndarray,
                                   strength: np.ndarray,
                                   blur_sigma_px: float = 0.0) -> np.ndarray:
    img = np.zeros((height, width), dtype=np.float32)
    if uv.shape[0] == 0:
        return img

    max_strength = max(float(np.max(strength)), 1e-6)
    amp = np.clip(strength / max_strength, 0.25, 1.0).astype(np.float32)

    for (u, v), r, a in zip(uv, radius_px, amp):
        rr = max(1, int(round(float(r))))
        uc = int(round(float(u)))
        vc = int(round(float(v)))
        if uc < -rr or uc >= width + rr or vc < -rr or vc >= height + rr:
            continue
        cv2.circle(img, (uc, vc), rr, float(a), thickness=-1, lineType=cv2.LINE_AA)

    if blur_sigma_px > 1e-8:
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=float(blur_sigma_px), sigmaY=float(blur_sigma_px))
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def rasterize_uniform_dot_texture(width: int,
                                  height: int,
                                  uv: np.ndarray,
                                  radius_px: np.ndarray,
                                  blur_sigma_px: float = 0.0) -> np.ndarray:
    img = np.zeros((height, width), dtype=np.float32)
    if uv.shape[0] == 0:
        return img

    for (u, v), r in zip(uv, radius_px):
        rr = max(1, int(round(float(r))))
        uc = int(round(float(u)))
        vc = int(round(float(v)))
        if uc < -rr or uc >= width + rr or vc < -rr or vc >= height + rr:
            continue
        cv2.circle(img, (uc, vc), rr, 1.0, thickness=-1, lineType=cv2.LINE_AA)

    if blur_sigma_px > 1e-8:
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=float(blur_sigma_px), sigmaY=float(blur_sigma_px))
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def rasterize_support_map(width: int,
                          height: int,
                          uv: np.ndarray,
                          radius_px: np.ndarray) -> np.ndarray:
    img = np.zeros((height, width), dtype=np.float32)
    for (u, v), r in zip(uv, radius_px):
        rr = max(1, int(round(float(r))))
        uc = int(round(float(u)))
        vc = int(round(float(v)))
        if uc < -rr or uc >= width + rr or vc < -rr or vc >= height + rr:
            continue
        cv2.circle(img, (uc, vc), rr, 1.0, thickness=-1, lineType=cv2.LINE_AA)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def overlay_uv_points(texture_u8: np.ndarray,
                      uv: np.ndarray,
                      color=(0, 255, 0)) -> np.ndarray:
    vis = cv2.cvtColor(texture_u8, cv2.COLOR_GRAY2BGR)
    for u, v in uv:
        cv2.circle(vis, (int(round(float(u))), int(round(float(v)))), 1, color, -1, lineType=cv2.LINE_AA)
    return vis


# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 2 projector seed from stage1 support masks + sparse geometry")
    ap.add_argument("--root", type=str, default="wall_capture", help="Root wall_capture directory")
    ap.add_argument("--session", type=str, default="latest", help="session_* dir name or 'latest'")
    ap.add_argument("--stage1_name", type=str, default="projector_stage1", help="Input stage1 subdir")
    ap.add_argument("--out_name", type=str, default="projector_stage2", help="Output subdir inside session")

    ap.add_argument("--projector_width", type=int, default=4096, help="Projector texture width")
    ap.add_argument("--projector_height", type=int, default=2048, help="Projector texture height")

    ap.add_argument("--yaw_deg", type=float, default=1.75,
                    help="Additional local yaw around projector +down axis [deg]")
    ap.add_argument("--pitch_deg", type=float, default=0.0,
                    help="Additional local pitch around projector +right axis [deg]")
    ap.add_argument("--roll_deg", type=float, default=0.0,
                    help="Additional local roll around projector +forward axis [deg]")

    ap.add_argument("--fov_margin_frac", type=float, default=0.06,
                    help="Extra margin around observed sparse angular span when FOV is auto")
    ap.add_argument("--fov_x_deg", type=float, default=None,
                    help="Horizontal projector FOV in degrees. If omitted together with --fov_y_deg, auto-FOV from sparse points is used.")
    ap.add_argument("--fov_y_deg", type=float, default=None,
                    help="Vertical projector FOV in degrees. If omitted together with --fov_x_deg, auto-FOV from sparse points is used.")

    ap.add_argument("--component_min_area_px", type=int, default=2,
                    help="Minimum support-mask component area to keep")
    ap.add_argument("--component_max_area_px", type=int, default=500,
                    help="Maximum support-mask component area to keep")
    ap.add_argument("--merge_radius_px", type=float, default=1.0,
                    help="UV-space merge radius for left/right dot fusion. Conservative default to avoid over-merging.")
    ap.add_argument("--uniform_radius_percentile", type=float, default=25.0,
                    help="Percentile of projected UV component radii used as common dot radius")
    ap.add_argument("--min_circle_gap_px", type=float, default=0.0,
                    help="Optional additional gap between circles after non-overlap filtering")
    ap.add_argument("--texture_blur_sigma_px", type=float, default=0.0,
                    help="Optional blur after rasterizing textures. Default: disabled.")
    ap.add_argument("--preview_percentile_high", type=float, default=99.9,
                    help="High percentile for weighted texture preview normalization")

    ap.add_argument("--export_flip_u", action="store_true",
                    help="Flip U when exporting projector texture PNGs")
    ap.add_argument("--export_flip_v", dest="export_flip_v", action="store_true", default=False,
                    help="Flip V when exporting projector texture PNGs")
    ap.add_argument("--no_export_flip_v", dest="export_flip_v", action="store_false",
                    help="Do not flip V during texture export (default)")

    args = ap.parse_args()

    root = Path(args.root)
    session_dir = resolve_session(root, args.session)
    stage1_dir = session_dir / args.stage1_name
    stage1_summary_path = stage1_dir / "stage1_summary.json"
    if not stage1_summary_path.exists():
        raise FileNotFoundError(stage1_summary_path)

    summary = load_json(stage1_summary_path)
    rect_npz_path = session_dir / summary["next_stage_inputs"]["rectification_data_npz"]
    match_npz_path = session_dir / summary["next_stage_inputs"]["matched_points_rectified_npz"]

    rect_npz = np.load(str(rect_npz_path))
    match_npz = np.load(str(match_npz_path))

    P1 = rect_npz["P1"].astype(np.float64)
    P2 = rect_npz["P2"].astype(np.float64)

    rect_left = rect_npz["rect_left"].astype(np.float32)
    rect_right = rect_npz["rect_right"].astype(np.float32)
    rect_lcn_left = rect_npz["rect_lcn_left"].astype(np.float32)
    rect_lcn_right = rect_npz["rect_lcn_right"].astype(np.float32)

    rect_support_left, rect_support_right = load_stage1_support_masks(rect_npz)

    left_xy = match_npz["left_xy"].astype(np.float64)
    right_xy = match_npz["right_xy"].astype(np.float64)
    disparity = match_npz["disparity"].astype(np.float64)

    baseline_m = float(summary["camera_and_plane"]["baseline_m"])
    plane_z_mm = float(summary["camera_and_plane"]["plane_z_mm"])
    expected_disp = float(summary["camera_and_plane"]["expected_disparity_px_from_plane_z"])

    out_dir = session_dir / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz_all = triangulate_rectified_points_positive_z(left_xy, disparity, P1, baseline_m)
    valid_xyz = xyz_all[:, 2] > 1e-9
    xyz_sparse = xyz_all[valid_xyz]
    left_xy = left_xy[valid_xyz]
    right_xy = right_xy[valid_xyz]
    disparity = disparity[valid_xyz]

    if xyz_sparse.shape[0] < 3:
        raise RuntimeError("Too few valid sparse stereo points after triangulation.")

    plane = fit_plane_svd(xyz_sparse)

    projector_pose = make_projector_pose_seed(
        xyz=xyz_sparse,
        yaw_deg=float(args.yaw_deg),
        pitch_deg=float(args.pitch_deg),
        roll_deg=float(args.roll_deg),
    )

    xyz_proj_sparse = transform_left_to_projector(
        xyz_left=xyz_sparse,
        C_left=projector_pose["C_left_m"],
        R_pl=projector_pose["R_projector_from_left"],
    )

    auto_fov_x_rad, auto_fov_y_rad = compute_fov_from_projector_points(
        xyz_proj_sparse,
        margin_frac=float(args.fov_margin_frac),
    )

    if (args.fov_x_deg is None) != (args.fov_y_deg is None):
        raise ValueError("Specify both --fov_x_deg and --fov_y_deg, or neither to use auto-FOV.")

    if args.fov_x_deg is None and args.fov_y_deg is None:
        fov_mode = "auto_from_sparse"
        fov_x_rad = auto_fov_x_rad
        fov_y_rad = auto_fov_y_rad
    else:
        fov_mode = "fixed_cli"
        fov_x_rad = np.deg2rad(float(args.fov_x_deg))
        fov_y_rad = np.deg2rad(float(args.fov_y_deg))

    Kp, fxp, fyp, cxp, cyp = intrinsics_from_fov(
        args.projector_width,
        args.projector_height,
        fov_x_rad,
        fov_y_rad,
    )

    comps_left = extract_support_components(
        rect_support_left,
        rect_left,
        min_area_px=int(args.component_min_area_px),
        max_area_px=int(args.component_max_area_px),
    )
    comps_right = extract_support_components(
        rect_support_right,
        rect_right,
        min_area_px=int(args.component_min_area_px),
        max_area_px=int(args.component_max_area_px),
    )

    plane_n = plane["normal"].reshape(3)
    plane_d = float(plane["offset_d"][0])

    O_left = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    O_right = np.array([baseline_m, 0.0, 0.0], dtype=np.float64)

    xyz_left_support, valid_xyz_left = pixel_rays_to_plane(
        comps_left["center_xy"], P1, O_left, plane_n, plane_d
    )
    xyz_right_support, valid_xyz_right = pixel_rays_to_plane(
        comps_right["center_xy"], P2, O_right, plane_n, plane_d
    )

    comps_left_xy = comps_left["center_xy"][valid_xyz_left]
    comps_left_r = comps_left["radius_px"][valid_xyz_left]
    comps_left_s = comps_left["mean_score"][valid_xyz_left]
    xyz_left_support = xyz_left_support[valid_xyz_left]

    comps_right_xy = comps_right["center_xy"][valid_xyz_right]
    comps_right_r = comps_right["radius_px"][valid_xyz_right]
    comps_right_s = comps_right["mean_score"][valid_xyz_right]
    xyz_right_support = xyz_right_support[valid_xyz_right]

    xyz_left_support_proj = transform_left_to_projector(
        xyz_left=xyz_left_support,
        C_left=projector_pose["C_left_m"],
        R_pl=projector_pose["R_projector_from_left"],
    )
    xyz_right_support_proj = transform_left_to_projector(
        xyz_left=xyz_right_support,
        C_left=projector_pose["C_left_m"],
        R_pl=projector_pose["R_projector_from_left"],
    )

    uv_left_cv, valid_uv_left = project_projector_uv(xyz_left_support_proj, Kp)
    uv_right_cv, valid_uv_right = project_projector_uv(xyz_right_support_proj, Kp)

    uv_left_cv = uv_left_cv[valid_uv_left]
    uv_right_cv = uv_right_cv[valid_uv_right]
    comps_left_xy = comps_left_xy[valid_uv_left]
    comps_right_xy = comps_right_xy[valid_uv_right]
    comps_left_r = comps_left_r[valid_uv_left]
    comps_right_r = comps_right_r[valid_uv_right]
    comps_left_s = comps_left_s[valid_uv_left]
    comps_right_s = comps_right_s[valid_uv_right]
    xyz_left_support = xyz_left_support[valid_uv_left]
    xyz_right_support = xyz_right_support[valid_uv_right]

    radius_uv_left = estimate_projector_radius_from_component(
        comps_left_xy, comps_left_r, P1, O_left, plane_n, plane_d,
        projector_pose["C_left_m"], projector_pose["R_projector_from_left"], Kp
    )[valid_uv_left]
    radius_uv_right = estimate_projector_radius_from_component(
        comps_right_xy, comps_right_r, P2, O_right, plane_n, plane_d,
        projector_pose["C_left_m"], projector_pose["R_projector_from_left"], Kp
    )[valid_uv_right]

    valid_r_left = radius_uv_left > 1e-6
    valid_r_right = radius_uv_right > 1e-6
    all_r = np.concatenate([radius_uv_left[valid_r_left], radius_uv_right[valid_r_right]], axis=0)
    if all_r.size == 0:
        uniform_radius = 2.0
    else:
        uniform_radius = float(np.percentile(all_r, float(args.uniform_radius_percentile)))
        uniform_radius = max(1.0, uniform_radius)

    uv_all_cv = np.concatenate([uv_left_cv, uv_right_cv], axis=0)
    r_all = np.concatenate([radius_uv_left, radius_uv_right], axis=0)
    s_all = np.concatenate([comps_left_s, comps_right_s], axis=0)
    src_all = np.concatenate([
        np.zeros((uv_left_cv.shape[0],), dtype=np.uint8),
        np.ones((uv_right_cv.shape[0],), dtype=np.uint8),
    ], axis=0)

    merged = cluster_uv_points(
        uv=uv_all_cv,
        radius_uv=r_all,
        strength=s_all,
        source_label=src_all,
        merge_radius_px=float(args.merge_radius_px),
    )

    merged_uv_cv = merged["uv"]
    merged_strength = merged["strength"]
    merged_source_flags = merged["source_flags"]
    merged_radius = np.full((merged_uv_cv.shape[0],), float(uniform_radius), dtype=np.float32)

    filtered = suppress_intersections_keep_radius(
        merged_uv_cv,
        merged_radius,
        merged_strength,
        merged_source_flags,
        merged["member_count"],
        min_gap_px=float(args.min_circle_gap_px),
    )

    final_uv_cv = filtered["uv"]
    final_strength = filtered["strength"]
    final_source_flags = filtered["source_flags"]
    final_radius = filtered["radius_px"]
    final_member_count = filtered["member_count"]

    final_uv_tex = convert_uv_for_texture_export(
        final_uv_cv,
        width=args.projector_width,
        height=args.projector_height,
        flip_u=bool(args.export_flip_u),
        flip_v=bool(args.export_flip_v),
    )

    is_left_only = final_source_flags == 1
    is_right_only = final_source_flags == 2
    is_shared = final_source_flags == 3

    texture_weighted = rasterize_weighted_dot_texture(
        args.projector_width,
        args.projector_height,
        final_uv_tex,
        final_radius,
        final_strength,
        blur_sigma_px=float(args.texture_blur_sigma_px),
    )
    texture_uniform = rasterize_uniform_dot_texture(
        args.projector_width,
        args.projector_height,
        final_uv_tex,
        final_radius,
        blur_sigma_px=float(args.texture_blur_sigma_px),
    )

    texture_weighted_u8 = normalize_u8(texture_weighted, 0.0, float(args.preview_percentile_high))
    texture_uniform_u8 = normalize_u8(texture_uniform, 0.0, 100.0)

    support_shared = rasterize_support_map(
        args.projector_width, args.projector_height,
        final_uv_tex[is_shared], final_radius[is_shared]
    )
    support_left_only = rasterize_support_map(
        args.projector_width, args.projector_height,
        final_uv_tex[is_left_only], final_radius[is_left_only]
    )
    support_right_only = rasterize_support_map(
        args.projector_width, args.projector_height,
        final_uv_tex[is_right_only], final_radius[is_right_only]
    )

    texture_debug = overlay_uv_points(texture_uniform_u8, final_uv_tex)

    cv2.imwrite(str(out_dir / "projector_texture.png"), texture_uniform_u8)
    cv2.imwrite(str(out_dir / "projector_texture_uniform.png"), texture_uniform_u8)
    cv2.imwrite(str(out_dir / "projector_texture_weighted.png"), texture_weighted_u8)
    cv2.imwrite(str(out_dir / "projector_texture_debug.png"), texture_debug)
    cv2.imwrite(str(out_dir / "projector_support_shared.png"), normalize_u8(support_shared))
    cv2.imwrite(str(out_dir / "projector_support_left_only.png"), normalize_u8(support_left_only))
    cv2.imwrite(str(out_dir / "projector_support_right_only.png"), normalize_u8(support_right_only))
    cv2.imwrite(str(out_dir / "rect_left_reference.png"), normalize_u8(rect_lcn_left))
    cv2.imwrite(str(out_dir / "rect_right_reference.png"), normalize_u8(rect_lcn_right))

    np.savez_compressed(
        str(out_dir / "projector_points_uv.npz"),
        sparse_left_xy=left_xy.astype(np.float32),
        sparse_right_xy=right_xy.astype(np.float32),
        sparse_disparity=disparity.astype(np.float32),
        sparse_xyz_left=xyz_sparse.astype(np.float32),

        left_component_xy=comps_left_xy.astype(np.float32),
        right_component_xy=comps_right_xy.astype(np.float32),
        left_component_uv_cv=uv_left_cv.astype(np.float32),
        right_component_uv_cv=uv_right_cv.astype(np.float32),
        left_component_radius_uv=radius_uv_left.astype(np.float32),
        right_component_radius_uv=radius_uv_right.astype(np.float32),
        left_component_strength=comps_left_s.astype(np.float32),
        right_component_strength=comps_right_s.astype(np.float32),

        merged_uv_pre_cv=merged_uv_cv.astype(np.float32),
        merged_radius_uv_pre=merged_radius.astype(np.float32),
        merged_strength_pre=merged_strength.astype(np.float32),
        merged_source_flags_pre=merged_source_flags.astype(np.uint8),
        merged_member_count_pre=merged["member_count"].astype(np.int32),

        merged_uv_cv=final_uv_cv.astype(np.float32),
        merged_uv_texture=final_uv_tex.astype(np.float32),
        merged_radius_uv=final_radius.astype(np.float32),
        merged_strength=final_strength.astype(np.float32),
        merged_source_flags=final_source_flags.astype(np.uint8),
        merged_member_count=final_member_count.astype(np.int32),
    )

    np.savez_compressed(
        str(out_dir / "plane_points_xyz.npz"),
        sparse_xyz_left=xyz_sparse.astype(np.float32),
        left_support_xyz_left=xyz_left_support.astype(np.float32),
        right_support_xyz_left=xyz_right_support.astype(np.float32),
        plane_center=plane["center"].astype(np.float32),
        plane_normal=plane["normal"].astype(np.float32),
        signed_distances=plane["signed_distances"].astype(np.float32),
    )

    seed = {
        "session_dir": str(session_dir),
        "source_stage1_summary": str(stage1_summary_path.relative_to(session_dir)),
        "method": "sparse_geometry_support_mask_uv_merge_single_plane_fixed_projector_pose_seed",
        "notes": [
            "Projector position is fixed internally from the D435 mechanical prior.",
            "Projector base orientation looks at the sparse point cloud center.",
            "Optional local yaw/pitch/roll corrections are applied on top of that seed orientation.",
            "FOV is auto-estimated from sparse projector-frame points by default.",
            "Fixed FOV can still be forced by passing both --fov_x_deg and --fov_y_deg.",
        ],
        "inputs": {
            "baseline_m": baseline_m,
            "plane_z_mm_from_stage1": plane_z_mm,
            "expected_disparity_px": expected_disp,
            "measured_disparity_stats_px": summary["matches"]["measured_disparity_stats_px"],
            "sparse_matched_count": int(xyz_sparse.shape[0]),
            "left_support_component_count": int(uv_left_cv.shape[0]),
            "right_support_component_count": int(uv_right_cv.shape[0]),
            "merged_projector_dot_count_pre": int(merged_uv_cv.shape[0]),
            "merged_projector_dot_count_final": int(final_uv_cv.shape[0]),
        },
        "projector_image": {
            "width": int(args.projector_width),
            "height": int(args.projector_height),
            "texture_path": str((out_dir / "projector_texture.png").relative_to(session_dir)),
            "texture_uniform_path": str((out_dir / "projector_texture_uniform.png").relative_to(session_dir)),
            "texture_weighted_path": str((out_dir / "projector_texture_weighted.png").relative_to(session_dir)),
            "texture_debug_path": str((out_dir / "projector_texture_debug.png").relative_to(session_dir)),
            "support_shared_path": str((out_dir / "projector_support_shared.png").relative_to(session_dir)),
            "support_left_only_path": str((out_dir / "projector_support_left_only.png").relative_to(session_dir)),
            "support_right_only_path": str((out_dir / "projector_support_right_only.png").relative_to(session_dir)),
            "default_texture_variant": "uniform",
            "uniform_dot_radius_uv_px": float(uniform_radius),
            "merge_radius_px": float(args.merge_radius_px),
            "min_circle_gap_px": float(args.min_circle_gap_px),
        },
        "texture_coordinate_convention": {
            "internal_uv": "cv_image_coords_x_right_y_down_origin_top_left",
            "exported_texture_uv": "png_export_coords_after_optional_flip",
            "export_flip_u": bool(args.export_flip_u),
            "export_flip_v": bool(args.export_flip_v),
        },
        "projector_intrinsics": {
            "fov_mode": fov_mode,
            "K": Kp.tolist(),
            "fx": float(fxp),
            "fy": float(fyp),
            "cx": float(cxp),
            "cy": float(cyp),
            "fov_x_deg": float(np.degrees(fov_x_rad)),
            "fov_y_deg": float(np.degrees(fov_y_rad)),
            "auto_fov_x_deg": None if auto_fov_x_rad is None else float(np.degrees(auto_fov_x_rad)),
            "auto_fov_y_deg": None if auto_fov_y_rad is None else float(np.degrees(auto_fov_y_rad)),
            "fov_margin_frac": float(args.fov_margin_frac),
        },
        "projector_pose_in_left_frame": {
            "C_left_m": projector_pose["C_left_m"].tolist(),
            "R_projector_from_left": projector_pose["R_projector_from_left"].tolist(),
            "right_axis_left": projector_pose["right_axis_left"].tolist(),
            "down_axis_left": projector_pose["down_axis_left"].tolist(),
            "forward_axis_left": projector_pose["forward_axis_left"].tolist(),
            "seed_right_axis_left": projector_pose["seed_right_axis_left"].tolist(),
            "seed_down_axis_left": projector_pose["seed_down_axis_left"].tolist(),
            "seed_forward_axis_left": projector_pose["seed_forward_axis_left"].tolist(),
            "cloud_center_left_m": projector_pose["cloud_center_left_m"].tolist(),
            "projector_x_from_left_m": float(DEFAULT_D435_PROJECTOR_X_FROM_LEFT_M),
            "projector_y_m": float(DEFAULT_D435_PROJECTOR_Y_M),
            "projector_z_m": float(DEFAULT_D435_PROJECTOR_Z_M),
            "projector_x_relative_to_right_m": float(DEFAULT_D435_PROJECTOR_X_FROM_LEFT_M - baseline_m),
            "yaw_deg": float(args.yaw_deg),
            "pitch_deg": float(args.pitch_deg),
            "roll_deg": float(args.roll_deg),
        },
        "observed_plane_in_left_frame": {
            "center_m": plane["center"].tolist(),
            "normal": plane["normal"].tolist(),
            "rmse_m": float(plane["rmse"][0]),
            "signed_distance_stats_m": robust_stats(plane["signed_distances"]),
            "sparse_xyz_stats": {
                "x_m": robust_stats(xyz_sparse[:, 0]),
                "y_m": robust_stats(xyz_sparse[:, 1]),
                "z_m": robust_stats(xyz_sparse[:, 2]),
            },
        },
        "component_reconstruction": {
            "left_radius_uv_stats": robust_stats(radius_uv_left),
            "right_radius_uv_stats": robust_stats(radius_uv_right),
            "merged_member_count_stats_pre": robust_stats(merged["member_count"]),
            "merged_strength_stats_pre": robust_stats(merged_strength),
            "final_member_count_stats": robust_stats(final_member_count),
            "final_strength_stats": robust_stats(final_strength),
        },
    }
    save_json(out_dir / "projector_seed.json", seed)

    print("Done.")
    print(f"Session                    : {session_dir}")
    print(f"Output                     : {out_dir}")
    print(f"Sparse matched points      : {xyz_sparse.shape[0]}")
    print(f"Left support components    : {uv_left_cv.shape[0]}")
    print(f"Right support components   : {uv_right_cv.shape[0]}")
    print(f"Merged dots pre/final      : {merged_uv_cv.shape[0]} / {final_uv_cv.shape[0]}")
    print(f"Shared / L-only / R-only   : {np.count_nonzero(is_shared)} / {np.count_nonzero(is_left_only)} / {np.count_nonzero(is_right_only)}")
    print(f"Plane rmse [mm]            : {1000.0 * float(plane['rmse'][0]):.3f}")
    print(f"Projector center [m]       : {projector_pose['C_left_m'].tolist()}")
    print(f"Angles yaw/pitch/roll [deg]: {args.yaw_deg:.4f}, {args.pitch_deg:.4f}, {args.roll_deg:.4f}")
    print(f"FOV mode                   : {fov_mode}")
    print(f"FOV x/y [deg]              : {np.degrees(fov_x_rad):.3f}, {np.degrees(fov_y_rad):.3f}")
    print(f"Uniform dot radius [px]    : {uniform_radius:.3f}")
    print(f"Export flip U / V          : {bool(args.export_flip_u)} / {bool(args.export_flip_v)}")
    print(f"Default texture            : {out_dir / 'projector_texture.png'}")


if __name__ == "__main__":
    main()
