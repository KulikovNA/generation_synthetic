#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1: prepare stereo observations for projector reconstruction / fitting.

This version keeps the old sparse branch:
    diff -> rectification -> peaks -> sparse stereo matches -> sparse triangulation

and adds a new dense branch:
    diff/lcn -> dense seed/support/mask/weight maps
    both in original and rectified image domains

The goal is:
1) keep reliable sparse correspondences for projector geometry fitting;
2) preserve dense observation fields for later dense projector-texture accumulation.

Place this script next to wall_capture/ and run it from the tools directory.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


# ---------------------------- utils ----------------------------

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


def load_gray_stack(paths: List[Path]) -> np.ndarray:
    arrs = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        if img.ndim != 2:
            raise RuntimeError(f"Expected grayscale image, got shape {img.shape} for {p}")
        arrs.append(img)
    return np.stack(arrs, axis=0)


def median_uint8_stack(stack: np.ndarray) -> np.ndarray:
    if stack.dtype != np.uint8:
        raise ValueError("median_uint8_stack expects uint8 stack")
    n = stack.shape[0]
    k = n // 2
    part_hi = np.partition(stack, k, axis=0)
    if n % 2 == 1:
        med = part_hi[k]
    else:
        part_lo = np.partition(stack, k - 1, axis=0)
        med = ((part_hi[k].astype(np.uint16) + part_lo[k - 1].astype(np.uint16)) // 2).astype(np.uint8)
    return med


def compute_mean_and_median(dir_path: Path) -> Tuple[np.ndarray, np.ndarray, List[Path]]:
    paths = sorted(dir_path.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNGs in {dir_path}")
    stack = load_gray_stack(paths)
    mean_img = np.mean(stack.astype(np.float32), axis=0)
    if stack.dtype == np.uint8:
        med_img = median_uint8_stack(stack)
    else:
        med_img = np.median(stack, axis=0)
    return mean_img, med_img, paths


def normalize_u8(img: np.ndarray, percentile_low: float = 0.0, percentile_high: float = 100.0) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.percentile(x, percentile_low))
    hi = float(np.percentile(x, percentile_high))
    if hi <= lo + 1e-12:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - lo) / (hi - lo)
    return np.clip(np.round(255.0 * y), 0, 255).astype(np.uint8)


def ensure_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return normalize_u8(img)


def normalize_positive_01(img: np.ndarray, percentile_high: float = 99.5) -> np.ndarray:
    x = np.clip(np.asarray(img, dtype=np.float32), 0.0, None)
    pos = x[x > 0]
    if pos.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    hi = float(np.percentile(pos, percentile_high))
    if hi <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    y = x / hi
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def binary_to_u8(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask, dtype=np.uint8) * 255).astype(np.uint8)


def parse_intrinsics(d: Dict) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    K = np.array([[float(d["fx"]), 0.0, float(d["ppx"])],
                  [0.0, float(d["fy"]), float(d["ppy"])],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    coeffs = np.array(d.get("coeffs", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1)
    size = (int(d["width"]), int(d["height"]))
    return K, coeffs, size


def parse_extrinsics_ir_right_to_left(d: Dict) -> Tuple[np.ndarray, np.ndarray]:
    if "rotation_row_major_3x3" in d:
        R = np.array(d["rotation_row_major_3x3"], dtype=np.float64).reshape(3, 3)
        t = np.array(d["translation_m"], dtype=np.float64).reshape(3, 1)
        return R, t
    if "T_target_from_source_4x4" in d:
        T = np.array(d["T_target_from_source_4x4"], dtype=np.float64).reshape(4, 4)
        return T[:3, :3].copy(), T[:3, 3:4].copy()
    raise KeyError("Unsupported extrinsics format in capture_meta.json")


def build_pattern_diff(med_on: np.ndarray, med_off: np.ndarray) -> np.ndarray:
    diff = med_on.astype(np.float32) - med_off.astype(np.float32)
    return np.clip(diff, 0.0, None)


def build_lcn(img: np.ndarray, small_sigma: float = 1.0, large_sigma: float = 12.0) -> np.ndarray:
    x = img.astype(np.float32)
    if small_sigma > 0:
        x = cv2.GaussianBlur(x, (0, 0), small_sigma)
    local_mean = cv2.GaussianBlur(x, (0, 0), large_sigma)
    local_var = cv2.GaussianBlur((x - local_mean) ** 2, (0, 0), large_sigma)
    local_std = np.sqrt(np.maximum(local_var, 1e-6))
    y = (x - local_mean) / local_std
    y = np.clip(y, 0.0, None)
    return y


def rectify_pair(diff_left: np.ndarray,
                 diff_right: np.ndarray,
                 K1: np.ndarray,
                 D1: np.ndarray,
                 K2: np.ndarray,
                 D2: np.ndarray,
                 R: np.ndarray,
                 t: np.ndarray):
    h, w = diff_left.shape[:2]
    image_size = (w, h)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0
    )
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    rect_left = cv2.remap(diff_left.astype(np.float32), map1x, map1y, interpolation=cv2.INTER_LINEAR)
    rect_right = cv2.remap(diff_right.astype(np.float32), map2x, map2y, interpolation=cv2.INTER_LINEAR)

    return {
        "rect_left": rect_left,
        "rect_right": rect_right,
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
        "roi1": roi1, "roi2": roi2,
        "map1x": map1x, "map1y": map1y, "map2x": map2x, "map2y": map2y,
    }


def detect_peaks_from_pattern(img: np.ndarray,
                              percentile: float = 99.6,
                              min_distance: int = 4,
                              min_score: float = 0.0) -> Dict[str, np.ndarray]:
    x = img.astype(np.float32)
    x = cv2.GaussianBlur(x, (0, 0), 0.9)

    pos = x[x > 0]
    if pos.size == 0:
        return {"xy": np.zeros((0, 2), np.float32), "score": np.zeros((0,), np.float32), "threshold": 0.0}

    thr = float(np.percentile(pos, percentile))
    thr = max(thr, float(min_score))

    k = 2 * min_distance + 1
    dil = cv2.dilate(x, np.ones((k, k), np.uint8))
    mask = (x >= thr) & (np.abs(x - dil) <= 1e-6)

    mask[:min_distance, :] = False
    mask[-min_distance:, :] = False
    mask[:, :min_distance] = False
    mask[:, -min_distance:] = False

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return {"xy": np.zeros((0, 2), np.float32), "score": np.zeros((0,), np.float32), "threshold": thr}

    scores = x[ys, xs].astype(np.float32)
    xy = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)

    order = np.argsort(-scores)
    xy = xy[order]
    scores = scores[order]

    return {"xy": xy, "score": scores, "threshold": thr}


def expected_disparity_px_from_plane_z(K_left: np.ndarray, baseline_m: float, plane_z_m: float) -> float:
    fx = float(K_left[0, 0])
    return fx * baseline_m / plane_z_m


def match_rectified_peaks(left_xy: np.ndarray,
                          left_score: np.ndarray,
                          right_xy: np.ndarray,
                          right_score: np.ndarray,
                          expected_disp_px: float,
                          row_tol_px: int = 2,
                          disp_tol_px: float = 8.0) -> Dict[str, np.ndarray]:
    if left_xy.size == 0 or right_xy.size == 0:
        empty = np.zeros((0, 2), np.float32)
        return {
            "left_xy": empty, "right_xy": empty,
            "left_score": np.zeros((0,), np.float32),
            "right_score": np.zeros((0,), np.float32),
            "disparity": np.zeros((0,), np.float32),
            "row_error": np.zeros((0,), np.float32),
        }

    buckets = {}
    for i, (x, y) in enumerate(right_xy):
        yi = int(round(float(y)))
        buckets.setdefault(yi, []).append(i)

    used_right = set()
    matched_l = []
    matched_r = []
    matched_ls = []
    matched_rs = []
    disparities = []
    row_errs = []

    for i, ((xl, yl), sl) in enumerate(zip(left_xy, left_score)):
        best = None
        for yy in range(int(round(yl)) - row_tol_px, int(round(yl)) + row_tol_px + 1):
            for j in buckets.get(yy, []):
                if j in used_right:
                    continue
                xr, yr = right_xy[j]
                d = float(xl - xr)
                if d < 0:
                    continue
                if abs(d - expected_disp_px) > disp_tol_px:
                    continue
                row_err = abs(float(yl - yr))
                cost = abs(d - expected_disp_px) + 0.5 * row_err - 0.001 * float(right_score[j])
                if best is None or cost < best[0]:
                    best = (cost, j, d, row_err)
        if best is None:
            continue
        _, j, d, row_err = best
        used_right.add(j)
        matched_l.append(left_xy[i])
        matched_r.append(right_xy[j])
        matched_ls.append(sl)
        matched_rs.append(right_score[j])
        disparities.append(d)
        row_errs.append(row_err)

    if not matched_l:
        empty = np.zeros((0, 2), np.float32)
        return {
            "left_xy": empty, "right_xy": empty,
            "left_score": np.zeros((0,), np.float32),
            "right_score": np.zeros((0,), np.float32),
            "disparity": np.zeros((0,), np.float32),
            "row_error": np.zeros((0,), np.float32),
        }

    return {
        "left_xy": np.asarray(matched_l, np.float32),
        "right_xy": np.asarray(matched_r, np.float32),
        "left_score": np.asarray(matched_ls, np.float32),
        "right_score": np.asarray(matched_rs, np.float32),
        "disparity": np.asarray(disparities, np.float32),
        "row_error": np.asarray(row_errs, np.float32),
    }


def reproject_points_with_Q(left_xy: np.ndarray, disparity: np.ndarray, Q: np.ndarray) -> np.ndarray:
    if left_xy.size == 0:
        return np.zeros((0, 3), np.float32)
    pts4 = np.concatenate([
        left_xy.astype(np.float64),
        disparity.reshape(-1, 1).astype(np.float64),
        np.ones((left_xy.shape[0], 1), np.float64)
    ], axis=1)
    XYZW = (Q @ pts4.T).T
    XYZ = XYZW[:, :3] / XYZW[:, 3:4]
    return XYZ.astype(np.float32)


def overlay_points(img_u8: np.ndarray, pts: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    if img_u8.ndim == 2:
        vis = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    else:
        vis = img_u8.copy()
    for x, y in pts:
        cv2.circle(vis, (int(round(float(x))), int(round(float(y)))), 2, color, -1, lineType=cv2.LINE_AA)
    return vis


def overlay_matches(left_img_u8: np.ndarray, right_img_u8: np.ndarray,
                    left_pts: np.ndarray, right_pts: np.ndarray,
                    max_draw: int = 3000) -> np.ndarray:
    left_bgr = cv2.cvtColor(left_img_u8, cv2.COLOR_GRAY2BGR)
    right_bgr = cv2.cvtColor(right_img_u8, cv2.COLOR_GRAY2BGR)
    canvas = np.concatenate([left_bgr, right_bgr], axis=1)
    offset_x = left_img_u8.shape[1]

    n = min(max_draw, left_pts.shape[0])
    if n == 0:
        return canvas

    idx = np.linspace(0, left_pts.shape[0] - 1, n, dtype=np.int32)
    rng = np.random.default_rng(123)
    for k in idx:
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        xl, yl = left_pts[k]
        xr, yr = right_pts[k]
        p1 = (int(round(float(xl))), int(round(float(yl))))
        p2 = (int(round(float(xr))) + offset_x, int(round(float(yr))))
        cv2.circle(canvas, p1, 2, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p2, 2, color, -1, lineType=cv2.LINE_AA)
        cv2.line(canvas, p1, p2, color, 1, lineType=cv2.LINE_AA)
    return canvas


def overlay_mask_on_gray(img_u8: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    vis = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    m = np.asarray(mask, dtype=bool)
    vis[m] = (0.6 * vis[m] + 0.4 * np.asarray(color, dtype=np.float32)).astype(np.uint8)
    return vis


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


def remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    m = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if min_area_px <= 1:
        return m.astype(bool)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m, dtype=np.uint8)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= int(min_area_px):
            keep[labels == i] = 1
    return keep.astype(bool)


def build_dense_observation_fields(
    diff: np.ndarray,
    lcn: np.ndarray,
    *,
    seed_percentile: float = 99.3,
    support_percentile: float = 88.0,
    support_radius_px: int = 3,
    min_component_area_px: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Dense observation extraction:
    - seed: strong lcn maxima/cores
    - support: broader positive diff support
    - dense mask: dilated seed intersected with support, plus the seed itself
    - dense weight: masked weighted combination of normalized diff and normalized lcn
    """
    diff = np.clip(np.asarray(diff, dtype=np.float32), 0.0, None)
    lcn = np.clip(np.asarray(lcn, dtype=np.float32), 0.0, None)

    diff_pos = diff[diff > 0]
    lcn_pos = lcn[lcn > 0]

    if diff_pos.size == 0 or lcn_pos.size == 0:
        z = np.zeros_like(diff, dtype=np.float32)
        zb = np.zeros_like(diff, dtype=bool)
        return {
            "seed_mask": zb,
            "support_mask": zb,
            "dense_mask": zb,
            "weight": z,
            "diff_norm": z,
            "lcn_norm": z,
            "seed_threshold": 0.0,
            "support_threshold": 0.0,
        }

    seed_threshold = float(np.percentile(lcn_pos, seed_percentile))
    support_threshold = float(np.percentile(diff_pos, support_percentile))

    seed_mask = lcn >= seed_threshold
    support_mask = diff >= support_threshold

    if support_radius_px > 0:
        k = 2 * int(support_radius_px) + 1
        grown_seed = cv2.dilate(seed_mask.astype(np.uint8), np.ones((k, k), np.uint8), iterations=1) > 0
    else:
        grown_seed = seed_mask.copy()

    dense_mask = (grown_seed & support_mask) | seed_mask
    dense_mask = remove_small_components(dense_mask, min_component_area_px)

    diff_norm = normalize_positive_01(diff, percentile_high=99.5)
    lcn_norm = normalize_positive_01(lcn, percentile_high=99.5)
    weight = dense_mask.astype(np.float32) * (0.75 * diff_norm + 0.25 * lcn_norm)

    return {
        "seed_mask": seed_mask.astype(bool),
        "support_mask": support_mask.astype(bool),
        "dense_mask": dense_mask.astype(bool),
        "weight": weight.astype(np.float32),
        "diff_norm": diff_norm.astype(np.float32),
        "lcn_norm": lcn_norm.astype(np.float32),
        "seed_threshold": seed_threshold,
        "support_threshold": support_threshold,
    }


# ---------------------------- main stage ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 1 extraction from wall_capture session")
    ap.add_argument("--root", type=str, default="wall_capture", help="Root wall_capture directory")
    ap.add_argument("--session", type=str, default="latest", help="session_* dir name or 'latest'")
    ap.add_argument("--out_name", type=str, default="projector_stage1", help="Output subdirectory inside session")
    ap.add_argument("--rear_to_surface_mm", type=float, default=None,
                    help="Measured distance from rear camera wall to plane, mm")
    ap.add_argument("--rear_to_depth_origin_mm", type=float, default=20.8,
                    help="Approx. rear-wall to depth-origin offset, mm")
    ap.add_argument("--plane_z_mm", type=float, default=None,
                    help="Directly specify plane depth from camera/depth origin, mm")

    # sparse branch
    ap.add_argument("--dot_percentile", type=float, default=99.6, help="Percentile threshold for peak detection")
    ap.add_argument("--min_peak_distance_px", type=int, default=4, help="NMS radius in px")
    ap.add_argument("--row_tol_px", type=int, default=2, help="Rectified row tolerance for stereo matching")
    ap.add_argument("--disp_tol_px", type=float, default=8.0, help="Allowed deviation from expected disparity")

    # dense branch
    ap.add_argument("--dense_seed_percentile", type=float, default=99.3,
                    help="LCN percentile for dense seed extraction")
    ap.add_argument("--dense_support_percentile", type=float, default=88.0,
                    help="Diff percentile for dense support extraction")
    ap.add_argument("--dense_support_radius_px", type=int, default=3,
                    help="Seed dilation radius before support intersection")
    ap.add_argument("--dense_min_component_area_px", type=int, default=3,
                    help="Remove connected components smaller than this area")

    args = ap.parse_args()

    root = Path(args.root)
    session_dir = resolve_session(root, args.session)
    meta_path = session_dir / "capture_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    meta = load_json(meta_path)

    out_dir = session_dir / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plane_z_mm is not None:
        plane_z_mm = float(args.plane_z_mm)
        plane_source = "user_plane_z_mm"
    elif args.rear_to_surface_mm is not None:
        plane_z_mm = float(args.rear_to_surface_mm) - float(args.rear_to_depth_origin_mm)
        plane_source = "rear_to_surface_mm - rear_to_depth_origin_mm"
    else:
        raise ValueError("Provide either --plane_z_mm or --rear_to_surface_mm")
    plane_z_m = plane_z_mm / 1000.0

    K1, D1, size1 = parse_intrinsics(meta["streams"]["IR_LEFT"])
    K2, D2, size2 = parse_intrinsics(meta["streams"]["IR_RIGHT"])
    if size1 != size2:
        raise RuntimeError(f"Left/right image sizes differ: {size1} vs {size2}")

    R_rl, t_rl = parse_extrinsics_ir_right_to_left(meta["extrinsics"]["IR_RIGHT_to_IR_LEFT"])
    baseline_m = float(np.linalg.norm(t_rl.reshape(3)))
    expected_disp_px = expected_disparity_px_from_plane_z(K1, baseline_m, plane_z_m)

    camera_params = {
        "K_left": K1.tolist(),
        "D_left": D1.tolist(),
        "K_right": K2.tolist(),
        "D_right": D2.tolist(),
        "R_left_from_right": R_rl.tolist(),
        "t_left_from_right_m": t_rl.reshape(3).tolist(),
        "baseline_m": baseline_m,
        "plane_z_mm": plane_z_mm,
        "plane_z_source": plane_source,
        "expected_disparity_px_from_plane_z": expected_disp_px,
    }
    save_json(out_dir / "camera_and_plane_params.json", camera_params)

    left_on_dir = session_dir / "left_on"
    left_off_dir = session_dir / "left_off"
    right_on_dir = session_dir / "right_on"
    right_off_dir = session_dir / "right_off"

    mean_l_on, med_l_on, left_on_paths = compute_mean_and_median(left_on_dir)
    mean_l_off, med_l_off, left_off_paths = compute_mean_and_median(left_off_dir)
    mean_r_on, med_r_on, right_on_paths = compute_mean_and_median(right_on_dir)
    mean_r_off, med_r_off, right_off_paths = compute_mean_and_median(right_off_dir)

    obs_dir = out_dir / "observations"
    obs_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(obs_dir / "left_mean_on.png"), normalize_u8(mean_l_on))
    cv2.imwrite(str(obs_dir / "left_mean_off.png"), normalize_u8(mean_l_off))
    cv2.imwrite(str(obs_dir / "right_mean_on.png"), normalize_u8(mean_r_on))
    cv2.imwrite(str(obs_dir / "right_mean_off.png"), normalize_u8(mean_r_off))
    cv2.imwrite(str(obs_dir / "left_median_on.png"), ensure_gray_u8(med_l_on))
    cv2.imwrite(str(obs_dir / "left_median_off.png"), ensure_gray_u8(med_l_off))
    cv2.imwrite(str(obs_dir / "right_median_on.png"), ensure_gray_u8(med_r_on))
    cv2.imwrite(str(obs_dir / "right_median_off.png"), ensure_gray_u8(med_r_off))

    diff_left = build_pattern_diff(med_l_on, med_l_off)
    diff_right = build_pattern_diff(med_r_on, med_r_off)
    lcn_left = build_lcn(diff_left)
    lcn_right = build_lcn(diff_right)

    dense_left = build_dense_observation_fields(
        diff_left,
        lcn_left,
        seed_percentile=float(args.dense_seed_percentile),
        support_percentile=float(args.dense_support_percentile),
        support_radius_px=int(args.dense_support_radius_px),
        min_component_area_px=int(args.dense_min_component_area_px),
    )
    dense_right = build_dense_observation_fields(
        diff_right,
        lcn_right,
        seed_percentile=float(args.dense_seed_percentile),
        support_percentile=float(args.dense_support_percentile),
        support_radius_px=int(args.dense_support_radius_px),
        min_component_area_px=int(args.dense_min_component_area_px),
    )

    cv2.imwrite(str(obs_dir / "left_diff_raw.png"), normalize_u8(diff_left))
    cv2.imwrite(str(obs_dir / "right_diff_raw.png"), normalize_u8(diff_right))
    cv2.imwrite(str(obs_dir / "left_diff_lcn.png"), normalize_u8(lcn_left))
    cv2.imwrite(str(obs_dir / "right_diff_lcn.png"), normalize_u8(lcn_right))

    cv2.imwrite(str(obs_dir / "left_dense_seed_mask.png"), binary_to_u8(dense_left["seed_mask"]))
    cv2.imwrite(str(obs_dir / "left_dense_support_mask.png"), binary_to_u8(dense_left["support_mask"]))
    cv2.imwrite(str(obs_dir / "left_dense_mask.png"), binary_to_u8(dense_left["dense_mask"]))
    cv2.imwrite(str(obs_dir / "left_dense_weight.png"), normalize_u8(dense_left["weight"], 0.0, 99.5))
    cv2.imwrite(str(obs_dir / "left_dense_overlay.png"),
                overlay_mask_on_gray(normalize_u8(diff_left), dense_left["dense_mask"], (0, 255, 0)))

    cv2.imwrite(str(obs_dir / "right_dense_seed_mask.png"), binary_to_u8(dense_right["seed_mask"]))
    cv2.imwrite(str(obs_dir / "right_dense_support_mask.png"), binary_to_u8(dense_right["support_mask"]))
    cv2.imwrite(str(obs_dir / "right_dense_mask.png"), binary_to_u8(dense_right["dense_mask"]))
    cv2.imwrite(str(obs_dir / "right_dense_weight.png"), normalize_u8(dense_right["weight"], 0.0, 99.5))
    cv2.imwrite(str(obs_dir / "right_dense_overlay.png"),
                overlay_mask_on_gray(normalize_u8(diff_right), dense_right["dense_mask"], (0, 255, 0)))

    np.savez_compressed(
        str(obs_dir / "pattern_observations.npz"),
        # old keys kept
        diff_left=diff_left.astype(np.float32),
        diff_right=diff_right.astype(np.float32),
        lcn_left=lcn_left.astype(np.float32),
        lcn_right=lcn_right.astype(np.float32),
        med_left_on=med_l_on.astype(np.uint8),
        med_left_off=med_l_off.astype(np.uint8),
        med_right_on=med_r_on.astype(np.uint8),
        med_right_off=med_r_off.astype(np.uint8),

        # new dense fields
        mask_left=dense_left["dense_mask"].astype(np.uint8),
        mask_right=dense_right["dense_mask"].astype(np.uint8),
        weight_left=dense_left["weight"].astype(np.float32),
        weight_right=dense_right["weight"].astype(np.float32),
        seed_mask_left=dense_left["seed_mask"].astype(np.uint8),
        seed_mask_right=dense_right["seed_mask"].astype(np.uint8),
        support_mask_left=dense_left["support_mask"].astype(np.uint8),
        support_mask_right=dense_right["support_mask"].astype(np.uint8),
        diff_norm_left=dense_left["diff_norm"].astype(np.float32),
        diff_norm_right=dense_right["diff_norm"].astype(np.float32),
        lcn_norm_left=dense_left["lcn_norm"].astype(np.float32),
        lcn_norm_right=dense_right["lcn_norm"].astype(np.float32),
    )

    save_json(obs_dir / "dense_observation_meta.json", {
        "dense_seed_percentile": float(args.dense_seed_percentile),
        "dense_support_percentile": float(args.dense_support_percentile),
        "dense_support_radius_px": int(args.dense_support_radius_px),
        "dense_min_component_area_px": int(args.dense_min_component_area_px),
        "left": {
            "seed_threshold": float(dense_left["seed_threshold"]),
            "support_threshold": float(dense_left["support_threshold"]),
            "seed_px_count": int(np.count_nonzero(dense_left["seed_mask"])),
            "support_px_count": int(np.count_nonzero(dense_left["support_mask"])),
            "dense_px_count": int(np.count_nonzero(dense_left["dense_mask"])),
        },
        "right": {
            "seed_threshold": float(dense_right["seed_threshold"]),
            "support_threshold": float(dense_right["support_threshold"]),
            "seed_px_count": int(np.count_nonzero(dense_right["seed_mask"])),
            "support_px_count": int(np.count_nonzero(dense_right["support_mask"])),
            "dense_px_count": int(np.count_nonzero(dense_right["dense_mask"])),
        },
    })

    rect = rectify_pair(diff_left, diff_right, K1, D1, K2, D2, R_rl, t_rl)
    rect_dir = out_dir / "rectified"
    rect_dir.mkdir(parents=True, exist_ok=True)
    rect_left = rect["rect_left"]
    rect_right = rect["rect_right"]

    cv2.imwrite(str(rect_dir / "left_rect_diff.png"), normalize_u8(rect_left))
    cv2.imwrite(str(rect_dir / "right_rect_diff.png"), normalize_u8(rect_right))

    rect_lcn_left = build_lcn(rect_left)
    rect_lcn_right = build_lcn(rect_right)
    cv2.imwrite(str(rect_dir / "left_rect_lcn.png"), normalize_u8(rect_lcn_left))
    cv2.imwrite(str(rect_dir / "right_rect_lcn.png"), normalize_u8(rect_lcn_right))

    rect_dense_left = build_dense_observation_fields(
        rect_left,
        rect_lcn_left,
        seed_percentile=float(args.dense_seed_percentile),
        support_percentile=float(args.dense_support_percentile),
        support_radius_px=int(args.dense_support_radius_px),
        min_component_area_px=int(args.dense_min_component_area_px),
    )
    rect_dense_right = build_dense_observation_fields(
        rect_right,
        rect_lcn_right,
        seed_percentile=float(args.dense_seed_percentile),
        support_percentile=float(args.dense_support_percentile),
        support_radius_px=int(args.dense_support_radius_px),
        min_component_area_px=int(args.dense_min_component_area_px),
    )

    cv2.imwrite(str(rect_dir / "left_rect_dense_seed_mask.png"), binary_to_u8(rect_dense_left["seed_mask"]))
    cv2.imwrite(str(rect_dir / "left_rect_dense_support_mask.png"), binary_to_u8(rect_dense_left["support_mask"]))
    cv2.imwrite(str(rect_dir / "left_rect_dense_mask.png"), binary_to_u8(rect_dense_left["dense_mask"]))
    cv2.imwrite(str(rect_dir / "left_rect_dense_weight.png"), normalize_u8(rect_dense_left["weight"], 0.0, 99.5))
    cv2.imwrite(str(rect_dir / "left_rect_dense_overlay.png"),
                overlay_mask_on_gray(normalize_u8(rect_left), rect_dense_left["dense_mask"], (0, 255, 0)))

    cv2.imwrite(str(rect_dir / "right_rect_dense_seed_mask.png"), binary_to_u8(rect_dense_right["seed_mask"]))
    cv2.imwrite(str(rect_dir / "right_rect_dense_support_mask.png"), binary_to_u8(rect_dense_right["support_mask"]))
    cv2.imwrite(str(rect_dir / "right_rect_dense_mask.png"), binary_to_u8(rect_dense_right["dense_mask"]))
    cv2.imwrite(str(rect_dir / "right_rect_dense_weight.png"), normalize_u8(rect_dense_right["weight"], 0.0, 99.5))
    cv2.imwrite(str(rect_dir / "right_rect_dense_overlay.png"),
                overlay_mask_on_gray(normalize_u8(rect_right), rect_dense_right["dense_mask"], (0, 255, 0)))

    np.savez_compressed(
        str(rect_dir / "rectification_data.npz"),
        # old keys kept
        R1=rect["R1"], R2=rect["R2"], P1=rect["P1"], P2=rect["P2"], Q=rect["Q"],
        map1x=rect["map1x"], map1y=rect["map1y"], map2x=rect["map2x"], map2y=rect["map2y"],
        rect_left=rect_left.astype(np.float32), rect_right=rect_right.astype(np.float32),
        rect_lcn_left=rect_lcn_left.astype(np.float32), rect_lcn_right=rect_lcn_right.astype(np.float32),

        # new dense rectified fields
        rect_mask_left=rect_dense_left["dense_mask"].astype(np.uint8),
        rect_mask_right=rect_dense_right["dense_mask"].astype(np.uint8),
        rect_weight_left=rect_dense_left["weight"].astype(np.float32),
        rect_weight_right=rect_dense_right["weight"].astype(np.float32),
        rect_seed_mask_left=rect_dense_left["seed_mask"].astype(np.uint8),
        rect_seed_mask_right=rect_dense_right["seed_mask"].astype(np.uint8),
        rect_support_mask_left=rect_dense_left["support_mask"].astype(np.uint8),
        rect_support_mask_right=rect_dense_right["support_mask"].astype(np.uint8),
    )
    save_json(rect_dir / "rectification_meta.json", {
        "roi1": [int(v) for v in rect["roi1"]],
        "roi2": [int(v) for v in rect["roi2"]],
        "expected_disparity_px_from_plane_z": expected_disp_px,
        "rect_dense": {
            "left_seed_px_count": int(np.count_nonzero(rect_dense_left["seed_mask"])),
            "left_support_px_count": int(np.count_nonzero(rect_dense_left["support_mask"])),
            "left_dense_px_count": int(np.count_nonzero(rect_dense_left["dense_mask"])),
            "right_seed_px_count": int(np.count_nonzero(rect_dense_right["seed_mask"])),
            "right_support_px_count": int(np.count_nonzero(rect_dense_right["support_mask"])),
            "right_dense_px_count": int(np.count_nonzero(rect_dense_right["dense_mask"])),
        }
    })

    # sparse branch stays intact
    left_peaks = detect_peaks_from_pattern(
        rect_lcn_left, percentile=args.dot_percentile, min_distance=args.min_peak_distance_px
    )
    right_peaks = detect_peaks_from_pattern(
        rect_lcn_right, percentile=args.dot_percentile, min_distance=args.min_peak_distance_px
    )

    peaks_dir = out_dir / "peaks"
    peaks_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(peaks_dir / "left_rect_peaks.png"),
                overlay_points(normalize_u8(rect_lcn_left), left_peaks["xy"], (0, 255, 0)))
    cv2.imwrite(str(peaks_dir / "right_rect_peaks.png"),
                overlay_points(normalize_u8(rect_lcn_right), right_peaks["xy"], (0, 255, 0)))

    np.savez_compressed(
        str(peaks_dir / "detected_peaks_rectified.npz"),
        left_xy=left_peaks["xy"], left_score=left_peaks["score"],
        right_xy=right_peaks["xy"], right_score=right_peaks["score"],
    )
    save_json(peaks_dir / "detected_peaks_meta.json", {
        "left_count": int(left_peaks["xy"].shape[0]),
        "right_count": int(right_peaks["xy"].shape[0]),
        "dot_percentile": float(args.dot_percentile),
        "min_peak_distance_px": int(args.min_peak_distance_px),
        "left_threshold": float(left_peaks.get("threshold", 0.0)),
        "right_threshold": float(right_peaks.get("threshold", 0.0)),
    })

    matches = match_rectified_peaks(
        left_xy=left_peaks["xy"],
        left_score=left_peaks["score"],
        right_xy=right_peaks["xy"],
        right_score=right_peaks["score"],
        expected_disp_px=expected_disp_px,
        row_tol_px=args.row_tol_px,
        disp_tol_px=args.disp_tol_px,
    )

    match_dir = out_dir / "matches"
    match_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(match_dir / "rectified_matches.png"),
                overlay_matches(normalize_u8(rect_lcn_left), normalize_u8(rect_lcn_right),
                                matches["left_xy"], matches["right_xy"]))

    xyz = reproject_points_with_Q(matches["left_xy"], matches["disparity"], rect["Q"])

    np.savez_compressed(
        str(match_dir / "matched_points_rectified.npz"),
        left_xy=matches["left_xy"],
        right_xy=matches["right_xy"],
        left_score=matches["left_score"],
        right_score=matches["right_score"],
        disparity=matches["disparity"],
        row_error=matches["row_error"],
        xyz_rectified=xyz,
    )

    illum = {}
    if xyz.shape[0] > 0:
        illum = {
            "x_m": robust_stats(xyz[:, 0]),
            "y_m": robust_stats(xyz[:, 1]),
            "z_m": robust_stats(xyz[:, 2]),
            "bbox_xy_m": {
                "x_min": float(np.min(xyz[:, 0])),
                "x_max": float(np.max(xyz[:, 0])),
                "y_min": float(np.min(xyz[:, 1])),
                "y_max": float(np.max(xyz[:, 1])),
            },
        }

    save_json(match_dir / "matched_points_meta.json", {
        "matched_count": int(matches["left_xy"].shape[0]),
        "row_tol_px": int(args.row_tol_px),
        "disp_tol_px": float(args.disp_tol_px),
        "expected_disparity_px_from_plane_z": float(expected_disp_px),
        "measured_disparity_stats_px": robust_stats(matches["disparity"]),
        "row_error_stats_px": robust_stats(matches["row_error"]),
        "triangulated_xyz_stats": illum,
    })

    summary = {
        "session_dir": str(session_dir),
        "used_capture_meta": str(meta_path),
        "image_count": {
            "left_on": len(left_on_paths),
            "left_off": len(left_off_paths),
            "right_on": len(right_on_paths),
            "right_off": len(right_off_paths),
        },
        "camera_and_plane": camera_params,
        "dense_observations": {
            "params": {
                "dense_seed_percentile": float(args.dense_seed_percentile),
                "dense_support_percentile": float(args.dense_support_percentile),
                "dense_support_radius_px": int(args.dense_support_radius_px),
                "dense_min_component_area_px": int(args.dense_min_component_area_px),
            },
            "left": {
                "seed_px_count": int(np.count_nonzero(dense_left["seed_mask"])),
                "support_px_count": int(np.count_nonzero(dense_left["support_mask"])),
                "dense_px_count": int(np.count_nonzero(dense_left["dense_mask"])),
            },
            "right": {
                "seed_px_count": int(np.count_nonzero(dense_right["seed_mask"])),
                "support_px_count": int(np.count_nonzero(dense_right["support_mask"])),
                "dense_px_count": int(np.count_nonzero(dense_right["dense_mask"])),
            },
            "rectified_left": {
                "seed_px_count": int(np.count_nonzero(rect_dense_left["seed_mask"])),
                "support_px_count": int(np.count_nonzero(rect_dense_left["support_mask"])),
                "dense_px_count": int(np.count_nonzero(rect_dense_left["dense_mask"])),
            },
            "rectified_right": {
                "seed_px_count": int(np.count_nonzero(rect_dense_right["seed_mask"])),
                "support_px_count": int(np.count_nonzero(rect_dense_right["support_mask"])),
                "dense_px_count": int(np.count_nonzero(rect_dense_right["dense_mask"])),
            },
        },
        "rectification": {
            "roi1": [int(v) for v in rect["roi1"]],
            "roi2": [int(v) for v in rect["roi2"]],
        },
        "peaks": {
            "left_count": int(left_peaks["xy"].shape[0]),
            "right_count": int(right_peaks["xy"].shape[0]),
        },
        "matches": {
            "matched_count": int(matches["left_xy"].shape[0]),
            "expected_disparity_px_from_plane_z": float(expected_disp_px),
            "measured_disparity_stats_px": robust_stats(matches["disparity"]),
            "row_error_stats_px": robust_stats(matches["row_error"]),
            "triangulated_xyz_stats": illum,
        },
        "next_stage_inputs": {
            "pattern_observations_npz": str((obs_dir / "pattern_observations.npz").relative_to(session_dir)),
            "rectification_data_npz": str((rect_dir / "rectification_data.npz").relative_to(session_dir)),
            "detected_peaks_rectified_npz": str((peaks_dir / "detected_peaks_rectified.npz").relative_to(session_dir)),
            "matched_points_rectified_npz": str((match_dir / "matched_points_rectified.npz").relative_to(session_dir)),
        }
    }
    save_json(out_dir / "stage1_summary.json", summary)

    print("Done.")
    print(f"Session               : {session_dir}")
    print(f"Output                : {out_dir}")
    print(f"Plane Z [mm]          : {plane_z_mm:.3f} ({plane_source})")
    print(f"Baseline [mm]         : {baseline_m * 1000.0:.3f}")
    print(f"Expected disp px      : {expected_disp_px:.3f}")
    print(f"Left dense px         : {np.count_nonzero(dense_left['dense_mask'])}")
    print(f"Right dense px        : {np.count_nonzero(dense_right['dense_mask'])}")
    print(f"Rect left dense px    : {np.count_nonzero(rect_dense_left['dense_mask'])}")
    print(f"Rect right dense px   : {np.count_nonzero(rect_dense_right['dense_mask'])}")
    print(f"Left peaks            : {left_peaks['xy'].shape[0]}")
    print(f"Right peaks           : {right_peaks['xy'].shape[0]}")
    print(f"Matches               : {matches['left_xy'].shape[0]}")


if __name__ == "__main__":
    main()