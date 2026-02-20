#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_align_rectify_vs_gt.py

Expected run_dir structure (new):
  run_dir/
    depth_gt/*.png                 (uint16, 0=invalid)  # GT aligned to RGB grid
    depth_sgm_align/*.png          (uint16, 0=invalid)  # PLAIN SGM aligned to RGB grid
    depth_rectify_sgm_align/*.png  (uint16, 0=invalid)  # RECTIFIED SGM aligned to RGB grid

    depth_sgm_raw/*.png            (uint16, 0=invalid)  # optional raw (plain grid)
    depth_rectify_sgm_raw/*.png    (uint16, 0=invalid)  # optional raw (rectified grid)

    disp_npy/*.npy                 (float32 disparity px)          [optional, plain]
    disp_rectify_npy/*.npy         (float32 disparity px)          [optional, rectified]

    ir_left/*.png|jpg              [optional]
    ir_right/*.png|jpg             [optional]
    rgb/*.jpg|png                  [optional]

What it does:
  - Computes metrics vs GT for BOTH predictors:
        plain  : depth_sgm_align
        rectify: depth_rectify_sgm_align
    on overlap(valid_pred & valid_gt)

  - Also reports raw validity/stats separately (no GT compare):
        depth_sgm_raw, depth_rectify_sgm_raw

  - Writes:
        run_dir/analysis/summary.json
        run_dir/analysis/per_frame.csv
        run_dir/analysis/vis/*.png  (worst frames by abs_p95 + low overlap + biggest delta)

Depth encoding:
  depth_u16 = round((depth_m * 1000) / depth_scale_mm)
  depth_m   = depth_u16 * depth_scale_mm / 1000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ------------------------- FS helpers -------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _list_pngs(dirpath: str) -> List[str]:
    if not dirpath or not os.path.isdir(dirpath):
        return []
    return sorted(glob(os.path.join(dirpath, "*.png")))


def _list_npys(dirpath: str) -> List[str]:
    if not dirpath or not os.path.isdir(dirpath):
        return []
    return sorted(glob(os.path.join(dirpath, "*.npy")))


def _read_u16_png(path: str) -> np.ndarray:
    x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if x is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if x.ndim == 3 and x.shape[2] == 1:
        x = x.squeeze(axis=2)
    if x.ndim != 2:
        raise RuntimeError(f"Expected single-channel PNG at {path}, got shape={x.shape}")
    if x.dtype != np.uint16:
        raise RuntimeError(f"Expected uint16 PNG at {path}, got {x.dtype}")
    return x


def _basename_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _try_read_rgb(path_no_ext: str) -> Optional[np.ndarray]:
    for ext in (".jpg", ".jpeg", ".png"):
        p = path_no_ext + ext
        if os.path.exists(p):
            return cv2.imread(p, cv2.IMREAD_COLOR)  # BGR
    return None


def _try_read_any_gray(path_no_ext: str) -> Optional[np.ndarray]:
    for ext in (".png", ".jpg", ".jpeg"):
        p = path_no_ext + ext
        if os.path.exists(p):
            x = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if x is None:
                continue
            if x.ndim == 3:
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            if x.dtype == np.uint16:
                return np.clip((x.astype(np.float32) / 256.0), 0, 255).astype(np.uint8)
            if x.dtype != np.uint8:
                return np.clip(x.astype(np.float32), 0, 255).astype(np.uint8)
            return x
    return None


# ------------------------- numeric helpers -------------------------

def u16_to_m(depth_u16: np.ndarray, depth_scale_mm: float) -> np.ndarray:
    return (depth_u16.astype(np.float32) * float(depth_scale_mm)) / 1000.0


def _robust_stats(x: np.ndarray) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"count": 0.0}
    return {
        "count": float(x.size),
        "min": float(np.min(x)),
        "p1": float(np.quantile(x, 0.01)),
        "p5": float(np.quantile(x, 0.05)),
        "p50": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def _quantile_safe(x: np.ndarray, q: float) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


# ------------------------- viz helpers -------------------------

def _colorize_depth(depth_m: np.ndarray, valid: np.ndarray, *, vmax: float) -> np.ndarray:
    """
    Near = bright, far = dark.
    """
    d = depth_m.astype(np.float32, copy=True)
    d[~np.isfinite(d)] = 0.0
    d[~valid] = 0.0

    vmax = float(max(vmax, 1e-6))
    v = 1.0 - np.clip(d / vmax, 0.0, 1.0)
    u8 = (v * 255.0 + 0.5).astype(np.uint8)
    u8[~valid] = 0
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


def _colorize_abs_err(abs_err_m: np.ndarray) -> np.ndarray:
    e = abs_err_m.astype(np.float32, copy=True)
    e[~np.isfinite(e)] = 0.0
    if np.any(e > 0):
        cap = float(np.quantile(e[e > 0], 0.99))
        cap = max(cap, 1e-6)
    else:
        cap = 1e-6
    u8 = (np.clip(e / cap, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


def _colorize_signed_err(err_m: np.ndarray) -> np.ndarray:
    e = err_m.astype(np.float32, copy=True)
    e[~np.isfinite(e)] = 0.0
    mag = np.abs(e)
    cap = float(np.quantile(mag[mag > 0], 0.99)) if np.any(mag > 0) else 1e-6
    cap = max(cap, 1e-6)

    x = np.clip(e / cap, -1.0, 1.0)  # [-1..1]
    r = np.clip(x, 0.0, 1.0)
    b = np.clip(-x, 0.0, 1.0)
    g = 1.0 - np.abs(x)
    bgr = np.stack([b, g, r], axis=-1)
    return (bgr * 255.0 + 0.5).astype(np.uint8)


def _disp_to_vis(disp: np.ndarray) -> np.ndarray:
    d = disp.astype(np.float32, copy=True)
    d[~np.isfinite(d)] = 0.0
    d[d <= 0.0] = 0.0
    if np.any(d > 0):
        vmax = float(np.quantile(d[d > 0], 0.99))
        vmax = max(vmax, 1e-6)
        u8 = (np.clip(d / vmax, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    else:
        u8 = np.zeros_like(d, dtype=np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


def _save_vis_bundle(
    vis_dir: str,
    fid: str,
    *,
    gt_m: np.ndarray,
    valid_gt: np.ndarray,
    plain_m: np.ndarray,
    valid_plain: np.ndarray,
    rect_m: Optional[np.ndarray],
    valid_rect: Optional[np.ndarray],
    depth_max_m: float,
    rgb_bgr: Optional[np.ndarray],
    irL_u8: Optional[np.ndarray],
    disp_plain: Optional[np.ndarray],
    disp_rect: Optional[np.ndarray],
) -> None:
    _ensure_dir(vis_dir)

    # common GT
    cv2.imwrite(os.path.join(vis_dir, f"{fid}_gt_depth.png"), _colorize_depth(gt_m, valid_gt, vmax=depth_max_m))
    cv2.imwrite(os.path.join(vis_dir, f"{fid}_valid_gt.png"), (valid_gt.astype(np.uint8) * 255))

    # plain
    both_p = valid_plain & valid_gt
    err_p = np.full_like(plain_m, np.nan, dtype=np.float32)
    err_p[both_p] = plain_m[both_p] - gt_m[both_p]
    abs_p = np.abs(err_p)

    cv2.imwrite(os.path.join(vis_dir, f"{fid}_plain_pred_depth.png"), _colorize_depth(plain_m, valid_plain, vmax=depth_max_m))
    cv2.imwrite(os.path.join(vis_dir, f"{fid}_valid_plain.png"), (valid_plain.astype(np.uint8) * 255))
    cv2.imwrite(os.path.join(vis_dir, f"{fid}_overlap_plain.png"), (both_p.astype(np.uint8) * 255))
    cv2.imwrite(os.path.join(vis_dir, f"{fid}_plain_abs_err.png"), _colorize_abs_err(abs_p))
    cv2.imwrite(os.path.join(vis_dir, f"{fid}_plain_signed_err.png"), _colorize_signed_err(err_p))

    # rectified (optional)
    if rect_m is not None and valid_rect is not None:
        both_r = valid_rect & valid_gt
        err_r = np.full_like(rect_m, np.nan, dtype=np.float32)
        err_r[both_r] = rect_m[both_r] - gt_m[both_r]
        abs_r = np.abs(err_r)

        cv2.imwrite(os.path.join(vis_dir, f"{fid}_rect_pred_depth.png"), _colorize_depth(rect_m, valid_rect, vmax=depth_max_m))
        cv2.imwrite(os.path.join(vis_dir, f"{fid}_valid_rect.png"), (valid_rect.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(vis_dir, f"{fid}_overlap_rect.png"), (both_r.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(vis_dir, f"{fid}_rect_abs_err.png"), _colorize_abs_err(abs_r))
        cv2.imwrite(os.path.join(vis_dir, f"{fid}_rect_signed_err.png"), _colorize_signed_err(err_r))

        # delta abs error (rect - plain) on common overlap to visualize improvement/regression
        common = both_p & both_r
        delta = np.full_like(gt_m, np.nan, dtype=np.float32)
        if np.any(common):
            delta[common] = (abs_r[common] - abs_p[common]).astype(np.float32)
        cv2.imwrite(os.path.join(vis_dir, f"{fid}_delta_abs_err_rect_minus_plain.png"), _colorize_signed_err(delta))

    if rgb_bgr is not None:
        cv2.imwrite(os.path.join(vis_dir, f"{fid}_rgb.png"), rgb_bgr)
    if irL_u8 is not None:
        cv2.imwrite(os.path.join(vis_dir, f"{fid}_ir_left.png"), irL_u8)

    if disp_plain is not None:
        cv2.imwrite(os.path.join(vis_dir, f"{fid}_disp_plain.png"), _disp_to_vis(disp_plain))
    if disp_rect is not None:
        cv2.imwrite(os.path.join(vis_dir, f"{fid}_disp_rect.png"), _disp_to_vis(disp_rect))


# ------------------------- run_dir detection -------------------------

def _find_run_dirs(output_root: str) -> List[str]:
    """
    Prefer directories containing:
      depth_gt + depth_sgm_align + depth_rectify_sgm_align
    Fallback:
      depth_gt + depth_sgm_align
    """
    if not os.path.isdir(output_root):
        return []

    strict: List[str] = []
    loose: List[str] = []

    for dirpath, dirnames, _ in os.walk(output_root):
        has_gt = "depth_gt" in dirnames and glob(os.path.join(dirpath, "depth_gt", "*.png"))
        has_plain = "depth_sgm_align" in dirnames and glob(os.path.join(dirpath, "depth_sgm_align", "*.png"))
        has_rect = "depth_rectify_sgm_align" in dirnames and glob(os.path.join(dirpath, "depth_rectify_sgm_align", "*.png"))

        if has_gt and has_plain and has_rect:
            strict.append(dirpath)
        elif has_gt and has_plain:
            loose.append(dirpath)

    strict.sort(key=lambda p: os.path.getmtime(p))
    loose.sort(key=lambda p: os.path.getmtime(p))

    return strict if strict else loose


def _default_run_dir(output_root: str) -> str:
    cands = _find_run_dirs(output_root)
    if not cands:
        raise RuntimeError(
            f"Could not find any run_dir under '{output_root}' containing depth_gt/*.png and depth_sgm_align/*.png"
        )
    return cands[-1]


# ------------------------- analysis core -------------------------

@dataclass
class VariantFrame:
    pred_valid_ratio: float
    gt_valid_ratio: float
    overlap_ratio: float
    frac_gt: float
    frac_pred: float

    pred_p50_m: float
    pred_p95_m: float
    gt_p50_m: float
    gt_p95_m: float

    mae_m: float
    rmse_m: float
    abs_p50_m: float
    abs_p95_m: float
    abs_p99_m: float
    abs_max_m: float

    bias_mean_m: float
    bias_p50_m: float

    out_1cm: float
    out_2cm: float
    out_5cm: float
    out_10cm: float


@dataclass
class FrameRow:
    frame_id: str
    plain: VariantFrame
    rectify: Optional[VariantFrame]

    diff_mae_m: float
    diff_abs_p95_m: float
    diff_bias_mean_m: float
    diff_overlap_ratio: float

    raw_plain_valid_ratio: Optional[float]
    raw_rect_valid_ratio: Optional[float]


def _compute_variant(pred_u16: np.ndarray, gt_u16: np.ndarray, depth_scale_mm: float) -> Tuple[VariantFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns:
      - VariantFrame metrics
      - abs_err array on overlap (1D) for global accumulation
      - signed_err array on overlap (1D) for global accumulation
    """
    pred_m = u16_to_m(pred_u16, depth_scale_mm)
    gt_m = u16_to_m(gt_u16, depth_scale_mm)

    valid_pred = pred_u16 > 0
    valid_gt = gt_u16 > 0
    overlap = valid_pred & valid_gt

    pred_valid_ratio = float(valid_pred.mean())
    gt_valid_ratio = float(valid_gt.mean())
    overlap_ratio = float(overlap.mean())

    pred_stats = _robust_stats(pred_m[valid_pred]) if np.any(valid_pred) else {"p50": float("nan"), "p95": float("nan")}
    gt_stats = _robust_stats(gt_m[valid_gt]) if np.any(valid_gt) else {"p50": float("nan"), "p95": float("nan")}

    frac_gt = float(overlap.sum()) / float(valid_gt.sum() + 1e-9)
    frac_pred = float(overlap.sum()) / float(valid_pred.sum() + 1e-9)

    abs_e_1d: Optional[np.ndarray] = None
    signed_1d: Optional[np.ndarray] = None

    if np.any(overlap):
        signed = (pred_m[overlap] - gt_m[overlap]).astype(np.float32)
        abs_e = np.abs(signed)

        mae = float(np.mean(abs_e))
        rmse = float(np.sqrt(np.mean(signed * signed)))

        abs_p50 = _quantile_safe(abs_e, 0.50)
        abs_p95 = _quantile_safe(abs_e, 0.95)
        abs_p99 = _quantile_safe(abs_e, 0.99)
        abs_max = float(np.max(abs_e))

        bias_mean = float(np.mean(signed))
        bias_p50 = _quantile_safe(signed, 0.50)

        out_1cm = float(np.mean(abs_e > 0.01))
        out_2cm = float(np.mean(abs_e > 0.02))
        out_5cm = float(np.mean(abs_e > 0.05))
        out_10cm = float(np.mean(abs_e > 0.10))

        abs_e_1d = abs_e
        signed_1d = signed
    else:
        mae = rmse = abs_p50 = abs_p95 = abs_p99 = abs_max = float("nan")
        bias_mean = bias_p50 = float("nan")
        out_1cm = out_2cm = out_5cm = out_10cm = float("nan")

    vf = VariantFrame(
        pred_valid_ratio=pred_valid_ratio,
        gt_valid_ratio=gt_valid_ratio,
        overlap_ratio=overlap_ratio,
        frac_gt=frac_gt,
        frac_pred=frac_pred,
        pred_p50_m=float(pred_stats.get("p50", float("nan"))),
        pred_p95_m=float(pred_stats.get("p95", float("nan"))),
        gt_p50_m=float(gt_stats.get("p50", float("nan"))),
        gt_p95_m=float(gt_stats.get("p95", float("nan"))),
        mae_m=mae,
        rmse_m=rmse,
        abs_p50_m=abs_p50,
        abs_p95_m=abs_p95,
        abs_p99_m=abs_p99,
        abs_max_m=abs_max,
        bias_mean_m=bias_mean,
        bias_p50_m=bias_p50,
        out_1cm=out_1cm,
        out_2cm=out_2cm,
        out_5cm=out_5cm,
        out_10cm=out_10cm,
    )
    return vf, abs_e_1d, signed_1d


def analyze(
    run_dir: str,
    *,
    depth_scale_mm: float,
    depth_max_m: float,
    vis_count: int,
) -> Tuple[Dict[str, object], List[FrameRow]]:
    run_dir = os.path.abspath(run_dir)

    gt_dir = os.path.join(run_dir, "depth_gt")
    plain_dir = os.path.join(run_dir, "depth_sgm_align")
    rect_dir = os.path.join(run_dir, "depth_rectify_sgm_align")

    raw_plain_dir = os.path.join(run_dir, "depth_sgm_raw")
    raw_rect_dir = os.path.join(run_dir, "depth_rectify_sgm_raw")

    disp_plain_dir = os.path.join(run_dir, "disp_npy")
    disp_rect_dir = os.path.join(run_dir, "disp_rectify_npy")

    rgb_dir = os.path.join(run_dir, "rgb")
    irL_dir = os.path.join(run_dir, "ir_left")

    gt_paths = _list_pngs(gt_dir)
    plain_paths = _list_pngs(plain_dir)
    rect_paths = _list_pngs(rect_dir)

    if not gt_paths:
        raise RuntimeError(f"No depth_gt/*.png in {gt_dir}")
    if not plain_paths:
        raise RuntimeError(f"No depth_sgm_align/*.png in {plain_dir}")

    gt_map = {_basename_no_ext(p): p for p in gt_paths}
    plain_map = {_basename_no_ext(p): p for p in plain_paths}

    have_rect = bool(rect_paths)
    rect_map = {_basename_no_ext(p): p for p in rect_paths} if have_rect else {}

    # frame ids:
    #   - always intersect gt & plain
    #   - if rectify exists -> intersect with rectify too (fair comparison)
    fids = sorted(set(gt_map.keys()) & set(plain_map.keys()))
    if have_rect:
        fids = sorted(set(fids) & set(rect_map.keys()))

    if not fids:
        raise RuntimeError("No matching frame ids between required folders (gt + plain [+ rectify])")

    # optional maps
    raw_plain_map = {_basename_no_ext(p): p for p in _list_pngs(raw_plain_dir)} if os.path.isdir(raw_plain_dir) else {}
    raw_rect_map = {_basename_no_ext(p): p for p in _list_pngs(raw_rect_dir)} if os.path.isdir(raw_rect_dir) else {}

    disp_plain_map = {_basename_no_ext(p): p for p in _list_npys(disp_plain_dir)} if os.path.isdir(disp_plain_dir) else {}
    disp_rect_map = {_basename_no_ext(p): p for p in _list_npys(disp_rect_dir)} if os.path.isdir(disp_rect_dir) else {}

    H0 = W0 = None

    rows: List[FrameRow] = []

    # global accumulators (pixelwise) for each variant
    plain_abs_all: List[np.ndarray] = []
    plain_signed_all: List[np.ndarray] = []
    rect_abs_all: List[np.ndarray] = []
    rect_signed_all: List[np.ndarray] = []

    # per-frame validity arrays for summaries
    plain_valid_ratios: List[float] = []
    rect_valid_ratios: List[float] = []
    gt_valid_ratios: List[float] = []
    plain_overlap_ratios: List[float] = []
    rect_overlap_ratios: List[float] = []

    # raw stats
    raw_plain_valid_ratios: List[float] = []
    raw_rect_valid_ratios: List[float] = []
    raw_plain_depth_vals: List[np.ndarray] = []
    raw_rect_depth_vals: List[np.ndarray] = []

    for fid in fids:
        gt_u16 = _read_u16_png(gt_map[fid])
        plain_u16 = _read_u16_png(plain_map[fid])

        if gt_u16.shape != plain_u16.shape:
            raise RuntimeError(
                f"Shape mismatch '{fid}': gt {gt_u16.shape} vs plain {plain_u16.shape}. "
                f"These must be RGB grid both."
            )

        rect_u16 = None
        if have_rect:
            rect_u16 = _read_u16_png(rect_map[fid])
            if rect_u16.shape != gt_u16.shape:
                raise RuntimeError(
                    f"Shape mismatch '{fid}': gt {gt_u16.shape} vs rectify {rect_u16.shape}. "
                    f"These must be RGB grid both."
                )

        if H0 is None:
            H0, W0 = gt_u16.shape

        # compute metrics (plain)
        plain_m, plain_abs, plain_signed = _compute_variant(plain_u16, gt_u16, depth_scale_mm)
        if plain_abs is not None:
            plain_abs_all.append(plain_abs)
        if plain_signed is not None:
            plain_signed_all.append(plain_signed)

        # compute metrics (rectify)
        rect_m: Optional[VariantFrame] = None
        rect_abs = rect_signed = None
        if rect_u16 is not None:
            rect_m, rect_abs, rect_signed = _compute_variant(rect_u16, gt_u16, depth_scale_mm)
            if rect_abs is not None:
                rect_abs_all.append(rect_abs)
            if rect_signed is not None:
                rect_signed_all.append(rect_signed)

        # diffs (rect - plain)
        if rect_m is not None:
            diff_mae = rect_m.mae_m - plain_m.mae_m if (np.isfinite(rect_m.mae_m) and np.isfinite(plain_m.mae_m)) else float("nan")
            diff_p95 = rect_m.abs_p95_m - plain_m.abs_p95_m if (np.isfinite(rect_m.abs_p95_m) and np.isfinite(plain_m.abs_p95_m)) else float("nan")
            diff_bias = rect_m.bias_mean_m - plain_m.bias_mean_m if (np.isfinite(rect_m.bias_mean_m) and np.isfinite(plain_m.bias_mean_m)) else float("nan")
            diff_ov = rect_m.overlap_ratio - plain_m.overlap_ratio
        else:
            diff_mae = diff_p95 = diff_bias = diff_ov = float("nan")

        # raw validity
        raw_plain_v = None
        if fid in raw_plain_map:
            ru = _read_u16_png(raw_plain_map[fid])
            rv = (ru > 0)
            raw_plain_v = float(rv.mean())
            raw_plain_valid_ratios.append(raw_plain_v)
            if np.any(rv):
                raw_plain_depth_vals.append(u16_to_m(ru[rv], depth_scale_mm))

        raw_rect_v = None
        if fid in raw_rect_map:
            ru = _read_u16_png(raw_rect_map[fid])
            rv = (ru > 0)
            raw_rect_v = float(rv.mean())
            raw_rect_valid_ratios.append(raw_rect_v)
            if np.any(rv):
                raw_rect_depth_vals.append(u16_to_m(ru[rv], depth_scale_mm))

        rows.append(FrameRow(
            frame_id=fid,
            plain=plain_m,
            rectify=rect_m,
            diff_mae_m=diff_mae,
            diff_abs_p95_m=diff_p95,
            diff_bias_mean_m=diff_bias,
            diff_overlap_ratio=diff_ov,
            raw_plain_valid_ratio=raw_plain_v,
            raw_rect_valid_ratio=raw_rect_v,
        ))

        # collect per-frame ratios
        plain_valid_ratios.append(plain_m.pred_valid_ratio)
        gt_valid_ratios.append(plain_m.gt_valid_ratio)  # gt ratio same for both variants
        plain_overlap_ratios.append(plain_m.overlap_ratio)

        if rect_m is not None:
            rect_valid_ratios.append(rect_m.pred_valid_ratio)
            rect_overlap_ratios.append(rect_m.overlap_ratio)

    # ---------------- summary helpers ----------------

    def _ratio_pack(v: List[float]) -> Dict[str, float]:
        a = np.asarray(v, dtype=np.float32)
        return {
            "mean": float(np.mean(a)),
            "p5": float(np.quantile(a, 0.05)),
            "p50": float(np.quantile(a, 0.50)),
            "p95": float(np.quantile(a, 0.95)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }

    def _error_pack(abs_list: List[np.ndarray], signed_list: List[np.ndarray]) -> Tuple[Dict[str, float], Dict[str, float]]:
        if not abs_list or not signed_list:
            return {"count": 0.0}, {"count": 0.0}
        abs_all = np.concatenate(abs_list, axis=0).astype(np.float32)
        signed_all = np.concatenate(signed_list, axis=0).astype(np.float32)

        abs_stats = _robust_stats(abs_all)
        signed_stats = _robust_stats(signed_all)

        abs_stats.update({
            "mae": float(np.mean(abs_all)),
            "rmse": float(np.sqrt(np.mean(signed_all * signed_all))),
            "out_1cm": float(np.mean(abs_all > 0.01)),
            "out_2cm": float(np.mean(abs_all > 0.02)),
            "out_5cm": float(np.mean(abs_all > 0.05)),
            "out_10cm": float(np.mean(abs_all > 0.10)),
        })
        signed_stats.update({
            "bias_mean": float(np.mean(signed_all)),
            "bias_p50": float(np.quantile(signed_all, 0.50)),
        })
        return abs_stats, signed_stats

    plain_abs_stats, plain_signed_stats = _error_pack(plain_abs_all, plain_signed_all)

    rect_abs_stats, rect_signed_stats = ({"count": 0.0}, {"count": 0.0})
    if have_rect:
        rect_abs_stats, rect_signed_stats = _error_pack(rect_abs_all, rect_signed_all)

    # deltas (rect - plain) on global summary (use scalar stats)
    delta_summary: Dict[str, float] = {}
    if have_rect and plain_abs_stats.get("count", 0.0) > 0 and rect_abs_stats.get("count", 0.0) > 0:
        # NOTE: these are deltas of summary numbers, NOT pixelwise delta distribution.
        for k in ("mae", "rmse", "p50", "p95", "p99", "bias_mean"):
            a = rect_abs_stats.get(k, None)
            b = plain_abs_stats.get(k, None)
            if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                delta_summary[f"abs_{k}_rect_minus_plain"] = float(a - b)
        a = rect_signed_stats.get("bias_mean", None)
        b = plain_signed_stats.get("bias_mean", None)
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
            delta_summary["signed_bias_mean_rect_minus_plain"] = float(a - b)

    summary: Dict[str, object] = {
        "run_dir": run_dir,
        "frame_count": int(len(rows)),
        "image_hw": [int(H0 or 0), int(W0 or 0)],
        "depth_scale_mm": float(depth_scale_mm),
        "depth_max_m": float(depth_max_m),

        "plain": {
            "pred_valid_ratio": _ratio_pack(plain_valid_ratios),
            "gt_valid_ratio": _ratio_pack(gt_valid_ratios),
            "overlap_ratio": _ratio_pack(plain_overlap_ratios),
            "abs_error_m": plain_abs_stats,
            "signed_error_m": plain_signed_stats,
            "raw_present": bool(bool(raw_plain_map)),
            "disp_present": bool(bool(disp_plain_map)),
        },

        "rectify": {
            "present": bool(have_rect),
            "pred_valid_ratio": _ratio_pack(rect_valid_ratios) if rect_valid_ratios else {"count": 0.0},
            "gt_valid_ratio": _ratio_pack(gt_valid_ratios),
            "overlap_ratio": _ratio_pack(rect_overlap_ratios) if rect_overlap_ratios else {"count": 0.0},
            "abs_error_m": rect_abs_stats,
            "signed_error_m": rect_signed_stats,
            "raw_present": bool(bool(raw_rect_map)),
            "disp_present": bool(bool(disp_rect_map)),
        },

        "delta_summary": delta_summary,
    }

    if raw_plain_valid_ratios:
        summary["plain"]["raw_valid_ratio"] = {
            "mean": float(np.mean(np.asarray(raw_plain_valid_ratios, dtype=np.float32))),
            "p50": float(np.quantile(np.asarray(raw_plain_valid_ratios, dtype=np.float32), 0.50)),
            "min": float(np.min(np.asarray(raw_plain_valid_ratios, dtype=np.float32))),
            "max": float(np.max(np.asarray(raw_plain_valid_ratios, dtype=np.float32))),
            "note": "raw plain grid (not compared to GT).",
        }
    if raw_plain_depth_vals:
        summary["plain"]["raw_depth_stats_m"] = _robust_stats(np.concatenate(raw_plain_depth_vals, axis=0).astype(np.float32))

    if raw_rect_valid_ratios:
        summary["rectify"]["raw_valid_ratio"] = {
            "mean": float(np.mean(np.asarray(raw_rect_valid_ratios, dtype=np.float32))),
            "p50": float(np.quantile(np.asarray(raw_rect_valid_ratios, dtype=np.float32), 0.50)),
            "min": float(np.min(np.asarray(raw_rect_valid_ratios, dtype=np.float32))),
            "max": float(np.max(np.asarray(raw_rect_valid_ratios, dtype=np.float32))),
            "note": "raw rectified grid (not compared to GT).",
        }
    if raw_rect_depth_vals:
        summary["rectify"]["raw_depth_stats_m"] = _robust_stats(np.concatenate(raw_rect_depth_vals, axis=0).astype(np.float32))

    # ---------------- visuals selection ----------------
    if vis_count > 0:
        # worst by abs_p95 (plain)
        finite_plain = [r for r in rows if np.isfinite(r.plain.abs_p95_m)]
        worst_plain = sorted(finite_plain, key=lambda r: r.plain.abs_p95_m, reverse=True)

        # low overlap (plain)
        low_overlap_plain = sorted(rows, key=lambda r: r.plain.overlap_ratio)

        picks: List[str] = []
        half = max(1, vis_count // 3)

        for r in worst_plain[:half]:
            picks.append(r.frame_id)
        for r in low_overlap_plain[:half]:
            picks.append(r.frame_id)

        if have_rect:
            finite_rect = [r for r in rows if r.rectify is not None and np.isfinite(r.rectify.abs_p95_m)]
            worst_rect = sorted(finite_rect, key=lambda r: r.rectify.abs_p95_m, reverse=True)
            for r in worst_rect[:half]:
                picks.append(r.frame_id)

            # biggest delta: rect - plain (positive = worse)
            finite_delta = [r for r in rows if np.isfinite(r.diff_abs_p95_m)]
            worst_delta = sorted(finite_delta, key=lambda r: r.diff_abs_p95_m, reverse=True)
            for r in worst_delta[: max(1, vis_count - len(picks))]:
                picks.append(r.frame_id)

        # uniq preserve order, clamp to vis_count
        seen = set()
        picks = [x for x in picks if not (x in seen or seen.add(x))]
        picks = picks[:vis_count]

        vis_dir = os.path.join(run_dir, "analysis", "vis")
        _ensure_dir(vis_dir)

        for fid in picks:
            gt_u16 = _read_u16_png(gt_map[fid])
            plain_u16 = _read_u16_png(plain_map[fid])

            gt_m = u16_to_m(gt_u16, depth_scale_mm)
            plain_m = u16_to_m(plain_u16, depth_scale_mm)
            valid_gt = gt_u16 > 0
            valid_plain = plain_u16 > 0

            rect_m = None
            valid_rect = None
            if have_rect:
                rect_u16 = _read_u16_png(rect_map[fid])
                rect_m = u16_to_m(rect_u16, depth_scale_mm)
                valid_rect = rect_u16 > 0

            rgb_bgr = _try_read_rgb(os.path.join(rgb_dir, fid)) if os.path.isdir(rgb_dir) else None
            irL_u8 = _try_read_any_gray(os.path.join(irL_dir, fid)) if os.path.isdir(irL_dir) else None

            disp_plain = None
            if fid in disp_plain_map:
                try:
                    disp_plain = np.load(disp_plain_map[fid]).astype(np.float32)
                except Exception:
                    disp_plain = None

            disp_rect = None
            if fid in disp_rect_map:
                try:
                    disp_rect = np.load(disp_rect_map[fid]).astype(np.float32)
                except Exception:
                    disp_rect = None

            _save_vis_bundle(
                vis_dir,
                fid,
                gt_m=gt_m,
                valid_gt=valid_gt,
                plain_m=plain_m,
                valid_plain=valid_plain,
                rect_m=rect_m,
                valid_rect=valid_rect,
                depth_max_m=depth_max_m,
                rgb_bgr=rgb_bgr,
                irL_u8=irL_u8,
                disp_plain=disp_plain,
                disp_rect=disp_rect,
            )

    return summary, rows


def write_outputs(run_dir: str, summary: Dict[str, object], rows: List[FrameRow]) -> None:
    out_dir = os.path.join(run_dir, "analysis")
    _ensure_dir(out_dir)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(out_dir, "per_frame.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_id",

            # plain
            "plain_pred_valid_ratio",
            "plain_gt_valid_ratio",
            "plain_overlap_ratio",
            "plain_frac_gt",
            "plain_frac_pred",
            "plain_mae_m",
            "plain_rmse_m",
            "plain_abs_p50_m",
            "plain_abs_p95_m",
            "plain_abs_p99_m",
            "plain_abs_max_m",
            "plain_bias_mean_m",
            "plain_bias_p50_m",
            "plain_out_1cm",
            "plain_out_2cm",
            "plain_out_5cm",
            "plain_out_10cm",

            # rectify
            "rect_pred_valid_ratio",
            "rect_overlap_ratio",
            "rect_mae_m",
            "rect_rmse_m",
            "rect_abs_p50_m",
            "rect_abs_p95_m",
            "rect_abs_p99_m",
            "rect_abs_max_m",
            "rect_bias_mean_m",
            "rect_bias_p50_m",
            "rect_out_1cm",
            "rect_out_2cm",
            "rect_out_5cm",
            "rect_out_10cm",

            # diffs (rect - plain)
            "diff_mae_m",
            "diff_abs_p95_m",
            "diff_bias_mean_m",
            "diff_overlap_ratio",

            # raw validity
            "raw_plain_valid_ratio",
            "raw_rect_valid_ratio",
        ])

        for r in rows:
            p = r.plain
            q = r.rectify

            def fmt(x: float) -> str:
                return "" if (x is None or not np.isfinite(x)) else f"{x:.6f}"

            w.writerow([
                r.frame_id,

                fmt(p.pred_valid_ratio),
                fmt(p.gt_valid_ratio),
                fmt(p.overlap_ratio),
                fmt(p.frac_gt),
                fmt(p.frac_pred),
                fmt(p.mae_m),
                fmt(p.rmse_m),
                fmt(p.abs_p50_m),
                fmt(p.abs_p95_m),
                fmt(p.abs_p99_m),
                fmt(p.abs_max_m),
                fmt(p.bias_mean_m),
                fmt(p.bias_p50_m),
                fmt(p.out_1cm),
                fmt(p.out_2cm),
                fmt(p.out_5cm),
                fmt(p.out_10cm),

                fmt(q.pred_valid_ratio) if q is not None else "",
                fmt(q.overlap_ratio) if q is not None else "",
                fmt(q.mae_m) if q is not None else "",
                fmt(q.rmse_m) if q is not None else "",
                fmt(q.abs_p50_m) if q is not None else "",
                fmt(q.abs_p95_m) if q is not None else "",
                fmt(q.abs_p99_m) if q is not None else "",
                fmt(q.abs_max_m) if q is not None else "",
                fmt(q.bias_mean_m) if q is not None else "",
                fmt(q.bias_p50_m) if q is not None else "",
                fmt(q.out_1cm) if q is not None else "",
                fmt(q.out_2cm) if q is not None else "",
                fmt(q.out_5cm) if q is not None else "",
                fmt(q.out_10cm) if q is not None else "",

                fmt(r.diff_mae_m),
                fmt(r.diff_abs_p95_m),
                fmt(r.diff_bias_mean_m),
                fmt(r.diff_overlap_ratio),

                "" if r.raw_plain_valid_ratio is None else fmt(r.raw_plain_valid_ratio),
                "" if r.raw_rect_valid_ratio is None else fmt(r.raw_rect_valid_ratio),
            ])


# ------------------------- CLI -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output_root",
        type=str,
        default="output",
        help="Root output folder (default: ./output). Script auto-picks newest run_dir under it.",
    )
    p.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Explicit run_dir. If set, overrides auto-detection.",
    )
    p.add_argument(
        "--depth_scale_mm",
        type=float,
        default=1.0,
        help="depth_scale_mm used in writer (default: 1.0).",
    )
    p.add_argument(
        "--depth_max",
        type=float,
        default=3.0,
        help="Depth max (meters) used for depth visualization scaling (default: 3.0).",
    )
    p.add_argument(
        "--vis_count",
        type=int,
        default=12,
        help="How many frames to visualize into run_dir/analysis/vis (default: 12).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir) if args.run_dir else os.path.abspath(_default_run_dir(args.output_root))

    print(f"[analyze] run_dir = {run_dir}")
    summary, rows = analyze(
        run_dir,
        depth_scale_mm=float(args.depth_scale_mm),
        depth_max_m=float(args.depth_max),
        vis_count=int(args.vis_count),
    )
    write_outputs(run_dir, summary, rows)

    print("\n=== SUMMARY ===")
    print("frames:", summary.get("frame_count"))
    print("plain.abs_error_m:", summary.get("plain", {}).get("abs_error_m", {}))
    print("plain.signed_error_m:", summary.get("plain", {}).get("signed_error_m", {}))

    if summary.get("rectify", {}).get("present", False):
        print("rect.abs_error_m:", summary.get("rectify", {}).get("abs_error_m", {}))
        print("rect.signed_error_m:", summary.get("rectify", {}).get("signed_error_m", {}))
        print("delta_summary:", summary.get("delta_summary", {}))
    else:
        print("[WARN] depth_rectify_sgm_align is missing -> rectified metrics skipped.")

    # strong sanity warnings
    plain_ov = summary.get("plain", {}).get("overlap_ratio", {}).get("mean", None)
    if plain_ov is not None and float(plain_ov) < 0.05:
        print("\n[WARN] plain overlap(valid_pred & valid_gt) < 5% in mean.")
        print("       Almost always means pred/gt are NOT in the same RGB grid.")

    print(f"\nWrote: {os.path.join(run_dir, 'analysis', 'summary.json')}")
    print(f"Wrote: {os.path.join(run_dir, 'analysis', 'per_frame.csv')}")
    if int(args.vis_count) > 0:
        print(f"Wrote visuals to: {os.path.join(run_dir, 'analysis', 'vis')}")


if __name__ == "__main__":
    main()
