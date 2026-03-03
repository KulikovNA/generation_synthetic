from __future__ import annotations

import os
from typing import Literal, Optional, Sequence

import cv2
import numpy as np


DepthSaveMode = Literal["u16", "heatmap"]


def save_rgb_ir_stereo_rectified(
    output_dir: str,
    rgb_ext: str,
    color_file_format: str,
    jpg_quality: int,
    data_rgbs: Sequence[np.ndarray],
    data_ir_lefts: Sequence[np.ndarray],
    data_ir_rights: Sequence[np.ndarray],
    depth_scale_mm: float,
    *,
    # --- RECTIFIED only ---
    depth_ir_left_rect_m: Sequence[np.ndarray],          # list[np.ndarray] depth (m) in rectified IR_LEFT grid
    depth_color_from_ir_rect_m: Sequence[np.ndarray],    # list[np.ndarray] depth (m) aligned to COLOR grid

    # --- GT (optional, usually aligned to COLOR grid) ---
    depth_gt_rgb_m: Optional[Sequence[np.ndarray]] = None,

    # --- disparity (optional, rectified) ---
    disp_rect_px: Optional[Sequence[np.ndarray]] = None,
    save_disp_png: bool = False,

    # --- depth save mode ---
    depth_save_mode: DepthSaveMode = "u16",

    # naming
    start_index: int = 0,
):
    """
    Saves always:
      rgb/000000.{rgb_ext}
      ir_left/000000.png
      ir_right/000000.png

    Depth folders (same folders, mode-dependent):
      depth_ir_left_rect/000000.png
      depth_color_from_ir_rect/000000.png
      depth_gt_rgb/000000.png   (if provided)

    If depth_save_mode == "u16":
      - saves uint16 depth PNG (metric depth)
    If depth_save_mode == "heatmap":
      - saves color heatmap PNG (uint8 BGR), invalid=black

    Disparity optionally:
      disp_rect_npy/000000.npy
      disp_rect_vis/000000.png  (if save_disp_png=True)
    """

    if depth_save_mode not in ("u16", "heatmap"):
        raise ValueError(f"Unknown depth_save_mode: {depth_save_mode}")

    n = len(data_rgbs)

    # -------------------- sanity checks --------------------
    if len(data_ir_lefts) != n or len(data_ir_rights) != n:
        raise ValueError("Input lists must have same length (rgbs, ir_left, ir_right).")

    if len(depth_ir_left_rect_m) != n:
        raise ValueError("depth_ir_left_rect_m length must match number of frames.")
    if len(depth_color_from_ir_rect_m) != n:
        raise ValueError("depth_color_from_ir_rect_m length must match number of frames.")

    if depth_gt_rgb_m is not None and len(depth_gt_rgb_m) != n:
        raise ValueError("depth_gt_rgb_m length must match number of frames.")

    if disp_rect_px is not None and len(disp_rect_px) != n:
        raise ValueError("disp_rect_px length must match number of frames.")

    # -------------------- dirs --------------------
    _ensure_dir(os.path.join(output_dir, "rgb"))
    _ensure_dir(os.path.join(output_dir, "ir_left"))
    _ensure_dir(os.path.join(output_dir, "ir_right"))

    _ensure_dir(os.path.join(output_dir, "depth_ir_left_rect"))
    _ensure_dir(os.path.join(output_dir, "depth_color_from_ir_rect"))

    if depth_gt_rgb_m is not None:
        _ensure_dir(os.path.join(output_dir, "depth_gt_rgb"))

    if disp_rect_px is not None:
        _ensure_dir(os.path.join(output_dir, "disp_rect_npy"))
        if save_disp_png:
            _ensure_dir(os.path.join(output_dir, "disp_rect_vis"))

    # -------------------- per-frame save --------------------
    for k in range(n):
        idx = start_index + k
        name = f"{idx:06d}"

        # RGB
        _save_rgb(
            os.path.join(output_dir, "rgb", f"{name}.{rgb_ext}"),
            data_rgbs[k],
            fmt=color_file_format,
            jpg_quality=jpg_quality,
        )

        # IR (gray u8)
        _save_gray_u8(os.path.join(output_dir, "ir_left", f"{name}.png"), data_ir_lefts[k])
        _save_gray_u8(os.path.join(output_dir, "ir_right", f"{name}.png"), data_ir_rights[k])

        # depth rectified IR_LEFT grid
        _save_depth(
            os.path.join(output_dir, "depth_ir_left_rect", f"{name}.png"),
            depth_ir_left_rect_m[k],
            depth_scale_mm=depth_scale_mm,
            mode=depth_save_mode,
        )

        # depth aligned into COLOR grid
        _save_depth(
            os.path.join(output_dir, "depth_color_from_ir_rect", f"{name}.png"),
            depth_color_from_ir_rect_m[k],
            depth_scale_mm=depth_scale_mm,
            mode=depth_save_mode,
        )

        # GT (optional)
        if depth_gt_rgb_m is not None:
            _save_depth(
                os.path.join(output_dir, "depth_gt_rgb", f"{name}.png"),
                depth_gt_rgb_m[k],
                depth_scale_mm=depth_scale_mm,
                mode=depth_save_mode,
            )

        # disparity (optional)
        if disp_rect_px is not None:
            disp = np.asarray(disp_rect_px[k], dtype=np.float32)
            np.save(os.path.join(output_dir, "disp_rect_npy", f"{name}.npy"), disp)
            if save_disp_png:
                vis = _disp_to_u8_vis(disp)
                cv2.imwrite(os.path.join(output_dir, "disp_rect_vis", f"{name}.png"), vis)


# -------------------- helpers --------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _as_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Accepts float [0..1] or uint8; returns uint8 RGB (3ch)."""
    img = np.asarray(img)
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
    f = fmt.upper()
    if f == "PNG":
        cv2.imwrite(path, bgr)
    elif f in ("JPG", "JPEG"):
        cv2.imwrite(path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
    else:
        raise ValueError(f"Unknown color format: {fmt}")


def _save_gray_u8(path: str, img: np.ndarray) -> None:
    """Save grayscale 8-bit PNG. Accepts float [0..1], uint8 RGB, uint8 gray."""
    _ensure_dir(os.path.dirname(path))
    img = np.asarray(img)

    if img.ndim == 3:
        rgb = _as_uint8_rgb(img)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        if img.dtype == np.uint8:
            gray = img
        else:
            gray = np.clip(img, 0.0, 1.0)
            gray = (gray * 255.0 + 0.5).astype(np.uint8)

    cv2.imwrite(path, gray)


def _save_png_u16(path: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    if arr.dtype != np.uint16:
        raise ValueError(f"_save_png_u16 expects uint16, got {arr.dtype}")
    _ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, arr)


def _save_depth(
    path: str,
    depth_m: np.ndarray,
    *,
    depth_scale_mm: float,
    mode: DepthSaveMode,
) -> None:
    if mode == "u16":
        depth_u16 = meters_to_depth_u16(depth_m, depth_scale_mm)
        _save_png_u16(path, depth_u16)
        return

    if mode == "heatmap":
        heat = depth_to_heatmap(depth_m)
        _ensure_dir(os.path.dirname(path))
        cv2.imwrite(path, heat)
        return

    raise ValueError(f"Unknown depth save mode: {mode}")


def meters_to_depth_u16(depth_m: np.ndarray, depth_scale_mm: float = 1.0) -> np.ndarray:
    """
    BOP-compatible:
      depth_u16 = round((depth_m * 1000) / depth_scale_mm)
      0 means invalid
    """
    if depth_scale_mm <= 0:
        raise ValueError("depth_scale_mm must be > 0")

    d = np.asarray(depth_m, dtype=np.float32)
    d = np.where(np.isfinite(d), d, 0.0)
    d = np.where(d > 0.0, d, 0.0)

    depth_mm = d * 1000.0
    scaled = depth_mm / float(depth_scale_mm)
    scaled = np.clip(scaled, 0.0, 65535.0)
    return (scaled + 0.5).astype(np.uint16)


def depth_to_heatmap(
    depth_m: np.ndarray,
    *,
    far_quantile: float = 0.99,
) -> np.ndarray:
    """
    Convert depth [m] to heatmap preview.

    - invalid = black
    - near = hot
    - far = cold

    Returns uint8 BGR image, ready for cv2.imwrite.
    """
    d = np.asarray(depth_m, dtype=np.float32).copy()
    if d.ndim != 2:
        raise ValueError(f"depth_m must be [H,W], got {d.shape}")

    d[~np.isfinite(d)] = 0.0
    valid = d > 0.0

    if not np.any(valid):
        return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)

    q = float(np.clip(far_quantile, 0.5, 1.0))
    z_far = float(np.quantile(d[valid], q))
    z_far = max(z_far, 1e-6)

    # near -> 1, far -> 0
    norm = np.zeros_like(d, dtype=np.float32)
    norm[valid] = 1.0 - np.clip(d[valid] / z_far, 0.0, 1.0)

    gray = (norm * 255.0 + 0.5).astype(np.uint8)
    gray[~valid] = 0

    heat = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    heat[~valid] = 0
    return heat


def _disp_to_u8_vis(disp: np.ndarray) -> np.ndarray:
    d = np.asarray(disp, dtype=np.float32).copy()
    d[~np.isfinite(d)] = 0.0
    d[d <= 0.0] = 0.0
    if np.max(d) <= 0:
        return np.zeros_like(d, dtype=np.uint8)
    vmax = np.quantile(d[d > 0], 0.99) if np.any(d > 0) else float(np.max(d))
    vmax = float(max(vmax, 1e-6))
    v = np.clip(d / vmax, 0.0, 1.0)
    return (v * 255.0 + 0.5).astype(np.uint8)