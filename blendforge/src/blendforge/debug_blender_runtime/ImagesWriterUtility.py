import os
import numpy as np
import cv2
from typing import Optional, Sequence


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

    # naming
    start_index: int = 0,     # если хочешь продолжать нумерацию
):
    """
    Saves always:
      rgb/000000.{rgb_ext}
      ir_left/000000.png
      ir_right/000000.png
      depth_ir_left_rect/000000.png          (u16) depth in rectified IR_LEFT grid
      depth_color_from_ir_rect/000000.png    (u16) depth aligned into COLOR grid

    Saves optionally:
      depth_gt_rgb/000000.png                (u16) GT depth aligned to COLOR grid
      disp_rect_npy/000000.npy               (float32 disparity)
      disp_rect_vis/000000.png               (u8 visualization, if save_disp_png=True)
    """

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
        d_ir_u16 = meters_to_depth_u16(depth_ir_left_rect_m[k], depth_scale_mm)
        _save_png_u16(os.path.join(output_dir, "depth_ir_left_rect", f"{name}.png"), d_ir_u16)

        # depth aligned into COLOR grid
        d_rgb_u16 = meters_to_depth_u16(depth_color_from_ir_rect_m[k], depth_scale_mm)
        _save_png_u16(os.path.join(output_dir, "depth_color_from_ir_rect", f"{name}.png"), d_rgb_u16)

        # GT (optional)
        if depth_gt_rgb_m is not None:
            gt_u16 = meters_to_depth_u16(depth_gt_rgb_m[k], depth_scale_mm)
            _save_png_u16(os.path.join(output_dir, "depth_gt_rgb", f"{name}.png"), gt_u16)

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
