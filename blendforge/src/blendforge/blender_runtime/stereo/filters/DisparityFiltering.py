#stereo/filters/DisparityFiltering.py
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from blendforge.blender_runtime.stereo.types.StereoTypes import FillMode

# optional ximgproc
try:
    import cv2.ximgproc as xip  # type: ignore
    _HAVE_XIMGPROC = True
except Exception:
    xip = None
    _HAVE_XIMGPROC = False


def has_ximgproc() -> bool:
    return bool(_HAVE_XIMGPROC)


def sanitize_disparity(disp: np.ndarray) -> np.ndarray:
    out = np.asarray(disp, dtype=np.float32).copy()
    out[~np.isfinite(out)] = 0.0
    out[out <= 0.0] = 0.0
    return out


def apply_wls_filter(
    dispL_i16: np.ndarray,
    dispR_i16: np.ndarray,
    left_u8: np.ndarray,
    matcher_left: Any,
    *,
    lambda_value: float = 80000.0,
    sigma_color: float = 1.2,
) -> np.ndarray:
    """
    Apply OpenCV ximgproc WLS filtering to disparity.
    Returns float32 disparity in px (not x16), invalid<=0 are preserved as 0 after sanitize.
    """
    if not _HAVE_XIMGPROC:
        raise RuntimeError("cv2.ximgproc is not available. Install opencv-contrib-python.")

    if left_u8.ndim != 2 or left_u8.dtype != np.uint8:
        raise ValueError("left_u8 must be grayscale uint8 [H,W]")

    wls = xip.createDisparityWLSFilter(matcher_left=matcher_left)
    wls.setLambda(float(lambda_value))
    wls.setSigmaColor(float(sigma_color))

    disp_f = wls.filter(dispL_i16, left_u8, None, dispR_i16).astype(np.float32) / 16.0
    return sanitize_disparity(disp_f)


def apply_speckle_filter(
    disp: np.ndarray,
    *,
    max_speckle_size: int = 100,
    max_diff_disp16: int = 16,
    new_val: int = 0,
) -> np.ndarray:
    """
    OpenCV speckle filtering on disparity.

    OpenCV filterSpeckles works on disparity in fixed-point int16 (x16 scale).
    """
    out = sanitize_disparity(disp)
    tmp = np.round(out * 16.0).astype(np.int16)

    cv2.filterSpeckles(
        tmp,
        int(new_val),
        int(max_speckle_size),
        int(max_diff_disp16),
    )

    out = tmp.astype(np.float32) / 16.0
    return sanitize_disparity(out)


def fill_disp_mean(disp: np.ndarray, k: int = 5, iters: int = 1) -> np.ndarray:
    out = disp.astype(np.float32, copy=True)
    k = int(max(3, k))
    if k % 2 == 0:
        k += 1

    for _ in range(int(max(0, iters))):
        hole = out <= 0.0
        if not hole.any():
            break
        valid = (out > 0.0).astype(np.float32)
        num = cv2.blur(out * valid, (k, k))
        den = cv2.blur(valid, (k, k))
        filled = num / (den + 1e-6)
        out[hole] = filled[hole]

    return sanitize_disparity(out)


def fill_disp_dilate_max(disp: np.ndarray, iters: int = 1) -> np.ndarray:
    out = disp.astype(np.float32, copy=True)
    mask = out <= 0.0
    if not mask.any():
        return sanitize_disparity(out)

    kernel = np.ones((3, 3), np.uint8)
    for _ in range(int(max(0, iters))):
        dil = cv2.dilate(out, kernel)
        out[mask] = dil[mask]
        mask = out <= 0.0
        if not mask.any():
            break

    return sanitize_disparity(out)


def fill_disparity(
    disp: np.ndarray,
    *,
    mode: FillMode = "none",
    iters: int = 0,
) -> np.ndarray:
    """
    Unified entry point for disparity hole filling.
    """
    if mode == "none" or int(iters) <= 0:
        return sanitize_disparity(disp)

    if mode == "mean":
        return fill_disp_mean(disp, k=5, iters=int(iters))

    if mode == "dilate_max":
        return fill_disp_dilate_max(disp, iters=int(iters))

    raise ValueError(f"Unknown FillMode: {mode}")