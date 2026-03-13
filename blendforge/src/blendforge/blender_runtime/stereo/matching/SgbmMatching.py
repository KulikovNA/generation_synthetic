from __future__ import annotations

from typing import Any, Tuple

import cv2
import numpy as np

# optional ximgproc
try:
    import cv2.ximgproc as xip  # type: ignore
    _HAVE_XIMGPROC = True
except Exception:
    xip = None
    _HAVE_XIMGPROC = False


def has_ximgproc() -> bool:
    return bool(_HAVE_XIMGPROC)


def create_sgbm(
    block_size: int,
    num_disparities: int,
    min_disparity: int,
    *,
    mode: int = cv2.STEREO_SGBM_MODE_HH,
    uniqueness_ratio: int = 10,
    speckle_window_size: int = 100,
    speckle_range: int = 2,
    disp12_max_diff: int = 1,
    pre_filter_cap: int = 63,
    p1_scale: float = 8.0,
    p2_scale: float = 32.0,
) -> cv2.StereoSGBM:
    block_size = int(block_size)
    if block_size % 2 == 0:
        raise ValueError("block_size must be odd.")
    if num_disparities <= 0 or num_disparities % 16 != 0:
        raise ValueError("num_disparities must be >0 and divisible by 16.")
    if float(p1_scale) <= 0:
        raise ValueError("p1_scale must be > 0.")
    if float(p2_scale) <= 0:
        raise ValueError("p2_scale must be > 0.")
    if float(p2_scale) <= float(p1_scale):
        raise ValueError("p2_scale must be > p1_scale.")

    cn = 1  # grayscale

    # decoupled regularization
    P1 = max(1, int(round(float(p1_scale) * cn * (block_size ** 2))))
    P2 = max(P1 + 1, int(round(float(p2_scale) * cn * (block_size ** 2))))

    return cv2.StereoSGBM_create(
        minDisparity=int(min_disparity),
        numDisparities=int(num_disparities),
        blockSize=int(block_size),
        P1=int(P1),
        P2=int(P2),
        disp12MaxDiff=int(disp12_max_diff),
        uniquenessRatio=int(uniqueness_ratio),
        speckleWindowSize=int(speckle_window_size),
        speckleRange=int(speckle_range),
        preFilterCap=int(pre_filter_cap),
        mode=int(mode),
    )


def compute_sgbm_disparities(
    left_u8: np.ndarray,
    right_u8: np.ndarray,
    *,
    block_size: int,
    num_disparities: int,
    min_disparity: int,
    mode: int = cv2.STEREO_SGBM_MODE_HH,
    uniqueness_ratio: int = 10,
    sgbm_speckle_window_size: int = 100,
    sgbm_speckle_range: int = 2,
    disp12_max_diff: int = 1,
    pre_filter_cap: int = 63,
    p1_scale: float = 8.0,
    p2_scale: float = 32.0,
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute left/right SGBM disparities.

    Returns:
        matcher_left,
        dispL_i16, dispR_i16,   # OpenCV fixed-point disparities (x16)
        dispL_f32, dispR_f32    # float32 disparities in pixels
    """
    if left_u8.ndim != 2 or right_u8.ndim != 2:
        raise ValueError("left_u8/right_u8 must be grayscale [H,W]")
    if left_u8.dtype != np.uint8 or right_u8.dtype != np.uint8:
        raise ValueError("left_u8/right_u8 must be uint8")
    if left_u8.shape != right_u8.shape:
        raise ValueError(f"left/right shape mismatch: {left_u8.shape} vs {right_u8.shape}")

    matcher_left = create_sgbm(
        block_size=block_size,
        num_disparities=num_disparities,
        min_disparity=min_disparity,
        mode=mode,
        uniqueness_ratio=uniqueness_ratio,
        speckle_window_size=sgbm_speckle_window_size,
        speckle_range=sgbm_speckle_range,
        disp12_max_diff=disp12_max_diff,
        pre_filter_cap=pre_filter_cap,
        p1_scale=p1_scale,
        p2_scale=p2_scale,
    )

    dispL_i16 = matcher_left.compute(left_u8, right_u8)

    if _HAVE_XIMGPROC:
        matcher_right = xip.createRightMatcher(matcher_left)
        dispR_i16 = matcher_right.compute(right_u8, left_u8)
    else:
        # fallback: same matcher with swapped inputs
        dispR_i16 = matcher_left.compute(right_u8, left_u8)

    dispL_f32 = dispL_i16.astype(np.float32) / 16.0
    dispR_f32 = dispR_i16.astype(np.float32) / 16.0

    dispL_f32[~np.isfinite(dispL_f32)] = 0.0
    dispR_f32[~np.isfinite(dispR_f32)] = 0.0

    return matcher_left, dispL_i16, dispR_i16, dispL_f32, dispR_f32