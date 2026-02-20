#stereo/filters/DepthFiltering.py

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from blendforge.blender_runtime.stereo.types.StereoTypes import DepthRangePolicy


def disparity_to_depth(
    disp_px: np.ndarray,
    fx: float,
    baseline_m: float,
) -> np.ndarray:
    """
    Convert disparity [px] -> depth [m]:
        Z = fx * B / d

    Invalid disparity (<=0 or non-finite) -> 0.
    """
    d = np.asarray(disp_px, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError(f"disp_px must be [H,W], got {d.shape}")

    fx = float(fx)
    baseline_m = float(baseline_m)
    if fx <= 0 or baseline_m <= 0:
        raise ValueError("fx and baseline_m must be > 0")

    depth = np.zeros_like(d, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0.0)
    depth[valid] = (fx * baseline_m) / d[valid]
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0
    return depth


def apply_depth_range_policy(
    depth_m: np.ndarray,
    depth_min: float,
    depth_max: float,
    policy: DepthRangePolicy = "zero",
) -> np.ndarray:
    """
    Validate/clamp depth to [depth_min, depth_max].

    policy == "zero":
        values outside range -> 0
    policy == "clamp":
        values outside range -> clipped to bounds
    """
    depth = np.asarray(depth_m, dtype=np.float32).copy()
    if depth.ndim != 2:
        raise ValueError(f"depth_m must be [H,W], got {depth.shape}")

    dmin = float(max(depth_min, 0.0))
    dmax = float(max(depth_max, dmin + 1e-6))

    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0

    valid = depth > 0.0
    if policy == "clamp":
        depth[valid] = np.clip(depth[valid], dmin, dmax)
    elif policy == "zero":
        bad = valid & ((depth < dmin) | (depth > dmax))
        depth[bad] = 0.0
    else:
        raise ValueError(f"Unknown depth range policy: {policy}")

    return depth


def fill_in_fast(
    depth_map: np.ndarray,
    max_depth: float = 100.0,
    custom_kernel: Optional[np.ndarray] = None,
    extrapolate: bool = False,
    blur_type: str = "bilateral",
) -> np.ndarray:
    """
    Simple fast depth completion (morphological), adapted from common RGB-D completion recipes.
    Expects depth in meters, invalid == 0.
    """
    FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
    FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
    FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

    if custom_kernel is None:
        custom_kernel = FULL_KERNEL_5

    depth_map = np.asarray(depth_map, dtype=np.float32).copy()
    eps = 1e-6

    # Invert valid depth so morphology fills holes in the "right direction"
    valid_pixels = depth_map > eps
    depth_map[valid_pixels] = float(max_depth) - depth_map[valid_pixels]

    # Dilate + close
    depth_map = cv2.dilate(depth_map, custom_kernel)
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = depth_map <= eps
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Optional extrapolation to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > eps, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]
        for col in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[col], col] = top_pixel_values[col]

        empty_pixels = depth_map <= eps
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    # Median
    depth_map = cv2.medianBlur(depth_map, 5)

    # Edge-preserving / smoothing blur
    if blur_type == "bilateral":
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == "gaussian":
        valid_pixels = depth_map > eps
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]
    elif blur_type not in ("none", ""):
        raise ValueError(f"Unknown blur_type: {blur_type}")

    # Invert back
    valid_pixels = depth_map > eps
    depth_map[valid_pixels] = float(max_depth) - depth_map[valid_pixels]

    # Cleanup
    depth_map[~np.isfinite(depth_map)] = 0.0
    depth_map[depth_map < 0.0] = 0.0
    return depth_map