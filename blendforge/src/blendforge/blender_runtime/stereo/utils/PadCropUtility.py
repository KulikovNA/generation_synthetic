# stereo/utils/PadCropUtility.py
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from blendforge.blender_runtime.stereo.StereoRectify import RectifyMaps


# -------------------- pad / crop --------------------

def compute_pad_left(num_disparities: int, min_disparity: int, block_size: int) -> int:
    """
    SGBM artifact fix: pad LEFT border so the algorithm can search disparities
    without hitting image boundary.

    Typical invalid band on the left is ~numDisparities. We pad slightly more.
    """
    nd = int(max(0, num_disparities))
    md = int(max(0, min_disparity))
    bs = int(max(1, block_size))
    return int(nd + md + (bs // 2) + 2)


def crop_w(arr2d: Optional[np.ndarray], pad_left: int, width_original: int) -> Optional[np.ndarray]:
    """
    Crop image/mask back after left padding.
    """
    if arr2d is None:
        return None
    if int(pad_left) <= 0:
        return arr2d
    return arr2d[:, int(pad_left): int(pad_left) + int(width_original)]


def pad_left_replicate(img2d: np.ndarray, pad_left: int) -> np.ndarray:
    """
    Pad left border using BORDER_REPLICATE (good for grayscale images before SGBM).
    """
    x = np.asarray(img2d)
    if x.ndim != 2:
        raise ValueError(f"pad_left_replicate expects [H,W], got {x.shape}")

    p = int(max(0, pad_left))
    if p == 0:
        return x
    return cv2.copyMakeBorder(x, 0, 0, p, 0, cv2.BORDER_REPLICATE)


def pad_left_false(mask2d: np.ndarray, pad_left: int) -> np.ndarray:
    """
    Pad left border with False (0) for boolean masks.
    """
    m = np.asarray(mask2d, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"pad_left_false expects [H,W], got {m.shape}")

    p = int(max(0, pad_left))
    if p == 0:
        return m
    m_u8 = cv2.copyMakeBorder(m.astype(np.uint8), 0, 0, p, 0, cv2.BORDER_CONSTANT, value=0)
    return m_u8.astype(bool)


# -------------------- rectify helpers (kept here for now by your request) --------------------

def get_rectify_left_maps(rectify_maps: RectifyMaps) -> Tuple[np.ndarray, np.ndarray]:
    cand = [
        ("mapLx", "mapLy"),
        ("map1x", "map1y"),
        ("left_map_x", "left_map_y"),
        ("left_mapx", "left_mapy"),
        ("Lx", "Ly"),
    ]
    for ax, ay in cand:
        mx = getattr(rectify_maps, ax, None)
        my = getattr(rectify_maps, ay, None)
        if mx is not None and my is not None:
            return mx, my
    raise AttributeError(
        "RectifyMaps does not expose LEFT remap matrices. "
        "Expected one of: mapLx/mapLy, map1x/map1y, left_map_x/left_map_y, ..."
    )


def rectify_single_channel(
    img: np.ndarray,
    rectify_maps: RectifyMaps,
    *,
    interp: int = cv2.INTER_NEAREST,
    border_val: float = 0.0,
) -> np.ndarray:
    mapx, mapy = get_rectify_left_maps(rectify_maps)
    x = np.asarray(img)
    if x.ndim != 2:
        raise ValueError(f"rectify_single_channel expects [H,W], got {x.shape}")
    return cv2.remap(
        x,
        mapx,
        mapy,
        interpolation=int(interp),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )


def fxB_from_rectify_maps_strict(rectify_maps: RectifyMaps) -> Tuple[float, float]:
    """
    Strictly require P1/P2 to compute rectified fx' and baseline B'.
      fx' = P1[0,0]
      B'  = |P2[0,3]| / fx'
    """
    P1 = getattr(rectify_maps, "P1", None)
    P2 = getattr(rectify_maps, "P2", None)
    if P1 is None or P2 is None:
        raise ValueError("RectifyMaps must contain P1 and P2 (from stereoRectify).")

    P1 = np.asarray(P1, dtype=np.float64)
    P2 = np.asarray(P2, dtype=np.float64)
    if P1.shape != (3, 4) or P2.shape != (3, 4):
        raise ValueError(f"Invalid P1/P2 shapes: P1={P1.shape}, P2={P2.shape}, expected (3,4).")

    fxp = float(P1[0, 0])
    if not np.isfinite(fxp) or fxp <= 0:
        raise ValueError(f"Invalid rectified fx': {fxp}")

    p203 = float(P2[0, 3])
    if not np.isfinite(p203):
        raise ValueError("Invalid P2[0,3] in rectify maps.")

    Bp = float(abs(p203) / fxp)
    if not np.isfinite(Bp) or Bp <= 0:
        raise ValueError(f"Invalid rectified baseline B': {Bp}")

    return fxp, Bp