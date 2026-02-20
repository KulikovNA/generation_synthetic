# stereo/masks/OverlapMask.py
from __future__ import annotations

import numpy as np


def build_overlap_mask_from_rectified_gt(
    depth_gt_rect_m: np.ndarray,
    fx_rect: float,
    baseline_rect_m: float,
    depth_min: float,
    depth_max: float,
) -> np.ndarray:
    """
    Physical overlap mask computed in RECTIFIED LEFT grid.

      d_gt = fx'*B'/Z
      xR = x - d_gt

    Keep pixels where:
      - Z valid and within [depth_min, depth_max]
      - xR in [0..W-1]
    """
    Z = np.asarray(depth_gt_rect_m, dtype=np.float32)
    if Z.ndim != 2:
        raise ValueError(f"depth_gt_rect_m must be [H,W], got {Z.shape}")

    fx = float(fx_rect)
    B = float(baseline_rect_m)
    if fx <= 0 or B <= 0:
        raise ValueError("fx_rect and baseline_rect_m must be > 0")

    H, W = Z.shape
    dmin = float(max(depth_min, 0.0))
    dmax = float(max(depth_max, dmin + 1e-6))

    validZ = np.isfinite(Z) & (Z >= dmin) & (Z <= dmax)

    d = np.zeros_like(Z, dtype=np.float32)
    d[validZ] = (fx * B) / (Z[validZ] + 1e-6)

    xs = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)
    xR = xs - d

    mask = validZ & np.isfinite(d) & (d > 0.0) & (xR >= 0.0) & (xR <= (W - 1))
    return mask