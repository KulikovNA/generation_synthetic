# stereo/masks/LrCheck.py

from __future__ import annotations

import numpy as np


def lr_consistency_mask_auto(
    dispL: np.ndarray,
    dispR: np.ndarray,
    thresh_px: float = 1.0,
    min_keep_ratio: float = 0.02,
) -> np.ndarray:
    """
    LR-check with auto convention detection (mapping/sign).
    Returns boolean mask in the same grid as dispL.

    Tries 4 hypotheses:
      - xR = x - dL  OR  xR = x + dL
      - dR convention is +dR OR -dR
    Chooses hypothesis with best keep ratio.
    """
    dL = np.asarray(dispL, dtype=np.float32)
    dR = np.asarray(dispR, dtype=np.float32)

    if dL.ndim != 2 or dR.ndim != 2:
        raise ValueError(f"dispL/dispR must be [H,W], got {dL.shape} and {dR.shape}")
    if dL.shape != dR.shape:
        raise ValueError(f"dispL/dispR shape mismatch: {dL.shape} vs {dR.shape}")

    H, W = dL.shape
    xs = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)
    ys = np.arange(H, dtype=np.int32)[:, None].repeat(W, axis=1)

    validL = np.isfinite(dL) & (dL > 0.0)
    if not np.any(validL):
        return np.zeros_like(validL, dtype=bool)

    dLr = np.rint(dL).astype(np.int32)

    best_keep = -1.0
    best_mask = np.zeros_like(validL, dtype=bool)

    for mapping_sign in (-1, +1):
        xR = xs + mapping_sign * dLr.astype(np.float32)
        xRi = np.clip(np.rint(xR).astype(np.int32), 0, W - 1)

        dR_samp = dR[ys, xRi]
        valid = validL & np.isfinite(dR_samp) & (dR_samp != 0.0)
        if not np.any(valid):
            continue

        for sign in (+1.0, -1.0):
            dR_eff = sign * dR_samp
            mask = valid & (np.abs(dL - dR_eff) <= float(thresh_px))
            keep = float(mask.sum()) / float(validL.sum())
            if keep > best_keep:
                best_keep = keep
                best_mask = mask

    if best_keep < float(min_keep_ratio):
        return validL  # fallback: don't throw everything away

    return best_mask