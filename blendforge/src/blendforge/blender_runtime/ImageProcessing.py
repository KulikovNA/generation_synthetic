from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def to_gray_u8(img: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale uint8.

    Accepts:
      - [H,W] grayscale
      - [H,W,3] RGB
      - [H,W,4] RGBA
    dtype:
      - uint8 (assumed 0..255)
      - float (assumed either 0..1 or 0..255-ish)

    Returns:
      - uint8 [H,W] grayscale
    """
    if img is None:
        raise ValueError("img is None")

    x = np.asarray(img)

    # If color -> RGB(A) -> gray
    if x.ndim == 3:
        if x.shape[2] == 4:
            x = x[:, :, :3]
        if x.shape[2] != 3:
            raise ValueError(f"Expected 3 or 4 channels, got {x.shape}")

        rgb8 = _as_uint8_rgb(x)
        return cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)

    # If already single channel
    if x.ndim != 2:
        raise ValueError(f"Expected [H,W] or [H,W,C], got {x.shape}")

    if x.dtype == np.uint8:
        return x

    # float/other -> uint8
    return _as_uint8_gray(x)


def preprocess_gray_u8(gray_u8: np.ndarray, mode: str = "clahe") -> np.ndarray:
    """
    Preprocess grayscale uint8 image before stereo matching.

    Modes:
      - "none"      : no-op
      - "clahe"     : CLAHE (good default for IR)
      - "equalize"  : global histogram equalization
    """
    if gray_u8 is None:
        raise ValueError("gray_u8 is None")

    g = np.asarray(gray_u8)
    if g.ndim != 2:
        raise ValueError(f"preprocess_gray_u8 expects [H,W], got {g.shape}")
    if g.dtype != np.uint8:
        raise ValueError(f"preprocess_gray_u8 expects uint8, got {g.dtype}")

    m = (mode or "none").lower()
    if m == "none":
        return g
    if m == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(g)
    if m == "equalize":
        return cv2.equalizeHist(g)

    # unknown mode -> no-op (или можно raise, но я бы оставил мягко)
    return g


# -------------------- helpers --------------------

def _as_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """
    Accepts RGB float/uint8, returns uint8 RGB.
    If float is assumed 0..1 or 0..255-ish.
    """
    x = np.asarray(img)

    if x.dtype == np.uint8:
        rgb8 = x
    else:
        xf = x.astype(np.float32, copy=False)
        xf = np.nan_to_num(xf, nan=0.0, posinf=0.0, neginf=0.0)

        # Heuristic: float in 0..1 or 0..255
        mx = float(xf.max()) if xf.size else 0.0
        if mx > 1.5:
            xf = np.clip(xf, 0.0, 255.0)
            rgb8 = (xf + 0.5).astype(np.uint8)
        else:
            xf = np.clip(xf, 0.0, 1.0)
            rgb8 = (xf * 255.0 + 0.5).astype(np.uint8)

    if rgb8.ndim != 3 or rgb8.shape[2] != 3:
        raise ValueError(f"_as_uint8_rgb expects [H,W,3], got {rgb8.shape}")
    return rgb8


def _as_uint8_gray(gray: np.ndarray) -> np.ndarray:
    """
    Accepts grayscale float/other, returns uint8 gray.
    If float is assumed 0..1 or 0..255-ish.
    """
    x = np.asarray(gray)

    if x.dtype == np.uint8:
        return x

    xf = x.astype(np.float32, copy=False)
    xf = np.nan_to_num(xf, nan=0.0, posinf=0.0, neginf=0.0)

    mx = float(xf.max()) if xf.size else 0.0
    if mx > 1.5:
        xf = np.clip(xf, 0.0, 255.0)
        return (xf + 0.5).astype(np.uint8)
    else:
        xf = np.clip(xf, 0.0, 1.0)
        return (xf * 255.0 + 0.5).astype(np.uint8)
