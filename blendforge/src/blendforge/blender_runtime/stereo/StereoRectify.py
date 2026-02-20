from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class RectifyMaps:
    # remap-таблицы
    map1x: np.ndarray
    map1y: np.ndarray
    map2x: np.ndarray
    map2y: np.ndarray

    # выход stereoRectify
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray

    image_size: Tuple[int, int]  # (W,H)
    alpha: float

    @property
    def fx_rect(self) -> float:
        return float(self.P1[0, 0])

    @property
    def fy_rect(self) -> float:
        return float(self.P1[1, 1])

    @property
    def cx_rect(self) -> float:
        return float(self.P1[0, 2])

    @property
    def cy_rect(self) -> float:
        return float(self.P1[1, 2])

    @property
    def baseline_m(self) -> float:
        fx = float(self.P2[0, 0])
        if abs(fx) < 1e-12:
            return float("nan")
        return float(abs(self.P2[0, 3]) / fx)



def build_rectify_maps(rectify: Dict[str, Any], out_size: Tuple[int, int]) -> RectifyMaps:
    """
    rectify dict must include:
      K_left, D_left, K_right, D_right, R, t
      alpha (0..1)
    out_size: (W,H)
    """
    W, H = int(out_size[0]), int(out_size[1])

    K1 = np.asarray(rectify["K_left"], dtype=np.float64)
    D1 = np.asarray(rectify["D_left"], dtype=np.float64).reshape(-1)
    K2 = np.asarray(rectify["K_right"], dtype=np.float64)
    D2 = np.asarray(rectify["D_right"], dtype=np.float64).reshape(-1)

    R = np.asarray(rectify["R"], dtype=np.float64)
    t = np.asarray(rectify["t"], dtype=np.float64).reshape(3,)

    alpha = float(rectify.get("alpha", 0.0))

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, (W, H),
        R, t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha,
        newImageSize=(W, H),
    )

    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (W, H), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (W, H), cv2.CV_32FC1)

    return RectifyMaps(
        map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        image_size=(W, H),
        alpha=alpha,
    )


def rectify_pair(left, right, maps, *, interpolation=cv2.INTER_LINEAR, border_value=0):
    l = cv2.remap(left, maps.map1x, maps.map1y, interpolation=interpolation,
                  borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    r = cv2.remap(right, maps.map2x, maps.map2y, interpolation=interpolation,
                  borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    return l, r
