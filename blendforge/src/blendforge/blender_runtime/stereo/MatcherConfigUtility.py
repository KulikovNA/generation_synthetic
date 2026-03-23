from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def cfg_get(section, key: str, default):
    if section is None:
        return default
    if isinstance(section, dict):
        return section.get(key, default)
    return getattr(section, key, default)


def sample_scalar_or_range(value, *, cast):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return cast(value)

    vals = list(value)
    if len(vals) == 0:
        raise ValueError("Numeric range must not be empty")
    if len(vals) == 1:
        return cast(vals[0])

    lo, hi = vals[0], vals[1]
    if cast is int:
        lo = int(round(lo))
        hi = int(round(hi))
        if hi < lo:
            lo, hi = hi, lo
        return int(np.random.randint(lo, hi + 1))

    lo = float(lo)
    hi = float(hi)
    if hi < lo:
        lo, hi = hi, lo
    return cast(np.random.uniform(lo, hi))


def normalize_matcher_value(name: str, value):
    if value is None:
        return None

    if name == "num_disparities":
        v = max(16, int(round(value)))
        return int(((v + 15) // 16) * 16)

    if name == "block_size":
        v = max(1, int(round(value)))
        if v % 2 == 0:
            v += 1
        return int(v)

    if name in {
        "min_disparity",
        "fill_iters",
        "uniqueness_ratio",
        "disp12_max_diff",
        "pre_filter_cap",
    }:
        return int(round(value))

    if name in {
        "depth_min",
        "depth_max",
        "lr_thresh_px",
        "lr_min_keep_ratio",
        "p1_scale",
        "p2_scale",
    }:
        return float(value)

    return value


def matcher_numeric_param(section, key: str, default, *, cast):
    raw = cfg_get(section, key, default)
    sampled = sample_scalar_or_range(raw, cast=cast)
    return normalize_matcher_value(key, sampled)


def resolve_sgbm_mode(value):
    if value is None:
        return cv2.STEREO_SGBM_MODE_HH
    if isinstance(value, int):
        return int(value)

    mode = str(value).strip().upper()
    mapping = {
        "SGBM": getattr(cv2, "STEREO_SGBM_MODE_SGBM", 0),
        "HH": getattr(cv2, "STEREO_SGBM_MODE_HH", 1),
        "3WAY": getattr(cv2, "STEREO_SGBM_MODE_SGBM_3WAY", 2),
        "HH4": getattr(cv2, "STEREO_SGBM_MODE_HH4", 3),
    }
    if mode not in mapping:
        raise ValueError(f"Unsupported sgbm_mode: {value}")
    return int(mapping[mode])


def build_matcher_kwargs(section, rs, *, stream: str = "IR_LEFT") -> dict[str, Any]:
    return {
        "depth_min": matcher_numeric_param(section, "depth_min_m", rs.depth_min_m, cast=float),
        "depth_max": matcher_numeric_param(section, "depth_max_m", rs.depth_max_m, cast=float),
        "min_disparity": matcher_numeric_param(section, "min_disparity", 0, cast=int),
        "num_disparities": matcher_numeric_param(
            section,
            "num_disparities",
            rs.recommend_num_disparities(stream=stream),
            cast=int,
        ),
        "block_size": matcher_numeric_param(section, "block_size", 7, cast=int),
        "preprocess": str(cfg_get(section, "preprocess", "clahe")),
        "use_wls": bool(cfg_get(section, "use_wls", False)),
        "lr_check": bool(cfg_get(section, "lr_check", False)),
        "lr_thresh_px": matcher_numeric_param(section, "lr_thresh_px", 1.0, cast=float),
        "lr_min_keep_ratio": matcher_numeric_param(section, "lr_min_keep_ratio", 0.02, cast=float),
        "speckle_filter": bool(cfg_get(section, "speckle_filter", False)),
        "fill_mode": str(cfg_get(section, "fill_mode", "none")),
        "fill_iters": matcher_numeric_param(section, "fill_iters", 0, cast=int),
        "depth_completion": bool(cfg_get(section, "depth_completion", False)),
        "sgbm_mode": resolve_sgbm_mode(cfg_get(section, "sgbm_mode", "HH")),
        "uniqueness_ratio": matcher_numeric_param(section, "uniqueness_ratio", 10, cast=int),
        "disp12_max_diff": matcher_numeric_param(section, "disp12_max_diff", 1, cast=int),
        "pre_filter_cap": matcher_numeric_param(section, "pre_filter_cap", 63, cast=int),
        "p1_scale": matcher_numeric_param(section, "p1_scale", 8.0, cast=float),
        "p2_scale": matcher_numeric_param(section, "p2_scale", 32.0, cast=float),
        "use_geom_mask_from_gt": bool(cfg_get(section, "use_geom_mask_from_gt", False)),
    }


__all__ = [
    "cfg_get",
    "sample_scalar_or_range",
    "normalize_matcher_value",
    "matcher_numeric_param",
    "resolve_sgbm_mode",
    "build_matcher_kwargs",
]
