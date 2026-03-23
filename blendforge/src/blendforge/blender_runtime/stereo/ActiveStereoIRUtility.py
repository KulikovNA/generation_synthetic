from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np

from blendforge.blender_runtime.camera.RealsenseProfileLoader import RealSenseProfile
from blendforge.blender_runtime.stereo.StereoPipline import stereo_global_matching_rectified
from blendforge.blender_runtime.stereo.StereoRectify import build_rectify_maps, rectify_pair


def build_rectify_from_rs(
    rs: RealSenseProfile,
    left: str = "IR_LEFT",
    right: str = "IR_RIGHT",
    *,
    use_distortion: bool = False,
) -> Dict[str, Any]:
    s_left = rs.get_stream(left)
    s_right = rs.get_stream(right)
    T_right_from_left = np.asarray(rs.get_T_cv(left, right), dtype=np.float64)
    if use_distortion:
        d_left = np.asarray(s_left.distortion_coeffs, dtype=np.float64).reshape(-1)
        d_right = np.asarray(s_right.distortion_coeffs, dtype=np.float64).reshape(-1)
    else:
        d_left = np.zeros(5, dtype=np.float64)
        d_right = np.zeros(5, dtype=np.float64)
    return {
        "K_left": np.asarray(s_left.K, dtype=np.float64),
        "D_left": d_left,
        "K_right": np.asarray(s_right.K, dtype=np.float64),
        "D_right": d_right,
        "R": T_right_from_left[:3, :3].copy(),
        "t": T_right_from_left[:3, 3].copy(),
        "alpha": 0.0,
        "image_size": (int(s_left.width), int(s_left.height)),
    }


def _to_float01_image(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    xf = x.astype(np.float32, copy=False)
    xf = np.nan_to_num(xf, nan=0.0, posinf=0.0, neginf=0.0)
    if float(xf.max()) > 1.5:
        xf = np.clip(xf, 0.0, 255.0) / 255.0
    else:
        xf = np.clip(xf, 0.0, 1.0)
    return xf


def _local_contrast_normalize_u8(gray01: np.ndarray) -> np.ndarray:
    x = np.asarray(gray01, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"LCN expects [H,W], got {x.shape}")

    mu = cv2.GaussianBlur(x, (0, 0), sigmaX=9.0, sigmaY=9.0, borderType=cv2.BORDER_REFLECT)
    sq_mu = cv2.GaussianBlur(x * x, (0, 0), sigmaX=9.0, sigmaY=9.0, borderType=cv2.BORDER_REFLECT)
    var = np.maximum(sq_mu - mu * mu, 0.0)
    std = np.sqrt(var + 1e-4)

    z = (x - mu) / std
    z = np.clip(z, -3.0, 3.0)
    y = (z + 3.0) / 6.0
    return (np.clip(y, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def rgb_to_intensity_u8(img: np.ndarray, mode: str = "bt601") -> np.ndarray:
    x = np.asarray(img)
    m = (mode or "bt601").strip().lower()

    if x.ndim == 2:
        if x.dtype == np.uint8 and m not in ("lcn", "bt601_lcn", "lcn_bt601"):
            return x
        xf = _to_float01_image(x)
        if m in ("lcn", "bt601_lcn", "lcn_bt601"):
            return _local_contrast_normalize_u8(xf)
        return (xf * 255.0 + 0.5).astype(np.uint8)

    if x.ndim != 3:
        raise ValueError(f"Expected RGB/RGBA image, got shape {x.shape}")

    if x.shape[2] == 4:
        x = x[:, :, :3]
    if x.shape[2] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {x.shape}")

    xf = _to_float01_image(x)
    if m in ("bt601", "luma", "opencv"):
        y = 0.299 * xf[:, :, 0] + 0.587 * xf[:, :, 1] + 0.114 * xf[:, :, 2]
    elif m == "bt709":
        y = 0.2126 * xf[:, :, 0] + 0.7152 * xf[:, :, 1] + 0.0722 * xf[:, :, 2]
    elif m == "mean":
        y = xf.mean(axis=2)
    elif m == "max":
        y = xf.max(axis=2)
    elif m in ("lcn", "bt601_lcn", "lcn_bt601"):
        y = 0.299 * xf[:, :, 0] + 0.587 * xf[:, :, 1] + 0.114 * xf[:, :, 2]
        return _local_contrast_normalize_u8(np.clip(y, 0.0, 1.0))
    else:
        raise ValueError(f"Unknown rgb_to_intensity mode: {mode}")

    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)


def convert_ir_frames_to_intensity(frames: Iterable[np.ndarray], mode: str = "bt601") -> list[np.ndarray]:
    return [rgb_to_intensity_u8(frame, mode=mode) for frame in frames]


def stereo_from_ir_pair(
    rs: RealSenseProfile,
    *,
    left_stream: str,
    right_stream: str,
    left_ir_u8: np.ndarray,
    right_ir_u8: np.ndarray,
    left_depth_gt_m: Optional[np.ndarray] = None,
    plane_distance_m: Optional[float] = None,
    use_distortion: bool = False,
    **matcher_kwargs,
) -> Dict[str, Any]:
    rectify_cfg = build_rectify_from_rs(rs, left_stream, right_stream, use_distortion=use_distortion)
    rectify_maps = build_rectify_maps(rectify_cfg, rectify_cfg["image_size"])
    left_rect_u8, right_rect_u8 = rectify_pair(left_ir_u8, right_ir_u8, rectify_maps)

    depth_min = float(matcher_kwargs.pop("depth_min", rs.depth_min_m))
    depth_max = float(matcher_kwargs.pop("depth_max", rs.depth_max_m))
    min_disparity = int(matcher_kwargs.pop("min_disparity", 0))
    z_min_for_num_disp = depth_min if plane_distance_m is None else max(depth_min, float(plane_distance_m) * 0.5)
    num_disparities = matcher_kwargs.pop(
        "num_disparities",
        rs.recommend_num_disparities(
            z_min_m=z_min_for_num_disp,
            min_disparity=min_disparity,
            stream=left_stream,
        ),
    )

    stereo_frames = [np.stack([left_ir_u8, right_ir_u8], axis=0)]
    depth_gt_frames = None if left_depth_gt_m is None else [np.asarray(left_depth_gt_m, dtype=np.float32)]

    matcher_cfg = {
        "rectify": rectify_cfg,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "depth_range_policy": "zero",
        "block_size": 7,
        "num_disparities": int(num_disparities),
        "min_disparity": min_disparity,
        "preprocess": "clahe",
        "use_wls": True,
        "lr_check": True,
        "lr_thresh_px": 1.0,
        "lr_min_keep_ratio": 0.02,
        "speckle_filter": True,
        "fill_mode": "none",
        "fill_iters": 0,
        "depth_completion": False,
        "depth_gt_frames": depth_gt_frames,
        "use_geom_mask_from_gt": left_depth_gt_m is not None,
    }
    matcher_cfg.update(matcher_kwargs)

    depth_rect_list, disp_rect_list = stereo_global_matching_rectified(stereo_frames, **matcher_cfg)
    depth_rect_m = np.asarray(depth_rect_list[0], dtype=np.float32)
    disp_rect_px = np.asarray(disp_rect_list[0], dtype=np.float32)
    valid = depth_rect_m > 0.0

    depth_stats = {
        "valid_fraction": float(np.count_nonzero(valid) / depth_rect_m.size),
        "mean_depth_m": 0.0 if not np.any(valid) else float(np.mean(depth_rect_m[valid])),
        "median_depth_m": 0.0 if not np.any(valid) else float(np.median(depth_rect_m[valid])),
        "mae_to_plane_m": None if plane_distance_m is None else (
            0.0 if not np.any(valid)
            else float(np.mean(np.abs(depth_rect_m[valid] - float(plane_distance_m))))
        ),
    }

    return {
        "rectify_cfg": rectify_cfg,
        "left_rect_u8": left_rect_u8,
        "right_rect_u8": right_rect_u8,
        "depth_rect_m": depth_rect_m,
        "disp_rect_px": disp_rect_px,
        "depth_stats": depth_stats,
    }


__all__ = [
    "build_rectify_from_rs",
    "rgb_to_intensity_u8",
    "convert_ir_frames_to_intensity",
    "stereo_from_ir_pair",
]
