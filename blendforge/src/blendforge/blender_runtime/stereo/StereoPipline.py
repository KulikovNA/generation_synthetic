from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from blendforge.blender_runtime.ImageProcessing import to_gray_u8, preprocess_gray_u8
from blendforge.blender_runtime.stereo.FrameSeries import FrameGrid, FrameSeries
from blendforge.blender_runtime.stereo.StereoRectify import build_rectify_maps, rectify_pair

from blendforge.blender_runtime.stereo.types.StereoTypes import (
    DepthRangePolicy,
    FillMode,
    StereoMatcherParams,
    StereoRectifiedCalib,
)

from blendforge.blender_runtime.stereo.utils.PadCropUtility import (
    compute_pad_left,
    crop_w,
    pad_left_false,
    pad_left_replicate,
    rectify_single_channel,
    fxB_from_rectify_maps_strict,
)

from blendforge.blender_runtime.stereo.matching.SgbmMatching import (
    compute_sgbm_disparities,
    has_ximgproc as sgbm_has_ximgproc,
)

from blendforge.blender_runtime.stereo.masks.LrCheck import lr_consistency_mask_auto
from blendforge.blender_runtime.stereo.masks.OverlapMask import build_overlap_mask_from_rectified_gt

from blendforge.blender_runtime.stereo.filters.DisparityFiltering import (
    apply_speckle_filter,
    apply_wls_filter,
    fill_disparity,
    has_ximgproc as disp_has_ximgproc,
    sanitize_disparity,
)
from blendforge.blender_runtime.stereo.filters.DepthFiltering import (
    apply_depth_range_policy,
    disparity_to_depth,
    fill_in_fast,
)


def _build_matcher_params(
    *,
    depth_range_policy: DepthRangePolicy = "zero",
    block_size: int = 7,
    num_disparities: int = 128,
    min_disparity: int = 0,
    preprocess: str = "clahe",
    use_wls: bool = True,
    lr_check: bool = True,
    lr_thresh_px: float = 1.0,
    lr_min_keep_ratio: float = 0.02,
    speckle_filter: bool = True,
    fill_mode: FillMode = "none",
    fill_iters: int = 0,
    depth_completion: bool = False,
    border_pad: bool = True,
    pad_left: Optional[int] = None,
    # SGBM advanced
    sgbm_mode: int = cv2.STEREO_SGBM_MODE_HH,
    uniqueness_ratio: int = 10,
    disp12_max_diff: int = 1,
    pre_filter_cap: int = 63,
    p1_scale: float = 8.0,
    p2_scale: float = 32.0,
) -> StereoMatcherParams:
    p = StereoMatcherParams(
        depth_range_policy=depth_range_policy,
        block_size=block_size,
        num_disparities=num_disparities,
        min_disparity=min_disparity,
        preprocess=preprocess,
        use_wls=use_wls,
        lr_check=lr_check,
        lr_thresh_px=lr_thresh_px,
        lr_min_keep_ratio=lr_min_keep_ratio,
        speckle_filter=speckle_filter,
        fill_mode=fill_mode,
        fill_iters=fill_iters,
        depth_completion=depth_completion,
        border_pad=border_pad,
        pad_left=pad_left,

        sgbm_mode=sgbm_mode,
        uniqueness_ratio=uniqueness_ratio,
        disp12_max_diff=disp12_max_diff,
        pre_filter_cap=pre_filter_cap,
        p1_scale=p1_scale,
        p2_scale=p2_scale,
    )
    p.validate()
    return p


def stereo_depth_from_rectified_pair(
    left_u8_rect: np.ndarray,
    right_u8_rect: np.ndarray,
    *,
    fx_rect: float,
    baseline_rect_m: float,
    depth_min: float,
    depth_max: float,
    depth_range_policy: DepthRangePolicy = "zero",
    block_size: int = 7,
    num_disparities: int = 128,
    min_disparity: int = 0,
    preprocess: str = "clahe",
    use_wls: bool = True,
    lr_check: bool = True,
    lr_thresh_px: float = 1.0,
    lr_min_keep_ratio: float = 0.02,
    speckle_filter: bool = True,
    fill_mode: FillMode = "none",
    fill_iters: int = 0,
    depth_completion: bool = False,
    border_pad: bool = True,
    pad_left: Optional[int] = None,
    # SGBM advanced
    sgbm_mode: int = cv2.STEREO_SGBM_MODE_HH,
    uniqueness_ratio: int = 10,
    disp12_max_diff: int = 1,
    pre_filter_cap: int = 63,
    p1_scale: float = 8.0,
    p2_scale: float = 32.0,
    geom_mask_rect: Optional[np.ndarray] = None,  # in rectified grid, before padding
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Core stereo on a single ALREADY-RECTIFIED pair.

    Returns:
      depth_m   [H,W] float32, invalid=0
      disp_px   [H,W] float32, invalid=0
    """
    if left_u8_rect.ndim != 2 or right_u8_rect.ndim != 2:
        raise ValueError("left/right must be grayscale [H,W].")
    if left_u8_rect.dtype != np.uint8 or right_u8_rect.dtype != np.uint8:
        raise ValueError("left/right must be uint8.")
    if left_u8_rect.shape != right_u8_rect.shape:
        raise ValueError(f"left/right shapes differ: {left_u8_rect.shape} vs {right_u8_rect.shape}")

    H0, W0 = left_u8_rect.shape

    calib = StereoRectifiedCalib(
        fx_rect=float(fx_rect),
        baseline_rect_m=float(baseline_rect_m),
    )
    calib.validate()

    params = _build_matcher_params(
        depth_range_policy=depth_range_policy,
        block_size=block_size,
        num_disparities=num_disparities,
        min_disparity=min_disparity,
        preprocess=preprocess,
        use_wls=use_wls,
        lr_check=lr_check,
        lr_thresh_px=lr_thresh_px,
        lr_min_keep_ratio=lr_min_keep_ratio,
        speckle_filter=speckle_filter,
        fill_mode=fill_mode,
        fill_iters=fill_iters,
        depth_completion=depth_completion,
        border_pad=border_pad,
        pad_left=pad_left,
        sgbm_mode=sgbm_mode,
        uniqueness_ratio=uniqueness_ratio,
        disp12_max_diff=disp12_max_diff,
        pre_filter_cap=pre_filter_cap,
        p1_scale=p1_scale,
        p2_scale=p2_scale,
    )

    dmin = float(max(depth_min, 0.0))
    dmax = float(max(depth_max, dmin + 1e-6))

    # preprocess
    left_p = preprocess_gray_u8(left_u8_rect, params.preprocess)
    right_p = preprocess_gray_u8(right_u8_rect, params.preprocess)

    # left padding (removes fixed invalid search band)
    padL = 0
    if params.border_pad:
        padL = int(
            compute_pad_left(params.num_disparities, params.min_disparity, params.block_size)
            if params.pad_left is None else params.pad_left
        )
        padL = max(0, padL)
        if padL > 0:
            left_p = pad_left_replicate(left_p, padL)
            right_p = pad_left_replicate(right_p, padL)

    # pad geometry mask consistently
    geom_mask_p: Optional[np.ndarray] = None
    if geom_mask_rect is not None:
        gm = np.asarray(geom_mask_rect, dtype=bool)
        if gm.shape != (H0, W0):
            raise ValueError(f"geom_mask_rect must be {(H0, W0)}, got {gm.shape}")
        geom_mask_p = pad_left_false(gm, padL) if padL > 0 else gm

    # SGBM disparities (left/right)
    matcher_left, dispL_i16, dispR_i16, dispL, dispR = compute_sgbm_disparities(
        left_p,
        right_p,
        block_size=params.block_size,
        num_disparities=params.num_disparities,
        min_disparity=params.min_disparity,
        mode=params.sgbm_mode,
        uniqueness_ratio=params.uniqueness_ratio,
        sgbm_speckle_window_size=params.speckle_window_size,
        sgbm_speckle_range=params.speckle_range,
        disp12_max_diff=params.disp12_max_diff,
        pre_filter_cap=params.pre_filter_cap,
        p1_scale=params.p1_scale,
        p2_scale=params.p2_scale,
    )

    dispL = sanitize_disparity(dispL)
    dispR = sanitize_disparity(dispR)

    if geom_mask_p is not None:
        dispL[~geom_mask_p] = 0.0
        dispR[~geom_mask_p] = 0.0

    # LR consistency
    if params.lr_check:
        mask_lr = lr_consistency_mask_auto(
            dispL,
            dispR,
            thresh_px=params.lr_thresh_px,
            min_keep_ratio=params.lr_min_keep_ratio,
        )
    else:
        mask_lr = dispL > 0.0

    if geom_mask_p is not None:
        mask_lr = mask_lr & geom_mask_p

    # WLS (optional)
    disp_work = dispL.copy()
    if params.use_wls:
        if disp_has_ximgproc() and sgbm_has_ximgproc():
            disp_work = apply_wls_filter(
                dispL_i16,
                dispR_i16,
                left_p,
                matcher_left,
                lambda_value=params.wls_lambda,
                sigma_color=params.wls_sigma_color,
            )
            if geom_mask_p is not None:
                disp_work[~geom_mask_p] = 0.0
        # else: silently fallback to raw dispL

    # apply LR mask
    disp_work[~mask_lr] = 0.0
    disp_work = sanitize_disparity(disp_work)

    # speckle filtering
    if params.speckle_filter:
        disp_work = apply_speckle_filter(
            disp_work,
            max_speckle_size=params.speckle_window_size,
            max_diff_disp16=16,
            new_val=0,
        )

    # optional hole filling
    disp_work = fill_disparity(
        disp_work,
        mode=params.fill_mode,
        iters=params.fill_iters,
    )

    disp_final = sanitize_disparity(disp_work)
    if geom_mask_p is not None:
        disp_final[~geom_mask_p] = 0.0

    # disparity -> depth
    depth = disparity_to_depth(
        disp_final,
        fx=calib.fx_rect,
        baseline_m=calib.baseline_rect_m,
    )

    # depth range policy
    depth = apply_depth_range_policy(
        depth,
        depth_min=dmin,
        depth_max=dmax,
        policy=params.depth_range_policy,
    )

    if geom_mask_p is not None:
        depth[~geom_mask_p] = 0.0

    # optional depth completion
    if params.depth_completion:
        depth = fill_in_fast(depth, max_depth=float(dmax))
        depth = apply_depth_range_policy(
            depth,
            depth_min=dmin,
            depth_max=dmax,
            policy=params.depth_range_policy,
        )
        if geom_mask_p is not None:
            depth[~geom_mask_p] = 0.0

    # crop back after padding
    disp_final = crop_w(disp_final, padL, W0)
    depth = crop_w(depth, padL, W0)

    # crop_w returns Optional[np.ndarray], but here inputs are never None
    if disp_final is None or depth is None:
        raise RuntimeError("Unexpected None after crop_w")

    return depth.astype(np.float32, copy=False), disp_final.astype(np.float32, copy=False)


def stereo_global_matching_rectified(
    stereo_frames: List[np.ndarray],
    *,
    rectify: Dict[str, Any],  # REQUIRED
    depth_min: float,
    depth_max: float,
    depth_range_policy: DepthRangePolicy = "zero",
    block_size: int = 7,
    num_disparities: int = 128,
    min_disparity: int = 0,
    preprocess: str = "clahe",
    use_wls: bool = True,
    lr_check: bool = True,
    lr_thresh_px: float = 1.0,
    lr_min_keep_ratio: float = 0.02,
    speckle_filter: bool = True,
    fill_mode: FillMode = "none",
    fill_iters: int = 0,
    depth_completion: bool = False,
    # SGBM advanced
    sgbm_mode: int = cv2.STEREO_SGBM_MODE_HH,
    uniqueness_ratio: int = 10,
    disp12_max_diff: int = 1,
    pre_filter_cap: int = 63,
    p1_scale: float = 8.0,
    p2_scale: float = 32.0,
    # optional physical overlap mask from GT depth (IR_LEFT render depth)
    depth_gt_frames: Optional[List[np.ndarray]] = None,  # ORIGINAL left grid
    use_geom_mask_from_gt: bool = True,
    # remove fixed SGBM left band
    border_pad: bool = True,
    pad_left: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Rectify-ALWAYS stereo pipeline for a series of stereo frames.

    Input:
      stereo_frames[idx] shape: [2,H,W,C] or [2,H,W]
        stereo[0] = left, stereo[1] = right (original IR grids)
    Output:
      depth_out, disp_out  (both in RECTIFIED LEFT grid)
    """
    if not stereo_frames:
        raise ValueError("stereo_frames is empty")

    if depth_gt_frames is not None and len(depth_gt_frames) != len(stereo_frames):
        raise ValueError(
            f"depth_gt_frames length must match stereo_frames: "
            f"{len(depth_gt_frames)} vs {len(stereo_frames)}"
        )

    first = stereo_frames[0]
    if first.ndim < 3 or first.shape[0] != 2:
        raise ValueError(f"Expected first frame shape [2,H,W,...], got {first.shape}")

    H, W = int(first.shape[1]), int(first.shape[2])

    # Build rectify maps once
    rectify_maps = build_rectify_maps(rectify, out_size=(W, H))

    # Strict rectified fx' and baseline B'
    fx_rect, baseline_rect_m = fxB_from_rectify_maps_strict(rectify_maps)

    # Outputs as FrameSeries (list-compatible), with grid metadata
    rect_grid = FrameGrid(name="IR_LEFT_RECT", rectify_meta=rectify_maps)
    depth_out = FrameSeries(grid=rect_grid)
    disp_out = FrameSeries(grid=rect_grid)

    for idx, stereo in enumerate(stereo_frames):
        if stereo.shape[0] != 2:
            raise ValueError(f"Frame idx={idx}: stereo.shape[0] must be 2, got {stereo.shape}")

        left = to_gray_u8(stereo[0])
        right = to_gray_u8(stereo[1])

        # Rectify pair ALWAYS
        left_r, right_r = rectify_pair(left, right, rectify_maps)

        # Optional physical overlap mask from GT depth (GT is in ORIGINAL left grid)
        geom_mask_rect: Optional[np.ndarray] = None
        if use_geom_mask_from_gt and depth_gt_frames is not None:
            gt = np.asarray(depth_gt_frames[idx], dtype=np.float32)
            if gt.shape != left.shape:
                raise ValueError(
                    f"GT depth shape mismatch at idx={idx}: gt {gt.shape} vs left {left.shape} "
                    f"(expected ORIGINAL left grid)."
                )

            gt_r = rectify_single_channel(
                gt,
                rectify_maps,
                interp=cv2.INTER_NEAREST,
                border_val=0.0,
            )

            geom_mask_rect = build_overlap_mask_from_rectified_gt(
                gt_r,
                fx_rect=float(fx_rect),
                baseline_rect_m=float(baseline_rect_m),
                depth_min=float(depth_min),
                depth_max=float(depth_max),
            )

        depth_m, disp_px = stereo_depth_from_rectified_pair(
            left_r,
            right_r,
            fx_rect=float(fx_rect),
            baseline_rect_m=float(baseline_rect_m),
            depth_min=float(depth_min),
            depth_max=float(depth_max),
            depth_range_policy=depth_range_policy,
            block_size=block_size,
            num_disparities=num_disparities,
            min_disparity=min_disparity,
            preprocess=preprocess,
            use_wls=use_wls,
            lr_check=lr_check,
            lr_thresh_px=lr_thresh_px,
            lr_min_keep_ratio=lr_min_keep_ratio,
            speckle_filter=speckle_filter,
            fill_mode=fill_mode,
            fill_iters=fill_iters,
            depth_completion=depth_completion,
            sgbm_mode=sgbm_mode,
            uniqueness_ratio=uniqueness_ratio,
            disp12_max_diff=disp12_max_diff,
            pre_filter_cap=pre_filter_cap,
            p1_scale=p1_scale,
            p2_scale=p2_scale,
            border_pad=border_pad,
            pad_left=pad_left,
            geom_mask_rect=geom_mask_rect,
        )

        depth_out.append(depth_m)
        disp_out.append(disp_px)

    return depth_out, disp_out