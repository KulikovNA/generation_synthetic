# blendforge/debug_blender_runtime/EvalDepth.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities for evaluating predicted depth maps against ground-truth depth.

This module is kept as a legacy/debug evaluation helper. The current
effective/random generator scripts do not depend on it directly.

Main idea:
- invalid / missing pixels are evaluated separately from numeric depth error;
- numeric metrics are computed only on overlap of valid GT and valid prediction;
- edge metrics are computed on GT depth discontinuities only.

Depth units:
- all depth arrays are expected in meters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


ArrayLike = Any


def to_frame_list(x: Optional[Sequence[np.ndarray]]) -> List[np.ndarray]:
    """
    Convert arbitrary list-like / FrameSeries-like input into a plain list of frames.

    Parameters
    ----------
    x:
        None, list, tuple, or any iterable container of depth frames.

    Returns
    -------
    List[np.ndarray]
    """
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return list(x)


def valid_depth_mask(
    depth_m: ArrayLike,
    depth_min_m: Optional[float] = None,
    depth_max_m: Optional[float] = None,
) -> np.ndarray:
    """
    Build validity mask for a depth map.

    A pixel is considered valid if:
    - finite
    - > 0
    - within [depth_min_m, depth_max_m] if limits are provided

    Parameters
    ----------
    depth_m:
        Depth map in meters.
    depth_min_m:
        Optional lower bound.
    depth_max_m:
        Optional upper bound.

    Returns
    -------
    np.ndarray
        Boolean mask of shape HxW.
    """
    d = np.asarray(depth_m, dtype=np.float32)
    mask = np.isfinite(d) & (d > 0.0)

    if depth_min_m is not None:
        mask &= d >= float(depth_min_m)
    if depth_max_m is not None:
        mask &= d <= float(depth_max_m)

    return mask


def make_edge_mask_from_gt(
    gt_m: ArrayLike,
    valid_gt: Optional[np.ndarray] = None,
    dilation_px: int = 1,
    percentile: float = 85.0,
) -> np.ndarray:
    """
    Estimate a mask of depth discontinuity zones from GT depth.

    The mask is built from the Sobel gradient magnitude over the GT depth map
    (invalid GT pixels are suppressed), then thresholded by percentile and
    optionally dilated.

    Parameters
    ----------
    gt_m:
        GT depth map in meters.
    valid_gt:
        Optional precomputed GT validity mask. If None, it will be computed.
    dilation_px:
        Radius of dilation. 0 means no dilation.
    percentile:
        Percentile used to threshold gradient magnitude over valid GT pixels.

    Returns
    -------
    np.ndarray
        Boolean edge mask of shape HxW.
    """
    gt = np.asarray(gt_m, dtype=np.float32)

    if valid_gt is None:
        valid_gt = valid_depth_mask(gt)

    if not np.any(valid_gt):
        return np.zeros_like(valid_gt, dtype=bool)

    z = gt.copy()
    z[~valid_gt] = 0.0

    gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    valid_grad = grad[valid_gt]
    if valid_grad.size < 16:
        return np.zeros_like(valid_gt, dtype=bool)

    thr = float(np.percentile(valid_grad, percentile))
    thr = max(thr, 1e-6)

    edge = grad >= thr

    if dilation_px > 0:
        k = 2 * int(dilation_px) + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        edge = cv2.dilate(edge.astype(np.uint8), kernel, iterations=1).astype(bool)

    return edge & valid_gt


def compute_depth_frame_metrics(
    pred_m: ArrayLike,
    gt_m: ArrayLike,
    depth_min_m: Optional[float] = None,
    depth_max_m: Optional[float] = None,
    edge_dilation_px: int = 1,
    edge_percentile: float = 85.0,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Compute metrics for one predicted depth frame against one GT depth frame.

    Metrics are split into two groups:
    1) validity / coverage metrics:
       - how many GT-valid pixels are recovered
       - how many predicted-valid pixels fall outside GT-valid region
    2) numeric error metrics:
       - only on overlap(valid_pred, valid_gt)

    Parameters
    ----------
    pred_m:
        Predicted depth map in meters.
    gt_m:
        GT depth map in meters.
    depth_min_m:
        Optional lower bound for valid pixels.
    depth_max_m:
        Optional upper bound for valid pixels.
    edge_dilation_px:
        Dilation radius for GT edge mask.
    edge_percentile:
        Percentile threshold for GT edge mask.

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, np.ndarray]]
        metrics:
            JSON-serializable dictionary of scalar metrics for this frame.
        aux:
            Raw vectors (errors, ratios, masks) useful for aggregation or debugging.
    """
    pred = np.asarray(pred_m, dtype=np.float32)
    gt = np.asarray(gt_m, dtype=np.float32)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    valid_pred = valid_depth_mask(pred, depth_min_m, depth_max_m)
    valid_gt = valid_depth_mask(gt, depth_min_m, depth_max_m)

    overlap = valid_pred & valid_gt
    missing = valid_gt & (~valid_pred)
    false_valid = valid_pred & (~valid_gt)

    gt_valid_px = int(valid_gt.sum())
    pred_valid_px = int(valid_pred.sum())
    overlap_px = int(overlap.sum())
    missing_px = int(missing.sum())
    false_valid_px = int(false_valid.sum())

    metrics: Dict[str, Any] = {
        "gt_valid_px": gt_valid_px,
        "pred_valid_px": pred_valid_px,
        "overlap_px": overlap_px,
        "missing_px": missing_px,
        "false_valid_px": false_valid_px,
        "coverage_recall": float(overlap_px / max(gt_valid_px, 1)),
        "valid_precision": float(overlap_px / max(pred_valid_px, 1)),
        "missing_ratio_from_gt": float(missing_px / max(gt_valid_px, 1)),
        "false_valid_ratio_from_pred": float(false_valid_px / max(pred_valid_px, 1)) if pred_valid_px > 0 else 0.0,
    }

    aux: Dict[str, np.ndarray] = {
        "valid_pred_mask": valid_pred,
        "valid_gt_mask": valid_gt,
        "overlap_mask": overlap,
        "missing_mask": missing,
        "false_valid_mask": false_valid,
        "abs_err": np.array([], dtype=np.float32),
        "signed_err": np.array([], dtype=np.float32),
        "rel_err": np.array([], dtype=np.float32),
        "ratio": np.array([], dtype=np.float32),
        "edge_abs_err": np.array([], dtype=np.float32),
        "edge_ratio": np.array([], dtype=np.float32),
        "edge_overlap_mask": np.zeros_like(overlap, dtype=bool),
    }

    if overlap_px == 0:
        metrics.update({
            "mae_m": None,
            "rmse_m": None,
            "medae_m": None,
            "p95_ae_m": None,
            "mean_signed_error_m": None,
            "mean_rel_error": None,
            "delta1": None,
            "edge_overlap_px": 0,
            "edge_mae_m": None,
            "edge_p95_ae_m": None,
            "edge_delta1": None,
        })
        return metrics, aux

    pred_o = pred[overlap]
    gt_o = gt[overlap]

    signed_err = pred_o - gt_o
    abs_err = np.abs(signed_err)
    rel_err = abs_err / np.maximum(gt_o, 1e-6)
    ratio = np.maximum(
        pred_o / np.maximum(gt_o, 1e-6),
        gt_o / np.maximum(pred_o, 1e-6),
    )

    metrics.update({
        "mae_m": float(abs_err.mean()),
        "rmse_m": float(np.sqrt(np.mean(signed_err ** 2))),
        "medae_m": float(np.median(abs_err)),
        "p95_ae_m": float(np.percentile(abs_err, 95.0)),
        "mean_signed_error_m": float(signed_err.mean()),
        "mean_rel_error": float(rel_err.mean()),
        "delta1": float(np.mean(ratio < 1.25)),
    })

    aux["abs_err"] = abs_err.astype(np.float32, copy=False)
    aux["signed_err"] = signed_err.astype(np.float32, copy=False)
    aux["rel_err"] = rel_err.astype(np.float32, copy=False)
    aux["ratio"] = ratio.astype(np.float32, copy=False)

    edge_mask = make_edge_mask_from_gt(
        gt,
        valid_gt=valid_gt,
        dilation_px=edge_dilation_px,
        percentile=edge_percentile,
    )
    edge_overlap = overlap & edge_mask
    edge_overlap_px = int(edge_overlap.sum())

    metrics["edge_overlap_px"] = edge_overlap_px
    aux["edge_overlap_mask"] = edge_overlap

    if edge_overlap_px == 0:
        metrics.update({
            "edge_mae_m": None,
            "edge_p95_ae_m": None,
            "edge_delta1": None,
        })
        return metrics, aux

    pred_e = pred[edge_overlap]
    gt_e = gt[edge_overlap]

    edge_abs_err = np.abs(pred_e - gt_e)
    edge_ratio = np.maximum(
        pred_e / np.maximum(gt_e, 1e-6),
        gt_e / np.maximum(pred_e, 1e-6),
    )

    metrics.update({
        "edge_mae_m": float(edge_abs_err.mean()),
        "edge_p95_ae_m": float(np.percentile(edge_abs_err, 95.0)),
        "edge_delta1": float(np.mean(edge_ratio < 1.25)),
    })

    aux["edge_abs_err"] = edge_abs_err.astype(np.float32, copy=False)
    aux["edge_ratio"] = edge_ratio.astype(np.float32, copy=False)

    return metrics, aux


def aggregate_depth_metrics(
    pred_frames: Sequence[ArrayLike],
    gt_frames: Sequence[ArrayLike],
    depth_min_m: Optional[float] = None,
    depth_max_m: Optional[float] = None,
    edge_dilation_px: int = 1,
    edge_percentile: float = 85.0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Aggregate depth metrics over a sequence of frames.

    This computes:
    - per-frame metrics for each pair
    - global aggregate metrics over concatenated valid overlap pixels

    Parameters
    ----------
    pred_frames:
        Sequence of predicted depth maps in meters.
    gt_frames:
        Sequence of GT depth maps in meters.
    depth_min_m:
        Optional lower bound for valid pixels.
    depth_max_m:
        Optional upper bound for valid pixels.
    edge_dilation_px:
        Dilation radius for GT edge mask.
    edge_percentile:
        Percentile threshold for GT edge mask.

    Returns
    -------
    Tuple[Dict[str, Any], List[Dict[str, Any]]]
        aggregate_metrics:
            Global metrics across all frames.
        per_frame_metrics:
            List of scalar metrics dictionaries, one per frame.
    """
    pred_list = to_frame_list(pred_frames)
    gt_list = to_frame_list(gt_frames)

    if len(pred_list) != len(gt_list):
        raise ValueError(
            f"Frame count mismatch: pred={len(pred_list)} vs gt={len(gt_list)}"
        )

    per_frame_metrics: List[Dict[str, Any]] = []

    total_gt_valid = 0
    total_pred_valid = 0
    total_overlap = 0
    total_missing = 0
    total_false_valid = 0
    total_edge_overlap = 0

    all_abs = []
    all_signed = []
    all_rel = []
    all_ratio = []
    all_edge_abs = []
    all_edge_ratio = []

    for i, (pred, gt) in enumerate(zip(pred_list, gt_list)):
        fm, aux = compute_depth_frame_metrics(
            pred_m=pred,
            gt_m=gt,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
            edge_dilation_px=edge_dilation_px,
            edge_percentile=edge_percentile,
        )
        fm["frame_idx"] = i
        per_frame_metrics.append(fm)

        total_gt_valid += fm["gt_valid_px"]
        total_pred_valid += fm["pred_valid_px"]
        total_overlap += fm["overlap_px"]
        total_missing += fm["missing_px"]
        total_false_valid += fm["false_valid_px"]
        total_edge_overlap += fm["edge_overlap_px"]

        if aux["abs_err"].size:
            all_abs.append(aux["abs_err"])
            all_signed.append(aux["signed_err"])
            all_rel.append(aux["rel_err"])
            all_ratio.append(aux["ratio"])

        if aux["edge_abs_err"].size:
            all_edge_abs.append(aux["edge_abs_err"])
            all_edge_ratio.append(aux["edge_ratio"])

    abs_all = np.concatenate(all_abs) if all_abs else np.array([], dtype=np.float32)
    signed_all = np.concatenate(all_signed) if all_signed else np.array([], dtype=np.float32)
    rel_all = np.concatenate(all_rel) if all_rel else np.array([], dtype=np.float32)
    ratio_all = np.concatenate(all_ratio) if all_ratio else np.array([], dtype=np.float32)

    edge_abs_all = np.concatenate(all_edge_abs) if all_edge_abs else np.array([], dtype=np.float32)
    edge_ratio_all = np.concatenate(all_edge_ratio) if all_edge_ratio else np.array([], dtype=np.float32)

    aggregate: Dict[str, Any] = {
        "frame_count": len(pred_list),
        "gt_valid_px": int(total_gt_valid),
        "pred_valid_px": int(total_pred_valid),
        "overlap_px": int(total_overlap),
        "missing_px": int(total_missing),
        "false_valid_px": int(total_false_valid),
        "edge_overlap_px": int(total_edge_overlap),
        "coverage_recall": float(total_overlap / max(total_gt_valid, 1)),
        "valid_precision": float(total_overlap / max(total_pred_valid, 1)),
        "missing_ratio_from_gt": float(total_missing / max(total_gt_valid, 1)),
        "false_valid_ratio_from_pred": float(total_false_valid / max(total_pred_valid, 1)) if total_pred_valid > 0 else 0.0,
    }

    if abs_all.size:
        aggregate.update({
            "mae_m": float(abs_all.mean()),
            "rmse_m": float(np.sqrt(np.mean(signed_all ** 2))),
            "medae_m": float(np.median(abs_all)),
            "p95_ae_m": float(np.percentile(abs_all, 95.0)),
            "mean_signed_error_m": float(signed_all.mean()),
            "mean_rel_error": float(rel_all.mean()),
            "delta1": float(np.mean(ratio_all < 1.25)),
        })
    else:
        aggregate.update({
            "mae_m": None,
            "rmse_m": None,
            "medae_m": None,
            "p95_ae_m": None,
            "mean_signed_error_m": None,
            "mean_rel_error": None,
            "delta1": None,
        })

    if edge_abs_all.size:
        aggregate.update({
            "edge_mae_m": float(edge_abs_all.mean()),
            "edge_p95_ae_m": float(np.percentile(edge_abs_all, 95.0)),
            "edge_delta1": float(np.mean(edge_ratio_all < 1.25)),
        })
    else:
        aggregate.update({
            "edge_mae_m": None,
            "edge_p95_ae_m": None,
            "edge_delta1": None,
        })

    return aggregate, per_frame_metrics


def rank_depth_runs(
    runs: Sequence[Dict[str, Any]],
    *,
    mae_weight: float = 1.0,
    edge_mae_weight: float = 0.5,
    coverage_penalty_weight: float = 2.0,
    missing_target_max: float = 0.50,
) -> List[Dict[str, Any]]:
    """
    Rank experimental runs by a simple scalar score.

    Lower score is better.

    Suggested usage:
    - compare sweep results after aggregate_depth_metrics(...)
    - keep numeric depth error low
    - penalize excessive missing pixels

    Score:
        score =
            mae_weight * mae_m
            + edge_mae_weight * edge_mae_m
            + coverage_penalty_weight * max(0, missing_ratio_from_gt - missing_target_max)

    Parameters
    ----------
    runs:
        Sequence of dictionaries. Each dict must contain at least:
        {
            "aggregate_metrics": {...}
        }
    mae_weight:
        Weight for global MAE.
    edge_mae_weight:
        Weight for edge MAE.
    coverage_penalty_weight:
        Penalty weight for excessive missing ratio.
    missing_target_max:
        Allowed upper target for missing_ratio_from_gt before penalty starts.

    Returns
    -------
    List[Dict[str, Any]]
        New list of runs sorted by ascending score.
        Each result contains added key "rank_score".
    """
    ranked = []

    for run in runs:
        agg = run.get("aggregate_metrics", {}) or {}

        mae = agg.get("mae_m", None)
        edge_mae = agg.get("edge_mae_m", None)
        missing_ratio = agg.get("missing_ratio_from_gt", None)

        if mae is None:
            mae = 1e9
        if edge_mae is None:
            edge_mae = 1e9
        if missing_ratio is None:
            missing_ratio = 1.0

        coverage_penalty = max(0.0, float(missing_ratio) - float(missing_target_max))

        score = (
            float(mae_weight) * float(mae)
            + float(edge_mae_weight) * float(edge_mae)
            + float(coverage_penalty_weight) * float(coverage_penalty)
        )

        item = dict(run)
        item["rank_score"] = float(score)
        ranked.append(item)

    ranked.sort(key=lambda x: x["rank_score"])
    return ranked
