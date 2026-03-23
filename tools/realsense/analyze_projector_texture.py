#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Анализ projector_texture.png.

Что считает скрипт:
1) бинаризует изображение и выделяет отдельные точки через connected components;
2) оценивает покрытие/плотность в скользящем окне;
3) оценивает расстояния между центрами ближайших точек;
4) строит статистику по вертикальным полосам (слева -> справа);
5) оценивает верхнюю и нижнюю огибающие по центрам точек, чтобы увидеть skew/наклон;
6) сохраняет PNG-диагностику и JSON со сводкой.

Типичный запуск:
python analyze_projector_texture.py \
  --image wall_capture/session_20260314_143949/projector_stage2/projector_texture.png

Пример с настройкой:
python analyze_projector_texture.py \
  --image wall_capture/session_20260314_143949/projector_stage2/projector_texture.png \
  --outdir wall_capture/session_20260314_143949/projector_stage2/analysis_texture \
  --window-w 192 --window-h 192 \
  --num-x-bins 48
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def robust_stats(x: np.ndarray) -> Dict:
    x = np.asarray(x)
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p5": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)),
        "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return img


def to_binary(img: np.ndarray,
              threshold: int | None,
              blur_ksize: int = 0,
              morph_open: int = 0) -> Tuple[np.ndarray, Dict]:
    work = img.copy()

    if blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        work = cv2.GaussianBlur(work, (blur_ksize, blur_ksize), 0)

    if threshold is None:
        thr_val, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold_used = int(thr_val)
        threshold_mode = "otsu"
    else:
        _, binary = cv2.threshold(work, int(threshold), 255, cv2.THRESH_BINARY)
        threshold_used = int(threshold)
        threshold_mode = "manual"

    if morph_open > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    meta = {
        "threshold_mode": threshold_mode,
        "threshold_value": threshold_used,
        "blur_ksize": int(blur_ksize),
        "morph_open": int(morph_open),
    }
    return binary, meta


def extract_components(binary: np.ndarray,
                       min_area_px: int = 1,
                       max_area_px: int = 10_000) -> Dict[str, np.ndarray]:
    m = (binary > 0).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)

    centers = []
    areas = []
    eq_r = []
    bboxes = []

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_area_px) or area > int(max_area_px):
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[i]

        centers.append([float(cx), float(cy)])
        areas.append(float(area))
        eq_r.append(float(np.sqrt(area / np.pi)))
        bboxes.append([x, y, w, h])

    if not centers:
        z2 = np.zeros((0, 2), dtype=np.float32)
        z1 = np.zeros((0,), dtype=np.float32)
        z4 = np.zeros((0, 4), dtype=np.int32)
        return {
            "centers_xy": z2,
            "areas_px": z1,
            "eq_radius_px": z1,
            "bboxes_xywh": z4,
            "labels": labels,
        }

    return {
        "centers_xy": np.asarray(centers, dtype=np.float32),
        "areas_px": np.asarray(areas, dtype=np.float32),
        "eq_radius_px": np.asarray(eq_r, dtype=np.float32),
        "bboxes_xywh": np.asarray(bboxes, dtype=np.int32),
        "labels": labels,
    }


def compute_window_density_maps(binary: np.ndarray,
                                centers_xy: np.ndarray,
                                window_w: int,
                                window_h: int) -> Dict[str, np.ndarray]:
    h, w = binary.shape
    occ = (binary > 0).astype(np.float32)
    kernel = np.ones((int(window_h), int(window_w)), dtype=np.float32)

    occ_sum = cv2.filter2D(occ, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    occ_density = occ_sum / float(window_h * window_w)

    dot_map = np.zeros((h, w), dtype=np.float32)
    if centers_xy.shape[0] > 0:
        x = np.clip(np.round(centers_xy[:, 0]).astype(int), 0, w - 1)
        y = np.clip(np.round(centers_xy[:, 1]).astype(int), 0, h - 1)
        dot_map[y, x] = 1.0

    dot_sum = cv2.filter2D(dot_map, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    dot_density = dot_sum / float(window_h * window_w)

    return {
        "occupancy_density": occ_density.astype(np.float32),
        "centroid_density": dot_density.astype(np.float32),
        "centroid_map": dot_map.astype(np.float32),
    }


def compute_nearest_neighbor_distances(centers_xy: np.ndarray) -> np.ndarray:
    n = centers_xy.shape[0]
    if n <= 1:
        return np.zeros((n,), dtype=np.float32)

    pts = centers_xy.astype(np.float32)
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(np.min(d2, axis=1))
    return nn.astype(np.float32)


def stripe_statistics(binary: np.ndarray,
                      centers_xy: np.ndarray,
                      nn_dist: np.ndarray,
                      num_x_bins: int,
                      envelope_q_low: float,
                      envelope_q_high: float,
                      min_points_per_bin: int = 8) -> Dict[str, np.ndarray]:
    h, w = binary.shape
    edges = np.linspace(0.0, float(w), int(num_x_bins) + 1)
    x_mid = 0.5 * (edges[:-1] + edges[1:])

    coverage = []
    count_density = []
    nn_median = []
    nn_p25 = []
    nn_p75 = []
    y_top = []
    y_bottom = []
    width = []
    counts = []

    occ = (binary > 0).astype(np.float32)

    x = centers_xy[:, 0] if centers_xy.shape[0] > 0 else np.zeros((0,), dtype=np.float32)
    y = centers_xy[:, 1] if centers_xy.shape[0] > 0 else np.zeros((0,), dtype=np.float32)

    for i in range(len(edges) - 1):
        x0 = int(round(edges[i]))
        x1 = int(round(edges[i + 1]))
        x1 = max(x1, x0 + 1)

        strip_occ = occ[:, x0:x1]
        coverage.append(float(np.mean(strip_occ)))

        m = (x >= edges[i]) & (x < edges[i + 1])
        xi = x[m]
        yi = y[m]
        nni = nn_dist[m] if nn_dist.shape[0] == x.shape[0] else np.zeros((0,), dtype=np.float32)
        counts.append(int(np.count_nonzero(m)))

        strip_area = float(h * max(1, x1 - x0))
        count_density.append(float(np.count_nonzero(m) / strip_area))

        if np.count_nonzero(m) >= max(2, min_points_per_bin):
            nn_median.append(float(np.median(nni)))
            nn_p25.append(float(np.percentile(nni, 25)))
            nn_p75.append(float(np.percentile(nni, 75)))

            top_i = float(np.percentile(yi, envelope_q_low))
            bottom_i = float(np.percentile(yi, envelope_q_high))
            y_top.append(top_i)
            y_bottom.append(bottom_i)
            width.append(float(bottom_i - top_i))
        else:
            nn_median.append(np.nan)
            nn_p25.append(np.nan)
            nn_p75.append(np.nan)
            y_top.append(np.nan)
            y_bottom.append(np.nan)
            width.append(np.nan)

    return {
        "x_edges": edges.astype(np.float32),
        "x_mid": x_mid.astype(np.float32),
        "coverage": np.asarray(coverage, dtype=np.float32),
        "count_density": np.asarray(count_density, dtype=np.float32),
        "nn_median": np.asarray(nn_median, dtype=np.float32),
        "nn_p25": np.asarray(nn_p25, dtype=np.float32),
        "nn_p75": np.asarray(nn_p75, dtype=np.float32),
        "y_top": np.asarray(y_top, dtype=np.float32),
        "y_bottom": np.asarray(y_bottom, dtype=np.float32),
        "width": np.asarray(width, dtype=np.float32),
        "counts": np.asarray(counts, dtype=np.int32),
    }


def fit_line_from_valid(x: np.ndarray, y: np.ndarray) -> Dict:
    m = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(m) < 2:
        return {
            "valid_count": int(np.count_nonzero(m)),
            "slope_px_per_px": None,
            "intercept_px": None,
            "angle_deg": None,
        }

    a, b = np.polyfit(x[m], y[m], 1)
    angle_deg = float(np.degrees(np.arctan(a)))
    return {
        "valid_count": int(np.count_nonzero(m)),
        "slope_px_per_px": float(a),
        "intercept_px": float(b),
        "angle_deg": angle_deg,
    }


def save_density_map(path: Path, arr: np.ndarray, cmap: int = cv2.COLORMAP_TURBO) -> None:
    x = np.asarray(arr, dtype=np.float32)
    if x.size == 0:
        raise ValueError("Пустая карта")
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi <= lo + 1e-12:
        out = np.zeros_like(x, dtype=np.uint8)
    else:
        out = np.clip(np.round(255.0 * (x - lo) / (hi - lo)), 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(out, cmap)
    cv2.imwrite(str(path), color)


def draw_centers_overlay(gray: np.ndarray,
                         centers_xy: np.ndarray,
                         y_top: np.ndarray,
                         y_bottom: np.ndarray,
                         x_mid: np.ndarray,
                         top_line: Dict,
                         bottom_line: Dict,
                         out_path: Path) -> None:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for u, v in centers_xy:
        cv2.circle(vis, (int(round(float(u))), int(round(float(v)))), 1, (0, 255, 0), -1, lineType=cv2.LINE_AA)

    for xm, yt, yb in zip(x_mid, y_top, y_bottom):
        if np.isfinite(yt):
            cv2.circle(vis, (int(round(float(xm))), int(round(float(yt)))), 3, (255, 0, 0), -1, lineType=cv2.LINE_AA)
        if np.isfinite(yb):
            cv2.circle(vis, (int(round(float(xm))), int(round(float(yb)))), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    h, w = gray.shape
    xs = np.array([0.0, float(w - 1)], dtype=np.float32)

    if top_line["slope_px_per_px"] is not None:
        ys = top_line["slope_px_per_px"] * xs + top_line["intercept_px"]
        p0 = (int(round(xs[0])), int(round(ys[0])))
        p1 = (int(round(xs[1])), int(round(ys[1])))
        cv2.line(vis, p0, p1, (255, 255, 0), 2, lineType=cv2.LINE_AA)

    if bottom_line["slope_px_per_px"] is not None:
        ys = bottom_line["slope_px_per_px"] * xs + bottom_line["intercept_px"]
        p0 = (int(round(xs[0])), int(round(ys[0])))
        p1 = (int(round(xs[1])), int(round(ys[1])))
        cv2.line(vis, p0, p1, (0, 255, 255), 2, lineType=cv2.LINE_AA)

    cv2.imwrite(str(out_path), vis)


def save_plots(out_path: Path,
               stripe: Dict[str, np.ndarray],
               top_line: Dict,
               bottom_line: Dict) -> None:
    x_mid = stripe["x_mid"]
    coverage = stripe["coverage"]
    nn_med = stripe["nn_median"]
    nn_p25 = stripe["nn_p25"]
    nn_p75 = stripe["nn_p75"]
    width = stripe["width"]
    counts = stripe["counts"]
    y_top = stripe["y_top"]
    y_bottom = stripe["y_bottom"]

    fig = plt.figure(figsize=(12, 12), dpi=140)

    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(x_mid, coverage, marker='o')
    ax1.set_title('Покрытие бинарной маски по вертикальным полосам')
    ax1.set_ylabel('coverage')
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(x_mid, nn_med, marker='o', label='median NN distance')
    ax2.fill_between(x_mid, nn_p25, nn_p75, alpha=0.25, label='25-75 percentile')
    ax2.set_title('Расстояние до ближайшей соседней точки по полосам')
    ax2.set_ylabel('px')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = fig.add_subplot(4, 1, 3)
    ax3.plot(x_mid, width, marker='o', label='top-bottom width')
    ax3.plot(x_mid, counts, marker='x', label='dot count per stripe')
    ax3.set_title('Ширина заполненной области и число точек по полосам')
    ax3.set_ylabel('px / count')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = fig.add_subplot(4, 1, 4)
    ax4.scatter(x_mid, y_top, s=18, label='top envelope')
    ax4.scatter(x_mid, y_bottom, s=18, label='bottom envelope')

    x_line = np.array([np.nanmin(x_mid), np.nanmax(x_mid)], dtype=np.float32)
    if top_line["slope_px_per_px"] is not None:
        y_line = top_line["slope_px_per_px"] * x_line + top_line["intercept_px"]
        ax4.plot(x_line, y_line, linewidth=2, label=f'top fit: {top_line["angle_deg"]:.3f} deg')
    if bottom_line["slope_px_per_px"] is not None:
        y_line = bottom_line["slope_px_per_px"] * x_line + bottom_line["intercept_px"]
        ax4.plot(x_line, y_line, linewidth=2, label=f'bottom fit: {bottom_line["angle_deg"]:.3f} deg')

    ax4.set_title('Огибающие верхней и нижней границы по полосам')
    ax4.set_xlabel('x [px]')
    ax4.set_ylabel('y [px]')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Анализ распределения точек в projector_texture.png")
    ap.add_argument("--image", type=str, required=True, help="Путь к projector_texture.png")
    ap.add_argument("--outdir", type=str, default="", help="Папка для результатов. По умолчанию создаётся рядом с изображением.")
    ap.add_argument("--threshold", type=int, default=None,
                    help="Порог бинаризации [0..255]. Если не задан, используется Otsu.")
    ap.add_argument("--blur-ksize", type=int, default=0, help="Gaussian blur перед порогом")
    ap.add_argument("--morph-open", type=int, default=0, help="Размер ядра открытия после порога")
    ap.add_argument("--min-area", type=int, default=1, help="Минимальная площадь компоненты")
    ap.add_argument("--max-area", type=int, default=1000, help="Максимальная площадь компоненты")
    ap.add_argument("--window-w", type=int, default=192, help="Ширина скользящего окна")
    ap.add_argument("--window-h", type=int, default=192, help="Высота скользящего окна")
    ap.add_argument("--num-x-bins", type=int, default=48, help="Число вертикальных полос")
    ap.add_argument("--envelope-q-low", type=float, default=5.0,
                    help="Нижний процентиль для верхней огибающей")
    ap.add_argument("--envelope-q-high", type=float, default=95.0,
                    help="Верхний процентиль для нижней огибающей")
    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = image_path.parent / "analysis_projector_texture"
    outdir.mkdir(parents=True, exist_ok=True)

    gray = load_gray(image_path)
    h, w = gray.shape

    binary, threshold_meta = to_binary(
        gray,
        threshold=args.threshold,
        blur_ksize=int(args.blur_ksize),
        morph_open=int(args.morph_open),
    )
    cv2.imwrite(str(outdir / "binary_mask.png"), binary)

    comps = extract_components(
        binary,
        min_area_px=int(args.min_area),
        max_area_px=int(args.max_area),
    )
    centers_xy = comps["centers_xy"]
    areas_px = comps["areas_px"]
    eq_radius_px = comps["eq_radius_px"]

    density = compute_window_density_maps(
        binary,
        centers_xy,
        window_w=int(args.window_w),
        window_h=int(args.window_h),
    )
    save_density_map(outdir / "occupancy_density_map.png", density["occupancy_density"])
    save_density_map(outdir / "centroid_density_map.png", density["centroid_density"])

    nn_dist = compute_nearest_neighbor_distances(centers_xy)

    stripe = stripe_statistics(
        binary,
        centers_xy,
        nn_dist,
        num_x_bins=int(args.num_x_bins),
        envelope_q_low=float(args.envelope_q_low),
        envelope_q_high=float(args.envelope_q_high),
        min_points_per_bin=8,
    )

    top_line = fit_line_from_valid(stripe["x_mid"], stripe["y_top"])
    bottom_line = fit_line_from_valid(stripe["x_mid"], stripe["y_bottom"])

    draw_centers_overlay(
        gray,
        centers_xy,
        stripe["y_top"],
        stripe["y_bottom"],
        stripe["x_mid"],
        top_line,
        bottom_line,
        outdir / "centers_and_envelopes_overlay.png",
    )
    save_plots(outdir / "stripe_statistics.png", stripe, top_line, bottom_line)

    left_mask = centers_xy[:, 0] < (0.5 * w) if centers_xy.shape[0] > 0 else np.zeros((0,), dtype=bool)
    right_mask = ~left_mask if centers_xy.shape[0] > 0 else np.zeros((0,), dtype=bool)

    summary = {
        "image": {
            "path": str(image_path),
            "width": int(w),
            "height": int(h),
        },
        "thresholding": threshold_meta,
        "components": {
            "count": int(centers_xy.shape[0]),
            "area_stats_px": robust_stats(areas_px),
            "equivalent_radius_stats_px": robust_stats(eq_radius_px),
            "nearest_neighbor_distance_stats_px": robust_stats(nn_dist),
            "nearest_neighbor_distance_left_half_stats_px": robust_stats(nn_dist[left_mask]),
            "nearest_neighbor_distance_right_half_stats_px": robust_stats(nn_dist[right_mask]),
        },
        "global_density": {
            "binary_coverage_ratio": float(np.mean(binary > 0)),
            "occupancy_density_map_stats": robust_stats(density["occupancy_density"]),
            "centroid_density_map_stats": robust_stats(density["centroid_density"]),
        },
        "stripe_diagnostics": {
            "num_x_bins": int(args.num_x_bins),
            "coverage_stats": robust_stats(stripe["coverage"]),
            "count_density_stats": robust_stats(stripe["count_density"]),
            "nn_median_stats": robust_stats(stripe["nn_median"][np.isfinite(stripe["nn_median"])]),
            "width_stats": robust_stats(stripe["width"][np.isfinite(stripe["width"])]),
            "leftmost_20pct": {
                "coverage_mean": float(np.nanmean(stripe["coverage"][:max(1, len(stripe["coverage"]) // 5)])),
                "nn_median_mean": float(np.nanmean(stripe["nn_median"][:max(1, len(stripe["nn_median"]) // 5)])),
                "width_mean": float(np.nanmean(stripe["width"][:max(1, len(stripe["width"]) // 5)])),
            },
            "rightmost_20pct": {
                "coverage_mean": float(np.nanmean(stripe["coverage"][-max(1, len(stripe["coverage"]) // 5):])),
                "nn_median_mean": float(np.nanmean(stripe["nn_median"][-max(1, len(stripe["nn_median"]) // 5):])),
                "width_mean": float(np.nanmean(stripe["width"][-max(1, len(stripe["width"]) // 5):])),
            },
            "top_envelope_line": top_line,
            "bottom_envelope_line": bottom_line,
        },
        "raw_arrays_preview": {
            "x_mid_px": stripe["x_mid"].tolist(),
            "coverage": stripe["coverage"].tolist(),
            "nn_median_px": np.nan_to_num(stripe["nn_median"], nan=-1.0).tolist(),
            "width_px": np.nan_to_num(stripe["width"], nan=-1.0).tolist(),
            "counts": stripe["counts"].tolist(),
            "y_top_px": np.nan_to_num(stripe["y_top"], nan=-1.0).tolist(),
            "y_bottom_px": np.nan_to_num(stripe["y_bottom"], nan=-1.0).tolist(),
        },
    }

    save_json(outdir / "summary.json", summary)

    print("Done.")
    print(f"Image: {image_path}")
    print(f"Outdir: {outdir}")
    print(f"Components: {centers_xy.shape[0]}")
    print(f"Global binary coverage: {np.mean(binary > 0):.6f}")
    if nn_dist.size > 0:
        print(f"Median NN distance [px]: {np.median(nn_dist):.4f}")
    if top_line['angle_deg'] is not None:
        print(f"Top envelope angle [deg]: {top_line['angle_deg']:.4f}")
    if bottom_line['angle_deg'] is not None:
        print(f"Bottom envelope angle [deg]: {bottom_line['angle_deg']:.4f}")


if __name__ == "__main__":
    main()
