#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TOP_METRICS = [
    "coverage_recall",
    "missing_ratio_from_gt",
    "mae_m",
    "rmse_m",
    "edge_mae_m",
    "p95_ae_m",
    "edge_p95_ae_m",
    "delta1",
    "edge_delta1",
]

LOWER_IS_BETTER = {
    "missing_ratio_from_gt",
    "false_valid_ratio_from_pred",
    "mae_m",
    "rmse_m",
    "medae_m",
    "p95_ae_m",
    "mean_signed_error_m",  # magnitude handled separately in score
    "mean_rel_error",
    "edge_mae_m",
    "edge_p95_ae_m",
}

PLOT_METRICS = [
    "mae_m",
    "edge_mae_m",
    "coverage_recall",
    "p95_ae_m",
    "score",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze sweep_results.jsonl from stereo parameter sweep."
    )
    p.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="Path to sweep_results.jsonl",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: <jsonl_parent>/analysis",
    )
    p.add_argument(
        "--baseline_param_name",
        type=str,
        default="baseline",
        help="Value of param_name used for baseline row",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="How many top runs to keep in summary tables",
    )
    return p.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
    return records


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if np.isfinite(v):
        return v
    return None


def _coerce_scalar(x: Any) -> Any:
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        if np.isfinite(v):
            return v
        return None
    return x


def _flatten_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "timestamp_utc": rec.get("timestamp_utc"),
        "param_name": rec.get("param_name"),
        "param_value": rec.get("param_value"),
        "experiment_dir": rec.get("experiment_dir"),
        "elapsed_sec": rec.get("elapsed_sec"),
    }

    agg = rec.get("aggregate_metrics", {}) or {}
    for k, v in agg.items():
        row[k] = _coerce_scalar(v)

    exp_dir = rec.get("experiment_dir")
    if exp_dir:
        metrics_path = Path(exp_dir) / "metrics.json"
        if metrics_path.exists():
            try:
                payload = _read_json(metrics_path)
                params = payload.get("params", {}) or {}
                for k, v in params.items():
                    row[f"cfg__{k}"] = _coerce_scalar(v)
                row["metrics_json_path"] = str(metrics_path)
            except Exception as e:
                row["metrics_json_path"] = str(metrics_path)
                row["metrics_json_load_error"] = str(e)
    return row


def load_runs(jsonl_path: Path) -> pd.DataFrame:
    records = _read_jsonl(jsonl_path)
    if not records:
        raise ValueError(f"No records found in {jsonl_path}")

    rows = [_flatten_record(r) for r in records]
    df = pd.DataFrame(rows)

    # helpful normalized columns for plotting/sorting
    df["param_value_num"] = pd.to_numeric(df["param_value"], errors="coerce")
    if "cfg__align_splat_2x2" in df.columns:
        df["cfg__align_splat_2x2"] = df["cfg__align_splat_2x2"].map(
            lambda x: bool(x) if pd.notna(x) else x
        )

    return df


def get_baseline_row(df: pd.DataFrame, baseline_param_name: str) -> pd.Series:
    baseline = df[df["param_name"] == baseline_param_name].copy()
    if baseline.empty:
        raise ValueError(
            f"Baseline row with param_name='{baseline_param_name}' not found."
        )
    if len(baseline) > 1:
        baseline = baseline.sort_values("timestamp_utc")
    return baseline.iloc[0]


def add_delta_columns(df: pd.DataFrame, baseline_row: pd.Series) -> pd.DataFrame:
    out = df.copy()
    delta_metrics = [
        "coverage_recall",
        "missing_ratio_from_gt",
        "valid_precision",
        "mae_m",
        "rmse_m",
        "p95_ae_m",
        "edge_mae_m",
        "edge_p95_ae_m",
        "delta1",
        "edge_delta1",
    ]

    for metric in delta_metrics:
        if metric not in out.columns:
            continue
        base = _safe_float(baseline_row.get(metric))
        if base is None:
            continue
        out[f"delta__{metric}"] = pd.to_numeric(out[metric], errors="coerce") - base

    return out


def add_score(df: pd.DataFrame, baseline_row: pd.Series) -> pd.DataFrame:
    out = df.copy()

    base_cov = _safe_float(baseline_row.get("coverage_recall")) or 0.0
    base_missing = _safe_float(baseline_row.get("missing_ratio_from_gt")) or 1.0

    mae = pd.to_numeric(out.get("mae_m"), errors="coerce").fillna(1e9)
    edge_mae = pd.to_numeric(out.get("edge_mae_m"), errors="coerce").fillna(1e9)
    p95 = pd.to_numeric(out.get("p95_ae_m"), errors="coerce").fillna(1e9)
    coverage = pd.to_numeric(out.get("coverage_recall"), errors="coerce").fillna(0.0)
    missing = pd.to_numeric(out.get("missing_ratio_from_gt"), errors="coerce").fillna(1.0)

    coverage_penalty = (base_cov - coverage).clip(lower=0.0)
    missing_penalty = (missing - base_missing).clip(lower=0.0)

    out["score"] = (
        1.0 * mae
        + 0.50 * edge_mae
        + 0.25 * p95
        + 2.00 * coverage_penalty
        + 1.50 * missing_penalty
    )

    # lower is better
    out["rank_global"] = out["score"].rank(method="min", ascending=True).astype(int)
    return out


def sort_group_for_param(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    if g["param_value_num"].notna().all():
        return g.sort_values("param_value_num")

    # bools often land as object strings in JSONL, sort False < True if possible
    unique_vals = list(g["param_value"].dropna().unique())
    if set(unique_vals).issubset({False, True, "False", "True", "false", "true"}):
        def _bool_key(v: Any) -> int:
            return 1 if str(v).lower() == "true" else 0
        return g.assign(_sort_key=g["param_value"].map(_bool_key)).sort_values("_sort_key").drop(columns=["_sort_key"])

    return g.sort_values("param_value")


def detect_constant_params(group: pd.DataFrame, varied_param_name: str) -> Dict[str, Any]:
    const: Dict[str, Any] = {}
    cfg_cols = [c for c in group.columns if c.startswith("cfg__")]
    varied_cfg_key = f"cfg__{varied_param_name}"

    for col in cfg_cols:
        if col == varied_cfg_key:
            continue
        vals = group[col].dropna().tolist()
        if not vals:
            continue
        uniq = []
        seen = set()
        for v in vals:
            k = json.dumps(v, sort_keys=True, ensure_ascii=False, default=str)
            if k not in seen:
                uniq.append(v)
                seen.add(k)
        if len(uniq) == 1:
            const[col.replace("cfg__", "")] = uniq[0]
    return const


def choose_best_in_group(group: pd.DataFrame) -> pd.Series:
    g = group.copy()
    if "score" in g.columns:
        g = g.sort_values("score", ascending=True)
    else:
        g = g.sort_values("mae_m", ascending=True)
    return g.iloc[0]


def _metric_direction(metric: str) -> str:
    return "min" if metric in LOWER_IS_BETTER else "max"


def _format_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(float(v)):
            return "nan"
        return f"{float(v):.6f}".rstrip("0").rstrip(".")
    return str(v)


def _format_constants_box(const_params: Dict[str, Any], max_lines: int = 12) -> str:
    if not const_params:
        return "fixed params: unavailable"
    items = [f"{k}={_format_scalar(v)}" for k, v in sorted(const_params.items())]
    if len(items) > max_lines:
        hidden = len(items) - max_lines
        items = items[:max_lines] + [f"... +{hidden} more"]
    return "fixed params\n" + "\n".join(items)


def _prepare_x(group: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    vals = group["param_value"].tolist()
    num = pd.to_numeric(pd.Series(vals), errors="coerce")
    if num.notna().all():
        x = num.to_numpy(dtype=float)
        labels = [_format_scalar(v) for v in vals]
        return x, labels

    # categorical fallback
    x = np.arange(len(vals), dtype=float)
    labels = [_format_scalar(v) for v in vals]
    return x, labels


def plot_metric_group(
    group: pd.DataFrame,
    baseline_row: pd.Series,
    param_name: str,
    out_dir: Path,
    const_params: Dict[str, Any],
) -> None:
    plot_dir = out_dir / "plots" / param_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    g = sort_group_for_param(group)
    x, labels = _prepare_x(g)

    for metric in PLOT_METRICS:
        if metric not in g.columns:
            continue

        y = pd.to_numeric(g[metric], errors="coerce")
        valid = y.notna().to_numpy()
        if not np.any(valid):
            continue

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(x[valid], y.to_numpy(dtype=float)[valid], marker="o")

        base_val = _safe_float(baseline_row.get(metric))
        if base_val is not None:
            ax.axhline(base_val, linestyle="--", linewidth=1.0)
            ax.text(
                0.99,
                0.98,
                f"baseline {metric}={_format_scalar(base_val)}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
            )

        best_idx = None
        if metric == "score" or _metric_direction(metric) == "min":
            best_idx = y.idxmin()
        else:
            best_idx = y.idxmax()

        if best_idx is not None and pd.notna(best_idx):
            best_row = g.loc[best_idx]
            best_x = x[list(g.index).index(best_idx)]
            best_y = float(best_row[metric])
            ax.scatter([best_x], [best_y], s=80, zorder=5)
            ax.annotate(
                f"best={_format_scalar(best_row['param_value'])}",
                (best_x, best_y),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=9,
            )

        ax.set_title(f"{param_name}: {metric}")
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)

        ax.text(
            1.01,
            0.02,
            _format_constants_box(const_params),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            family="monospace",
        )

        fig.tight_layout()
        fig.savefig(plot_dir / f"{metric}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def write_group_outputs(
    df: pd.DataFrame,
    baseline_row: pd.Series,
    out_dir: Path,
    top_k: int,
) -> pd.DataFrame:
    groups_dir = out_dir / "per_parameter"
    groups_dir.mkdir(parents=True, exist_ok=True)

    best_rows: List[Dict[str, Any]] = []

    for param_name in sorted(x for x in df["param_name"].dropna().unique() if x != "baseline"):
        group = df[df["param_name"] == param_name].copy()
        if group.empty:
            continue

        group = sort_group_for_param(group)
        const_params = detect_constant_params(group, varied_param_name=param_name)

        group["rank_in_param_by_score"] = group["score"].rank(method="min", ascending=True).astype(int)
        group["rank_in_param_by_mae"] = pd.to_numeric(group["mae_m"], errors="coerce").rank(method="min", ascending=True)
        group = group.sort_values(["score", "mae_m"], ascending=[True, True])

        param_dir = groups_dir / param_name
        param_dir.mkdir(parents=True, exist_ok=True)

        # full table
        group.to_csv(param_dir / "all_runs.csv", index=False)

        # concise table
        concise_cols = [
            c for c in [
                "param_name", "param_value", "score",
                "coverage_recall", "delta__coverage_recall",
                "missing_ratio_from_gt", "delta__missing_ratio_from_gt",
                "mae_m", "delta__mae_m",
                "rmse_m", "delta__rmse_m",
                "edge_mae_m", "delta__edge_mae_m",
                "p95_ae_m", "delta__p95_ae_m",
                "delta1", "delta__delta1",
                "edge_delta1", "delta__edge_delta1",
                "elapsed_sec", "experiment_dir"
            ] if c in group.columns
        ]
        group[concise_cols].to_csv(param_dir / "summary.csv", index=False)

        with open(param_dir / "fixed_other_params.json", "w", encoding="utf-8") as f:
            json.dump(const_params, f, ensure_ascii=False, indent=2)

        best = choose_best_in_group(group)
        best_rows.append(best.to_dict())

        # top-k markdown-ish text summary
        with open(param_dir / "top_runs.txt", "w", encoding="utf-8") as f:
            f.write(f"Parameter: {param_name}\n")
            f.write("Fixed other params:\n")
            for k, v in sorted(const_params.items()):
                f.write(f"  - {k} = {_format_scalar(v)}\n")
            f.write("\nTop runs:\n")
            preview = group.head(top_k)
            for _, row in preview.iterrows():
                f.write(
                    f"  * value={_format_scalar(row['param_value'])} | "
                    f"score={_format_scalar(row.get('score'))} | "
                    f"mae={_format_scalar(row.get('mae_m'))} | "
                    f"edge_mae={_format_scalar(row.get('edge_mae_m'))} | "
                    f"coverage={_format_scalar(row.get('coverage_recall'))} | "
                    f"missing={_format_scalar(row.get('missing_ratio_from_gt'))}\n"
                )

        plot_metric_group(group, baseline_row, param_name, out_dir, const_params)

    best_df = pd.DataFrame(best_rows)
    if not best_df.empty:
        best_df = best_df.sort_values("score", ascending=True)
        best_df.to_csv(groups_dir / "best_per_parameter.csv", index=False)
    return best_df


def write_global_outputs(
    df: pd.DataFrame,
    baseline_row: pd.Series,
    out_dir: Path,
    top_k: int,
    best_per_param: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "all_runs_flat.csv", index=False)

    baseline_df = df[df["param_name"] == baseline_row["param_name"]].copy()
    baseline_df.to_csv(out_dir / "baseline.csv", index=False)

    ranked = df[df["param_name"] != baseline_row["param_name"]].copy()
    ranked = ranked.sort_values(["score", "mae_m"], ascending=[True, True])
    ranked.to_csv(out_dir / "all_runs_ranked.csv", index=False)
    ranked.head(top_k).to_csv(out_dir / "top_overall.csv", index=False)

    # Human-readable markdown summary
    summary_md = out_dir / "summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Sweep analysis summary\n\n")

        f.write("## Baseline\n\n")
        for metric in TOP_METRICS:
            if metric in baseline_row.index:
                f.write(f"- {metric}: {_format_scalar(baseline_row.get(metric))}\n")
        f.write("\n")

        f.write("## Top overall (excluding baseline)\n\n")
        if ranked.empty:
            f.write("No non-baseline runs found.\n\n")
        else:
            preview_cols = [
                c for c in [
                    "param_name", "param_value", "score",
                    "coverage_recall", "missing_ratio_from_gt",
                    "mae_m", "edge_mae_m", "p95_ae_m", "experiment_dir"
                ] if c in ranked.columns
            ]
            f.write(ranked.head(top_k)[preview_cols].to_string(index=False))
            f.write("\n\n")

        f.write("## Best run for each varied parameter\n\n")
        if best_per_param.empty:
            f.write("No per-parameter groups found.\n")
        else:
            preview_cols = [
                c for c in [
                    "param_name", "param_value", "score",
                    "coverage_recall", "missing_ratio_from_gt",
                    "mae_m", "edge_mae_m", "p95_ae_m", "experiment_dir"
                ] if c in best_per_param.columns
            ]
            f.write(best_per_param[preview_cols].to_string(index=False))
            f.write("\n")


def main() -> None:
    args = parse_args()

    jsonl_path = Path(args.jsonl).expanduser().resolve()
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else jsonl_path.parent / "analysis"
    )

    df = load_runs(jsonl_path)
    baseline_row = get_baseline_row(df, args.baseline_param_name)
    df = add_delta_columns(df, baseline_row)
    df = add_score(df, baseline_row)

    best_per_param = write_group_outputs(df, baseline_row, out_dir, top_k=args.top_k)
    write_global_outputs(df, baseline_row, out_dir, top_k=args.top_k, best_per_param=best_per_param)

    print(f"[OK] Analysis written to: {out_dir}")
    print(f"[OK] Main summary: {out_dir / 'summary.md'}")
    print(f"[OK] Ranked table: {out_dir / 'all_runs_ranked.csv'}")


if __name__ == "__main__":
    main()
