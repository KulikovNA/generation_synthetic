from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check fragment registration transform-chain sanity")
    parser.add_argument("--scene_dir", type=str, required=True, help="Path to scene_XXXXXX directory")
    parser.add_argument("--tolerance_m", type=float, default=1e-4)
    return parser.parse_args()


def transform_points(mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pts_h = np.concatenate([pts.astype(np.float64), np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return (mat.astype(np.float64) @ pts_h.T).T[:, :3]


def main() -> int:
    args = parse_args()
    scene_dir = os.path.abspath(args.scene_dir)
    gt_path = os.path.join(scene_dir, "gt_annotations.json")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(gt_path)

    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    max_error = 0.0
    checked = 0
    per_frame = []

    for frame in gt.get("frames", []):
        vp_path = os.path.join(scene_dir, frame["visible_points"])
        if not os.path.isfile(vp_path):
            raise FileNotFoundError(vp_path)
        visible = np.load(vp_path)
        shell_indices = visible["shell_indices"].astype(np.int64)
        if shell_indices.size == 0:
            per_frame.append((frame["frame_id"], 0, 0.0))
            continue

        fragment_tfms: Dict[int, np.ndarray] = {
            int(item["fragment_id"]): np.asarray(item["T_C_from_O"], dtype=np.float64)
            for item in frame.get("fragments", [])
        }

        frame_errors = []
        for fragment_id in np.unique(visible["fragment_id"][shell_indices]):
            fragment_id = int(fragment_id)
            if fragment_id not in fragment_tfms:
                continue
            idx = shell_indices[visible["fragment_id"][shell_indices] == fragment_id]
            pred_c = transform_points(fragment_tfms[fragment_id], visible["points_O"][idx])
            err = np.linalg.norm(pred_c - visible["points_C"][idx], axis=1)
            if err.size:
                frame_errors.append(err)

        if frame_errors:
            frame_err = np.concatenate(frame_errors)
            checked += int(frame_err.size)
            frame_max = float(frame_err.max())
            max_error = max(max_error, frame_max)
            per_frame.append((frame["frame_id"], int(frame_err.size), frame_max))
        else:
            per_frame.append((frame["frame_id"], 0, 0.0))

    print(f"Checked shell points: {checked}")
    for frame_id, count, frame_max in per_frame:
        print(f"frame_{int(frame_id):06d}: shell_points={count}, max_error_m={frame_max:.8g}")
    print(f"max_error_m={max_error:.8g}, tolerance_m={float(args.tolerance_m):.8g}")

    return 0 if max_error <= float(args.tolerance_m) else 2


if __name__ == "__main__":
    sys.exit(main())
