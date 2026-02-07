#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_bop.py — контур проверки датасета
(поддержка .png/.jpg/.jpeg/.exr; каталоги сцен 000000, 000001, …; параллельная обработка; прогресс-бар)

Коды возврата:
 0 — PASS (критических ошибок нет), 2 — FAIL (есть Critical).
"""
import argparse
import csv
import json
import os
import sys
import time
from glob import glob
from typing import Any, Dict, List, Tuple
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed

# опциональные зависимости
try:
    import yaml
except Exception:
    yaml = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # прогресс-бар необязателен

import numpy as np
import cv2

SUPPORTED_EXTS = ("png", "jpg", "jpeg", "exr")

# ---------------------------- утилиты IO ---------------------------- #

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def load_json(p: str) -> Any:
    with open(p, "r") as f:
        return json.load(f)

def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def write_empty_issues(path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write("level,scene,im_id,check,msg\n")

def blake_hash_file(path: str) -> str:
    import hashlib
    h = hashlib.blake2b(digest_size=32)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# ------------------------- матем. утилиты -------------------------- #

def mat3_is_rotation(R: np.ndarray, tol: float = 1e-3) -> bool:
    if R.shape != (3, 3):
        return False
    should_I = R.T @ R
    I = np.eye(3)
    det = np.linalg.det(R)
    return np.allclose(should_I, I, atol=tol) and abs(det - 1.0) < 1e-2

def bbox_area(b: List[float]) -> float:
    return max(0.0, float(b[2])) * max(0.0, float(b[3]))

def bbox_clip(b: List[float], W: int, H: int) -> List[float]:
    x, y, w, h = map(float, b)
    x2, y2 = x + w, y + h
    x = max(0.0, min(W - 1.0, x))
    y = max(0.0, min(H - 1.0, y))
    x2 = max(0.0, min(float(W), x2))
    y2 = max(0.0, min(float(H), y2))
    return [x, y, max(0.0, x2 - x), max(0.0, y2 - y)]

def iou_bbox(a: List[float], b: List[float]) -> float:
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0.0 else 0.0

def mask_subset_ratio(mask_visib_path: str, mask_full_path: str) -> float:
    mv = cv2.imread(mask_visib_path, cv2.IMREAD_UNCHANGED)
    mf = cv2.imread(mask_full_path, cv2.IMREAD_UNCHANGED)
    if mv is None or mf is None:
        return 0.0
    mv_bin = (mv > 0).astype(np.uint8)
    mf_bin = (mf > 0).astype(np.uint8)
    inter = (mv_bin & mf_bin).sum()
    vis_cnt = mv_bin.sum()
    return (inter / vis_cnt) if vis_cnt > 0 else 1.0

# -------- авто-детект имён кадров и масок (паддинг/расширение) ----- #

@lru_cache(maxsize=None)
def index_numeric_stems(dir_path: str) -> Dict[int, str]:
    p = Path(dir_path)
    if not p.exists() or not p.is_dir():
        return {}
    idx: Dict[int, str] = {}
    for ext in SUPPORTED_EXTS:
        for fp in p.glob(f"*.{ext}"):
            try:
                k = int(fp.stem.lstrip("0") or "0")
            except ValueError:
                continue
            if k not in idx:
                idx[k] = str(fp)
    return idx

def resolve_frame_file(scene_dir: str, subdir: str, im_id: int) -> str:
    base = os.path.join(scene_dir, subdir)
    return index_numeric_stems(base).get(int(im_id), "")

def find_mask(scene_dir: str, subdir: str, im_id: int, inst_idx: int) -> str:
    base = os.path.join(scene_dir, subdir)
    # строгий вариант
    for ext in SUPPORTED_EXTS:
        cand = os.path.join(base, f"{im_id:06d}_{inst_idx:06d}.{ext}")
        if os.path.exists(cand):
            return cand
    # свободный: *{im_id}*_{inst_idx}.ext
    cands = []
    for ext in SUPPORTED_EXTS:
        cands.extend(glob(os.path.join(base, f"*{im_id}*_{inst_idx}.{ext}")))
    return cands[0] if cands else ""

# ---------------------------- проверка сцены ------------------------ #

def validate_scene(scene_dir: str, thresholds: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
    """Возвращает (metrics_rows, issues_rows, perf_info)."""
    t0 = time.time()
    metrics: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []

    p_gt = os.path.join(scene_dir, "scene_gt.json")
    p_gtinfo = os.path.join(scene_dir, "scene_gt_info.json")
    p_camera = os.path.join(scene_dir, "scene_camera.json")
    for p in (p_gt, p_gtinfo, p_camera):
        if not os.path.exists(p):
            issues.append(dict(level="Critical", scene=scene_dir, im_id="*", check="files",
                               msg=f"missing {os.path.basename(p)}"))
            perf = dict(scene=scene_dir, seconds=round(time.time()-t0,3), frames=0)
            return metrics, issues, perf

    gt = load_json(p_gt)
    gti = load_json(p_gtinfo)
    cam = load_json(p_camera)

    try:
        im_ids = sorted([int(k) for k in gt.keys()])
    except Exception:
        im_ids = sorted(list(gt.keys()))

    frames_count = 0

    for im_id in im_ids:
        key = str(im_id)
        rgb_path = resolve_frame_file(scene_dir, "rgb", int(im_id))
        # depth не обязателен
        # depth_path = resolve_frame_file(scene_dir, "depth", int(im_id))

        if not rgb_path or not os.path.exists(rgb_path):
            issues.append(dict(level="Critical", scene=scene_dir, im_id=im_id, check="files",
                               msg="missing rgb (auto-detect failed)"))
            continue

        rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        if rgb is None:
            issues.append(dict(level="Critical", scene=scene_dir, im_id=im_id, check="files",
                               msg=f"rgb unreadable: {os.path.basename(rgb_path)}"))
            continue
        H, W = rgb.shape[:2]

        if key not in cam:
            issues.append(dict(level="Critical", scene=scene_dir, im_id=im_id, check="camera",
                               msg="no camera for im_id"))
            continue
        K = np.array(cam[key].get("cam_K")).reshape(3, 3)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        if not (fx > 0.0 and fy > 0.0):
            issues.append(dict(level="Critical", scene=scene_dir, im_id=im_id, check="camera",
                               msg="fx/fy <= 0"))

        tol = float(thresholds["camera"]["cx_cy_tol_px"])
        if abs(cx - (W / 2.0)) > tol or abs(cy - (H / 2.0)) > tol:
            issues.append(dict(level="Major", scene=scene_dir, im_id=im_id, check="camera",
                               msg=f"cx/cy far from center (tol={tol}px)"))

        if key not in gt:
            issues.append(dict(level="Minor", scene=scene_dir, im_id=im_id, check="gt",
                               msg="no objects for im_id"))
            continue

        inst_list = gt[key]
        for inst_idx, inst in enumerate(inst_list):
            obj_id = inst.get("obj_id")
            R_raw = inst.get("rot_matrix") or inst.get("cam_R_m2c")
            t_raw = inst.get("cam_t_m2c") or inst.get("trans_vec")
            try:
                R = np.array(R_raw, dtype=np.float64).reshape(3, 3)
            except Exception:
                R = np.zeros((3, 3), dtype=np.float64)
            try:
                t = np.array(t_raw, dtype=np.float64).reshape(3)
            except Exception:
                t = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

            if not mat3_is_rotation(R):
                issues.append(dict(level="Critical", scene=scene_dir, im_id=im_id, check="pose",
                                   msg=f"R not rotation (obj {obj_id})"))
            if not np.all(np.isfinite(t)):
                issues.append(dict(level="Critical", scene=scene_dir, im_id=im_id, check="pose",
                                   msg=f"t not finite (obj {obj_id})"))

            # gt_info + маски
            if key in gti and inst_idx < len(gti[key]):
                info = gti[key][inst_idx]
                bbox_v = list(info.get("bbox_visib", [0, 0, 0, 0]))
                bbox_o = list(info.get("bbox_obj", [0, 0, 0, 0]))

                for name, bb in (("bbox_visib", bbox_v), ("bbox_obj", bbox_o)):
                    try:
                        finite = all(np.isfinite([float(x) for x in bb]))
                    except Exception:
                        finite = False
                    if not finite:
                        issues.append(dict(level="Critical", scene=scene_dir, im_id=im_id, check="bbox",
                                           msg=f"{name} not finite (obj {obj_id})"))
                        continue
                    clipped = bbox_clip(bb, W, H)
                    if bbox_area(clipped) <= 0.0:
                        issues.append(dict(level="Major", scene=scene_dir, im_id=im_id, check="bbox",
                                           msg=f"{name} area<=0 (obj {obj_id})"))
                    if iou_bbox(bb, clipped) < 1.0:
                        issues.append(dict(level="Major", scene=scene_dir, im_id=im_id, check="bbox",
                                           msg=f"{name} out of image (obj {obj_id})"))

                # маски (если есть)
                mvis = find_mask(scene_dir, "mask_visib", int(im_id), int(inst_idx))
                mfull = find_mask(scene_dir, "mask", int(im_id), int(inst_idx))
                if mvis and mfull:
                    ratio = mask_subset_ratio(mvis, mfull)
                    if ratio < float(thresholds["masks"]["visib_in_full_ratio"]):
                        issues.append(dict(level="Critical", scene=scene_dir, im_id=im_id, check="masks",
                                           msg=f"mask_visib not subset (ratio={ratio:.3f})"))
                    mv = cv2.imread(mvis, cv2.IMREAD_UNCHANGED)
                    mf = cv2.imread(mfull, cv2.IMREAD_UNCHANGED)
                    visib_fract_est = (mv > 0).sum() / max(1, (mf > 0).sum())
                    visib_fract_gt = float(info.get("visib_fract", visib_fract_est))
                    if abs(visib_fract_est - visib_fract_gt) > float(thresholds["masks"]["visib_fract_tol"]):
                        issues.append(dict(level="Major", scene=scene_dir, im_id=im_id, check="masks",
                                           msg=f"visib_fract mismatch est={visib_fract_est:.3f} gt={visib_fract_gt:.3f}"))
            else:
                issues.append(dict(level="Major", scene=scene_dir, im_id=im_id, check="gt_info",
                                   msg=f"missing gt_info for inst {inst_idx}"))

        metrics.append(dict(metric="frames_checked", scene=scene_dir, im_id=int(im_id), value=1))
        frames_count += 1

    perf = dict(scene=scene_dir, seconds=round(time.time() - t0, 3), frames=frames_count)
    return metrics, issues, perf

# ------------------------------- main ------------------------------ #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True,
                    help="Корень BOP: внутри подкаталоги сцен (например, 000000, 000001, ...)")
    ap.add_argument("--out_dir", required=True, help="Куда писать отчёты/метрики")
    ap.add_argument("--thresholds", default=None, help="YAML/JSON с порогами (опционально)")
    ap.add_argument("--check_hashes", action="store_true", help="Подсчитать хэши rgb/mask/mask_visib (дольше)")
    ap.add_argument("--workers", type=int, default=1, help="Кол-во параллельных процессов (по сценам)")
    ap.add_argument("--progress", action="store_true", help="Показывать прогресс-бар по сценам")
    args = ap.parse_args()

    # пороги по умолчанию
    thresholds: Dict[str, Any] = {
        "camera": {"cx_cy_tol_px": 30.0},
        "masks": {"visib_in_full_ratio": 0.98, "visib_fract_tol": 0.03},
    }

    if args.thresholds and os.path.exists(args.thresholds):
        ext = os.path.splitext(args.thresholds)[1].lower()
        with open(args.thresholds, "r") as fh:
            if ext == ".json":
                thresholds.update(json.load(fh))
            elif ext in (".yml", ".yaml"):
                if yaml is None:
                    raise RuntimeError("PyYAML не установлен. Установи его или используй .json.")
                thresholds.update(yaml.safe_load(fh))
            else:
                raise ValueError(f"Неизвестное расширение порогов: {ext}")

    ensure_dir(args.out_dir)
    t0 = time.time()

    # собираем список сцен
    scenes = sorted(p for p in glob(os.path.join(args.dataset_root, "*")) if os.path.isdir(p))
    if not scenes:
        has_core = all(os.path.exists(os.path.join(args.dataset_root, f))
                       for f in ("scene_gt.json", "scene_camera.json", "scene_gt_info.json"))
        if has_core:
            scenes = [args.dataset_root]

    print(f"[SDQS] dataset_root: {args.dataset_root}")
    print(f"[SDQS] scenes found: {len(scenes)}; workers: {args.workers}")

    all_metrics: List[Dict[str, Any]] = []
    all_issues: List[Dict[str, Any]] = []
    perf_rows: List[Dict[str, Any]] = []

    # параллельная обработка сцен
    if args.workers > 1:
        # ограничим лишний параллелизм в OpenCV/NumPy
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(validate_scene, s, thresholds): s for s in scenes}
            iterator = as_completed(futures)
            if args.progress and tqdm is not None:
                iterator = tqdm(iterator, total=len(futures), desc="Validate scenes", unit="scene")
            for fut in iterator:
                s = futures[fut]
                try:
                    m, iss, perf = fut.result()
                except Exception as e:
                    m, iss, perf = [], [dict(level="Critical", scene=s, im_id="*", check="exception", msg=str(e))], dict(scene=s, seconds=0.0, frames=0)
                all_metrics.extend(m); all_issues.extend(iss); perf_rows.append(perf)
    else:
        iterator = scenes
        if args.progress and tqdm is not None:
            iterator = tqdm(scenes, desc="Validate scenes", unit="scene")
        for s in iterator:
            m, iss, perf = validate_scene(s, thresholds)
            all_metrics.extend(m); all_issues.extend(iss); perf_rows.append(perf)

    # (опц.) хэши (делаем в главном процессе, чтобы не раздувать IO)
    hash_rows: List[Dict[str, Any]] = []
    if args.check_hashes:
        files = []
        for sub in ("rgb", "mask", "mask_visib"):
            for ext in SUPPORTED_EXTS:
                files.extend(glob(os.path.join(args.dataset_root, "*", sub, f"*.{ext}")))
        iterator = files
        if args.progress and tqdm is not None:
            iterator = tqdm(files, desc="Hashing", unit="file")
        for fp in iterator:
            try:
                hash_rows.append(dict(file=fp, blake2b=blake_hash_file(fp)))
            except Exception as e:
                all_issues.append(dict(level="Major", scene="*", im_id="*", check="hash", msg=f"hash failed {fp}: {e}"))

    has_critical = any(i.get("level") == "Critical" for i in all_issues)

    # запись артефактов
    write_csv(os.path.join(args.out_dir, "metrics", "basic.csv"), all_metrics)
    write_csv(os.path.join(args.out_dir, "metrics", "perf.csv"),  perf_rows)
    write_csv(os.path.join(args.out_dir, "metrics", "hashes.csv"), hash_rows)

    issues_path = os.path.join(args.out_dir, "issues.csv")
    if all_issues:
        write_csv(issues_path, all_issues)
    else:
        write_empty_issues(issues_path)

    run = dict(
        dataset_root=args.dataset_root,
        out_dir=args.out_dir,
        thresholds=thresholds,
        scenes=len(scenes),
        issues=dict(
            total=len(all_issues),
            critical=sum(1 for i in all_issues if i["level"] == "Critical"),
            major=sum(1 for i in all_issues if i["level"] == "Major"),
            minor=sum(1 for i in all_issues if i["level"] == "Minor"),
        ),
        perf=dict(
            total_seconds=round(time.time() - t0, 3),
            scenes=len(scenes),
        ),
    )
    with open(os.path.join(args.out_dir, "run.json"), "w") as f:
        json.dump(run, f, indent=2, ensure_ascii=False)

    if has_critical:
        print("[SDQS] FAIL: есть критические ошибки. См. issues.csv и run.json.")
        sys.exit(2)
    print("[SDQS] PASS: критических ошибок нет.")
    sys.exit(0)

if __name__ == "__main__":
    main()
