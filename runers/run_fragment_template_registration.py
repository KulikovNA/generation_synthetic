from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import traceback
from typing import Any, Dict, Mapping

from blendforge.host.FiletoDict import Config


BLENDER_SEED_MAX = 0x7FFFFFFF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fragment_template_registration scene jobs")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config file")
    return parser.parse_args()


def to_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_plain(v) for v in value]
    if isinstance(value, tuple):
        return [to_plain(v) for v in value]
    return value


def get_split_cfg(cfg_data: Dict[str, Any], split: str) -> Dict[str, Any]:
    splits = cfg_data.get("splits", {}) or {}
    split_cfg = dict(splits.get(split, {}) or {})
    split_cfg.setdefault("num_scenes", cfg_data.get("num_scenes", 1))
    split_cfg.setdefault("num_frames_per_scene", cfg_data.get("num_frames_per_scene", 1))
    split_cfg.setdefault("num_scene_workers", cfg_data.get("num_scene_workers", 1))
    return split_cfg


def seed_value_is_random(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"none", "null", "random", "auto"}:
        return True
    return False


def random_seed_for_blender() -> int:
    return int(secrets.randbelow(BLENDER_SEED_MAX + 1))


def list_existing_scene_ids(split_dir: str) -> list[int]:
    if not os.path.isdir(split_dir):
        return []
    scene_ids = []
    for name in os.listdir(split_dir):
        match = re.fullmatch(r"scene_(\d{6})", name)
        if match is not None and os.path.isdir(os.path.join(split_dir, name)):
            scene_ids.append(int(match.group(1)))
    return sorted(scene_ids)


def choose_first_scene_id(cfg_data: Dict[str, Any], split: str) -> int:
    requested_offset = int(cfg_data.get("scene_id_offset", 0))
    if bool(cfg_data.get("overwrite_scene", False)):
        return requested_offset

    split_dir = os.path.join(cfg_data["output_dir"], split)
    existing = list_existing_scene_ids(split_dir)
    if not existing:
        return requested_offset
    return max(requested_offset, existing[-1] + 1)


def write_scene_config(base_cfg_data: Dict[str, Any], scene_id: int, split: str, split_cfg: Dict[str, Any], worker_id: int) -> Dict[str, Any]:
    scene_cfg = json.loads(json.dumps(base_cfg_data))
    scene_cfg["split"] = split
    scene_cfg["dataset_type"] = split
    scene_cfg["scene_id"] = int(scene_id)
    scene_cfg["num_frames_per_scene"] = int(split_cfg["num_frames_per_scene"])
    scene_cfg["process_id"] = str(worker_id)
    scene_cfg["temp_dir_rgb"] = tempfile.mkdtemp(prefix=f"fragment_rgb_scene{scene_id:06d}_")
    scene_cfg["temp_dir_segmap"] = tempfile.mkdtemp(prefix=f"fragment_seg_scene{scene_id:06d}_")
    if seed_value_is_random(scene_cfg.get("seed", 13)):
        scene_cfg["resolved_seed"] = random_seed_for_blender()
        scene_cfg["seed_is_random"] = True
    else:
        scene_cfg.pop("resolved_seed", None)
        scene_cfg["seed_is_random"] = False

    cfg_dir = os.path.join(scene_cfg["output_dir"], "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    scene_cfg_path = os.path.join(cfg_dir, f"fragment_registration_{split}_scene{scene_id:06d}.json")
    with open(scene_cfg_path, "w", encoding="utf-8") as f:
        json.dump(scene_cfg, f, indent=2, ensure_ascii=False)

    return {
        "config_path": scene_cfg_path,
        "temp_dir_rgb": scene_cfg["temp_dir_rgb"],
        "temp_dir_segmap": scene_cfg["temp_dir_segmap"],
    }


def run_scene_job(job: Dict[str, Any]) -> int:
    env = os.environ.copy()
    env.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

    scene_cfg_paths = write_scene_config(
        base_cfg_data=job["base_cfg_data"],
        scene_id=int(job["scene_id"]),
        split=str(job["split"]),
        split_cfg=job["split_cfg"],
        worker_id=int(job["worker_id"]),
    )

    cmd = [
        "blenderproc",
        "run",
        "scenarios/fragment_template_registration/main.py",
        "--config_file",
        scene_cfg_paths["config_path"],
    ]

    try:
        print(f"[fragment_template_registration] Start scene_{int(job['scene_id']):06d}")
        subprocess.run(cmd, env=env, check=True)
        print(f"[fragment_template_registration] Done scene_{int(job['scene_id']):06d}")
        return int(job["scene_id"])
    finally:
        for key in ["temp_dir_rgb", "temp_dir_segmap"]:
            path = scene_cfg_paths.get(key)
            if path and os.path.exists(path):
                shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    cfg = Config(args.config_path)
    cfg_data = to_plain(cfg._data)

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "configs"), exist_ok=True)

    split = str(cfg_data.get("split", "train"))
    split_cfg = get_split_cfg(cfg_data, split)
    num_scenes = int(split_cfg["num_scenes"])
    num_workers = max(1, int(split_cfg["num_scene_workers"]))
    first_scene_id = choose_first_scene_id(cfg_data, split)

    jobs = []
    for local_scene_idx in range(num_scenes):
        scene_id = first_scene_id + local_scene_idx
        worker_id = local_scene_idx % num_workers
        jobs.append(
            {
                "base_cfg_data": cfg_data,
                "split": split,
                "split_cfg": split_cfg,
                "scene_id": scene_id,
                "worker_id": worker_id,
            }
        )

    print(
        "[fragment_template_registration] "
        f"split={split}, scenes={num_scenes}, first_scene_id={first_scene_id}, "
        f"frames_per_scene={split_cfg['num_frames_per_scene']}, workers={num_workers}"
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_scene_job, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        traceback.print_exc()
        raise
