from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import signal
import shutil
import subprocess
import tempfile
import time
import traceback
from typing import Any, Dict, List, Mapping

from blendforge.host.FiletoDict import Config


BLENDER_SEED_MAX = 0x7FFFFFFF
POLL_INTERVAL_SEC = 0.5
TERMINATE_TIMEOUT_SEC = 8.0
KILL_TIMEOUT_SEC = 4.0
OUTPUT_VALIDATION_TIMEOUT_SEC = 60.0
OUTPUT_VALIDATION_POLL_SEC = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fragment_template_registration scene jobs")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config file")
    return parser.parse_args()


def signal_to_keyboard_interrupt(signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt(f"Interrupted by signal {signum}")


def install_interrupt_handlers() -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, signal_to_keyboard_interrupt)
        except ValueError:
            pass


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


def cleanup_scene_temp_dirs(scene_cfg_paths: Mapping[str, str]) -> None:
    for key in ["temp_dir_rgb", "temp_dir_segmap"]:
        path = scene_cfg_paths.get(key)
        if path and os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)


def scene_dir_for_job(job: Mapping[str, Any]) -> str:
    return os.path.join(
        str(job["base_cfg_data"]["output_dir"]),
        str(job["split"]),
        f"scene_{int(job['scene_id']):06d}",
    )


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def scene_output_complete(job: Mapping[str, Any]) -> tuple[bool, str]:
    scene_dir = scene_dir_for_job(job)
    if not os.path.isdir(scene_dir):
        return False, f"missing scene dir: {scene_dir}"

    required_scene_files = [
        "gt_annotations.json",
        "coco_annotations.json",
        "scene_meta.json",
        "camera_info.json",
        os.path.join("fragments", "fragment_annotations.json"),
    ]
    for rel_path in required_scene_files:
        path = os.path.join(scene_dir, rel_path)
        if not os.path.isfile(path):
            return False, f"missing file: {path}"

    try:
        gt = load_json(os.path.join(scene_dir, "gt_annotations.json"))
    except Exception as exc:
        return False, f"cannot read gt_annotations.json: {exc}"

    frames = list(gt.get("frames", []))
    expected_frames = int(job["split_cfg"].get("num_frames_per_scene", 0))
    if expected_frames > 0 and len(frames) != expected_frames:
        return False, f"expected {expected_frames} frames, got {len(frames)}"

    write_flags = dict(job["base_cfg_data"].get("write_flags", {}) or {})
    frame_path_keys = []
    if write_flags.get("write_rgb", True):
        frame_path_keys.append("image")
    if write_flags.get("write_depth", True):
        frame_path_keys.append("depth")
    if write_flags.get("write_instance_masks", True):
        frame_path_keys.append("instance_mask")
    if write_flags.get("write_surface_masks", True):
        frame_path_keys.append("surface_mask")
    if write_flags.get("write_visible_points", True):
        frame_path_keys.append("visible_points")

    for frame in frames:
        frame_id = int(frame.get("frame_id", -1))
        for key in frame_path_keys:
            rel_path = frame.get(key)
            if not rel_path:
                return False, f"frame {frame_id} missing path key: {key}"
            path = os.path.join(scene_dir, rel_path)
            if not os.path.isfile(path):
                return False, f"frame {frame_id} missing file: {path}"

    return True, "complete"


def wait_for_scene_output_complete(
    job: Mapping[str, Any],
    *,
    timeout_sec: float = OUTPUT_VALIDATION_TIMEOUT_SEC,
    poll_sec: float = OUTPUT_VALIDATION_POLL_SEC,
) -> tuple[bool, str]:
    started = time.monotonic()
    last_reason = "not checked"
    while True:
        complete, reason = scene_output_complete(job)
        if complete:
            return True, reason
        last_reason = reason
        if time.monotonic() - started >= timeout_sec:
            return False, last_reason
        time.sleep(poll_sec)


def terminate_process_group(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=TERMINATE_TIMEOUT_SEC)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=KILL_TIMEOUT_SEC)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            pass


def terminate_active_jobs(active_jobs: List[Dict[str, Any]]) -> None:
    for active in active_jobs:
        proc = active.get("proc")
        if proc is not None:
            terminate_process_group(proc)
    for active in active_jobs:
        cleanup_scene_temp_dirs(active.get("scene_cfg_paths", {}))


def start_scene_job(job: Dict[str, Any]) -> Dict[str, Any]:
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

    print(f"[fragment_template_registration] Start scene_{int(job['scene_id']):06d}")
    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            start_new_session=True,
        )
    except BaseException:
        cleanup_scene_temp_dirs(scene_cfg_paths)
        raise
    return {
        "job": job,
        "proc": proc,
        "scene_cfg_paths": scene_cfg_paths,
    }


def run_jobs(jobs: List[Dict[str, Any]], num_workers: int) -> List[int]:
    pending = list(jobs)
    active: List[Dict[str, Any]] = []
    completed: List[int] = []

    try:
        while pending or active:
            while pending and len(active) < num_workers:
                active.append(start_scene_job(pending.pop(0)))

            for active_job in list(active):
                proc = active_job["proc"]
                returncode = proc.poll()
                if returncode is None:
                    continue

                active.remove(active_job)
                scene_id = int(active_job["job"]["scene_id"])
                complete, reason = wait_for_scene_output_complete(active_job["job"])
                cleanup_scene_temp_dirs(active_job["scene_cfg_paths"])

                if returncode != 0:
                    if complete:
                        completed.append(scene_id)
                        print(
                            "[fragment_template_registration] "
                            f"scene_{scene_id:06d} returned code {returncode}, "
                            "but output validation passed; treating as completed"
                        )
                        continue

                    print(
                        "[fragment_template_registration] "
                        f"scene_{scene_id:06d} failed with return code {returncode}; "
                        f"output validation failed ({reason}); stopping active jobs"
                    )
                    terminate_active_jobs(active)
                    raise subprocess.CalledProcessError(returncode, proc.args)

                if not complete:
                    print(
                        "[fragment_template_registration] "
                        f"scene_{scene_id:06d} returned code 0, but output validation failed "
                        f"after {OUTPUT_VALIDATION_TIMEOUT_SEC:.0f}s ({reason}); stopping active jobs"
                    )
                    terminate_active_jobs(active)
                    raise RuntimeError(f"scene_{scene_id:06d} output is incomplete: {reason}")

                completed.append(scene_id)
                print(f"[fragment_template_registration] Done scene_{scene_id:06d}")

            if pending or active:
                time.sleep(POLL_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("[fragment_template_registration] Interrupted, stopping active blenderproc jobs...")
        terminate_active_jobs(active)
        raise

    return completed


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

    run_jobs(jobs, num_workers=num_workers)


if __name__ == "__main__":
    try:
        install_interrupt_handlers()
        main()
    except KeyboardInterrupt as exc:
        print(f"[fragment_template_registration] Interrupted: {exc}")
        raise SystemExit(130)
    except Exception as exc:
        print(f"Error: {exc}")
        traceback.print_exc()
        raise
