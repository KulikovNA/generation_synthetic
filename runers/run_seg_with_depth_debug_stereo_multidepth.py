import argparse
import json
import multiprocessing
import os
import random
import signal
import shutil
import subprocess
import tempfile
import time
import traceback

from blendforge.host.FiletoDict import Config


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help="Path to config file", type=str)
args = parser.parse_args()


TMP_ROOT_PREFIX = "gsynth_bproc_tmp_"
STALE_TMP_ROOT_MIN_AGE_SEC = int(os.environ.get("GSYNTH_STALE_TMP_ROOT_MIN_AGE_SEC", str(6 * 3600)))


def _signal_to_keyboard_interrupt(signum, _frame):
    raise KeyboardInterrupt(f"Interrupted by signal {signum}")


def _install_interrupt_handlers():
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _signal_to_keyboard_interrupt)
        except ValueError:
            pass


def _cleanup_stale_runner_tmp_roots():
    tmp_root = tempfile.gettempdir()
    now = time.time()
    try:
        entries = list(os.scandir(tmp_root))
    except OSError:
        return

    for entry in entries:
        if not entry.is_dir(follow_symlinks=False):
            continue
        if not entry.name.startswith(TMP_ROOT_PREFIX):
            continue
        try:
            stat = entry.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        age_sec = now - max(float(stat.st_mtime), float(stat.st_ctime))
        if age_sec < STALE_TMP_ROOT_MIN_AGE_SEC:
            continue
        try:
            shutil.rmtree(entry.path)
            print(f"Удалена старая временная директория раннера: {entry.path}")
        except OSError:
            pass


def _terminate_process_group(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        except ProcessLookupError:
            pass


def _terminate_mp_processes(processes):
    for proc in processes:
        if getattr(proc, "_popen", None) is not None and proc.is_alive():
            proc.terminate()
    for proc in processes:
        if getattr(proc, "_popen", None) is not None:
            proc.join(timeout=10)


def _normalize_retryable_exit_codes(value) -> set[int]:
    if value is None:
        raw_values = [245]
    elif isinstance(value, (int, float, str)):
        raw_values = [value]
    else:
        raw_values = list(value)

    normalized = set()
    for raw in raw_values:
        code = int(raw)
        normalized.add(code)
        if code < 0:
            normalized.add((256 + code) % 256)
    return normalized


def _is_retryable_returncode(returncode: int, retryable_codes: set[int]) -> bool:
    if returncode in retryable_codes:
        return True
    if returncode < 0 and ((256 + returncode) % 256) in retryable_codes:
        return True
    return False


def _compute_retry_delay_sec(
    attempt_idx: int,
    *,
    base_delay_sec: float,
    multiplier: float,
    jitter_sec: float,
) -> float:
    delay = max(0.0, float(base_delay_sec)) * (max(1.0, float(multiplier)) ** max(0, attempt_idx - 1))
    if jitter_sec > 0:
        delay += random.uniform(0.0, float(jitter_sec))
    return float(delay)


def _describe_returncode(returncode: int) -> str:
    if returncode == 0:
        return "exit code 0"
    if returncode < 0:
        return f"signal {-returncode} ({signal.Signals(-returncode).name})"
    wrapped_negative = returncode - 256
    if wrapped_negative < 0:
        sig_num = -wrapped_negative
        try:
            sig_name = signal.Signals(sig_num).name
        except ValueError:
            sig_name = "UNKNOWN"
        return f"exit code {returncode} (likely wrapped {wrapped_negative} -> signal {sig_num} / {sig_name})"
    return f"exit code {returncode}"


def env_generic(index_device, cfg):
    _install_interrupt_handlers()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(index_device)
    retry_total_attempts = max(1, int(getattr(cfg, "blenderproc_retry_total_attempts", 3)))
    retry_base_delay_sec = float(getattr(cfg, "blenderproc_retry_backoff_sec", 15.0))
    retry_multiplier = float(getattr(cfg, "blenderproc_retry_backoff_multiplier", 1.8))
    retry_jitter_sec = float(getattr(cfg, "blenderproc_retry_backoff_jitter_sec", 5.0))
    retryable_exit_codes = _normalize_retryable_exit_codes(
        getattr(cfg, "blenderproc_retryable_exit_codes", [245])
    )

    process_on_gpu = []
    owned_tmp_roots = []
    for process in range(cfg.parallel_process_on_one_gpu):
        cfg._add_new_item(index_device=str(index_device))
        cfg._add_new_item(process_id=str(process))

        tmp_root = tempfile.mkdtemp(prefix=f"{TMP_ROOT_PREFIX}d{cfg.index_device}p{cfg.process_id}_")
        owned_tmp_roots.append(tmp_root)
        temp_dir_rgb = tempfile.mkdtemp(
            prefix=f"blender_rgb_id{cfg.index_device}ip{cfg.process_id}_",
            dir=tmp_root,
        )
        temp_dir_segmap = tempfile.mkdtemp(
            prefix=f"blender_segmap_id{cfg.index_device}ip{cfg.process_id}_",
            dir=tmp_root,
        )
        cfg._add_new_item(temp_dir_rgb=temp_dir_rgb)
        cfg._add_new_item(temp_dir_segmap=temp_dir_segmap)
        cfg._add_new_item(file_prefix_rgb=f"rgb_id{cfg.index_device}ip{cfg.process_id}_")
        cfg._add_new_item(file_prefix_segmap=f"segmap_id{cfg.index_device}ip{cfg.process_id}_")

        print("__INPUT DATA__")
        print("#########################################")
        print(f"From device_id = {index_device}, process_id = {process}")
        print("#########################################")
        print(cfg._data)

        path_to_cfg = os.path.join(cfg.output_dir, "configs", f"cfg_dev{index_device:03d}_proc{process:03d}.json")
        with open(path_to_cfg, "w", encoding="utf-8") as file:
            file.write(json.dumps(cfg._data, indent=4, ensure_ascii=False))

        proc_env = env.copy()
        proc_env["TMPDIR"] = tmp_root
        proc_env["TMP"] = tmp_root
        proc_env["TEMP"] = tmp_root

        cmd = [
            "blenderproc",
            "run",
            "--temp-dir",
            tmp_root,
            "scenarios/seg_with_depth/debug_stereo_multidepth.py",
            "--config_file",
            path_to_cfg,
        ]
        process_on_gpu.append(
            multiprocessing.Process(
                target=run_proc,
                args=(
                    cmd,
                    proc_env,
                    tmp_root,
                    cfg.runs_process,
                    retry_total_attempts,
                    retry_base_delay_sec,
                    retry_multiplier,
                    retry_jitter_sec,
                    retryable_exit_codes,
                    f"device={cfg.index_device} process={cfg.process_id}",
                ),
            )
        )

    try:
        for process in process_on_gpu:
            process.start()
        for process in process_on_gpu:
            process.join()
    finally:
        _terminate_mp_processes(process_on_gpu)
        for tmp_root in owned_tmp_roots:
            check_and_remove_dir(tmp_root)


def check_and_remove_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"Директория {dir_path} была удалена")
    else:
        print(f"Директория {dir_path} была удалена ранее")


def run_proc(
    cmd,
    env,
    tmp_root,
    runs_process,
    retry_total_attempts,
    retry_base_delay_sec,
    retry_multiplier,
    retry_jitter_sec,
    retryable_exit_codes,
    worker_label,
):
    _install_interrupt_handlers()
    active_proc = None
    try:
        for run_idx in range(int(runs_process)):
            attempt_idx = 1
            while True:
                active_proc = subprocess.Popen(
                    cmd,
                    env=env,
                    start_new_session=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert active_proc.stdout is not None
                try:
                    for line in active_proc.stdout:
                        print(line, end="")
                    retcode = active_proc.wait()
                finally:
                    active_proc = None

                if retcode == 0:
                    break

                print(f"[CrashExit] {worker_label} -> {_describe_returncode(retcode)}")

                if attempt_idx >= int(retry_total_attempts) or not _is_retryable_returncode(
                    retcode, retryable_exit_codes
                ):
                    raise subprocess.CalledProcessError(retcode, cmd)

                delay_sec = _compute_retry_delay_sec(
                    attempt_idx,
                    base_delay_sec=retry_base_delay_sec,
                    multiplier=retry_multiplier,
                    jitter_sec=retry_jitter_sec,
                )
                print(
                    "[Retry] "
                    f"{worker_label} | run={run_idx + 1}/{int(runs_process)} | "
                    f"attempt={attempt_idx}/{int(retry_total_attempts)} failed with exit code {retcode}. "
                    f"Sleeping {delay_sec:.1f}s before retry."
                )
                time.sleep(delay_sec)
                attempt_idx += 1
    except KeyboardInterrupt:
        print("Процесс был прерван, останавливаю blenderproc и очищаю временные директории")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    finally:
        _terminate_process_group(active_proc)
        check_and_remove_dir(tmp_root)


if __name__ == "__main__":
    cfg = None
    cfg_dir = None
    process_on_pc = []
    try:
        _install_interrupt_handlers()
        _cleanup_stale_runner_tmp_roots()
        cfg = Config(args.config_path)
        print(cfg._data)

        os.makedirs(cfg.output_dir, exist_ok=True)
        cfg_dir = os.path.join(cfg.output_dir, "configs")
        os.makedirs(cfg_dir, exist_ok=True)

        for index in range(cfg.num_gpus):
            process_on_pc.append(multiprocessing.Process(target=env_generic, args=(index, cfg)))

        for proc in process_on_pc:
            proc.start()

        for proc in process_on_pc:
            proc.join()

    except Exception as e:
        print(f"Произошла ошибка: {e}")
        traceback.print_exc()

    finally:
        _terminate_mp_processes(process_on_pc)
        if cfg is not None and cfg.save_config is False and cfg_dir is not None:
            if os.path.exists(cfg_dir):
                shutil.rmtree(cfg_dir)
