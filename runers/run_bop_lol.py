#!/usr/bin/env python3
# run_lol.py — оркестратор BlenderProc-сценария (LOL-style)
import argparse, os, subprocess, multiprocessing, json, tempfile, shutil, traceback, math, uuid, sys
from datetime import datetime
from blendforge.host.FiletoDict import Config

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', required=True, type=str, help='Path to config file')
args = parser.parse_args()

# ---------- утилиты ----------
def _mkdir(p):
    os.makedirs(p, exist_ok=True); return p

def _write_json(path, data: dict):
    _mkdir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path

def _rm_tree(p):
    if p and os.path.exists(p):
        shutil.rmtree(p, ignore_errors=True)

def _cfg_set(cfg: Config, **kwargs):
    for k, v in kwargs.items():
        cfg._add_new_item(**{k: v})

def _derive_runs_process(cfg):
    # Совместимость: если есть cfg.runs_process — используем его.
    # Иначе берём cfg.runs (сколько сцен) и исполняем ровно столько раз.
    if hasattr(cfg, 'runs_process'):
        return int(cfg.runs_process)
    return int(getattr(cfg, 'runs', 1))

# ---------- рабочая функция ----------
def run_proc(cmd: str, env: dict, temp_dir_rgb: str, temp_dir_seg: str, times: int, log_dir: str):
    """Запускает сценарий times раз; чистит временные каталоги в конце."""
    try:
        for i in range(times):
            tag = f"[{datetime.utcnow().isoformat()}] iter={i+1}/{times}"
            print(tag, 'CMD:', cmd, flush=True)
            # можно добавить timeout=... при необходимости
            subprocess.run(cmd, env=env, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Return code={e.returncode}", flush=True)
    except Exception as e:
        print(f"[EXC] {e}", flush=True)
        traceback.print_exc()
    finally:
        _rm_tree(temp_dir_rgb)
        _rm_tree(temp_dir_seg)
        print(f"[CLEAN] removed temp dirs for PID={os.getpid()}", flush=True)

def env_generic(index_device: int, cfg_path: str):
    # ВАЖНО: читаем конфиг "локально" в каждом процессе, чтобы не было race на объекте cfg
    cfg = Config(cfg_path)

    # GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(index_device)
    
    # Базовые каталоги вывода
    out_root = getattr(cfg, 'output_dir', os.path.join(os.getcwd(), 'out'))
    cfg_dir  = _mkdir(os.path.join(out_root, 'configs'))
    logs_dir = _mkdir(os.path.join(out_root, 'logs'))

    # Сколько раз запускать сценарий на каждом worker-е
    times = _derive_runs_process(cfg)

    procs = []
    for process in range(int(cfg.parallel_process_on_one_gpu)):
        # Обновляем cfg префиксами
        _cfg_set(cfg, index_device=str(index_device))
        _cfg_set(cfg, process_id=str(process))

        # Временные каталоги (делаем уникальными через UUID)
        uid = uuid.uuid4().hex[:8]
        tdir_rgb = tempfile.mkdtemp(prefix=f'bproc_rgb_d{index_device}p{process}_{uid}_')
        tdir_seg = tempfile.mkdtemp(prefix=f'bproc_seg_d{index_device}p{process}_{uid}_')
        _cfg_set(cfg, temp_dir_rgb=tdir_rgb)
        _cfg_set(cfg, temp_dir_seg=tdir_seg)

        # Локальный конфиг для сценария
        per_cfg_path = os.path.join(cfg_dir, f"cfg_dev{index_device:03d}_proc{process:03d}.json")
        _write_json(per_cfg_path, cfg._data)

        # Путь к сценарию
        mode = getattr(cfg, 'mode', 'llie_lol')
        scenario_main = os.path.join('scenarios', mode, 'main.py')
        if not os.path.exists(scenario_main):
            print(f"[WARN] Scenario not found: {scenario_main}", flush=True)

        # Команда
        cmd = f"blenderproc run {scenario_main} --config_file {per_cfg_path}"

        # Небольшая диагностическая печать
        print("#########################################")
        print(f"GPU={index_device}  WORKER={process}")
        print("CFG:", per_cfg_path)
        print("TMP:", tdir_rgb, "|", tdir_seg)
        print("CMD:", cmd)
        print("#########################################", flush=True)

        p = multiprocessing.Process(
            target=run_proc,
            args=(cmd, env, tdir_rgb, tdir_seg, times, logs_dir)
        )
        procs.append(p)

    for p in procs:
        p.start()

# ---------- main ----------
if __name__ == '__main__':
    try:
        cfg = Config(args.config_path)
        print("[CFG ROOT]", cfg._data)

        # Гарантируем базовые каталоги
        out_root = getattr(cfg, 'output_dir', os.path.join(os.getcwd(), 'out'))
        _mkdir(out_root)
        _mkdir(os.path.join(out_root, 'configs'))

        # Запускаем по количеству GPU
        pool = []
        for gpu_idx in range(int(cfg.num_gpus)):
            pool.append(multiprocessing.Process(target=env_generic, args=(gpu_idx, args.config_path)))
        for p in pool: p.start()
        for p in pool: p.join()

    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()
        sys.exit(1)
