import argparse
import json
import multiprocessing
import os
import shutil
import subprocess
import tempfile
import traceback

from blendforge.host.FiletoDict import Config


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help="Path to config file", type=str)
args = parser.parse_args()


def env_generic(index_device, cfg):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(index_device)

    process_on_gpu = []

    for process in range(cfg.parallel_process_on_one_gpu):
        cfg._add_new_item(index_device=str(index_device))
        cfg._add_new_item(process_id=str(process))
        temp_dir_rgb = tempfile.mkdtemp(prefix=f"blender_rgb_id{cfg.index_device}ip{cfg.process_id}")
        temp_dir_segmap = tempfile.mkdtemp(prefix=f"blender_segmap_id{cfg.index_device}ip{cfg.process_id}")
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

        cmd = (
            "blenderproc run scenarios/bop/main_stereo_multidepth.py "
            f"--config_file {path_to_cfg}"
        )
        process_on_gpu.append(multiprocessing.Process(target=run_proc, args=(cmd, env, temp_dir_rgb, temp_dir_segmap, cfg.runs_process)))

    for process in process_on_gpu:
        process.start()


def check_and_remove_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"Директория {dir_path} была удалена")
    else:
        print(f"Директория {dir_path} была удалена ранее")


def run_proc(cmd, env, temp_dir_rgb, temp_dir_segmap, runs_process):
    try:
        for _ in range(int(runs_process)):
            subprocess.run(cmd, env=env, shell=True, check=True)
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    finally:
        check_and_remove_dir(temp_dir_rgb)
        check_and_remove_dir(temp_dir_segmap)


def create_models_folder(dataset_parent_path: str, output_dir: str, dataset: str):
    models_dir = os.path.join(dataset_parent_path, dataset, "models")
    dataset_dir = os.path.join(output_dir, dataset)
    target_models_dir = os.path.join(dataset_dir, "models")

    os.makedirs(dataset_dir, exist_ok=True)
    if os.path.exists(models_dir) and not os.path.exists(target_models_dir):
        shutil.copytree(models_dir, target_models_dir)


if __name__ == "__main__":
    cfg = None
    cfg_dir = None
    try:
        cfg = Config(args.config_path)
        print(cfg._data)

        os.makedirs(cfg.output_dir, exist_ok=True)
        cfg_dir = os.path.join(cfg.output_dir, "configs")
        os.makedirs(cfg_dir, exist_ok=True)

        create_models_folder(cfg.dataset_parent_path, cfg.output_dir, cfg.bop_dataset_name)

        process_on_pc = []
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
        if cfg is not None and cfg.save_config is False and cfg_dir is not None:
            if os.path.exists(cfg_dir):
                shutil.rmtree(cfg_dir)
