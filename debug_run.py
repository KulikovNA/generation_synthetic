import shutil
from utils.config.FiletoDict import Config
import tempfile
import os
import subprocess

def main(): 
    config_path = "scenarios/deformebel_segmentation/debug_config.json"
    cfg = Config(config_path)
    # создание временной директории для хранения rgbmap
    temp_dir_rgb = tempfile.mkdtemp(prefix = cfg.temp_dir_rgb)
    # создание временной директории для хранения segmap
    temp_dir_segmap = tempfile.mkdtemp(prefix = cfg.temp_dir_segmap)

    cmd = f"blenderproc run debug_run.py " \
              f"--config_file {config_path} " 
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(0)

    subprocess.run(cmd, env=env, shell=True, check=True)
    # Удаление временных директорий
    check_and_remove_dir(temp_dir_rgb)
    check_and_remove_dir(temp_dir_segmap)

def check_and_remove_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"Директория {dir_path} была удалена")
    else:
        print(f"Директория {dir_path} была удалена ранее")

if __name__ == "__main__":
    main()