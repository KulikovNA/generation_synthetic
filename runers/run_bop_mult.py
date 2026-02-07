# This module makes it easy to write user-friendly command-line interfaces.
import argparse
# This module provides a portable way of using operating system dependent functionality
import os
# The subprocess module allows you to spawn new processes, connect to their input/output/error pipes, and obtain their return codes
import subprocess
# multiprocessing is a package that supports spawning processes using an API similar to the threading module
import multiprocessing
import time
import tempfile

import traceback
import shutil
from blendforge.host.FiletoDict import Config
import json


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', 
                    help='Path to config file',
                    type=str)
args = parser.parse_args()


def env_generic(index_device, cfg):
    # видеокарта 
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(index_device)

    process_on_gpu = []

    for process in range(cfg.parallel_process_on_one_gpu):
        # создание временной директории для хранения rgbmap
        temp_dir_rgb = tempfile.mkdtemp(prefix = f'blender_rgb_id{index_device}ip{process}')
        # создание временной директории для хранения segmap
        temp_dir_segmap = tempfile.mkdtemp(prefix = f'blender_segmap_id{index_device}ip{process}')
        
        # добавляем в конфигуратор путь к временным директориям
        cfg._add_new_item(temp_dir_rgb = temp_dir_rgb)
        cfg._add_new_item(temp_dir_segmap = temp_dir_segmap)

        cfg._add_new_item(file_prefix_rgb = f'rgb_id{index_device}ip{process}_')
        cfg._add_new_item(file_prefix_segmap = f'segmap_id{index_device}ip{process}_')

        print("__INPUT DATA__")
        print("#########################################")
        print(cfg._data)
        print("#########################################")

        path_to_cfg = os.path.join(cfg.output_dir, "configs", f"cfg_dev{index_device:03d}_proc{process:03d}.json") 
        with open(path_to_cfg, 'w') as file:  
            data_as_str = json.dumps(cfg._data, indent=4)  
            file.write(data_as_str)
        
        # передаем сценарию ссылку на cfg
        cmd = f"blenderproc run scenarios/{cfg.mode}/main.py " \
              f"--config_file {path_to_cfg} "
        
        process_on_gpu.append(multiprocessing.Process(target=run_proc, args=(cmd, env, temp_dir_rgb, temp_dir_segmap)))
    
    for process in process_on_gpu:
        process.start()

def check_and_remove_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"Директория {dir_path} была удалена")
    else:
        print(f"Директория {dir_path} была удалена ранее")

# Функция для запуска процесса
def run_proc(cmd, env, temp_dir_rgb, temp_dir_segmap):
    try:
        # Запускаем процесс
        subprocess.run(cmd, env=env, shell=True, check=True)
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    finally:
        # Удаление временных директорий
        check_and_remove_dir(temp_dir_rgb)
        check_and_remove_dir(temp_dir_segmap)


def create_folder_models(bop_parent_path: str, output_dir: str, dataset: str = ""):
        bop_dir = os.path.join(bop_parent_path, dataset)
        models_dir = os.path.join(bop_dir, 'models')
        
        dataset_dir = os.path.join(output_dir, dataset)
        target_models_dir = os.path.join(dataset_dir, 'models')

        if not os.path.exists(target_models_dir):
            os.makedirs(target_models_dir)

        os.system(f'cp -r {models_dir} {dataset_dir}')


# launch runners assigned to each GPU
if __name__ == '__main__':
    try: 
        # данные из config file 
        cfg = Config(args.config_path)
        
        # создаем директорию 
        if not os.path.exists(cfg.output_dir):
            os.makedirs(cfg.output_dir)
        # создаем директорию для файлов конфигурации 
        cfg_dir = os.path.join(cfg.output_dir, 'configs')
        if not os.path.exists(cfg_dir):
            os.makedirs(cfg_dir)

        create_folder_models(cfg.bop_parent_path, cfg.output_dir, cfg.bop_dataset_name)
        runs = cfg.num_chunk*1000/(cfg.parallel_process_on_one_gpu*cfg.num_gpus*25) # 20 количество снимков на позу 
        cfg._add_new_item(runs = runs)

        process_on_one_gpu = []
        
        for index in range(cfg.num_gpus):
            process_on_one_gpu.append(multiprocessing.Process(target=env_generic, args=(index, cfg,)))

        for proc in process_on_one_gpu:     
                proc.start()
            
        for proc in process_on_one_gpu:
            proc.join()

    except Exception as e:
        print(f"Произошла ошибка: {e}")
        traceback.print_exc()
    
    finally:
        # Удаление директории с конфигураторами
        if cfg.save_config is False:
            if os.path.exists(cfg_dir):
                shutil.rmtree(cfg_dir)
    