# This module makes it easy to write user-friendly command-line interfaces.
import argparse
# This module provides a portable way of using operating system dependent functionality
import os
# The subprocess module allows you to spawn new processes, connect to their input/output/error pipes, and obtain their return codes
import subprocess
# multiprocessing is a package that supports spawning processes using an API similar to the threading module
import multiprocessing

import json
import tempfile
import shutil

import traceback

from blendforge.host.FiletoDict import Config

# parse command line args 
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', 
                    help='Path to config file',
                    type=str)
args = parser.parse_args()

def env_generic(index_device, 
                cfg):
    
    # видеокарта 
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(index_device)
    
    process_on_gpu = []

    # для каждого рабочего процесса в девайсе
    for process in range(cfg.parallel_process_on_one_gpu):
         
        # добавляем в конфигуратор префиксы 
        cfg._add_new_item(index_device = str(index_device))
        cfg._add_new_item(process_id = str(process))

        # создание временной директории для хранения rgbmap
        temp_dir_rgb = tempfile.mkdtemp(prefix = f'blender_rgb_id{cfg.index_device}ip{cfg.process_id}')
        # создание временной директории для хранения segmap
        temp_dir_segmap = tempfile.mkdtemp(prefix = f'blender_segmap_id{cfg.index_device}ip{cfg.process_id}')
        
        # добавляем в конфигуратор путь к временным директориям
        cfg._add_new_item(temp_dir_rgb = temp_dir_rgb)
        cfg._add_new_item(temp_dir_segmap = temp_dir_segmap)

        print("__INPUT DATA__")
        print("#########################################")
        print(f"From device_id = {index_device}, processe_id = {process}")
        print("#########################################")
        print(cfg._data)
        

        path_to_cfg = os.path.join(cfg.output_dir, "configs", f"cfg_dev{index_device:03d}_proc{process:03d}.json") 
        with open(path_to_cfg, 'w') as file:  
            data_as_str = json.dumps(cfg._data, indent=4)  
            file.write(data_as_str)

        """# создаем файл с параметрами cfg
        with tempfile.NamedTemporaryFile(suffix=".json", mode='wb', delete=False) as tmp:
            data_as_str = json.dumps(cfg._data)
            tmp.write(data_as_str.encode())
            tmp_path = tmp.name """
        # передаем сценарию ссылку на cfg
        
        cmd = f"blenderproc run scenarios/{cfg.mode}/main.py " \
              f"--config_file {path_to_cfg} " 
        
        process_on_gpu.append(multiprocessing.Process(target=run_proc, args=(cmd, env, temp_dir_rgb, temp_dir_segmap, cfg.runs_process)))
    
    for process in process_on_gpu:
        process.start()

def check_and_remove_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"Директория {dir_path} была удалена")
    else:
        print(f"Директория {dir_path} была удалена ранее")

# Функция для запуска процесса
def run_proc(cmd, env, temp_dir_rgb, temp_dir_segmap, runs_process):
    try:
        for i in range(runs_process):
            # Запускаем процесс
            subprocess.run(cmd, env=env, shell=True, check=True)
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    finally:
        # Удаление временных директорий
        check_and_remove_dir(temp_dir_rgb)
        check_and_remove_dir(temp_dir_segmap)

# launch runners assigned to each GPU
if __name__ == '__main__':
    try:
        # данные из config file 
        cfg = Config(args.config_path)
        print(cfg._data)
        # создаем директорию 
        if not os.path.exists(cfg.output_dir):
            os.makedirs(cfg.output_dir)
        # создаем директорию для файлов конфигурации 
        cfg_dir = os.path.join(cfg.output_dir, 'configs')
        if not os.path.exists(cfg_dir):
            os.makedirs(cfg_dir)
        
        process_on_pc = []

        for index in range(cfg.num_gpus):
            process_on_pc.append(multiprocessing.Process(target=env_generic, args=(index, cfg,)))
        
        for proc in process_on_pc:     
            proc.start()
        
        for proc in process_on_pc:
            proc.join()

    except Exception as e:
        print(f"Произошла ошибка: {e}")
        traceback.print_exc()
    
    finally:
        # Удаление директории с конфигураторами
        if cfg.save_config is False:
            if os.path.exists(cfg_dir):
                shutil.rmtree(cfg_dir)
