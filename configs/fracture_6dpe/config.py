mode =  "fracture_6dpe"

# количество изображений: 
# parallel_process_on_one_gpu * num_gpus * max_poses_cam * max_runs_one_fracture * runs_process 
# ресурсные ограничения
parallel_process_on_one_gpu = 1
num_gpus = 1
# параметры сценария
poses_cam = 5
runs_one_fracture = 1 
# сколько раз вызовем сценарий
runs_process = 14 
dataset_type = "train"

dataset_parent_path = "prepared"
bop_dataset_name = "differSmall"
# путь к текстурам 
cc_textures = dict(
    cc_textures_path = "resources/textures_2k_plate",
    cc_textures_object_path = "resources/textures_1k_modules"
    )
output_dir = "output/fract_6dpe"

save_config = True
max_amount_of_samples = None
id_target_obj = 3

probability_drop = 0.70
