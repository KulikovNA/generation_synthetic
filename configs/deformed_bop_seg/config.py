mode =  "deformed_bop_seg"

# количество изображений: 
# parallel_process_on_one_gpu * num_gpus * max_poses_cam * max_runs_one_fracture * runs_process 
# ресурсные ограничения
parallel_process_on_one_gpu = 1
num_gpus = 1
# параметры сценария
poses_cam = 15
runs_one_fracture = 5 
# сколько раз вызовем сценарий
runs_process = 14 
dataset_type = "test"

dataset_parent_path = "/home/nikita/data_generator/generation_dataset/syntetic_datasets/prepared"
bop_dataset_name = "differBig"
bop_toolkit_path = "/home/nikita/data_generator/generation_dataset/bop_toolkit"
# путь к текстурам 
cc_textures = dict(
    cc_textures_path = "/home/nikita/data_generator/generation_dataset/syntetic_datasets/resources/textures_2k_plate",
    cc_textures_object_path = "/home/nikita/data_generator/generation_dataset/syntetic_datasets/resources/textures_1k_modules"
    )
output_dir = "/home/nikita/data_generator/test3/"

save_config = True
max_amount_of_samples = None
id_target_obj = 3

probability_drop = 0.70
