mode =  "bop_seg"

# количество изображений: 
# parallel_process_on_one_gpu * num_gpus * max_poses_cam * max_runs_one_fracture * runs_process 
# ресурсные ограничения
parallel_process_on_one_gpu = 1
num_gpus = 1
#-- параметры сценария -- 
# количество новых положений камеры
poses_cam = 15
# количество итераций с одними и теми же объектами
runs = 5 
# сколько раз вызовем сценарий
runs_process = 1 
dataset_type = "test"

#dataset_parent_path = "/home/nikita/data_generator/generation_dataset/syntetic_datasets/prepared"
dataset_parent_path = "prepared"
bop_dataset_name = "differBig"
#bop_toolkit_path = "/home/nikita/data_generator/generation_dataset/bop_toolkit"
# путь к текстурам 
cc_textures = dict(
    cc_textures_path = "resources/textures_2k_plate",
    cc_textures_object_path = "resources/textures_1k_modules"
    )

output_dir = "output/bop_seg/"

save_config = True
max_amount_of_samples = 50

probability_drop = 0.70
