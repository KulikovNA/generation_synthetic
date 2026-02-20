mode = "deformed_bop_seg"
# Минимальный конфиг для main_rgbd_coco.py
camera_path = "/home/nikita/data_generator/generation_dataset/generation_synthetic/prepared/differBig/d435_intrinsics_extrinsics.json"
dataset_parent_path = "/home/nikita/data_generator/generation_dataset/generation_synthetic/prepared"      # где лежит BOP датасет
bop_dataset_name = "differBig"                         # имя папки BOP внутри dataset_parent_path
num_gpus = 1
parallel_process_on_one_gpu = 1
poses_cam = 10                                     # сколько поз камеры сгенерировать (int)
probability_drop = 0.5                             # вероятность режима drop (float 0..1)

max_amount_of_samples = 50                        # Cycles samples (int) или null для random

cc_textures = dict(cc_textures_path = "/home/nikita/data_generator/generation_dataset/generation_synthetic/resources/textures_2k_plate")   # путь к CCTextures (폴ки материалов)
