mode = "bop_lol"
split = "train"           # "train" | "val" | "test"

# ==== ПУТИ ====
dataset_parent_path = "/home/nikita/data_generator/generation_dataset/syntetic_datasets/prepared"
bop_dataset_name    = "differBig"
bop_toolkit_path    = "/home/nikita/data_generator/generation_dataset/bop_toolkit"

cc_textures = dict(
    cc_textures_path = "/home/nikita/data_generator/generation_dataset/syntetic_datasets/resources/textures_2k_plate",
    cc_textures_object_path = "/home/nikita/data_generator/generation_dataset/syntetic_datasets/resources/textures_1k_modules"
)

# Корень датасета LOL-стиля (внутри скрипт создаст <split>/input, <split>/target, pairs.txt, meta.jsonl)
output_dir_lol = "/home/nikita/data_generator/lol_data/"

# ==== ПАРАМЕТРЫ РЕНДЕРА / СЦЕН ====
poses_cam = 15      # поз на сцену
runs      = 2      # количество сцен 

# диап. затемнения входа (EV для input); случайное значение берётся из диапазона
ev_low_range = [-4.0, -2.0]

# max samples на кадр (None => случайно 300..1000 внутри скрипта)
max_amount_of_samples = None

# Вероятность «свалки» (drop-посадка объектов в кучу)
probability_drop = 0.70

# ==== ПРОЧЕЕ / ОРКЕСТРАЦИЯ ====
save_config = True
num_gpus = 1
parallel_process_on_one_gpu = 2
index_device = 0
process_id = 0

ev_mode = "camera"           # или "light"
jpeg_input_quality = 85      # опционально, иначе PNG
color_view_transform = "Filmic"  # или "Standard"

