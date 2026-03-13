mode = "deformed_bop_seg"

# -------------------- base scene / camera --------------------

camera_path = "/home/nikita/data_generator/generation_dataset/generation_synthetic/prepared/differBig/d435_intrinsics_extrinsics.json"
dataset_parent_path = "/home/nikita/data_generator/generation_dataset/generation_synthetic/prepared"
bop_dataset_name = "differBig"

num_gpus = 1
parallel_process_on_one_gpu = 1

poses_cam = 10
probability_drop = 0.5

max_amount_of_samples = 50

cc_textures = dict(
    cc_textures_path="/home/nikita/data_generator/generation_dataset/generation_synthetic/resources/textures_2k_plate"
)

# -------------------- output / reproducibility --------------------

output_dir = "/home/nikita/data_generator/generation_dataset/generation_synthetic/tools/output/d435_stage1a_geom"
depth_scale_mm = 1.0
debug_max_frames = 10
sweep_pose_count = 50

# если хочешь воспроизводимость sweep по позам камеры / сцене
np_random_seed = 42

# использовать GT-маску физического overlap при sweep
use_geom_mask_from_gt = True

# -------------------- Stage 1A baseline: ONLY core search geometry --------------------

# фиксированный режим матчера для этого этапа
sgm_mode = "SGBM"

# базовые стартовые значения
sgm_block_size = 7
sgm_min_disparity = 0
stereo_preprocess = "clahe"

# если строку не указывать, скрипт возьмет rs.recommend_num_disparities(...)
sgm_num_disparities = 112

# -------------------- fixed advanced SGBM params (NOT swept at stage 1A) --------------------

sgm_uniqueness_ratio = 10
sgm_disp12_max_diff = 1
sgm_pre_filter_cap = 63
sgm_p1_scale = 8.0
sgm_p2_scale = 32.0

# -------------------- eval params --------------------

eval_edge_dilation_px = 1
eval_edge_percentile = 85.0

# -------------------- Stage 1A sweeps --------------------
# только базовая геометрия поиска

sweep_block_size_values = [5, 7, 9, 11]
sweep_min_disparity_values = [-2, 0, 2]
sweep_num_disparities_values = [80, 96, 112, 128]
sweep_preprocess_values = ["clahe", "none"]