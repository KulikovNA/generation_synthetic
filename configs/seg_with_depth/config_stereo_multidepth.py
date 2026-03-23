mode = "seg_with_depth_stereo_multidepth"

parallel_process_on_one_gpu = 1
num_gpus = 1

poses_cam = 10
runs_one_fracture = 1
runs_process = 1
dataset_type = "test"

dataset_parent_path = "prepared"
bop_dataset_name = "differBig"
camera_profile_json = "prepared/differBig/d435_effective_projector_640x480.json"

cc_textures = dict(
    cc_textures_path="resources/textures_2k_plate_valid",
    cc_textures_object_path="resources/textures_1k_modules",
)

output_dir = "output/seg_with_depth_stereo_multidepth"
save_config = True

probability_drop = 0.50
object_model_unit = "cm"
max_amount_of_samples = 50

color_file_format = "JPEG"
jpg_quality = 95
depth_scale_mm = 1.0

rgb_render = dict(
    light_energy_range=[300.0, 1200.0],
    max_amount_of_samples_range=[50, 1000],
)

stereo_render = dict(
    projector_energy_range=[50.0, 150.0],
    ir_light_energy_range=[150.0, 300.0],
    max_amount_of_samples_range=[40, 200],
)

stereo_output = dict(
    save_ir_pairs=True,
    depth_value_mode="target_z",
    splat_2x2=True,
)

matcher_effective = dict(
    rgb_to_intensity_mode="lcn",
    plane_distance_m=None,
    depth_min_m=0.2,
    depth_max_m=10.0,
    min_disparity=[0, 4],
    num_disparities=[48, 96],
    block_size=[5, 9],
    preprocess="none",
    use_geom_mask_from_gt=False,
    use_wls=False,
    lr_check=True,
    lr_thresh_px=3.0,
    lr_min_keep_ratio=0.2,
    speckle_filter=True,
    fill_mode="none",
    fill_iters=0,
    depth_completion=False,
    sgbm_mode="HH",
    uniqueness_ratio=5,
    disp12_max_diff=1,
    pre_filter_cap=63,
    p1_scale=[8.0, 16.0],
    p2_scale=[32.0, 48.0],
)

matcher_random = matcher_effective

random_projector_pattern = dict(
    seed_mode="random",
    seed_value=12345,
    dot_count_range=[4000, 5000],
    dot_radius_px_range=[5.0, 10.0],
    min_sep_px_range=[1.0, 10.0],
    dot_sigma_px_range=[0.1, 2.2],
    enforce_non_overlap=True,
    non_overlap_margin_px=0.5,
)

material_randomization = dict(
    prob_make_random_material=0.5,
    make_random_material_allowed=[
        "metal",
        "dirty_metal",
        "cast_iron",
        "steel",
        "brushed_steel",
        "galvanized_steel",
        "blackened_steel",
        "plastic_new",
        "plastic_old",
    ],
)

fracture = dict(
    source_limit_and_count_choices=[2, 3, 4, 5],
    source_noise_range=[0.0, 0.007],
    cell_scale_range=[0.75, 1.5],
    fracture_scale=0.05,
    seed_range=[2, 40],
)
