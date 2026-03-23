mode = "deformed_bop_seg"

camera_profile_json = "/home/nikita/data_generator/generation_dataset/generation_synthetic/prepared/differBig/d435_effective_projector_640x480.json"
dataset_parent_path = "/home/nikita/data_generator/generation_dataset/generation_synthetic/prepared"
bop_dataset_name = "differBig"

num_gpus = 1
parallel_process_on_one_gpu = 1
poses_cam = 1
probability_drop = 0.0

max_amount_of_samples = 50

output_dir = "/home/nikita/data_generator/generation_dataset/generation_synthetic/tools/output/differBig/alignment_research_fixed"

cc_textures = dict(
    cc_textures_path="/home/nikita/data_generator/generation_dataset/generation_synthetic/resources/textures_2k_plate_valid"
)

fixed_texture = dict(
    texture_root="/home/nikita/data_generator/generation_dataset/generation_synthetic/resources/textures_2k_plate_valid",
    texture_name="Tiles074",
)

effective_projector_render = dict(
    projector_energy_range=[100.0],
    ir_light_energy_range=[250.0],
)

rgb_render = dict(
    light_energy_range=[400.0],
)

fixed_scene = dict(
    scene_seed=1234,
    camera_location=[0.55, -0.3, 0.55],
    point_of_interest=[0.0, 0.0, 0.10],
    rotation_factor=5.0,
    light_location=[0.20, -0.25, 4.10],
    light_color=[1.0, 0.97, 0.92],
    light_plane_emission_strength=4.0,
    light_plane_emission_color=[0.90, 0.90, 0.90, 1.0],
)

matcher_data = dict(
    rgb_to_intensity_mode="lcn",
    plane_distance_m=None,

    depth_min_m=0.4,
    depth_max_m=10.0,
    min_disparity=2,
    num_disparities=80,
    block_size=5,
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
    p1_scale=12.0,
    p2_scale=32.0,
)

alignment_research = dict(
    pred_rectify_mode="on",
    gt_rectify_mode="off",
    splat_2x2=True,
    rectify_use_distortion=False,
    save_source_z=True,
    save_target_z=True,
)
