mode = "deformed_bop_seg"

camera_profile_json = "/home/nikita/data_generator/generation_dataset/generation_synthetic/prepared/differBig/d435_effective_projector_640x480.json"
dataset_parent_path = "/home/nikita/data_generator/generation_dataset/generation_synthetic/prepared"
bop_dataset_name = "differBig"
num_gpus = 1
parallel_process_on_one_gpu = 1
poses_cam = 1
probability_drop = 0.0

max_amount_of_samples = 50
 
cc_textures = dict(
    cc_textures_path="/home/nikita/data_generator/generation_dataset/generation_synthetic/resources/textures_2k_plate"
)

output_dir = "/home/nikita/data_generator/generation_dataset/generation_synthetic/tools/output/differBig/texture_sweep_fixed"

effective_projector_render = dict(
    projector_energy_range=[150.0],
    ir_light_energy_range=[300.0],
)

rgb_render = dict(
    light_energy_range=[400.0],
)

fixed_scene = dict(
    scene_seed=1234,
    camera_location=[0.95, -1.20, 0.75],
    point_of_interest=[0.0, 0.0, 0.10],
    rotation_factor=5.0,
    light_location=[0.20, -0.35, 4.10],
    light_color=[1.0, 0.97, 0.92],
    light_plane_emission_strength=4.0,
    light_plane_emission_color=[0.90, 0.90, 0.90, 1.0],
)

texture_sweep = dict(
    limit=None,
)
