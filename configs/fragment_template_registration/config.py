mode = "fragment_template_registration"

dataset_parent_path = "prepared"
bop_dataset_name = "differBig"
object_model_unit = "cm"
manifold = True

output_dir = "output/fragment_template_registration"
save_config = True

# Split orchestration. Run the runner separately for train/val/test.
split = "test"
num_scenes = 4
num_frames_per_scene = 5
num_scene_workers = 2
scene_id_offset = 0
# None means: generate a random per-scene seed in the runner and save it
# into the generated scene config/metadata. Use an int for reproducible
# seed + scene_id behavior.
seed = None
overwrite_scene = False
overwrite_models = False

# Supported selector modes: filename, category_id, dataset_index.
# If several values are provided, the scenario cycles them by scene_id.
object_selector = {
    "mode": "filename",
    "values": ["obj_000004.ply"],
}

cc_textures = dict(
    cc_textures_path="resources/textures_2k_plate",
    cc_textures_object_path="resources/textures_1k_modules",
)

fracture = {
    "source_limit_and_count": 4,
    "source_noise": 0.001,
    "cell_scale": [1.0, 1.0, 1.0],
    "margin": 0.001,
    # Matches seg_with_depth fragment size, but is baked into fragment meshes
    # and the exported digital twin after Voronoi fracture.
    "fragment_scale": 0.05,
    "max_attempts": 4,
}

layout = {
    "mode": "drop",  # static | scatter | drop
    "simulate_physics": True,
    "max_pose_tries": 1000,
    "min_simulation_time": 3,
    "max_simulation_time": 35,
    "check_object_interval": 2,
    "substeps_per_frame": 30,
    "solver_iters": 30,
}

camera_sampling = {
    "center": [0.0, 0.0, 0.1],
    "poi": [0.0, 0.0, 0.0],
    "radius_min": 0.4,
    "radius_max": 1.0,
    "elevation_min": 5,
    "elevation_max": 89,
    "rotation_factor": 11.0,
}

render = {
    # Use either a fixed value, e.g. 50, or an inclusive random range, e.g. [32, 96].
    "max_amount_of_samples": 50,
}

depth_scale_mm = 1.0
visible_points_pixel_stride = 1

surface_labeling = {
    "method": "distance_and_normal_to_digital_twin",
    "distance_threshold_m": 0.0005,
    "normal_cos_threshold": 0.75,
    "samples_per_face": 5,
    "unknown_policy": "ignore",
    # Removes tiny fracture islands fully or mostly surrounded by shell faces.
    # This filters labeling speckles before samples, masks and visible_points are derived.
    "cleanup_small_fracture_components": True,
    "min_fracture_component_faces": 12,
    "min_fracture_component_area": 0.0,
    "small_component_shell_neighbor_ratio": 0.6,
}

fragment_filter = {
    # Tiny fragments stay in RGB/depth as scene distractors, but are excluded
    # from COCO, GT poses, instance/surface masks and visible_points.
    "enabled": True,
    "min_vertices": 200,
    "min_faces": 400,
    "ignore_reason": "small_fragment",
}

write_flags = {
    "write_rgb": True,
    "write_depth": True,
    "write_instance_masks": True,
    "write_surface_masks": True,
    "write_visible_points": True,
}
