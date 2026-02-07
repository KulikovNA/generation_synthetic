import math
from os.path import join

CUSTOM_OBJ_IDS = {
    "custom": list(range(1, 4)),
    "differ": list(range(1, 19)),
    "differBig": list(range(1, 10)),
    "differSmall": [1],
    "databot": list(range(1, 7)),
    "new_data": list(range(1, 9)),
}

CUSTOM_SYMMETRIC_OBJ_IDS = {
    "custom": [],
    "differ": [],
    "differBig": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "differSmall": [1],
    "databot": [1, 2, 3, 4, 5],
    "new_data": [1, 3, 4, 5, 6],
}

def _ranges_default():
    return dict(
        depth_range=(638.38, 775.97),
        azimuth_range=(0, 2 * math.pi),
        elev_range=(-0.5 * math.pi, 0.5 * math.pi),
    )

CUSTOM_SPLITS = {
    "custom": dict(
        scene_ids={"train": [], "val": [1], "test": [1]},
        im_size=(1280, 960),
        test_ranges=_ranges_default(),
    ),
    "differ": dict(
        scene_ids={"train": list(range(1, 19)), "val": list(range(1, 19)), "test": list(range(1, 19))},
        im_size=(640, 480),
        test_ranges=_ranges_default(),
    ),
    "differBig": dict(
        scene_ids={"train": list(range(1, 30)), "val": list(range(1, 30)), "test": list(range(1, 30))},
        im_size=(640, 480),
        test_ranges=_ranges_default(),
    ),
    "differSmall": dict(
        scene_ids={"train": list(range(1, 30)), "val": list(range(1, 30)), "test": list(range(1, 30))},
        im_size=(640, 480),
        test_ranges=_ranges_default(),
    ),
    "databot": dict(
        scene_ids={"train": list(range(1, 30)), "val": list(range(1, 30)), "test": list(range(1, 30))},
        im_size=(640, 480),
        test_ranges=_ranges_default(),
    ),
    "new_data": dict(
        scene_ids={"train": list(range(1, 30)), "val": list(range(1, 30)), "test": list(range(1, 30))},
        im_size=(640, 480),
        test_ranges=_ranges_default(),
    ),
}

def patch_bop_dataset_params():
    """
    Patch bop_toolkit_lib.dataset_params to support custom datasets
    while staying compatible with newer bop_toolkit that has extra fields
    (eval_sensor/eval_modality/supported_error_types, multi-sensor datasets, etc.)
    """
    import bop_toolkit_lib.dataset_params as dp

    if getattr(dp, "_CUSTOM_DATASETS_PATCHED", False):
        return

    orig_get_model_params = dp.get_model_params
    orig_get_split_params = dp.get_split_params

    def get_model_params(datasets_path, dataset_name, model_type=None):
        if dataset_name in CUSTOM_OBJ_IDS:
            # Use an existing dataset as a "template" to preserve any new keys/behavior.
            base = orig_get_model_params(datasets_path, "lm", model_type=model_type)

            obj_ids = CUSTOM_OBJ_IDS[dataset_name]
            sym_ids = CUSTOM_SYMMETRIC_OBJ_IDS.get(dataset_name, [])

            models_folder_name = "models"
            if model_type is not None:
                models_folder_name += "_" + model_type
            models_path = join(datasets_path, dataset_name, models_folder_name)

            base["obj_ids"] = obj_ids
            base["symmetric_obj_ids"] = sym_ids
            base["model_tpath"] = join(models_path, "obj_{obj_id:06d}.ply")
            base["models_info_path"] = join(models_path, "models_info.json")
            return base

        return orig_get_model_params(datasets_path, dataset_name, model_type=model_type)

    def get_split_params(datasets_path, dataset_name, split, split_type=None):
        if dataset_name in CUSTOM_SPLITS:
            cfg = CUSTOM_SPLITS[dataset_name]

            # Create a classic-BOP template dict (has supported_error_types, eval_sensor, etc. in new versions)
            base = orig_get_split_params(datasets_path, "lm", split, split_type=split_type)

            base["name"] = dataset_name
            base["split"] = split
            base["split_type"] = split_type
            base["base_path"] = join(datasets_path, dataset_name)

            # Your dataset specifics
            base["scene_ids"] = cfg["scene_ids"][split]
            base["im_size"] = cfg["im_size"]

            # optional ranges
            base["depth_range"] = None
            base["azimuth_range"] = None
            base["elev_range"] = None
            if split == "test":
                base.update(cfg.get("test_ranges", {}))

            # Now rewrite paths to point to your dataset folder structure (classic BOP layout)
            base_path = join(datasets_path, dataset_name)
            split_path = join(base_path, split)
            if split_type is not None:
                # keep same rule as official: pbr adds suffix and may override scene_ids.
                if split_type == "pbr":
                    base["scene_ids"] = list(range(50))
                split_path += "_" + split_type

            base["split_path"] = split_path

            # Determine extensions: follow official rules for pbr/jpg, itodd/tif, etc.
            # For your datasets we keep png by default, unless you explicitly use pbr.
            rgb_ext = ".jpg" if (split_type == "pbr") else ".png"
            gray_ext = ".png"
            depth_ext = ".png"

            base.update({
                "gray_tpath": join(split_path, "{scene_id:06d}", "gray", "{im_id:06d}" + gray_ext),
                "rgb_tpath": join(split_path, "{scene_id:06d}", "rgb", "{im_id:06d}" + rgb_ext),
                "depth_tpath": join(split_path, "{scene_id:06d}", "depth", "{im_id:06d}" + depth_ext),
                "scene_camera_tpath": join(split_path, "{scene_id:06d}", "scene_camera.json"),
                "scene_gt_tpath": join(split_path, "{scene_id:06d}", "scene_gt.json"),
                "scene_gt_info_tpath": join(split_path, "{scene_id:06d}", "scene_gt_info.json"),
                "scene_gt_coco_tpath": join(split_path, "{scene_id:06d}", "scene_gt_coco.json"),
                "mask_tpath": join(split_path, "{scene_id:06d}", "mask", "{im_id:06d}_{gt_id:06d}.png"),
                "mask_visib_tpath": join(split_path, "{scene_id:06d}", "mask_visib", "{im_id:06d}_{gt_id:06d}.png"),
            })

            # Ensure classic format invariants for downstream helpers
            base["eval_sensor"] = None
            base["eval_modality"] = None
            base["im_modalities"] = ["rgb", "depth"]

            return base

        return orig_get_split_params(datasets_path, dataset_name, split, split_type=split_type)

    dp.get_model_params = get_model_params
    dp.get_split_params = get_split_params
    dp._CUSTOM_DATASETS_PATCHED = True
