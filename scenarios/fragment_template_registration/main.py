import blenderproc as bproc

import argparse
import math
import os
import re
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

import bpy
import numpy as np
from addon_utils import enable
from mathutils import Euler, Matrix, Vector

from blendforge.host.FiletoDict import Config
from blendforge.blender_runtime.CustomFractureUtills import fracture_object_with_cell
from blendforge.blender_runtime.CustomLightSetting import TemperatureToRGBConverter
from blendforge.blender_runtime.CustomLoadMesh import load_objs
from blendforge.blender_runtime.CustomMaterial import make_random_material
from blendforge.blender_runtime.utils import (
    sample_pose_func,
    sample_pose_func_drop,
)
from blendforge.blender_runtime.writer.FragmentRegistrationWriter import (
    DigitalTwinSurface,
    FragmentRegistrationWriter,
)


BLENDER_SEED_MAX = 0x7FFFFFFF


def parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fragment registration dataset scene generator")
    parser.add_argument("--config_file", type=str, required=True)
    return parser.parse_args(args)


def cfg_to_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: cfg_to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cfg_to_plain(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cfg_to_plain(v) for v in value)
    return value


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    return cfg._data.get(key, default) if hasattr(cfg, "_data") else getattr(cfg, key, default)


def nested_cfg(cfg: Any, key: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    value = cfg_get(cfg, key, default or {})
    return dict(cfg_to_plain(value or {}))


def seed_value_is_random(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"none", "null", "random", "auto"}:
        return True
    return False


def random_seed_for_blender() -> int:
    return int.from_bytes(os.urandom(4), byteorder="little", signed=False) % (BLENDER_SEED_MAX + 1)


def normalize_seed_for_blender(value: Any, *, name: str) -> int:
    seed = int(value)
    if seed < 0:
        raise ValueError(f"{name} must be >= 0")
    return seed % (BLENDER_SEED_MAX + 1)


def resolve_scene_seed(cfg: Any, scene_id: int) -> tuple[int, Optional[int], bool, str]:
    raw_seed = cfg_get(cfg, "seed", 13)
    if seed_value_is_random(raw_seed):
        resolved_seed = cfg_get(cfg, "resolved_seed", None)
        if seed_value_is_random(resolved_seed):
            return random_seed_for_blender(), None, True, "runtime_random"
        return normalize_seed_for_blender(resolved_seed, name="resolved_seed"), None, True, "runner_resolved_seed"

    base_seed = normalize_seed_for_blender(raw_seed, name="seed")
    return normalize_seed_for_blender(base_seed + int(scene_id), name="seed + scene_id"), base_seed, False, "base_seed_plus_scene_id"


def resolve_fracture_seed(fracture_cfg: Mapping[str, Any], scene_seed: int) -> tuple[int, str]:
    if "seed" not in fracture_cfg:
        return int(scene_seed), "scene_seed"
    raw_seed = fracture_cfg.get("seed")
    if seed_value_is_random(raw_seed):
        return random_seed_for_blender(), "runtime_random"
    return normalize_seed_for_blender(raw_seed, name="fracture.seed"), "fracture_config_seed"


def object_unit_scale_to_scene_unit(object_model_unit: str) -> float:
    unit_to_scale = {"m": 1.0, "dm": 0.1, "cm": 0.01, "mm": 0.001}
    if object_model_unit not in unit_to_scale:
        raise ValueError(f"Unsupported object_model_unit: {object_model_unit}")
    return unit_to_scale[object_model_unit]


def apply_scale_to_mesh_object(mesh_obj: Any) -> None:
    bpy_obj = mesh_obj.blender_obj if hasattr(mesh_obj, "blender_obj") else mesh_obj
    bpy.ops.object.select_all(action="DESELECT")
    bpy_obj.select_set(True)
    bpy.context.view_layer.objects.active = bpy_obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy_obj.data.update()


def duplicate_bpy_mesh_object(bpy_obj: bpy.types.Object, name: str) -> bpy.types.Object:
    dup = bpy_obj.copy()
    dup.data = bpy_obj.data.copy()
    dup.name = name
    dup.data.name = f"{name}_mesh"
    bpy.context.collection.objects.link(dup)
    dup.hide_render = True
    dup.hide_viewport = True
    return dup


def remove_object_by_name_if_exists(name: str) -> None:
    obj = bpy.data.objects.get(name)
    if obj is not None:
        bpy.data.objects.remove(obj, do_unlink=True)


def scale_mesh_vertices(bpy_obj: bpy.types.Object, scale: float) -> None:
    scale = float(scale)
    if np.isclose(scale, 1.0):
        return
    for vertex in bpy_obj.data.vertices:
        vertex.co *= scale
    bpy_obj.data.update(calc_edges=True)


def triangulate_bpy_object(bpy_obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    bpy_obj.select_set(True)
    bpy.context.view_layer.objects.active = bpy_obj
    modifier = bpy_obj.modifiers.new(name="FragmentRegistrationTriangulate", type="TRIANGULATE")
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy_obj.data.update(calc_edges=True)


def select_single_object(cfg: Config) -> Any:
    bop_dataset_path = os.path.join(cfg.dataset_parent_path, cfg.bop_dataset_name)
    selector = nested_cfg(cfg, "object_selector", {"mode": "category_id", "values": [cfg_get(cfg, "id_target_obj", 1)]})
    selector_mode = str(selector.get("mode", "category_id"))
    values = list(selector.get("values", []))
    if not values:
        raise ValueError("object_selector.values must contain at least one item")
    scene_id = int(cfg_get(cfg, "scene_id", 0))
    selected_value = values[scene_id % len(values)]

    if selector_mode == "category_id":
        obj_id = int(selected_value)
        loaded = load_objs(
            bop_dataset_path=bop_dataset_path,
            obj_ids=[obj_id],
            sample_objects=False,
            num_of_objs_to_sample=1,
            additional_scale=None,
            manifold=bool(cfg_get(cfg, "manifold", True)),
            object_model_unit=str(cfg_get(cfg, "object_model_unit", "cm")),
        )
        return loaded[0]

    if selector_mode in ("filename", "name"):
        obj_id = object_id_from_filename(str(selected_value), bop_dataset_path)
        loaded = load_objs(
            bop_dataset_path=bop_dataset_path,
            obj_ids=[obj_id],
            sample_objects=False,
            num_of_objs_to_sample=1,
            additional_scale=None,
            manifold=bool(cfg_get(cfg, "manifold", True)),
            object_model_unit=str(cfg_get(cfg, "object_model_unit", "cm")),
        )
        return loaded[0]

    if selector_mode == "dataset_index":
        loaded = load_objs(
            bop_dataset_path=bop_dataset_path,
            sample_objects=False,
            additional_scale=None,
            manifold=bool(cfg_get(cfg, "manifold", True)),
            object_model_unit=str(cfg_get(cfg, "object_model_unit", "cm")),
        )
        index = int(selected_value)
        if index < 0 or index >= len(loaded):
            raise IndexError(f"dataset_index {index} is out of range for {len(loaded)} loaded objects")
        selected = loaded[index]
        for obj in loaded:
            if obj != selected and obj.blender_obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj.blender_obj, do_unlink=True)
        return selected

    raise ValueError(f"Unsupported object_selector.mode: {selector_mode}")


def object_id_from_filename(value: str, bop_dataset_path: str) -> int:
    name = os.path.basename(value)
    stem = os.path.splitext(name)[0]
    match = re.fullmatch(r"obj_(\d+)", stem)
    if match is None:
        raise ValueError(f"Expected BOP model filename like obj_000004.ply, got: {value}")
    model_path = os.path.join(bop_dataset_path, "models", f"{stem}.ply")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    return int(match.group(1))


def create_room(cc_textures_path: Optional[str]) -> List[Any]:
    room_planes = [
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1]),
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[0, -3, 3], rotation=[-1.570796, 0, 0]),
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[0, 3, 3], rotation=[1.570796, 0, 0]),
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[3, 0, 3], rotation=[0, -1.570796, 0]),
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[-3, 0, 3], rotation=[0, 1.570796, 0]),
    ]
    for plane in room_planes:
        plane.enable_rigidbody(
            False,
            collision_shape="BOX",
            friction=100.0,
            linear_damping=0.99,
            angular_damping=0.99,
        )

    if cc_textures_path:
        try:
            cc_textures = bproc.loader.load_ccmaterials(cc_textures_path)
            if cc_textures:
                material = np.random.choice(cc_textures)
                for plane in room_planes:
                    plane.replace_materials(material)
        except Exception as exc:
            print(f"[fragment_template_registration] Could not load cc textures: {exc}")

    return room_planes


def create_light() -> Any:
    diap_tem = [5500, 6500]
    colour = [TemperatureToRGBConverter(tem) for tem in diap_tem]
    light = bproc.types.Light()
    light.set_energy(float(np.random.uniform(150, 1000)))
    light.set_color(np.random.uniform(colour[0], colour[1]))
    light.set_location(
        bproc.sampler.shell(
            center=[0, 0, 4],
            radius_min=0.05,
            radius_max=1.0,
            elevation_min=-1,
            elevation_max=1,
            uniform_volume=True,
        )
    )
    return light


def assign_fragment_metadata(fragments: Sequence[Any], category_id: int) -> None:
    for idx, frag in enumerate(sorted(fragments, key=lambda item: item.blender_obj.name)):
        frag.blender_obj.name = f"fragment_{idx:04d}"
        frag.blender_obj.data.name = f"fragment_{idx:04d}_mesh"
        frag.set_cp("category_id", int(category_id))
        frag.set_cp("fragment_id", int(idx))
        if not frag.has_cp("fracture_method"):
            frag.set_cp("fracture_method", "voronoi")


def canonical_T_O_from_F_by_name(fragments: Sequence[Any], T_W_from_O: np.ndarray) -> Dict[str, np.ndarray]:
    T_O_from_W = np.linalg.inv(T_W_from_O.astype(np.float64))
    out: Dict[str, np.ndarray] = {}
    for frag in fragments:
        T_W_from_F = np.asarray(frag.get_local2world_mat(), dtype=np.float64)
        out[frag.blender_obj.name] = T_O_from_W @ T_W_from_F
    return out


def apply_fragment_output_scale(
    fragments: Sequence[Any],
    raw_T_O_from_F_by_name: Mapping[str, np.ndarray],
    T_W_from_O: np.ndarray,
    fragment_output_scale: float,
) -> Dict[str, np.ndarray]:
    scaled_T_O_from_F_by_name: Dict[str, np.ndarray] = {}

    for frag in fragments:
        frag_bpy = frag.blender_obj
        raw_T_O_from_F = np.asarray(raw_T_O_from_F_by_name[frag_bpy.name], dtype=np.float64)
        scaled_T_O_from_F = raw_T_O_from_F.copy()
        scaled_T_O_from_F[:3, 3] *= float(fragment_output_scale)

        scale_mesh_vertices(frag_bpy, fragment_output_scale)

        T_W_from_F_scaled = np.asarray(T_W_from_O, dtype=np.float64) @ scaled_T_O_from_F
        frag_bpy.matrix_world = Matrix(T_W_from_F_scaled.tolist())
        bpy.context.view_layer.update()

        scaled_T_O_from_F_by_name[frag_bpy.name] = scaled_T_O_from_F

    return scaled_T_O_from_F_by_name


def set_random_fragment_materials(fragments: Sequence[Any]) -> None:
    for idx, frag in enumerate(fragments):
        frag.enable_rigidbody(True, friction=100.0, linear_damping=0.99, angular_damping=0.99)
        frag.set_shading_mode("auto")
        mat, _style = make_random_material(allowed=["plastic_new"], name_prefix=f"fragment_{idx:04d}")
        mats = frag.get_materials()
        if not mats:
            frag.set_material(0, mat)
        else:
            for mat_idx in range(len(mats)):
                frag.set_material(mat_idx, mat)


def layout_fragments(fragments: Sequence[Any], cfg: Config) -> None:
    layout_cfg = nested_cfg(cfg, "layout", {})
    mode = str(layout_cfg.get("mode", "drop"))
    simulate_physics = bool(layout_cfg.get("simulate_physics", True))

    if mode == "static":
        return
    if mode == "scatter":
        pose_func = sample_pose_func
    elif mode == "drop":
        pose_func = sample_pose_func_drop
    else:
        raise ValueError(f"Unsupported layout.mode: {mode}")

    bproc.object.sample_poses(
        objects_to_sample=list(fragments),
        sample_pose_func=pose_func,
        max_tries=int(layout_cfg.get("max_pose_tries", 1000)),
    )

    if simulate_physics:
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=float(layout_cfg.get("min_simulation_time", 3)),
            max_simulation_time=float(layout_cfg.get("max_simulation_time", 35)),
            check_object_interval=float(layout_cfg.get("check_object_interval", 2)),
            substeps_per_frame=int(layout_cfg.get("substeps_per_frame", 30)),
            solver_iters=int(layout_cfg.get("solver_iters", 30)),
        )


def add_camera_poses(num_frames: int, cfg: Config) -> None:
    camera_cfg = nested_cfg(cfg, "camera_sampling", {})
    center = np.asarray(camera_cfg.get("center", [0.0, 0.0, 0.1]), dtype=np.float32)
    poi = np.asarray(camera_cfg.get("poi", [0.0, 0.0, 0.0]), dtype=np.float32)
    radius_min = float(camera_cfg.get("radius_min", 0.40))
    radius_max = float(camera_cfg.get("radius_max", 1.0))
    elevation_min = float(camera_cfg.get("elevation_min", 5))
    elevation_max = float(camera_cfg.get("elevation_max", 89))
    rotation_factor = float(camera_cfg.get("rotation_factor", 11.0))

    for _ in range(num_frames):
        current_radius_max = round(float(np.random.uniform(radius_min, radius_max)), 2)
        cam_loc = bproc.sampler.shell(
            center=center,
            radius_min=radius_min,
            radius_max=current_radius_max,
            elevation_min=elevation_min,
            elevation_max=elevation_max,
            uniform_volume=False,
        )

        forward = poi - cam_loc
        dist = float(np.linalg.norm(forward))
        look_quat = Vector(forward).to_track_quat("-Z", "Y")
        R = look_quat.to_matrix()

        max_angle_deg = rotation_factor * dist
        rx = math.radians(float(np.random.uniform(-max_angle_deg, max_angle_deg)))
        ry = math.radians(float(np.random.uniform(-max_angle_deg, max_angle_deg)))
        rz = math.radians(float(np.random.uniform(-max_angle_deg, max_angle_deg)))
        R @= Euler((rx, ry, rz), "XYZ").to_matrix()

        cam2world = bproc.math.build_transformation_mat(cam_loc, np.asarray(R))
        bproc.camera.add_camera_pose(cam2world)


def render_scene(cfg: Config) -> Dict[str, Any]:
    render_cfg = nested_cfg(cfg, "render", {})
    max_samples = int(render_cfg.get("max_amount_of_samples", cfg_get(cfg, "max_amount_of_samples", 50)))
    bproc.renderer.set_max_amount_of_samples(max_samples)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_segmentation_output(
        map_by=[
            "category_id",
            "instance",
            "fragment_id",
            "fracture_uid",
            "fracture_seed",
            "fracture_method",
        ],
        default_values={
            "category_id": 0,
            "fragment_id": -1,
            "fracture_uid": "",
            "fracture_seed": 0,
            "fracture_method": "",
        },
        output_dir=cfg_get(cfg, "temp_dir_segmap", None),
        file_prefix=f"segmap_scene{int(cfg_get(cfg, 'scene_id', 0)):06d}_",
    )
    return bproc.renderer.render(
        output_dir=cfg_get(cfg, "temp_dir_rgb", None),
        file_prefix=f"rgb_scene{int(cfg_get(cfg, 'scene_id', 0)):06d}_",
    )


def main(args: Optional[Sequence[str]] = None) -> None:
    bproc.init()
    enable("object_fracture_cell")
    bproc.utility.reset_keyframes()

    if args is None:
        args = sys.argv[1:]
    parsed = parse_args(args)
    cfg = Config(parsed.config_file)

    scene_id = int(cfg_get(cfg, "scene_id", 0))
    split = str(cfg_get(cfg, "split", "train"))
    num_frames = int(cfg_get(cfg, "num_frames_per_scene", 1))
    seed, base_config_seed, seed_is_random, seed_source = resolve_scene_seed(cfg, scene_id)
    np.random.seed(seed)

    writer = FragmentRegistrationWriter(
        dataset_root=cfg.output_dir,
        split=split,
        scene_id=scene_id,
        overwrite_scene=bool(cfg_get(cfg, "overwrite_scene", False)),
        depth_scale_mm=float(cfg_get(cfg, "depth_scale_mm", 1.0)),
    )

    cc_textures = nested_cfg(cfg, "cc_textures", {})
    create_room(cc_textures.get("cc_textures_path"))
    light = create_light()

    fracture_cfg = nested_cfg(cfg, "fracture", {})
    fragment_output_scale = float(fracture_cfg.get("fragment_scale", 1.0) or 1.0)
    if fragment_output_scale <= 0.0:
        raise ValueError("fracture.fragment_scale must be > 0")

    source_obj = select_single_object(cfg)
    apply_scale_to_mesh_object(source_obj)
    triangulate_bpy_object(source_obj.blender_obj)

    category_id = int(source_obj.get_cp("category_id"))
    object_id = f"object_{category_id:06d}"
    T_W_from_O = np.asarray(source_obj.get_local2world_mat(), dtype=np.float64)

    digital_twin_bpy = duplicate_bpy_mesh_object(source_obj.blender_obj, f"{object_id}_digital_twin")
    digital_twin_name = str(digital_twin_bpy.name)
    scale_mesh_vertices(digital_twin_bpy, fragment_output_scale)
    source_unit_scale = object_unit_scale_to_scene_unit(str(cfg_get(cfg, "object_model_unit", "cm")))
    model_metadata = {
        "category_id": int(category_id),
        "source_object_model_unit": str(cfg_get(cfg, "object_model_unit", "cm")),
        "source_object_unit_scale_to_scene_unit": float(source_unit_scale),
        "fragment_output_scale": float(fragment_output_scale),
        "effective_source_vertex_scale_to_scene_unit": float(source_unit_scale * fragment_output_scale),
        "source_scale_applied_to_mesh": True,
        "fragment_output_scale_baked_to_mesh": True,
        "coordinate_frame": "O",
        "units": "scene_unit",
    }
    object_model_rel_from_split, object_model_sha256 = writer.write_object_model_ply(
        digital_twin_bpy,
        object_id,
        model_metadata=model_metadata,
        overwrite_model=bool(cfg_get(cfg, "overwrite_models", False)),
    )
    digital_twin_surface = DigitalTwinSurface(digital_twin_bpy)

    bop_dataset_path = os.path.join(cfg.dataset_parent_path, cfg.bop_dataset_name)
    bproc.loader.load_bop_intrinsics(bop_dataset_path=bop_dataset_path)
    fracture_seed, fracture_seed_source = resolve_fracture_seed(fracture_cfg, seed)

    fragments = fracture_object_with_cell(
        bpy_obj=source_obj.blender_obj,
        source_limit_and_count=int(fracture_cfg.get("source_limit_and_count", 4)),
        source_noise=float(fracture_cfg.get("source_noise", 0.001)),
        cell_scale=tuple(fracture_cfg.get("cell_scale", [1.0, 1.0, 1.0])),
        margin=float(fracture_cfg.get("margin", 0.001)),
        scale=None,
        seed=fracture_seed,
        max_attempts=int(fracture_cfg.get("max_attempts", 4)),
    )
    if not fragments:
        raise RuntimeError("Fracture did not produce any valid fragments")

    assign_fragment_metadata(fragments, category_id=category_id)
    for frag in fragments:
        triangulate_bpy_object(frag.blender_obj)

    raw_T_O_from_F_by_name = canonical_T_O_from_F_by_name(fragments, T_W_from_O=T_W_from_O)
    T_O_from_F_by_name = apply_fragment_output_scale(
        fragments=fragments,
        raw_T_O_from_F_by_name=raw_T_O_from_F_by_name,
        T_W_from_O=T_W_from_O,
        fragment_output_scale=fragment_output_scale,
    )

    set_random_fragment_materials(fragments)
    layout_fragments(fragments, cfg)

    surface_labeling_cfg = nested_cfg(cfg, "surface_labeling", {})
    fragment_filter_cfg = nested_cfg(cfg, "fragment_filter", {})
    fragment_records, face_labels_by_fragment, _samples_by_fragment = writer.write_fragment_assets(
        fragments=fragments,
        object_id=object_id,
        object_model_rel_from_split=object_model_rel_from_split,
        object_category_id=category_id,
        digital_twin_surface=digital_twin_surface,
        T_O_from_F_by_name=T_O_from_F_by_name,
        surface_labeling_cfg=surface_labeling_cfg,
        fragment_filter_cfg=fragment_filter_cfg,
        fracture_metadata={
            "method": "voronoi",
            "seed": int(fracture_seed),
            "seed_source": fracture_seed_source,
            "scene_seed": int(seed),
            "scene_seed_source": seed_source,
            "scene_seed_is_random": bool(seed_is_random),
            "base_config_seed": base_config_seed,
            "scene_id": int(scene_id),
            "source_limit_and_count": int(fracture_cfg.get("source_limit_and_count", 4)),
            "source_noise": float(fracture_cfg.get("source_noise", 0.001)),
            "cell_scale": [float(v) for v in fracture_cfg.get("cell_scale", [1.0, 1.0, 1.0])],
            "margin": float(fracture_cfg.get("margin", 0.001)),
            "fragment_scale": float(fragment_output_scale),
            "max_attempts": int(fracture_cfg.get("max_attempts", 4)),
        },
        geometry_metadata={
            "units": "scene_unit",
            "source_object_model_unit": str(cfg_get(cfg, "object_model_unit", "cm")),
            "source_object_unit_scale_to_scene_unit": float(source_unit_scale),
            "source_scale_applied_to_mesh": True,
            "fragment_output_scale": float(fragment_output_scale),
            "effective_source_vertex_scale_to_scene_unit": float(source_unit_scale * fragment_output_scale),
            "fragment_output_scale_baked_to_mesh": True,
            "post_fracture_object_scale_applied": False,
        },
    )
    T_O_from_F_by_fragment = {
        int(record["fragment_id"]): np.asarray(record["T_O_from_F"], dtype=np.float64)
        for record in fragment_records
    }
    remove_object_by_name_if_exists(digital_twin_name)

    add_camera_poses(num_frames, cfg)
    data = render_scene(cfg)

    writer.write_frames(
        colors=data["colors"],
        depths_m=data["depth"],
        instance_segmaps=data["instance_segmaps"],
        instance_attribute_maps=data["instance_attribute_maps"],
        fragments=fragments,
        fragment_records=fragment_records,
        face_labels_by_fragment=face_labels_by_fragment,
        T_O_from_F_by_fragment=T_O_from_F_by_fragment,
        write_flags=nested_cfg(cfg, "write_flags", {}),
        visible_points_pixel_stride=int(cfg_get(cfg, "visible_points_pixel_stride", 1)),
    )

    writer.write_camera_info()
    writer.write_scene_meta(
        object_id=object_id,
        object_model_rel_from_split=object_model_rel_from_split,
        num_fragments=len(fragments),
        num_frames=num_frames,
        extra={
            "units": "scene_unit",
            "object_category_id": int(category_id),
            "object_model_sha256": object_model_sha256,
            "source_object_model_unit": str(cfg_get(cfg, "object_model_unit", "cm")),
            "source_object_unit_scale_to_scene_unit": float(source_unit_scale),
            "source_scale_applied_to_mesh": True,
            "fragment_output_scale": float(fragment_output_scale),
            "effective_source_vertex_scale_to_scene_unit": float(source_unit_scale * fragment_output_scale),
            "fragment_output_scale_baked_to_mesh": True,
            "post_fracture_object_scale_applied": False,
            "matrix_convention": "BOP/OpenCV, column-vector homogeneous transforms",
            "seed": int(seed),
            "seed_source": seed_source,
            "seed_is_random": bool(seed_is_random),
            "base_config_seed": base_config_seed,
            "fracture_seed": int(fracture_seed),
            "fracture_seed_source": fracture_seed_source,
            "num_fragments_total": int(len(fragments)),
            "num_fragments_annotated": int(len(fragment_records)),
            "num_fragments_ignored": int(len(fragments) - len(fragment_records)),
            "fragment_filter": dict(fragment_filter_cfg),
        },
    )

    if light is not None:
        light.delete()


if __name__ == "__main__":
    main()
