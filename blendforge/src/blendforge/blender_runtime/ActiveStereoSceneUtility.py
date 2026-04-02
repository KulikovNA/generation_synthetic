from __future__ import annotations

import os

import blenderproc as bproc
import bpy
import numpy as np

from blendforge.blender_runtime.CustomLightSetting import TemperatureToRGBConverter
from blendforge.blender_runtime.utils import build_lookat_pose_cam
from blendforge.blender_runtime.camera.RealsenseProfileLoader import RealSenseProfile


def _section_get(section, key: str, default):
    if section is None:
        return default
    if isinstance(section, dict):
        return section.get(key, default)
    return getattr(section, key, default)


def _sample_uniform_from_range(value, default):
    if value is None:
        value = default

    if isinstance(value, (int, float)):
        return float(value)

    vals = list(value)
    if len(vals) == 0:
        vals = list(default)
    if len(vals) == 1:
        return float(vals[0])

    lo = float(vals[0])
    hi = float(vals[1])
    if hi < lo:
        lo, hi = hi, lo
    return float(np.random.uniform(lo, hi))


def _sample_int_from_range(value, default):
    if value is None:
        value = default

    if isinstance(value, (int, float)):
        return int(round(value))

    vals = list(value)
    if len(vals) == 0:
        vals = list(default)
    if len(vals) == 1:
        return int(round(vals[0]))

    lo = int(round(vals[0]))
    hi = int(round(vals[1]))
    if hi < lo:
        lo, hi = hi, lo
    return int(np.random.randint(lo, hi + 1))


def _normalize_rgb_triplet(value, default):
    raw = default if value is None else value
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"RGB triplet must contain exactly 3 values, got {arr.tolist()}")
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def _sample_rgb_from_range(value, default):
    if value is None:
        return tuple(float(v) for v in _normalize_rgb_triplet(default, default))

    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        return tuple(float(v) for v in _normalize_rgb_triplet(arr, default))

    if arr.ndim == 2 and arr.shape[0] == 1:
        return tuple(float(v) for v in _normalize_rgb_triplet(arr[0], default))

    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError(
            "RGB color range must be either [r, g, b] or [[r_lo, g_lo, b_lo], [r_hi, g_hi, b_hi]]"
        )

    lo = _normalize_rgb_triplet(arr[0], default)
    hi = _normalize_rgb_triplet(arr[1], default)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    return tuple(float(v) for v in np.random.uniform(lo, hi))


def build_room_and_lights(cfg):
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

    light_plane = bproc.object.create_primitive("PLANE", scale=[1, 1, 1], location=[0, 0, 5])
    light_plane.set_name("light_plane")
    light_plane_material = bproc.material.create("light_material")
    light_plane_material.make_emissive(
        emission_strength=np.random.uniform(3, 6),
        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]),
    )
    light_plane.replace_materials(light_plane_material)

    cc_textures = bproc.loader.load_ccmaterials(cfg.cc_textures.cc_textures_path)
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)

    diap_tem = [5500, 6500]
    colour = [TemperatureToRGBConverter(tem) for tem in diap_tem]
    light_energy = float(np.random.uniform(150, 1000))

    light_point = bproc.types.Light()
    light_point.set_energy(light_energy)
    light_point.set_color(np.random.uniform(colour[0], colour[1]))
    light_point.set_location(
        bproc.sampler.shell(
            center=[0, 0, 4],
            radius_min=0.05,
            radius_max=1.0,
            elevation_min=-1,
            elevation_max=1,
            uniform_volume=True,
        )
    )
    return room_planes, light_point, light_plane_material


def build_rig_poses(cfg, rs: RealSenseProfile):
    rig_poses_ir_left: list[np.ndarray] = []
    rotation_factor = 9.0

    for _ in range(int(cfg.poses_cam)):
        radius_max = round(float(np.random.uniform(0.75, 2.0)), 2)
        cam_loc = bproc.sampler.shell(
            center=[0, 0, 0.1],
            radius_min=0.50,
            radius_max=radius_max,
            elevation_min=5,
            elevation_max=89,
            uniform_volume=False,
        )
        poi = np.array([0, 0, 0], dtype=np.float32)
        cam2world = build_lookat_pose_cam(cam_loc, poi, rotation_factor=rotation_factor)
        rig_poses_ir_left.append(cam2world)

    T_color_from_ir_left = rs.get_T_blender("IR_LEFT", "COLOR")
    T_color_from_ir_right = rs.get_T_blender("IR_RIGHT", "COLOR")
    T_ir_left_from_color = np.linalg.inv(T_color_from_ir_left)

    rig_poses_color = [T_w_irL @ T_ir_left_from_color for T_w_irL in rig_poses_ir_left]
    rig_poses_ir_right = [T_w_c @ T_color_from_ir_right for T_w_c in rig_poses_color]

    return rig_poses_ir_left, rig_poses_ir_right, rig_poses_color


def sample_effective_energy_config(cfg) -> dict:
    legacy_section = getattr(cfg, "effective_projector_render", None)
    stereo_section = getattr(cfg, "stereo_render", None)
    rgb_section = getattr(cfg, "rgb_render", None)
    return {
        "projector_energy": _sample_uniform_from_range(
            _section_get(
                stereo_section,
                "projector_energy_range",
                _section_get(legacy_section, "projector_energy_range", [50.0, 150.0]),
            ),
            [50.0, 150.0],
        ),
        "projector_color_rgb": _sample_rgb_from_range(
            _section_get(
                stereo_section,
                "projector_color_range",
                _section_get(legacy_section, "projector_color_range", [1.0, 1.0, 1.0]),
            ),
            [1.0, 1.0, 1.0],
        ),
        "rgb_light_energy": _sample_uniform_from_range(
            _section_get(
                rgb_section,
                "light_energy_range",
                _section_get(legacy_section, "rgb_light_energy_range", [150.0, 400.0]),
            ),
            [150.0, 400.0],
        ),
        "ir_light_energy": _sample_uniform_from_range(
            _section_get(
                stereo_section,
                "ir_light_energy_range",
                _section_get(legacy_section, "ir_light_energy_range", [10.0, 60.0]),
            ),
            [10.0, 60.0],
        ),
    }


def sample_render_sample_config(cfg) -> dict:
    rgb_section = getattr(cfg, "rgb_render", None)
    stereo_section = getattr(cfg, "stereo_render", None)
    fallback = getattr(cfg, "max_amount_of_samples", None)

    if fallback is None:
        fallback = [30, 80]

    rgb_spp = _sample_int_from_range(
        _section_get(rgb_section, "max_amount_of_samples_range", fallback),
        fallback,
    )
    stereo_spp = _sample_int_from_range(
        _section_get(stereo_section, "max_amount_of_samples_range", fallback),
        fallback,
    )

    return {
        "rgb_max_amount_of_samples": max(1, int(rgb_spp)),
        "stereo_max_amount_of_samples": max(1, int(stereo_spp)),
    }


def _find_node_input(node, socket_name: str):
    for sock in node.inputs:
        if sock.name == socket_name:
            return sock
    return None


def _iter_world_background_nodes():
    world = getattr(bpy.context.scene, "world", None)
    if world is None or not getattr(world, "use_nodes", False) or world.node_tree is None:
        return []
    return [node for node in world.node_tree.nodes if node.type == "BACKGROUND"]


def _iter_material_emission_nodes(material):
    mat = getattr(material, "blender_obj", material)
    if mat is None or not getattr(mat, "use_nodes", False) or mat.node_tree is None:
        return []
    return [node for node in mat.node_tree.nodes if node.type == "EMISSION"]


def _resolve_world_strength_socket(node_name: str):
    world = getattr(bpy.context.scene, "world", None)
    if world is None or not getattr(world, "use_nodes", False) or world.node_tree is None:
        return None
    node = world.node_tree.nodes.get(node_name)
    if node is None:
        return None
    return _find_node_input(node, "Strength")


def _resolve_material_strength_socket(material_name: str, node_name: str):
    mat = bpy.data.materials.get(material_name)
    if mat is None or not getattr(mat, "use_nodes", False) or mat.node_tree is None:
        return None
    node = mat.node_tree.nodes.get(node_name)
    if node is None:
        return None
    return _find_node_input(node, "Strength")


def capture_render_lighting_state(light_point, emissive_material) -> dict:
    point_light_obj = getattr(light_point, "blender_obj", None)
    point_light_name = None if point_light_obj is None else str(point_light_obj.name)
    point_energy = None if point_light_obj is None else float(point_light_obj.data.energy)
    emissive_material_obj = getattr(emissive_material, "blender_obj", emissive_material)
    emissive_material_name = None if emissive_material_obj is None else str(emissive_material_obj.name)

    world_strengths = []
    for node in _iter_world_background_nodes():
        sock = _find_node_input(node, "Strength")
        if sock is not None:
            world_strengths.append((str(node.name), float(sock.default_value)))

    emissive_strengths = []
    for node in _iter_material_emission_nodes(emissive_material):
        sock = _find_node_input(node, "Strength")
        if sock is not None:
            emissive_strengths.append((str(node.name), float(sock.default_value)))

    return {
        "point_light_name": point_light_name,
        "point_energy": point_energy,
        "emissive_material_name": emissive_material_name,
        "world_strengths": world_strengths,
        "emissive_strengths": emissive_strengths,
    }


def apply_lighting_energy(state: dict, light_energy: float) -> None:
    target_energy = max(0.0, float(light_energy))
    point_light_name = state.get("point_light_name")
    point_light_obj = None if not point_light_name else bpy.data.objects.get(point_light_name)
    base_energy = state.get("point_energy")

    scale = 1.0
    if base_energy is not None:
        base_energy = float(base_energy)
        if base_energy > 1e-8:
            scale = target_energy / base_energy
        else:
            scale = 0.0

    if point_light_obj is not None:
        point_light_obj.data.energy = target_energy

    for node_name, base_value in state.get("world_strengths", []):
        sock = _resolve_world_strength_socket(node_name)
        if sock is not None:
            sock.default_value = float(base_value) * scale

    material_name = state.get("emissive_material_name")
    for node_name, base_value in state.get("emissive_strengths", []):
        sock = None if not material_name else _resolve_material_strength_socket(material_name, node_name)
        if sock is not None:
            sock.default_value = float(base_value) * scale


def restore_render_lighting_state(state: dict):
    point_light_name = state.get("point_light_name")
    point_light_obj = None if not point_light_name else bpy.data.objects.get(point_light_name)
    point_energy = state.get("point_energy")
    if point_light_obj is not None and point_energy is not None:
        point_light_obj.data.energy = float(point_energy)

    for node_name, base_value in state.get("world_strengths", []):
        sock = _resolve_world_strength_socket(node_name)
        if sock is not None:
            sock.default_value = float(base_value)

    material_name = state.get("emissive_material_name")
    for node_name, base_value in state.get("emissive_strengths", []):
        sock = None if not material_name else _resolve_material_strength_socket(material_name, node_name)
        if sock is not None:
            sock.default_value = float(base_value)


__all__ = [
    "build_room_and_lights",
    "build_rig_poses",
    "sample_effective_energy_config",
    "sample_render_sample_config",
    "capture_render_lighting_state",
    "apply_lighting_energy",
    "restore_render_lighting_state",
]
