#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import blenderproc as bproc

import math
import numpy as np
import bpy

from mathutils import Matrix

from blendforge.blender_runtime.camera.RealsenseProfileLoader import RealSenseProfile
from blendforge.blender_runtime.camera.ProjectorPatternUtility import (
    apply_pattern_postprocess,
    load_or_generate_projector_pattern,
)

# -------------------- projector helpers --------------------

def get_or_create_mount_empty(name: str = "rs_projector_mount"):
    if name in bpy.data.objects:
        return bpy.data.objects[name]
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = "PLAIN_AXES"
    bpy.context.collection.objects.link(empty)
    return empty


def get_or_create_projector_socket(name: str = "rs_projector_socket"):
    if name in bpy.data.objects:
        return bpy.data.objects[name]
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = "SINGLE_ARROW"
    bpy.context.collection.objects.link(empty)
    return empty


def animate_mount_from_cam2world(empty_obj, cam2world_list: list[np.ndarray]):
    for i, T in enumerate(cam2world_list):
        empty_obj.matrix_world = Matrix(T.tolist())
        empty_obj.keyframe_insert(data_path="location", frame=i)
        empty_obj.keyframe_insert(data_path="rotation_euler", frame=i)


def fov_from_K(K: np.ndarray, W: int, H: int) -> tuple[float, float]:
    """
    Compute pinhole FOV (rad) from intrinsics.
    """
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    fov_h = 2.0 * math.atan(W / (2.0 * fx))
    fov_v = 2.0 * math.atan(H / (2.0 * fy))
    return fov_h, fov_v


def set_projector_fov(proj, fov_h_rad: float, fov_v_rad: float):
    """
    В BlenderProc 2.7.1 setup_as_projector() создаёт node tree с Value-узлами:
      label='Focal Length' и label='Focal Length * Ratio'
    Мы их переписываем под FOV ПРОЕКТОРА, чтобы проектор не зависел от камеры.
    """
    # В оригинале focal_length = 2*tan(fov/2), r = H/W, fr = focal_length*r
    # Для заданных (fov_h, fov_v) делаем r = tan(fov_v/2)/tan(fov_h/2)
    F = 2.0 * math.tan(float(fov_h_rad) / 2.0)
    denom = math.tan(float(fov_h_rad) / 2.0)
    if abs(denom) < 1e-12:
        ratio = 1.0
    else:
        ratio = math.tan(float(fov_v_rad) / 2.0) / denom

    f_node, fr_node = _get_projector_fov_nodes(proj)

    f_node.outputs[0].default_value = float(F)
    fr_node.outputs[0].default_value = float(F * ratio)

    # Blender SPOT uses a circular cone, so it must cover the diagonal of the
    # rectangular projector frustum or the 4 pattern corners get clipped.
    tan_h = math.tan(float(fov_h_rad) / 2.0)
    tan_v = math.tan(float(fov_v_rad) / 2.0)
    diag_full_angle = 2.0 * math.atan(math.sqrt(tan_h * tan_h + tan_v * tan_v))

    light_data = proj.blender_obj.data
    light_data.spot_size = float(min(math.pi, diag_full_angle * 1.01))
    light_data.spot_blend = 0.0
    light_data.shadow_soft_size = 0.0


def _get_projector_node_tree(proj):
    node_tree = getattr(proj.blender_obj.data, "node_tree", None)
    if node_tree is None:
        raise RuntimeError("Projector light has no node tree; setup_as_projector() likely failed.")
    return node_tree


def _find_value_node_by_label(proj, label: str):
    node_tree = _get_projector_node_tree(proj)
    matches = [
        n for n in node_tree.nodes
        if n.type == "VALUE" and getattr(n, "label", "") == label
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one projector VALUE node with label '{label}', found {len(matches)}."
        )
    return matches[0]


def _get_projector_fov_nodes(proj):
    return (
        _find_value_node_by_label(proj, "Focal Length"),
        _find_value_node_by_label(proj, "Focal Length * Ratio"),
    )


def _get_projector_texture_nodes(proj):
    node_tree = _get_projector_node_tree(proj)
    return [
        n for n in node_tree.nodes
        if n.type == "TEX_IMAGE" and getattr(n, "label", "") == "Texture Image"
    ]


def assign_unique_projector_pattern_image(proj, pattern_img: np.ndarray, *, image_name: str):
    tex_nodes = _get_projector_texture_nodes(proj)
    if len(tex_nodes) != 1:
        raise RuntimeError(
            f"Expected exactly one projector TEX_IMAGE node with label 'Texture Image', found {len(tex_nodes)}."
        )

    img = bpy.data.images.new(
        name=image_name,
        width=int(pattern_img.shape[1]),
        height=int(pattern_img.shape[0]),
        alpha=True,
    )
    pixels = np.asarray(pattern_img)
    if pixels.dtype == np.uint8:
        pixels = pixels.astype(np.float32) / 255.0
    else:
        pixels = pixels.astype(np.float32, copy=False)
        if float(pixels.max()) > 1.0:
            pixels = np.clip(pixels, 0.0, 255.0) / 255.0
        else:
            pixels = np.clip(pixels, 0.0, 1.0)
    img.pixels = pixels.ravel()

    # BlenderProc 2.7.1 binds bpy.data.images['pattern'] inside setup_as_projector(),
    # so replace that shared datablock with a projector-specific one.
    tex_nodes[0].image = img
    return img


def cleanup_projector_pattern_images(proj):
    for node in _get_projector_texture_nodes(proj):
        img = getattr(node, "image", None)
        node.image = None
        if img is not None and img.users == 0:
            bpy.data.images.remove(img)


def _cfg_get(config, key: str, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _retarget_projector_copy_transforms(proj, target_obj):
    constraints = getattr(proj.blender_obj, "constraints", None)
    if constraints is None:
        raise RuntimeError("Projector object has no constraints collection.")

    matches = [c for c in constraints if c.type == "COPY_TRANSFORMS"]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one COPY_TRANSFORMS constraint on projector, found {len(matches)}."
        )
    matches[0].target = target_obj


def _local_transform_components(T_mount_from_projector_blender: np.ndarray):
    M = Matrix(np.asarray(T_mount_from_projector_blender, dtype=np.float64).tolist())
    loc, rot, _scale = M.decompose()
    rot_euler = rot.to_euler("XYZ")
    return (
        np.asarray(loc[:], dtype=np.float64),
        np.asarray(rot_euler[:], dtype=np.float64),
    )


def attach_socket_to_mount(socket_obj, mount_obj, T_mount_from_projector_blender: np.ndarray):
    socket_obj.parent = mount_obj
    socket_obj.matrix_parent_inverse = Matrix.Identity(4)
    socket_obj.matrix_local = Matrix(np.asarray(T_mount_from_projector_blender, dtype=np.float64).tolist())
    return socket_obj


def resolve_projector_target(
    rs: RealSenseProfile,
    mount_obj,
    *,
    socket_name: str = "rs_projector_socket",
):
    """
    Legacy projector binding path used by the old single_depth_gen branch.

    Current effective/random runtime paths use create_projector_from_runtime_config()
    and pass the final local transform explicitly, so they do not need this helper.
    """
    if not rs.has_projector():
        zero = np.zeros(3, dtype=np.float64)
        return mount_obj, {
            "local_transform_applied": False,
            "projector_socket_name": mount_obj.name,
            "local_transform_mode": "legacy_fallback",
            "local_translation": zero.copy(),
            "local_rotation_euler_rad": zero.copy(),
        }

    T_mount_from_projector = rs.get_projector_local_transform_blender()
    socket_obj = get_or_create_projector_socket(socket_name)
    attach_socket_to_mount(socket_obj, mount_obj, T_mount_from_projector)

    local_translation, local_rotation_euler_rad = _local_transform_components(T_mount_from_projector)
    mode = "explicit" if rs.has_projector_local_transform() else "midpoint"

    return socket_obj, {
        "local_transform_applied": True,
        "projector_socket_name": socket_obj.name,
        "local_transform_mode": mode,
        "local_translation": local_translation,
        "local_rotation_euler_rad": local_rotation_euler_rad,
    }


def create_projector_from_runtime_config(
    mount_obj,
    config,
    *,
    socket_name: str = "rs_projector_socket_runtime",
):
    pattern_img = load_or_generate_projector_pattern(
        path=_cfg_get(config, "pattern_path"),
        width=int(_cfg_get(config, "pattern_width")),
        height=int(_cfg_get(config, "pattern_height")),
        dot_count=int(_cfg_get(config, "dot_count")),
        min_sep_px=_cfg_get(config, "pattern_min_sep_px"),
        dot_radius_px=_cfg_get(config, "pattern_dot_radius_px"),
        seed=_cfg_get(config, "pattern_seed"),
        base_dir=_cfg_get(config, "pattern_base_dir"),
    )
    pattern_img = apply_pattern_postprocess(
        pattern_img,
        flip_u=bool(_cfg_get(config, "flip_u", False)),
        flip_v=bool(_cfg_get(config, "flip_v", False)),
        dot_sigma_px=_cfg_get(config, "pattern_dot_sigma_px"),
    )

    proj = bproc.types.Light()
    proj.set_type("SPOT")
    proj.set_energy(float(_cfg_get(config, "energy")))
    proj.setup_as_projector(pattern_img)
    assign_unique_projector_pattern_image(
        proj,
        pattern_img,
        image_name=f"{socket_name}_pattern",
    )

    T_mount_from_projector = _cfg_get(config, "local_transform_blender_4x4_final", None)
    if T_mount_from_projector is None:
        T_mount_from_projector = _cfg_get(config, "local_transform_blender_4x4_base", None)

    target_obj = mount_obj
    if T_mount_from_projector is not None:
        socket_obj = get_or_create_projector_socket(socket_name)
        attach_socket_to_mount(socket_obj, mount_obj, np.asarray(T_mount_from_projector, dtype=np.float64))
        target_obj = socket_obj

    _retarget_projector_copy_transforms(proj, target_obj)
    set_projector_fov(
        proj,
        math.radians(float(_cfg_get(config, "fov_h_deg"))),
        math.radians(float(_cfg_get(config, "fov_v_deg"))),
    )
    return proj, pattern_img


def create_realsense_ir_projector(
    rs: RealSenseProfile,
    mount_obj,
    *,
    dots: int | None = None,
    energy: float | None = None,
):
    """
    Legacy projector factory used by the old single_depth_gen branch.

    Effective/random runtime paths should use create_projector_from_runtime_config()
    so projector geometry, pattern source, flips, blur, and local mount transform
    all flow through one explicit runtime config.
    """
    # --- get projector params from JSON (preferred) or fallback ---
    if hasattr(rs, "has_projector") and rs.has_projector():
        pr = rs.get_projector()
        fov_h = pr.fov_h_rad
        fov_v = pr.fov_v_rad
        pat_w = int(pr.pattern_w)
        pat_h = int(pr.pattern_h)
        dots_v = int(pr.dot_count if dots is None else dots)
        energy_v = float(pr.energy if energy is None else energy)
        pattern_path = getattr(pr, "pattern_path", None)
        pattern_seed = getattr(pr, "pattern_seed", None)
        pattern_min_sep_px = getattr(pr, "pattern_min_sep_px", None)
        pattern_dot_radius_px = getattr(pr, "pattern_dot_radius_px", None)
        pattern_dot_sigma_px = getattr(pr, "pattern_dot_sigma_px", None)
        projector_metadata = getattr(pr, "metadata", {}) or {}
        flip_u = bool(projector_metadata.get("flip_u", False))
        flip_v = bool(projector_metadata.get("flip_v", False))
    else:
        # fallback: take IR_LEFT intrinsics as proxy (not desired, but keeps compatibility)
        s = rs.get_stream("IR_LEFT")
        fov_h, fov_v = fov_from_K(s.K, s.width, s.height)
        pat_w, pat_h = s.width, s.height
        dots_v = int(25600 if dots is None else dots)
        energy_v = float(3000.0 if energy is None else energy)
        pattern_path = None
        pattern_seed = 0
        pattern_min_sep_px = None
        pattern_dot_radius_px = 1.0
        pattern_dot_sigma_px = None
        flip_u = False
        flip_v = False

    # pattern resolution is independent from render resolution
    pattern_img = load_or_generate_projector_pattern(
        path=pattern_path,
        width=pat_w,
        height=pat_h,
        dot_count=dots_v,
        min_sep_px=pattern_min_sep_px,
        dot_radius_px=pattern_dot_radius_px,
        seed=pattern_seed,
        base_dir=getattr(rs, "profile_base_dir", None),
    )
    pattern_img = apply_pattern_postprocess(
        pattern_img,
        flip_u=flip_u,
        flip_v=flip_v,
        dot_sigma_px=pattern_dot_sigma_px,
    )

    proj = bproc.types.Light()
    proj.set_type("SPOT")
    proj.set_energy(float(energy_v))

    # Let BlenderProc build the correct node tree + texture wiring.
    # It will *initially* use camera fov, but then we override the nodes.
    proj.setup_as_projector(pattern_img)

    # Re-target COPY_TRANSFORMS constraint to mount/socket, not the active camera
    target_obj, _binding = resolve_projector_target(rs, mount_obj)
    _retarget_projector_copy_transforms(proj, target_obj)

    # Now: override projector fov in nodes -> decouple from camera completely
    set_projector_fov(proj, fov_h, fov_v)

    return proj
