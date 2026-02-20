#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import blenderproc as bproc

import math
import numpy as np
import bpy

from mathutils import Matrix

from blendforge.blender_runtime.camera.RealsenseProfileLoader import RealSenseProfile

# -------------------- projector helpers --------------------

def get_or_create_mount_empty(name: str = "rs_projector_mount"):
    if name in bpy.data.objects:
        return bpy.data.objects[name]
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = "PLAIN_AXES"
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

    nt = proj.blender_obj.data.node_tree
    nodes = nt.nodes

    f_node = None
    fr_node = None
    for n in nodes:
        if n.type == "VALUE" and getattr(n, "label", "") == "Focal Length":
            f_node = n
        if n.type == "VALUE" and getattr(n, "label", "") == "Focal Length * Ratio":
            fr_node = n

    if f_node is None or fr_node is None:
        # Если BlenderProc поменял имена/labels — лучше упасть явно
        raise RuntimeError("Projector node-tree does not contain expected Value nodes "
                           "('Focal Length', 'Focal Length * Ratio').")

    f_node.outputs[0].default_value = float(F)
    fr_node.outputs[0].default_value = float(F * ratio)

    # Геометрия cone (spot_size) тоже должна соответствовать проектору
    # Берём чуть больше максимального угла, чтобы не резать края cookie.
    proj.blender_obj.data.spot_size = float(max(fov_h_rad, fov_v_rad) * 1.02)


def create_realsense_ir_projector(
    rs: RealSenseProfile,
    mount_obj,
    *,
    dots: int | None = None,
    energy: float | None = None,
):
    """
    Создаёт SPOT-проектор с cookie-паттерном.
    Главное: FOV проектора берётся ИЗ JSON (rs.projector), а не из камеры.
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
    else:
        # fallback: take IR_LEFT intrinsics as proxy (not desired, but keeps compatibility)
        s = rs.get_stream("IR_LEFT")
        fov_h, fov_v = fov_from_K(s.K, s.width, s.height)
        pat_w, pat_h = s.width, s.height
        dots_v = int(25600 if dots is None else dots)
        energy_v = float(3000.0 if energy is None else energy)

    # pattern resolution is independent from render resolution
    pattern_img = bproc.utility.generate_random_pattern_img(pat_w, pat_h, dots_v)

    proj = bproc.types.Light()
    proj.set_type("SPOT")
    proj.set_energy(float(energy_v))

    # Let BlenderProc build the correct node tree + texture wiring.
    # It will *initially* use camera fov, but then we override the nodes.
    proj.setup_as_projector(pattern_img)

    # Re-target COPY_TRANSFORMS constraint to mount (rig), not the active camera
    c = proj.blender_obj.constraints.get("Copy Transforms")
    if c is not None:
        c.target = mount_obj
    else:
        # fallback: search any COPY_TRANSFORMS constraint
        for cc in proj.blender_obj.constraints:
            if cc.type == "COPY_TRANSFORMS":
                cc.target = mount_obj
                break

    # Now: override projector fov in nodes -> decouple from camera completely
    set_projector_fov(proj, fov_h, fov_v)

    return proj
