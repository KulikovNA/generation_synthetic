from __future__ import annotations

import random
import uuid

import numpy as np
import bpy

from blendforge.blender_runtime.camera.ActiveStereoProjectorRuntime import (
    get_mount_world_pose,
    get_stream_world_pose_from_anchor,
    render_single_stream_with_projector,
    resolve_projector_runtime_config,
)
from blendforge.blender_runtime.camera import RealsenseProjectorUtility as projector_util


def _cfg_get(section, key: str, default):
    if section is None:
        return default
    if isinstance(section, dict):
        return section.get(key, default)
    return getattr(section, key, default)


def _sample_from_range(value, *, cast=float):
    if isinstance(value, (int, float)):
        return cast(value)
    vals = list(value)
    if len(vals) == 0:
        raise ValueError("Range must not be empty")
    if len(vals) == 1:
        return cast(vals[0])
    lo = vals[0]
    hi = vals[1]
    if cast is int:
        lo = int(round(lo))
        hi = int(round(hi))
        if hi < lo:
            lo, hi = hi, lo
        return int(np.random.randint(lo, hi + 1))
    lo = float(lo)
    hi = float(hi)
    if hi < lo:
        lo, hi = hi, lo
    return cast(np.random.uniform(lo, hi))


def _resolve_non_overlap_min_sep(cfg_section, dot_radius_px: float, requested_min_sep_px: float) -> tuple[float, dict]:
    enforce_non_overlap = bool(_cfg_get(cfg_section, "enforce_non_overlap", True))
    non_overlap_margin_px = float(_cfg_get(cfg_section, "non_overlap_margin_px", 0.0))
    drawn_radius_px = max(1, int(round(float(dot_radius_px))))

    effective_min_sep_px = float(requested_min_sep_px)
    if enforce_non_overlap:
        effective_min_sep_px = max(
            effective_min_sep_px,
            float(2 * drawn_radius_px) + float(non_overlap_margin_px),
        )

    info = {
        "enforce_non_overlap": enforce_non_overlap,
        "non_overlap_margin_px": float(non_overlap_margin_px),
        "drawn_radius_px": int(drawn_radius_px),
        "requested_min_sep_px": float(requested_min_sep_px),
        "effective_min_sep_px": float(effective_min_sep_px),
    }
    return float(effective_min_sep_px), info


def build_random_projector_config(rs, overrides, cfg) -> dict:
    base_cfg = resolve_projector_runtime_config(rs, overrides, fallback_left_stream="IR_LEFT").to_dict()
    section = getattr(cfg, "random_projector_pattern", None)

    seed_mode = str(_cfg_get(section, "seed_mode", "random")).strip().lower()
    if seed_mode == "fixed":
        pattern_seed = int(_cfg_get(section, "seed_value", 12345))
    elif seed_mode == "random":
        pattern_seed = int(random.SystemRandom().randrange(0, 2**31 - 1))
    else:
        raise ValueError(f"Unsupported random_projector_pattern.seed_mode: {seed_mode}")

    dot_count = _sample_from_range(_cfg_get(section, "dot_count_range", [4000, 7000]), cast=int)
    dot_radius_px = _sample_from_range(_cfg_get(section, "dot_radius_px_range", [1.0, 2.0]), cast=float)
    requested_min_sep_px = _sample_from_range(_cfg_get(section, "min_sep_px_range", [3.0, 5.0]), cast=float)
    min_sep_px, _non_overlap_info = _resolve_non_overlap_min_sep(section, dot_radius_px, requested_min_sep_px)

    dot_sigma_range = _cfg_get(section, "dot_sigma_px_range", None)
    dot_sigma_px = None
    if dot_sigma_range not in (None, ""):
        dot_sigma_px = _sample_from_range(dot_sigma_range, cast=float)

    base_cfg["pattern_path"] = None
    base_cfg["pattern_base_dir"] = None
    base_cfg["pattern_seed"] = int(pattern_seed)
    base_cfg["dot_count"] = int(dot_count)
    base_cfg["pattern_dot_radius_px"] = float(dot_radius_px)
    base_cfg["pattern_min_sep_px"] = float(min_sep_px)
    base_cfg["pattern_dot_sigma_px"] = dot_sigma_px

    return base_cfg


def render_stereo_branch_pair(
    rs,
    *,
    left_pose: np.ndarray,
    overrides,
    stereo_branch: str,
    cfg,
    output_dir: str | None = None,
    file_prefix_base: str | None = None,
):
    right_pose = get_stream_world_pose_from_anchor(rs, "IR_RIGHT", "IR_LEFT", left_pose)
    mount_world_pose = get_mount_world_pose(
        rs,
        anchor_stream="IR_LEFT",
        secondary_stream="IR_RIGHT",
        anchor_world_pose=left_pose,
    )

    render_uid = uuid.uuid4().hex[:12]
    mount_name = f"rs_projector_mount_{stereo_branch}_{render_uid}"
    socket_name = f"rs_projector_socket_{stereo_branch}_{render_uid}"
    mount = projector_util.get_or_create_mount_empty(mount_name)

    if stereo_branch == "effective":
        projector_cfg = resolve_projector_runtime_config(rs, overrides, fallback_left_stream="IR_LEFT").to_dict()
    elif stereo_branch == "random_pattern":
        projector_cfg = build_random_projector_config(rs, overrides, cfg)
    else:
        raise ValueError(f"Unsupported stereo_branch: {stereo_branch}")

    try:
        left_render = render_single_stream_with_projector(
            rs,
            "IR_LEFT",
            left_pose,
            mount,
            mount_world_pose,
            projector_cfg,
            socket_name=socket_name,
            output_dir=output_dir,
            file_prefix=file_prefix_base,
        )
        right_render = render_single_stream_with_projector(
            rs,
            "IR_RIGHT",
            right_pose,
            mount,
            mount_world_pose,
            projector_cfg,
            socket_name=socket_name,
            output_dir=output_dir,
            file_prefix=file_prefix_base,
        )
    finally:
        socket_obj = bpy.data.objects.get(socket_name)
        if socket_obj is not None:
            bpy.data.objects.remove(socket_obj, do_unlink=True)
        mount_obj = bpy.data.objects.get(mount_name)
        if mount_obj is not None:
            bpy.data.objects.remove(mount_obj, do_unlink=True)

    return {
        "left_colors": left_render["colors"],
        "right_colors": right_render["colors"],
        "left_depth_m": left_render["depth"],
        "left_world_pose": left_pose,
        "right_world_pose": right_pose,
        "mount_world_pose": mount_world_pose,
    }


__all__ = [
    "build_random_projector_config",
    "render_stereo_branch_pair",
]
