#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import blenderproc as bproc  # <-- ВАЖНО: первый импорт

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import cv2

from blendforge.host.FiletoDict import Config
from blendforge.blender_runtime.CustomLightSetting import TemperatureToRGBConverter
from blendforge.blender_runtime.CustomMaterial import make_random_material
from blendforge.blender_runtime.CustomLoadMesh import load_objs
from blendforge.blender_runtime.utils import (
    sample_pose_func_drop,
    sample_pose_func,
    build_lookat_pose_cam,
)

from blendforge.blender_runtime.camera.RealsenseProfileLoader import RealSenseProfile
from blendforge.blender_runtime.camera.RealsenseProjectorUtility import (
    get_or_create_mount_empty,
    animate_mount_from_cam2world,
    create_realsense_ir_projector,
)

# если функция у тебя переехала в StereoPipline.py — поменяй import здесь
from blendforge.blender_runtime.stereo.StereoMatching import stereo_global_matching_rectified
from blendforge.blender_runtime.stereo.DepthAlignment import align_depth_series_ir_left_to_color

from blendforge.debug_blender_runtime.EvalDepth import (
    aggregate_depth_metrics,
    to_frame_list,
    valid_depth_mask,
)


# -------------------- args --------------------

def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Stage 1A: sweep only core SGBM search geometry (block_size, min_disparity, num_disparities, preprocess)"
    )
    p.add_argument("--config_file", type=str, required=True)
    return p.parse_args(argv)


# -------------------- small utils --------------------

def _np5(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    out = np.zeros(5, dtype=np.float64)
    if a.size:
        out[: min(5, a.size)] = a[: min(5, a.size)]
    return out


def _json_dump(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _value_tag(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        s = f"{float(v):.6f}".rstrip("0").rstrip(".")
        return s.replace(".", "p")
    return str(v).replace(" ", "_")


def _to_bgr_u8(img) -> np.ndarray:
    a = np.asarray(img)

    if a.ndim == 2:
        x = a.astype(np.float32)
        if x.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)
        if np.issubdtype(x.dtype, np.floating) and x.max() <= 1.5:
            x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
        return x

    if a.ndim != 3:
        raise ValueError(f"Unsupported image shape: {a.shape}")

    if a.shape[2] == 4:
        a = a[..., :3]

    x = a.astype(np.float32)
    if x.max() <= 1.5:
        x = x * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)

    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def _depth_to_u16(depth_m, depth_scale_mm: float) -> np.ndarray:
    d = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0.0)
    out = np.zeros_like(d, dtype=np.float32)
    out[valid] = d[valid] * 1000.0 / float(depth_scale_mm)
    return np.clip(out, 0.0, 65535.0).astype(np.uint16)


def _depth_heatmap(depth_m, depth_min_m: float, depth_max_m: float) -> np.ndarray:
    d = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0.0)

    norm = np.zeros_like(d, dtype=np.float32)
    den = max(float(depth_max_m) - float(depth_min_m), 1e-6)
    norm[valid] = (d[valid] - float(depth_min_m)) / den
    norm = np.clip(norm, 0.0, 1.0)

    gray = (norm * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def _disp_heatmap(disp_px) -> np.ndarray:
    d = np.asarray(disp_px, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0.0)

    gray = np.zeros_like(d, dtype=np.uint8)
    if np.any(valid):
        v = d[valid]
        lo = float(np.percentile(v, 2.0))
        hi = float(np.percentile(v, 98.0))
        if hi <= lo:
            hi = lo + 1e-6
        x = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
        gray = (x * 255.0).astype(np.uint8)

    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def _dedupe_keep_order(values):
    uniq = []
    seen = set()
    for v in values:
        key = (type(v).__name__, str(v))
        if key not in seen:
            uniq.append(v)
            seen.add(key)
    return uniq


def _normalize_sgbm_mode_value(x):
    if isinstance(x, str):
        s = x.strip().upper()
        aliases = {
            "MODE_SGBM": "SGBM",
            "MODE_HH": "HH",
            "MODE_SGBM_3WAY": "SGBM_3WAY",
            "MODE_HH4": "HH4",
            "3WAY": "SGBM_3WAY",
        }
        return aliases.get(s, s)
    return x


def _resolve_sgbm_mode(x) -> int:
    if isinstance(x, str):
        s = _normalize_sgbm_mode_value(x)
        if s == "SGBM":
            return int(cv2.STEREO_SGBM_MODE_SGBM)
        if s == "HH":
            return int(cv2.STEREO_SGBM_MODE_HH)
        if s == "SGBM_3WAY" and hasattr(cv2, "STEREO_SGBM_MODE_SGBM_3WAY"):
            return int(cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        if s == "HH4" and hasattr(cv2, "STEREO_SGBM_MODE_HH4"):
            return int(cv2.STEREO_SGBM_MODE_HH4)
        raise ValueError(f"Unknown sgbm_mode: {x}")
    return int(x)


def _snapshot_params_for_json(params: dict) -> dict:
    return {
        "depth_min_m": float(params["depth_min_m"]),
        "depth_max_m": float(params["depth_max_m"]),
        "block_size": int(params["block_size"]),
        "num_disparities": int(params["num_disparities"]),
        "min_disparity": int(params["min_disparity"]),
        "preprocess": str(params["preprocess"]),

        # fixed at stage 1A
        "sgbm_mode": str(params["sgbm_mode"]),
        "uniqueness_ratio": int(params["uniqueness_ratio"]),
        "disp12_max_diff": int(params["disp12_max_diff"]),
        "pre_filter_cap": int(params["pre_filter_cap"]),
        "p1_scale": float(params["p1_scale"]),
        "p2_scale": float(params["p2_scale"]),

        # filters fixed off
        "use_wls": bool(params["use_wls"]),
        "lr_check": bool(params["lr_check"]),
        "lr_thresh_px": float(params["lr_thresh_px"]),
        "lr_min_keep_ratio": float(params["lr_min_keep_ratio"]),
        "speckle_filter": bool(params["speckle_filter"]),
        "fill_mode": str(params["fill_mode"]),
        "fill_iters": int(params["fill_iters"]),
        "depth_completion": bool(params["depth_completion"]),

        # align fixed
        "align_depth_value_mode": str(params["align_depth_value_mode"]),
        "align_splat_2x2": bool(params["align_splat_2x2"]),
        "align_rectify_mode": str(params["align_rectify_mode"]),

        # eval
        "edge_dilation_px": int(params.get("edge_dilation_px", 1)),
        "edge_percentile": float(params.get("edge_percentile", 85.0)),
    }


def _normalize_experiment_params(params: dict) -> dict:
    p = dict(params)

    p["block_size"] = max(3, int(p["block_size"]))
    if p["block_size"] % 2 == 0:
        p["block_size"] += 1

    nd = max(16, int(p["num_disparities"]))
    nd = (nd // 16) * 16
    if nd < 16:
        nd = 16
    p["num_disparities"] = nd

    p["min_disparity"] = int(p["min_disparity"])
    p["preprocess"] = str(p["preprocess"])

    # fixed advanced SGBM on stage 1A
    p["sgbm_mode"] = _normalize_sgbm_mode_value(p["sgbm_mode"])
    p["uniqueness_ratio"] = max(0, int(p["uniqueness_ratio"]))
    p["disp12_max_diff"] = int(p["disp12_max_diff"])
    p["pre_filter_cap"] = max(1, int(p["pre_filter_cap"]))
    p["p1_scale"] = max(1e-6, float(p["p1_scale"]))
    p["p2_scale"] = float(p["p2_scale"])
    if p["p2_scale"] <= p["p1_scale"]:
        raise ValueError("For stage 1A fixed params: p2_scale must be > p1_scale")

    return p


def _build_rectify_from_rs(
    rs: RealSenseProfile,
    left: str = "IR_LEFT",
    right: str = "IR_RIGHT",
    *,
    use_distortion: bool = False,
):
    sL = rs.get_stream(left)
    sR = rs.get_stream(right)

    K1 = np.asarray(sL.K, dtype=np.float64)
    K2 = np.asarray(sR.K, dtype=np.float64)

    if use_distortion:
        D1 = _np5(getattr(sL, "distortion_coeffs", [0, 0, 0, 0, 0]))
        D2 = _np5(getattr(sR, "distortion_coeffs", [0, 0, 0, 0, 0]))
    else:
        D1 = np.zeros(5, dtype=np.float64)
        D2 = np.zeros(5, dtype=np.float64)

    T_c_l = np.asarray(rs.get_T_cv(left, "COLOR"), dtype=np.float64)
    T_c_r = np.asarray(rs.get_T_cv(right, "COLOR"), dtype=np.float64)

    T_r_l = np.linalg.inv(T_c_r) @ T_c_l
    R = T_r_l[:3, :3].copy()
    t = T_r_l[:3, 3].copy()

    return {
        "K_left": K1,
        "D_left": D1,
        "K_right": K2,
        "D_right": D2,
        "R": R,
        "t": t,
        "alpha": 0.0,
        "image_size": (int(sL.width), int(sL.height)),
    }


def _save_debug_images(
    experiment_dir: Path,
    rgb_frames,
    ir_left_frames,
    ir_right_frames,
    depth_pred_rgb_frames,
    depth_gt_rgb_frames,
    disp_rect_frames,
    *,
    depth_min_m: float,
    depth_max_m: float,
    depth_scale_mm: float,
    debug_max_frames=None,
):
    n = len(depth_pred_rgb_frames)
    if debug_max_frames is not None:
        n = min(n, int(debug_max_frames))

    for i in range(n):
        frame_dir = experiment_dir / f"frame_{i:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        rgb = rgb_frames[i]
        ir_l = ir_left_frames[i]
        ir_r = ir_right_frames[i]
        depth_pred = depth_pred_rgb_frames[i]
        depth_gt = depth_gt_rgb_frames[i]
        disp = disp_rect_frames[i]

        valid_pred = (valid_depth_mask(depth_pred, depth_min_m, depth_max_m).astype(np.uint8) * 255)
        valid_gt = (valid_depth_mask(depth_gt, depth_min_m, depth_max_m).astype(np.uint8) * 255)

        cv2.imwrite(str(frame_dir / "rgb.jpg"), _to_bgr_u8(rgb))
        cv2.imwrite(str(frame_dir / "ir_left.jpg"), _to_bgr_u8(ir_l))
        cv2.imwrite(str(frame_dir / "ir_right.jpg"), _to_bgr_u8(ir_r))
        cv2.imwrite(str(frame_dir / "depth_pred_u16.png"), _depth_to_u16(depth_pred, depth_scale_mm))
        cv2.imwrite(str(frame_dir / "depth_gt_u16.png"), _depth_to_u16(depth_gt, depth_scale_mm))
        cv2.imwrite(str(frame_dir / "depth_pred_heatmap.png"), _depth_heatmap(depth_pred, depth_min_m, depth_max_m))
        cv2.imwrite(str(frame_dir / "depth_gt_heatmap.png"), _depth_heatmap(depth_gt, depth_min_m, depth_max_m))
        cv2.imwrite(str(frame_dir / "disp_rect_heatmap.png"), _disp_heatmap(disp))
        cv2.imwrite(str(frame_dir / "valid_pred_mask.png"), valid_pred)
        cv2.imwrite(str(frame_dir / "valid_gt_mask.png"), valid_gt)


def _num_disp_values_around(base: int, delta_steps=(-32, -16, 0, 16, 32)):
    vals = []
    for d in delta_steps:
        v = int(base) + int(d)
        v = max(16, v)
        v = (v // 16) * 16
        if v < 16:
            v = 16
        vals.append(v)
    return sorted(set(vals))


def _cfg_list_or_default(cfg, attr_name: str, default_values):
    x = getattr(cfg, attr_name, None)
    if x is None:
        return list(default_values)
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _build_stage1a_sweeps(cfg, base_params: dict):
    """
    Stage 1A:
    ONLY core search geometry:
      - block_size
      - min_disparity
      - num_disparities
      - preprocess
    Everything else is fixed.
    """
    sweeps = {}

    sweeps["block_size"] = [
        int(v) for v in _cfg_list_or_default(
            cfg,
            "sweep_block_size_values",
            [5, 7, 9, 11],
        )
    ]

    sweeps["min_disparity"] = [
        int(v) for v in _cfg_list_or_default(
            cfg,
            "sweep_min_disparity_values",
            [-2, 0, 2],
        )
    ]

    sweeps["num_disparities"] = [
        int(v) for v in _cfg_list_or_default(
            cfg,
            "sweep_num_disparities_values",
            _num_disp_values_around(int(base_params["num_disparities"]), delta_steps=(-32, -16, 0, 16)),
        )
    ]

    sweeps["preprocess"] = [
        str(v) for v in _cfg_list_or_default(
            cfg,
            "sweep_preprocess_values",
            ["clahe", "none"],
        )
    ]

    for k, vals in list(sweeps.items()):
        sweeps[k] = _dedupe_keep_order(vals)

    return sweeps


def _run_one_experiment(
    *,
    rs: RealSenseProfile,
    rectify_dict: dict,
    stereo_frames,
    depth_gt_ir_frames,
    depth_gt_rgb_frames,
    rgb_frames,
    ir_left_frames,
    ir_right_frames,
    output_root: Path,
    params: dict,
    param_name: str,
    param_value,
    depth_scale_mm: float,
    use_geom_mask_from_gt: bool,
    debug_max_frames=None,
):
    params = _normalize_experiment_params(params)

    exp_tag = f"{param_name}__{_value_tag(param_value)}"
    exp_dir = output_root / param_name / exp_tag
    exp_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    depth_rect_m, disp_rect_px = stereo_global_matching_rectified(
        stereo_frames,
        rectify=rectify_dict,
        depth_min=float(params["depth_min_m"]),
        depth_max=float(params["depth_max_m"]),
        depth_range_policy="zero",

        # swept at stage 1A
        block_size=int(params["block_size"]),
        num_disparities=int(params["num_disparities"]),
        min_disparity=int(params["min_disparity"]),
        preprocess=str(params["preprocess"]),

        # fixed at stage 1A
        sgbm_mode=_resolve_sgbm_mode(params["sgbm_mode"]),
        uniqueness_ratio=int(params["uniqueness_ratio"]),
        disp12_max_diff=int(params["disp12_max_diff"]),
        pre_filter_cap=int(params["pre_filter_cap"]),
        p1_scale=float(params["p1_scale"]),
        p2_scale=float(params["p2_scale"]),

        # filters forced off
        use_wls=bool(params["use_wls"]),
        lr_check=bool(params["lr_check"]),
        lr_thresh_px=float(params["lr_thresh_px"]),
        lr_min_keep_ratio=float(params["lr_min_keep_ratio"]),
        speckle_filter=bool(params["speckle_filter"]),
        fill_mode=str(params["fill_mode"]),
        fill_iters=int(params["fill_iters"]),
        depth_completion=bool(params["depth_completion"]),

        depth_gt_frames=depth_gt_ir_frames if (use_geom_mask_from_gt and depth_gt_ir_frames is not None) else None,
        use_geom_mask_from_gt=bool(use_geom_mask_from_gt),
        border_pad=True,
        pad_left=None,
    )

    depth_pred_rgb_m, _ = align_depth_series_ir_left_to_color(
        rs,
        depth_rect_m,
        depth_value_mode=str(params["align_depth_value_mode"]),
        splat_2x2=bool(params["align_splat_2x2"]),
        rectify_mode=str(params["align_rectify_mode"]),
    )

    depth_pred_rgb_frames = to_frame_list(depth_pred_rgb_m)
    disp_rect_frames = to_frame_list(disp_rect_px)

    aggregate_metrics, per_frame_metrics = aggregate_depth_metrics(
        depth_pred_rgb_frames,
        depth_gt_rgb_frames,
        depth_min_m=float(params["depth_min_m"]),
        depth_max_m=float(params["depth_max_m"]),
        edge_dilation_px=int(params.get("edge_dilation_px", 1)),
        edge_percentile=float(params.get("edge_percentile", 85.0)),
    )

    _save_debug_images(
        experiment_dir=exp_dir,
        rgb_frames=rgb_frames,
        ir_left_frames=ir_left_frames,
        ir_right_frames=ir_right_frames,
        depth_pred_rgb_frames=depth_pred_rgb_frames,
        depth_gt_rgb_frames=depth_gt_rgb_frames,
        disp_rect_frames=disp_rect_frames,
        depth_min_m=float(params["depth_min_m"]),
        depth_max_m=float(params["depth_max_m"]),
        depth_scale_mm=float(depth_scale_mm),
        debug_max_frames=debug_max_frames,
    )

    elapsed_s = time.time() - t0
    params_json = _snapshot_params_for_json(params)

    metrics_payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "stage1a_sgbm_core_geom",
        "param_name": param_name,
        "param_value": param_value,
        "experiment_dir": str(exp_dir),
        "params": params_json,
        "aggregate_metrics": aggregate_metrics,
        "per_frame_metrics": per_frame_metrics,
        "elapsed_sec": float(elapsed_s),
    }

    _json_dump(exp_dir / "metrics.json", metrics_payload)

    summary_record = {
        "timestamp_utc": metrics_payload["timestamp_utc"],
        "stage": metrics_payload["stage"],
        "param_name": param_name,
        "param_value": param_value,
        "experiment_dir": str(exp_dir),
        "params": params_json,
        "aggregate_metrics": aggregate_metrics,
        "elapsed_sec": float(elapsed_s),
    }

    return metrics_payload, summary_record


# -------------------- scene helpers --------------------

def _build_room_and_lights(cfg):
    room_planes = [
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1]),
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[0, -3, 3], rotation=[-1.570796, 0, 0]),
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[0,  3, 3], rotation=[ 1.570796, 0, 0]),
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[3,  0, 3], rotation=[0, -1.570796, 0]),
        bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[-3, 0, 3], rotation=[0,  1.570796, 0]),
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
    return light_point


def _prepare_objects(cfg):
    sampled_objs = load_objs(
        bop_dataset_path=os.path.join(cfg.dataset_parent_path, cfg.bop_dataset_name),
        mm2m=None,
        sample_objects=False,
        num_of_objs_to_sample=9,
        additional_scale=None,
        manifold=True,
        object_model_unit="mm",
    )

    chosen_pose_func = sample_pose_func_drop if np.random.rand() < cfg.probability_drop else sample_pose_func

    bproc.object.sample_poses(
        objects_to_sample=sampled_objs,
        sample_pose_func=chosen_pose_func,
        max_tries=1000,
    )

    for j, obj in enumerate(sampled_objs):
        obj.enable_rigidbody(True, friction=100.0, linear_damping=0.99, angular_damping=0.99)
        obj.set_shading_mode("auto")

        mat, _style = make_random_material(allowed=["plastic_new"], name_prefix=f"obj_{j:06d}")
        mats = obj.get_materials()
        if not mats:
            obj.set_material(0, mat)
        else:
            for i in range(len(mats)):
                obj.set_material(i, mat)

    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=3,
        max_simulation_time=35,
        check_object_interval=2,
        substeps_per_frame=30,
        solver_iters=30,
    )


def _build_rig_poses(cfg, rs: RealSenseProfile, pose_count=None):
    rig_poses_ir_left: list[np.ndarray] = []
    rotation_factor = 9.0

    n_poses = int(pose_count if pose_count is not None else cfg.poses_cam)

    for _ in range(n_poses):
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


def _pick_mount_poses(rs: RealSenseProfile, rig_poses_ir_left, rig_poses_ir_right, rig_poses_color):
    mount_frame = "IR_LEFT"
    if hasattr(rs, "has_projector") and rs.has_projector():
        if hasattr(rs, "get_projector_mount_frame"):
            mount_frame = rs.get_projector_mount_frame()

    if mount_frame in ("IR_LEFT", "DEPTH"):
        return rig_poses_ir_left
    if mount_frame == "IR_RIGHT":
        return rig_poses_ir_right
    if mount_frame == "COLOR":
        return rig_poses_color
    return rig_poses_ir_left


# -------------------- main --------------------

def main(argv=None):
    bproc.init()
    bproc.utility.reset_keyframes()

    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    cfg = Config(args.config_file)

    seed = getattr(cfg, "np_random_seed", None)
    if seed is not None:
        np.random.seed(int(seed))

    rs = RealSenseProfile.from_json(cfg.camera_path)

    max_spp = int(np.random.randint(300, 1000)) if getattr(cfg, "max_amount_of_samples", None) is None else int(cfg.max_amount_of_samples)
    bproc.renderer.set_max_amount_of_samples(max_spp)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_normals_output()

    light_point = _build_room_and_lights(cfg)
    proj = None
    mount = None

    try:
        _prepare_objects(cfg)

        pose_count = int(getattr(cfg, "sweep_pose_count", 50))
        rig_poses_ir_left, rig_poses_ir_right, rig_poses_color = _build_rig_poses(cfg, rs, pose_count=pose_count)

        output_dir = Path(getattr(cfg, "output_dir", "test"))
        output_dir.mkdir(parents=True, exist_ok=True)

        depth_scale_mm = float(getattr(cfg, "depth_scale_mm", 1.0))
        debug_max_frames = getattr(cfg, "debug_max_frames", None)

        depth_min_m = float(rs.depth_min_m)
        depth_max_m = float(rs.depth_max_m)

        # -------------------- Stage 1A baseline: ONLY core geometry --------------------
        block_size = int(getattr(cfg, "sgm_block_size", getattr(cfg, "sgm_window_size", 7)))
        min_disparity = int(getattr(cfg, "sgm_min_disparity", 0))
        preprocess = str(getattr(cfg, "stereo_preprocess", "clahe"))

        # fixed advanced params for this stage
        sgbm_mode = _normalize_sgbm_mode_value(getattr(cfg, "sgm_mode", "SGBM"))
        uniqueness_ratio = int(getattr(cfg, "sgm_uniqueness_ratio", 10))
        disp12_max_diff = int(getattr(cfg, "sgm_disp12_max_diff", 1))
        pre_filter_cap = int(getattr(cfg, "sgm_pre_filter_cap", 63))
        p1_scale = float(getattr(cfg, "sgm_p1_scale", 8.0))
        p2_scale = float(getattr(cfg, "sgm_p2_scale", 32.0))

        # filters forced OFF
        use_wls = False
        lr_check = False
        lr_thresh_px = float(getattr(cfg, "lr_thresh_px", 1.0))
        lr_min_keep_ratio = float(getattr(cfg, "lr_min_keep_ratio", 0.02))
        speckle_filter = False
        fill_mode = "none"
        fill_iters = 0
        depth_completion = False

        use_geom_mask_from_gt = bool(getattr(cfg, "use_geom_mask_from_gt", True))

        rectify_use_distortion = False
        rectify_dict = _build_rectify_from_rs(
            rs, left="IR_LEFT", right="IR_RIGHT", use_distortion=rectify_use_distortion
        )

        mount = get_or_create_mount_empty("rs_projector_mount")
        mount_poses = _pick_mount_poses(rs, rig_poses_ir_left, rig_poses_ir_right, rig_poses_color)

        # PASS 1: IR_LEFT
        bproc.utility.reset_keyframes()
        rs.set_bproc_intrinsics("IR_LEFT")
        for i, T in enumerate(rig_poses_ir_left):
            bproc.camera.add_camera_pose(T, frame=i)

        animate_mount_from_cam2world(mount, mount_poses)
        proj = create_realsense_ir_projector(rs, mount, dots=None, energy=None)
        data_ir_left = bproc.renderer.render()

        # PASS 2: IR_RIGHT
        bproc.utility.reset_keyframes()
        rs.set_bproc_intrinsics("IR_RIGHT")
        for i, T in enumerate(rig_poses_ir_right):
            bproc.camera.add_camera_pose(T, frame=i)

        animate_mount_from_cam2world(mount, mount_poses)
        data_ir_right = bproc.renderer.render()

        proj.delete()
        proj = None

        if hasattr(rs, "set_bproc_stereo_from_ir"):
            rs.set_bproc_stereo_from_ir(convergence_distance=1.0)
        else:
            bproc.camera.set_stereo_parameters("PARALLEL", 1.0, float(rs.baseline_m))

        ir_left_frames = data_ir_left["colors"]
        ir_right_frames = data_ir_right["colors"]
        stereo_frames = [np.stack([a, b], axis=0) for a, b in zip(ir_left_frames, ir_right_frames)]

        depth_gt_ir_frames = data_ir_left.get("depth", None)
        if depth_gt_ir_frames is None:
            raise RuntimeError("IR_LEFT render did not return GT depth, but sweep requires GT comparison.")

        num_disparities = int(getattr(
            cfg,
            "sgm_num_disparities",
            rs.recommend_num_disparities(
                z_min_m=rs.depth_min_m,
                min_disparity=min_disparity,
                stream="IR_LEFT",
            )
        ))
        num_disparities = max(16, (int(num_disparities) // 16) * 16)

        print(
            "[Stage 1A] depth_range_m:",
            f"min={depth_min_m:.3f}, max={depth_max_m:.3f} | "
            f"block_size={block_size}, min_disp={min_disparity}, num_disp={num_disparities}, preprocess={preprocess} | "
            f"sgbm_mode={sgbm_mode} (fixed), uniqueness={uniqueness_ratio} (fixed), "
            f"disp12={disp12_max_diff} (fixed), preFilterCap={pre_filter_cap} (fixed), "
            f"p1_scale={p1_scale} (fixed), p2_scale={p2_scale} (fixed) | "
            f"filters=OFF | poses={pose_count}"
        )

        # PASS 3: RGB
        bproc.utility.reset_keyframes()
        rs.set_bproc_intrinsics("COLOR")
        for i, T in enumerate(rig_poses_color):
            bproc.camera.add_camera_pose(T, frame=i)

        data_rgb = bproc.renderer.render()
        rgb_frames = data_rgb["colors"]

        depth_gt_rgb_m, _ = align_depth_series_ir_left_to_color(
            rs,
            depth_gt_ir_frames,
            depth_value_mode="source_z",
            splat_2x2=True,
            rectify_mode="off",
        )
        depth_gt_rgb_frames = to_frame_list(depth_gt_rgb_m)

        base_params = {
            "depth_min_m": depth_min_m,
            "depth_max_m": depth_max_m,

            # swept at stage 1A
            "block_size": block_size,
            "num_disparities": num_disparities,
            "min_disparity": min_disparity,
            "preprocess": preprocess,

            # fixed advanced params
            "sgbm_mode": sgbm_mode,
            "uniqueness_ratio": uniqueness_ratio,
            "disp12_max_diff": disp12_max_diff,
            "pre_filter_cap": pre_filter_cap,
            "p1_scale": p1_scale,
            "p2_scale": p2_scale,

            # filters forced off
            "use_wls": use_wls,
            "lr_check": lr_check,
            "lr_thresh_px": lr_thresh_px,
            "lr_min_keep_ratio": lr_min_keep_ratio,
            "speckle_filter": speckle_filter,
            "fill_mode": fill_mode,
            "fill_iters": fill_iters,
            "depth_completion": depth_completion,

            # align fixed from previous conclusion
            "align_depth_value_mode": "source_z",
            "align_splat_2x2": True,
            "align_rectify_mode": "on",

            # eval
            "edge_dilation_px": int(getattr(cfg, "eval_edge_dilation_px", 1)),
            "edge_percentile": float(getattr(cfg, "eval_edge_percentile", 85.0)),
        }

        base_params = _normalize_experiment_params(base_params)

        sweep_root = output_dir / "stage1a_sgbm_core_geom"
        sweep_root.mkdir(parents=True, exist_ok=True)
        results_jsonl = sweep_root / "sweep_results.jsonl"

        sweeps = _build_stage1a_sweeps(cfg, base_params)

        _json_dump(
            sweep_root / "run_manifest.json",
            {
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "stage": "stage1a_sgbm_core_geom",
                "pose_count": int(pose_count),
                "depth_min_m": float(depth_min_m),
                "depth_max_m": float(depth_max_m),
                "depth_scale_mm": float(depth_scale_mm),
                "rectify_use_distortion": bool(rectify_use_distortion),
                "filters_disabled": True,
                "base_params": _snapshot_params_for_json(base_params),
                "sweep_plan": sweeps,
            }
        )

        print("[Baseline] running base configuration...")
        _, baseline_summary = _run_one_experiment(
            rs=rs,
            rectify_dict=rectify_dict,
            stereo_frames=stereo_frames,
            depth_gt_ir_frames=depth_gt_ir_frames,
            depth_gt_rgb_frames=depth_gt_rgb_frames,
            rgb_frames=rgb_frames,
            ir_left_frames=ir_left_frames,
            ir_right_frames=ir_right_frames,
            output_root=sweep_root,
            params=base_params.copy(),
            param_name="baseline",
            param_value="base",
            depth_scale_mm=depth_scale_mm,
            use_geom_mask_from_gt=use_geom_mask_from_gt,
            debug_max_frames=debug_max_frames,
        )
        _append_jsonl(results_jsonl, baseline_summary)

        for param_name, values in sweeps.items():
            print(f"[Sweep] {param_name}: {values}")
            for v in values:
                exp_params = base_params.copy()
                exp_params[param_name] = v
                exp_params = _normalize_experiment_params(exp_params)

                print(f"  -> {param_name} = {v}")

                _, summary_record = _run_one_experiment(
                    rs=rs,
                    rectify_dict=rectify_dict,
                    stereo_frames=stereo_frames,
                    depth_gt_ir_frames=depth_gt_ir_frames,
                    depth_gt_rgb_frames=depth_gt_rgb_frames,
                    rgb_frames=rgb_frames,
                    ir_left_frames=ir_left_frames,
                    ir_right_frames=ir_right_frames,
                    output_root=sweep_root,
                    params=exp_params,
                    param_name=param_name,
                    param_value=v,
                    depth_scale_mm=depth_scale_mm,
                    use_geom_mask_from_gt=use_geom_mask_from_gt,
                    debug_max_frames=debug_max_frames,
                )
                _append_jsonl(results_jsonl, summary_record)

        print(f"[DONE] Sweep results appended to: {results_jsonl}")

    finally:
        if proj is not None:
            try:
                proj.delete()
            except Exception:
                pass
        if light_point is not None:
            try:
                light_point.delete()
            except Exception:
                pass


if __name__ == "__main__":
    main()