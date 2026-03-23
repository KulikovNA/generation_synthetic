#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3: export Blender- and runtime-friendly configuration from stage2 projector seed.

What this script does:
1) Loads capture_meta.json and projector_stage2/projector_seed.json.
2) Exports exact Blender-facing camera/projector transforms and intrinsics helpers.
3) Chooses the final projector texture variant (uniform by default) and also preserves
   weighted/debug/support texture references.
4) Computes projector local transform relative to the chosen mount frame (IR_LEFT by default).
5) Writes:
   - blender_projector_config.json
   - blender_setup_notes.txt
   - realsense_effective_projector_profile.json

Notes:
- World frame is chosen to coincide with LEFT IR CV frame.
- This is an effective calibration/export for simulation, not a unique metric calibration.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


# ---------------------------- io ----------------------------

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def to_list(x):
    return np.asarray(x, dtype=np.float64).tolist()


def latest_session(root: Path) -> Path:
    sessions = sorted([p for p in root.glob("session_*") if p.is_dir()])
    if not sessions:
        raise FileNotFoundError(f"No session_* directories found in: {root}")
    return sessions[-1]


# ---------------------------- geometry helpers ----------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    if n <= 1e-12:
        raise ValueError("Zero-length vector cannot be normalized")
    return v / n


def parse_ir_intrinsics(d: Dict) -> Dict:
    return {
        "width": int(d["width"]),
        "height": int(d["height"]),
        "fx": float(d["fx"]),
        "fy": float(d["fy"]),
        "cx": float(d["ppx"]),
        "cy": float(d["ppy"]),
        "distortion_model": d.get("model", "brown_conrady"),
        "distortion_coeffs": list(d.get("coeffs", [0.0, 0.0, 0.0, 0.0, 0.0])),
    }


def parse_T_left_from_right(extr_dict: Dict) -> np.ndarray:
    if "T_target_from_source_4x4" in extr_dict:
        return np.array(extr_dict["T_target_from_source_4x4"], dtype=np.float64).reshape(4, 4)

    if "rotation_row_major_3x3" in extr_dict and "translation_m" in extr_dict:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.array(extr_dict["rotation_row_major_3x3"], dtype=np.float64).reshape(3, 3)
        T[:3, 3] = np.array(extr_dict["translation_m"], dtype=np.float64).reshape(3)
        return T

    raise KeyError("Unsupported extrinsics format for IR_RIGHT_to_IR_LEFT")


def build_blender_camera_matrix_from_cv_pose(C_world, right_axis_world, down_axis_world, forward_axis_world):
    """
    CV camera axes:
      +X = right
      +Y = down
      +Z = forward

    Blender camera/projector local axes:
      +X = right
      +Y = up
      -Z = forward

    Object world basis columns:
      col0 = right_world
      col1 = up_world   = -down_world
      col2 = back_world = -forward_world
    """
    right = normalize(np.asarray(right_axis_world, dtype=np.float64))
    down = normalize(np.asarray(down_axis_world, dtype=np.float64))
    forward = normalize(np.asarray(forward_axis_world, dtype=np.float64))

    up = -down
    back = -forward

    M = np.eye(4, dtype=np.float64)
    M[:3, 0] = right
    M[:3, 1] = up
    M[:3, 2] = back
    M[:3, 3] = np.asarray(C_world, dtype=np.float64).reshape(3)
    return M


def build_intrinsics_for_blender(width, height, fx, fy, cx, cy):
    """
    Export exact K-related values in a Blender-friendly way.

    Trick:
    choose sensor_width_mm = width and sensor_height_mm = height.
    Then lens_mm can numerically equal focal length in pixels.
    Ratios remain correct, which is what matters.
    """
    width = float(width)
    height = float(height)

    return {
        "image_width_px": int(width),
        "image_height_px": int(height),
        "fx_px": float(fx),
        "fy_px": float(fy),
        "cx_px": float(cx),
        "cy_px": float(cy),
        "sensor_width_mm": float(width),
        "sensor_height_mm": float(height),
        "lens_mm_x_equiv": float(fx),
        "lens_mm_y_equiv": float(fy),
        "shift_x": float(-((float(cx) - (width / 2.0)) / width)),
        "shift_y": float(((float(cy) - (height / 2.0)) / height)),
        "sensor_fit_recommendation": "HORIZONTAL",
    }


def choose_projector_texture_paths(proj_img: Dict) -> Tuple[str, str, str]:
    """
    Return:
      default_texture_path
      weighted_texture_path
      debug_texture_path
    """
    default_texture = proj_img.get("texture_uniform_path") or proj_img.get("texture_path")
    weighted_texture = proj_img.get("texture_weighted_path") or proj_img.get("texture_path")
    debug_texture = proj_img.get("texture_debug_path") or proj_img.get("texture_path")

    if default_texture is None:
        raise KeyError("Could not resolve default projector texture path from projector_seed.json")

    return str(default_texture), str(weighted_texture), str(debug_texture)


def build_runtime_projector_intrinsics(width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> Dict:
    return {
        "width": int(width),
        "height": int(height),
        "fx": float(fx),
        "fy": float(fy),
        "ppx": float(cx),
        "ppy": float(cy),
        "distortion_model": "brown_conrady",
        "distortion_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
    }


# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Export Blender/runtime config from projector_seed.json")
    ap.add_argument("--root", type=str, default="wall_capture", help="Root capture directory")
    ap.add_argument("--session", type=str, default="latest", help="Session name or 'latest'")
    ap.add_argument("--stage2_name", type=str, default="projector_stage2", help="Stage2 directory name")
    ap.add_argument("--out_name", type=str, default="projector_stage3", help="Stage3 output directory name")
    ap.add_argument("--mount_frame", type=str, default="IR_LEFT", choices=["IR_LEFT"], help="Projector mount frame used for runtime export")
    args = ap.parse_args()

    root = Path(args.root)
    session_dir = latest_session(root) if args.session == "latest" else (root / args.session)
    if not session_dir.exists():
        raise FileNotFoundError(session_dir)

    seed_path = session_dir / args.stage2_name / "projector_seed.json"
    meta_path = session_dir / "capture_meta.json"
    out_dir = session_dir / args.out_name
    ensure_dir(out_dir)

    seed = load_json(seed_path)
    meta = load_json(meta_path)

    ir_left = parse_ir_intrinsics(meta["streams"]["IR_LEFT"])
    ir_right = parse_ir_intrinsics(meta["streams"]["IR_RIGHT"])
    T_left_from_right = parse_T_left_from_right(meta["extrinsics"]["IR_RIGHT_to_IR_LEFT"])

    width = int(ir_left["width"])
    height = int(ir_left["height"])

    K_left_bl = build_intrinsics_for_blender(
        width=width,
        height=height,
        fx=ir_left["fx"],
        fy=ir_left["fy"],
        cx=ir_left["cx"],
        cy=ir_left["cy"],
    )
    K_right_bl = build_intrinsics_for_blender(
        width=width,
        height=height,
        fx=ir_right["fx"],
        fy=ir_right["fy"],
        cx=ir_right["cx"],
        cy=ir_right["cy"],
    )

    # World frame == LEFT IR CV frame
    left_C = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    left_right_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    left_down_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    left_forward_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    M_left_blender = build_blender_camera_matrix_from_cv_pose(
        left_C, left_right_axis, left_down_axis, left_forward_axis
    )

    right_C_in_left = T_left_from_right[:3, 3]
    M_right_blender = build_blender_camera_matrix_from_cv_pose(
        right_C_in_left, left_right_axis, left_down_axis, left_forward_axis
    )

    proj_pose = seed["projector_pose_in_left_frame"]
    proj_intr = seed["projector_intrinsics"]
    proj_img = seed["projector_image"]

    C_proj = np.array(proj_pose["C_left_m"], dtype=np.float64)
    right_axis = normalize(np.array(proj_pose["right_axis_left"], dtype=np.float64))
    down_axis = normalize(np.array(proj_pose["down_axis_left"], dtype=np.float64))
    forward_axis = normalize(np.array(proj_pose["forward_axis_left"], dtype=np.float64))

    M_proj_blender = build_blender_camera_matrix_from_cv_pose(
        C_proj, right_axis, down_axis, forward_axis
    )

    proj_width = int(proj_img["width"])
    proj_height = int(proj_img["height"])

    K_proj_bl = build_intrinsics_for_blender(
        width=proj_width,
        height=proj_height,
        fx=float(proj_intr["fx"]),
        fy=float(proj_intr["fy"]),
        cx=float(proj_intr["cx"]),
        cy=float(proj_intr["cy"]),
    )

    # local transform relative to mount frame object in Blender convention
    if args.mount_frame != "IR_LEFT":
        raise NotImplementedError("Only IR_LEFT mount_frame is implemented in this exporter")

    M_mount_blender = M_left_blender.copy()
    T_mount_from_projector_blender = np.linalg.inv(M_mount_blender) @ M_proj_blender

    # Mirror heuristic
    right_dot_left_x = float(np.dot(right_axis, np.array([1.0, 0.0, 0.0])))
    mirror_ambiguity_horizontal = right_dot_left_x < 0.0

    default_texture_path, weighted_texture_path, debug_texture_path = choose_projector_texture_paths(proj_img)

    # some stage2 variants may not have all these keys
    uniform_dot_radius_uv_px = proj_img.get("uniform_dot_radius_uv_px")
    safe_uniform_dot_radius_uv_px = proj_img.get("safe_uniform_dot_radius_uv_px", uniform_dot_radius_uv_px)
    merge_radius_px = proj_img.get("merge_radius_px")
    min_circle_gap_px = proj_img.get("min_circle_gap_px", proj_img.get("non_overlap_gap_px"))

    support_shared_path = proj_img.get("support_shared_path")
    support_left_only_path = proj_img.get("support_left_only_path")
    support_right_only_path = proj_img.get("support_right_only_path")

    notes = [
        "World frame is chosen to coincide with the LEFT IR CV frame.",
        "CV frame convention used here: +X right, +Y down, +Z forward.",
        "Blender camera/projector local convention: +X right, +Y up, -Z forward.",
        "Left camera is placed at origin; right camera uses capture_meta baseline.",
        "Projector config is an effective seed from a single plane, not a unique calibration.",
        "Default projector texture is the UNIFORM variant. Weighted/debug texture is kept separately.",
    ]
    if mirror_ambiguity_horizontal:
        notes.append("Horizontal mirror ambiguity detected: projector right-axis points mostly toward -X of the left camera.")
        notes.append("If projected dots appear horizontally mirrored, flip projector texture U or invert projector image X-axis.")

    blender_config = {
        "session_dir": str(session_dir),
        "source_files": {
            "projector_seed_json": str(seed_path.relative_to(session_dir)),
            "capture_meta_json": str(meta_path.relative_to(session_dir)),
        },
        "conventions": {
            "reference_world_frame": "LEFT_IR_CV_FRAME",
            "cv_axes": {"x": "right", "y": "down", "z": "forward"},
            "blender_camera_local_axes": {"x": "right", "y": "up", "minus_z": "forward"},
        },
        "left_camera": {
            "intrinsics": K_left_bl,
            "distortion_model": ir_left["distortion_model"],
            "distortion_coeffs": ir_left["distortion_coeffs"],
            "world_matrix_blender": to_list(M_left_blender),
        },
        "right_camera": {
            "intrinsics": K_right_bl,
            "distortion_model": ir_right["distortion_model"],
            "distortion_coeffs": ir_right["distortion_coeffs"],
            "T_left_from_right_4x4_m": to_list(T_left_from_right),
            "world_matrix_blender": to_list(M_right_blender),
            "baseline_m": float(np.linalg.norm(right_C_in_left)),
        },
        "projector": {
            "intrinsics": K_proj_bl,
            "fov_x_deg": float(proj_intr["fov_x_deg"]),
            "fov_y_deg": float(proj_intr["fov_y_deg"]),
            "default_texture_path": default_texture_path,
            "uniform_texture_path": proj_img.get("texture_uniform_path", default_texture_path),
            "weighted_texture_path": weighted_texture_path,
            "texture_debug_path": debug_texture_path,
            "support_shared_path": support_shared_path,
            "support_left_only_path": support_left_only_path,
            "support_right_only_path": support_right_only_path,
            "default_texture_variant": proj_img.get("default_texture_variant", "uniform"),
            "uniform_dot_radius_uv_px": uniform_dot_radius_uv_px,
            "safe_uniform_dot_radius_uv_px": safe_uniform_dot_radius_uv_px,
            "merge_radius_px": merge_radius_px,
            "min_circle_gap_px": min_circle_gap_px,
            "location_left_frame_m": to_list(C_proj),
            "right_axis_left": to_list(right_axis),
            "down_axis_left": to_list(down_axis),
            "forward_axis_left": to_list(forward_axis),
            "world_matrix_blender": to_list(M_proj_blender),
            "local_transform_blender_4x4_relative_to_mount": to_list(T_mount_from_projector_blender),
            "mount_frame": args.mount_frame,
            "mirror_ambiguity_horizontal": bool(mirror_ambiguity_horizontal),
            "suggested_blender_light_type": "SPOT_OR_CUSTOM_PROJECTOR",
        },
        "observed_plane": seed.get("observed_plane_in_left_frame", {}),
        "notes": notes,
    }

    save_json(out_dir / "blender_projector_config.json", blender_config)

    # Runtime-friendly profile export for direct reuse in your RealSenseProfile loader
    runtime_profile = {
        "device": {
            "family": "Intel RealSense D400",
            "model": "D435",
            "profile_name": f"{width}x{height}_effective_projector_seed",
            "units": {"translation": "meters", "intrinsics": "pixels"},
        },
        "stream_index_map": {
            "IR_LEFT": 1,
            "IR_RIGHT": 2,
            "note": "RealSense SDK: infrared(1)=Left, infrared(2)=Right",
        },
        "streams": {
            "IR_LEFT": {
                "width": width,
                "height": height,
                "intrinsics": {
                    "fx": ir_left["fx"],
                    "fy": ir_left["fy"],
                    "ppx": ir_left["cx"],
                    "ppy": ir_left["cy"],
                    "distortion_model": ir_left["distortion_model"],
                    "distortion_coeffs": ir_left["distortion_coeffs"],
                },
            },
            "IR_RIGHT": {
                "width": width,
                "height": height,
                "intrinsics": {
                    "fx": ir_right["fx"],
                    "fy": ir_right["fy"],
                    "ppx": ir_right["cx"],
                    "ppy": ir_right["cy"],
                    "distortion_model": ir_right["distortion_model"],
                    "distortion_coeffs": ir_right["distortion_coeffs"],
                },
            },
            "DEPTH": {
                "width": width,
                "height": height,
                "intrinsics": {
                    "fx": ir_left["fx"],
                    "fy": ir_left["fy"],
                    "ppx": ir_left["cx"],
                    "ppy": ir_left["cy"],
                    "distortion_model": ir_left["distortion_model"],
                    "distortion_coeffs": ir_left["distortion_coeffs"],
                },
                "note": "DEPTH intrinsics are exported equal to IR_LEFT for this effective profile",
            },
        },
        "extrinsics": {
            "IR_RIGHT_to_IR_LEFT": {
                "source_frame": "IR_RIGHT",
                "target_frame": "IR_LEFT",
                "T_target_from_source_4x4": to_list(T_left_from_right),
            },
        },
        "derived": {
            "baseline_m": float(np.linalg.norm(T_left_from_right[:3, 3])),
            "baseline_mm": float(1000.0 * np.linalg.norm(T_left_from_right[:3, 3])),
            "how_computed": "baseline = norm(translation of IR_RIGHT_to_IR_LEFT)",
        },
        "depth_range_m": {
            "min": 0.2,
            "max": 10.0,
            "note": "Generic default; adjust for your stereo/depth pipeline if needed.",
        },
        "stereo": {
            "num_disparities_margin": 1.15,
            "num_disparities_clamp": [64, 256],
            "note": "Heuristic for OpenCV SGBM: numDisp ~= ceil((fx*B/z_min)*margin) rounded to /16.",
        },
        "projector": {
            "type": "effective_projector_seed",
            "mount_mode": "explicit_local_transform",
            "mount_frame": args.mount_frame,
            "local_transform_4x4": to_list(T_mount_from_projector_blender),
            "fov_h_deg": float(proj_intr["fov_x_deg"]),
            "fov_v_deg": float(proj_intr["fov_y_deg"]),
            "energy": 3000.0,
            "pattern_image": {
                "width": proj_width,
                "height": proj_height,
                "path": default_texture_path,
                "path_uniform": proj_img.get("texture_uniform_path", default_texture_path),
                "path_weighted": weighted_texture_path,
                "path_debug": debug_texture_path,
                "support_shared_path": support_shared_path,
                "support_left_only_path": support_left_only_path,
                "support_right_only_path": support_right_only_path,
                "dot_radius_uv_px": safe_uniform_dot_radius_uv_px,
                "default_variant": proj_img.get("default_texture_variant", "uniform"),
            },
            "mirror_ambiguity_horizontal": bool(mirror_ambiguity_horizontal),
        },
        "export_debug": {
            "projector_world_matrix_blender": to_list(M_proj_blender),
            "mount_world_matrix_blender": to_list(M_mount_blender),
        },
        "notes": notes,
    }

    save_json(out_dir / "realsense_effective_projector_profile.json", runtime_profile)

    notes_txt = "\n".join([
        "Blender / runtime export notes",
        "==============================",
        "",
        f"Session: {session_dir.name}",
        "",
        "Use LEFT IR as the scene reference frame.",
        "Left camera world matrix is exported directly.",
        "Right camera world matrix is exported directly.",
        "Projector world matrix and local transform relative to IR_LEFT mount are exported directly.",
        "",
        "For camera/projector intrinsics in Blender:",
        "  sensor_width_mm  = image_width_px",
        "  sensor_height_mm = image_height_px",
        "  lens_mm_x_equiv  = fx_px",
        "  shift_x          = -(cx - W/2)/W",
        "  shift_y          =  (cy - H/2)/H",
        "",
        "Default projector texture = uniform texture.",
        "Weighted texture is kept for debug/confidence inspection.",
        "",
        "If the projected pattern looks horizontally mirrored,",
        "flip the projector texture in U or invert the projector image X-axis.",
        "",
    ])
    (out_dir / "blender_setup_notes.txt").write_text(notes_txt, encoding="utf-8")

    print(out_dir / "blender_projector_config.json")
    print(out_dir / "realsense_effective_projector_profile.json")
    print(out_dir / "blender_setup_notes.txt")


if __name__ == "__main__":
    main()
