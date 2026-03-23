#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import blenderproc as bproc


import argparse
import sys
from pathlib import Path

import bpy
import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BLENDFORGE_SRC = REPO_ROOT / "blendforge" / "src"
if str(BLENDFORGE_SRC) not in sys.path:
    sys.path.insert(0, str(BLENDFORGE_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blendforge.blender_runtime.camera.RealsenseProfileLoader import RealSenseProfile
from blendforge.blender_runtime.camera.ActiveStereoProjectorRuntime import (
    ProjectorOverrides,
    render_active_stereo_pair,
)
from blendforge.blender_runtime.stereo.ActiveStereoIRUtility import rgb_to_intensity_u8


DEFAULT_PLANE_DISTANCE_M = 1.0042
DEFAULT_RENDER_SAMPLES = 64
DEFAULT_LEFT_STREAM = "IR_LEFT"
DEFAULT_RIGHT_STREAM = "IR_RIGHT"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render one simplified active-stereo wall capture using the public runtime API."
    )
    p.add_argument("--camera_profile_json", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--plane_distance_m", type=float, default=DEFAULT_PLANE_DISTANCE_M)
    p.add_argument("--plane_width_m", type=float, default=3.5)
    p.add_argument("--plane_height_m", type=float, default=3.5)
    p.add_argument("--render_samples", type=int, default=DEFAULT_RENDER_SAMPLES)
    p.add_argument("--left_stream", type=str, default=DEFAULT_LEFT_STREAM)
    p.add_argument("--right_stream", type=str, default=DEFAULT_RIGHT_STREAM)
    p.add_argument(
        "--energy",
        type=float,
        default=None,
        help="Override projector energy from the profile JSON.",
    )
    return p.parse_args(_strip_blender_args(argv))


def _strip_blender_args(argv: list[str]) -> list[str]:
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return argv


def ensure_output_dir(path: str) -> Path:
    out_dir = Path(path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def cleanup_default_scene() -> None:
    for obj in list(bpy.data.objects):
        if obj.type in {"MESH", "LIGHT"} and obj.name in {"Cube", "Light"}:
            bpy.data.objects.remove(obj, do_unlink=True)


def configure_renderer(samples: int) -> None:
    bpy.context.scene.render.engine = "CYCLES"
    bproc.renderer.set_max_amount_of_samples(max(1, int(samples)))
    scene = bpy.context.scene
    scene.render.film_transparent = False

    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg = next((n for n in world.node_tree.nodes if n.type == "BACKGROUND"), None)
    if bg is not None:
        bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        bg.inputs["Strength"].default_value = 0.0


def create_neutral_plane_material(name: str = "projector_wall_material"):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    node_tree = material.node_tree
    bsdf = next((n for n in node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        raise RuntimeError("Could not find Principled BSDF for projector wall material.")
    bsdf.inputs["Base Color"].default_value = (0.82, 0.82, 0.82, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.92
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.05
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.05
    return material


def create_wall_plane(
    *,
    distance_m: float,
    width_m: float,
    height_m: float,
):
    plane = bproc.object.create_primitive(
        "PLANE",
        scale=[float(width_m) * 0.5, float(height_m) * 0.5, 1.0],
        location=[0.0, 0.0, -float(distance_m)],
        rotation=[0.0, 0.0, 0.0],
    )
    plane.set_name("projector_wall")
    material = create_neutral_plane_material()
    plane.blender_obj.data.materials.clear()
    plane.blender_obj.data.materials.append(material)
    return plane


def save_png(path: Path, img: np.ndarray) -> None:
    arr = np.asarray(img)
    if arr.ndim == 2:
        data = arr.astype(np.uint8, copy=False)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        data = cv2.cvtColor(arr.astype(np.uint8, copy=False), cv2.COLOR_RGB2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        data = cv2.cvtColor(arr.astype(np.uint8, copy=False), cv2.COLOR_RGBA2BGRA)
    else:
        raise ValueError(f"Unsupported image shape for save_png: {arr.shape}")
    cv2.imwrite(str(path), data)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)

    bproc.init()
    bproc.utility.reset_keyframes()
    cleanup_default_scene()
    configure_renderer(args.render_samples)
    create_wall_plane(
        distance_m=float(args.plane_distance_m),
        width_m=float(args.plane_width_m),
        height_m=float(args.plane_height_m),
    )

    rs = RealSenseProfile.from_json(args.camera_profile_json)
    if not rs.has_stream(args.left_stream):
        raise KeyError(f"Left stream '{args.left_stream}' not found in profile.")
    if not rs.has_stream(args.right_stream):
        raise KeyError(f"Right stream '{args.right_stream}' not found in profile.")

    render_data = render_active_stereo_pair(
        rs,
        left_stream=args.left_stream,
        right_stream=args.right_stream,
        overrides=ProjectorOverrides(projector_energy=args.energy),
        mount_name="rs_projector_mount_wall_render",
        socket_name="rs_projector_socket_wall_render",
    )

    left_ir = rgb_to_intensity_u8(render_data["left_colors"])
    right_ir = rgb_to_intensity_u8(render_data["right_colors"])

    out_dir = ensure_output_dir(args.output_dir)
    save_png(out_dir / "left_pattern_wall.png", left_ir)
    save_png(out_dir / "right_pattern_wall.png", right_ir)

    print(f"Saved: {out_dir / 'left_pattern_wall.png'}")
    print(f"Saved: {out_dir / 'right_pattern_wall.png'}")


if __name__ == "__main__":
    main()
