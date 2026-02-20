#!/usr/bin/env python3
# view_rgbd_open3d.py

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError as e:
    raise SystemExit(
        "Open3D not installed. Install with:\n"
        "  pip install open3d\n"
        "or (conda):\n"
        "  conda install -c conda-forge open3d\n"
    ) from e


def load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_image_entry(coco: dict, image_id: int) -> dict:
    for it in coco.get("images", []):
        if int(it.get("id", -1)) == int(image_id):
            return it
    raise KeyError(f"image_id={image_id} not found")


def stable_color_from_int(k: int) -> np.ndarray:
    """Deterministic pseudo-random color in [0,1] from integer id."""
    # simple hash
    x = (k * 2654435761) & 0xFFFFFFFF
    r = ((x >> 16) & 255) / 255.0
    g = ((x >> 8) & 255) / 255.0
    b = (x & 255) / 255.0
    # avoid too-dark colors
    c = np.array([r, g, b], dtype=np.float32)
    return 0.25 + 0.75 * c


def build_point_cloud_numpy(
    rgb_bgr: np.ndarray,
    depth_u16: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    depth_scale_mm: float,
    *,
    valid_mask: np.ndarray | None = None,
    instance_mask_u8: np.ndarray | None = None,
    color_mode: str = "rgb",   # "rgb" | "instance"
    zmin: float | None = None,
    zmax: float | None = None,
    stride: int = 1,
):
    """
    depth_u16: uint16, where depth_mm = u16 * depth_scale_mm (BOP-style)
    meters: z = u16 * depth_scale_mm / 1000
    """
    assert depth_u16.dtype == np.uint16

    H, W = depth_u16.shape[:2]
    s = max(1, int(stride))

    # subsample grid
    vv = np.arange(0, H, s, dtype=np.int32)
    uu = np.arange(0, W, s, dtype=np.int32)
    u, v = np.meshgrid(uu, vv)  # (h', w')

    d = depth_u16[v, u].astype(np.float32)
    z = d * (float(depth_scale_mm) / 1000.0)  # meters

    valid = d > 0
    if valid_mask is not None:
        valid &= (valid_mask[v, u] > 0)

    if zmin is not None:
        valid &= (z >= float(zmin))
    if zmax is not None:
        valid &= (z <= float(zmax))

    if not np.any(valid):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    u_valid = u[valid].astype(np.float32)
    v_valid = v[valid].astype(np.float32)
    z_valid = z[valid].astype(np.float32)

    # pinhole backprojection
    x = (u_valid - float(cx)) * z_valid / float(fx)
    y = (v_valid - float(cy)) * z_valid / float(fy)

    pts = np.stack([x, y, z_valid], axis=1).astype(np.float32)

    if color_mode == "instance":
        if instance_mask_u8 is None:
            raise ValueError("color_mode=instance requires instance_mask_u8")
        inst = instance_mask_u8[v, u][valid].astype(np.int32)
        cols = np.stack([stable_color_from_int(int(k)) for k in inst], axis=0).astype(np.float32)
    else:
        # RGB colors from image
        bgr = rgb_bgr[v, u][valid].astype(np.float32) / 255.0
        cols = bgr[:, ::-1]  # to RGB in [0..1]

    return pts, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/home/nikita/data_generator/generation_dataset/generation_synthetic/output/seg_with_depth/differBig/2026-02-11/train", 
                    help="split dir with coco_annotations.json")
    ap.add_argument("--image-id", type=int, default=0)
    ap.add_argument("--use-inpainted", action="store_true", help="use depth_inpainted instead of depth_raw")
    ap.add_argument("--use-valid-mask", action="store_true", help="apply mask_valid/<id>.png")
    ap.add_argument("--color", choices=["rgb", "instance"], default="rgb", help="color mode")
    ap.add_argument("--use-instance-mask", action="store_true", help="load masks/<id>.png for instance coloring")
    ap.add_argument("--zmin", type=float, default=None, help="min depth (m) for filtering/view")
    ap.add_argument("--zmax", type=float, default=None, help="max depth (m) for filtering/view")
    ap.add_argument("--stride", type=int, default=2, help="pixel subsampling stride")
    ap.add_argument("--voxel", type=float, default=0.0, help="voxel downsample size (m), 0 disables")
    ap.add_argument("--flip", action="store_true", help="flip Y,Z like many Open3D examples for nicer view")
    ap.add_argument("--save-ply", type=str, default=None, help="path to save point cloud .ply")
    args = ap.parse_args()

    root = Path(args.root)
    coco = load_json(root / "coco_annotations.json")
    cam_info = load_json(root / "camera_info.json")

    img = get_image_entry(coco, args.image_id)

    rgb_path = root / img["file_name"]
    depth_path = root / (img["depth_inpainted_file"] if args.use_inpainted else img["depth_raw_file"])

    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise SystemExit(f"Failed to read RGB: {rgb_path}")

    depth_u16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_u16 is None:
        raise SystemExit(f"Failed to read depth: {depth_path}")
    if depth_u16.dtype != np.uint16:
        raise SystemExit(f"Depth must be uint16 PNG, got {depth_u16.dtype} at {depth_path}")

    # intrinsics: prefer camera_info.json; fallback to per-image fields if needed
    fx = float(cam_info.get("fx", img.get("fx")))
    fy = float(cam_info.get("fy", img.get("fy")))
    cx = float(cam_info.get("cx", img.get("cx")))
    cy = float(cam_info.get("cy", img.get("cy")))

    # depth_scale meaning in your writer:
    # depth_mm = depth_u16 * depth_scale_mm
    depth_scale_mm = float(img.get("depth_scale", cam_info.get("depth_scale", 1.0)))

    valid_mask = None
    if args.use_valid_mask and "valid_depth_mask_file" in img:
        vm_path = root / img["valid_depth_mask_file"]
        vm = cv2.imread(str(vm_path), cv2.IMREAD_UNCHANGED)
        if vm is None:
            raise SystemExit(f"Failed to read valid mask: {vm_path}")
        valid_mask = vm

    inst_mask = None
    if args.use_instance_mask and "instances_mask_file" in img:
        im_path = root / img["instances_mask_file"]
        im = cv2.imread(str(im_path), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise SystemExit(f"Failed to read instance mask: {im_path}")
        inst_mask = im.astype(np.uint8)

    if args.color == "instance" and inst_mask is None:
        raise SystemExit("For --color instance you must pass --use-instance-mask (and have instances_mask_file).")

    pts, cols = build_point_cloud_numpy(
        rgb, depth_u16,
        fx, fy, cx, cy,
        depth_scale_mm,
        valid_mask=valid_mask,
        instance_mask_u8=inst_mask,
        color_mode=args.color,
        zmin=args.zmin,
        zmax=args.zmax,
        stride=args.stride,
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

    if args.voxel and args.voxel > 0:
        pcd = pcd.voxel_down_sample(float(args.voxel))

    if args.flip:
        # flip Y and Z to make it "upright" (common Open3D convention)
        pcd.transform(np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1],
        ], dtype=np.float64))

    if args.save_ply:
        out = Path(args.save_ply)
        out.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(out), pcd)
        print("Saved:", out)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    o3d.visualization.draw_geometries([pcd, frame])


if __name__ == "__main__":
    main()
