#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np
import cv2
import pyrealsense2 as rs


def make_dir(p: str):
    os.makedirs(p, exist_ok=True)


def try_set_emitter(dev: rs.device, enabled: bool) -> bool:
    try:
        depth_sensor = dev.first_depth_sensor()
        depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if enabled else 0.0)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--serial", type=str, default="", help="Optional device serial")
    ap.add_argument("--timeout_ms", type=int, default=15000, help="wait_for_frames timeout")
    ap.add_argument("--out", type=str, default="realsense_dump", help="Folder to save frames (key 's')")
    ap.add_argument("--normalize_ir", action="store_true", help="Auto-normalize IR for display")
    ap.add_argument("--depth_max_m", type=float, default=10.0, help="Max depth (meters) for visualization")
    ap.add_argument("--no_align", action="store_true", help="Do NOT align depth to color (default aligns)")
    args = ap.parse_args()

    pipeline = rs.pipeline()
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)

    # IR streams
    cfg.enable_stream(rs.stream.infrared, 1, args.w, args.h, rs.format.y8, args.fps)  # left
    cfg.enable_stream(rs.stream.infrared, 2, args.w, args.h, rs.format.y8, args.fps)  # right

    # DEPTH stream
    cfg.enable_stream(rs.stream.depth, args.w, args.h, rs.format.z16, args.fps)

    # COLOR stream (try bgr8, fallback rgb8)
    want_bgr = True
    try:
        cfg.enable_stream(rs.stream.color, args.w, args.h, rs.format.bgr8, args.fps)
        profile = pipeline.start(cfg)
    except Exception:
        want_bgr = False
        pipeline = rs.pipeline()
        cfg = rs.config()
        if args.serial:
            cfg.enable_device(args.serial)

        cfg.enable_stream(rs.stream.infrared, 1, args.w, args.h, rs.format.y8, args.fps)
        cfg.enable_stream(rs.stream.infrared, 2, args.w, args.h, rs.format.y8, args.fps)
        cfg.enable_stream(rs.stream.depth, args.w, args.h, rs.format.z16, args.fps)
        cfg.enable_stream(rs.stream.color, args.w, args.h, rs.format.rgb8, args.fps)
        profile = pipeline.start(cfg)

    dev = profile.get_device()

    emitter_on = True
    emitter_supported = try_set_emitter(dev, emitter_on)

    # Align depth -> color (default)
    align = None if args.no_align else rs.align(rs.stream.color)

    # depth scale (z16 -> meters)
    depth_sensor = dev.first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    print(f"Depth scale: {depth_scale} m per unit (z16)")

    print("Controls:  q/ESC = quit,  s = save frames,  e = toggle emitter (if supported)")
    print("Emitter toggle supported." if emitter_supported else "Emitter toggle NOT supported.")
    print(f"COLOR format: {'bgr8' if want_bgr else 'rgb8->bgr'}")
    print(f"Depth align: {'OFF' if args.no_align else 'ON (depth->color)'}")

    make_dir(args.out)

    # --- Create windows (fixed size WxH) ---
    win_c = f"COLOR {args.w}x{args.h}"
    win_l = f"IR LEFT {args.w}x{args.h}"
    win_r = f"IR RIGHT {args.w}x{args.h}"
    win_d = f"DEPTH {args.w}x{args.h}"

    for wname in (win_c, win_l, win_r, win_d):
        cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(wname, args.w, args.h)

    last_t = time.time()

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(args.timeout_ms)
            except RuntimeError:
                continue

            if align is not None:
                frames = align.process(frames)

            f_color = frames.get_color_frame()
            f_depth = frames.get_depth_frame()
            f_ir1 = frames.get_infrared_frame(1)
            f_ir2 = frames.get_infrared_frame(2)

            if not (f_color and f_depth and f_ir1 and f_ir2):
                continue

            color = np.asanyarray(f_color.get_data())
            ir1 = np.asanyarray(f_ir1.get_data())
            ir2 = np.asanyarray(f_ir2.get_data())
            depth_z16 = np.asanyarray(f_depth.get_data())  # uint16 z16

            if not want_bgr:
                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

            # IR display
            if args.normalize_ir:
                ir1_disp = cv2.normalize(ir1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                ir2_disp = cv2.normalize(ir2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                ir1_disp = ir1
                ir2_disp = ir2

            # Depth visualization (colorized)
            depth_m = depth_z16.astype(np.float32) * depth_scale
            depth_m = np.where(np.isfinite(depth_m), depth_m, 0.0)
            depth_m[(depth_m <= 0.0) | (depth_m > args.depth_max_m)] = 0.0

            # normalize to 0..255 for colormap (0 = invalid)
            depth_u8 = np.zeros_like(depth_z16, dtype=np.uint8)
            valid = depth_m > 0.0
            if np.any(valid):
                depth_u8[valid] = np.clip((depth_m[valid] / args.depth_max_m) * 255.0, 0, 255).astype(np.uint8)

            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
            depth_color[~valid] = (0, 0, 0)

            # FPS overlay (only on COLOR)
            now = time.time()
            dt = now - last_t
            last_t = now
            fps = 1.0 / dt if dt > 1e-6 else 0.0
            cv2.putText(
                color,
                f"FPS: {fps:5.1f} | emitter: {'ON' if emitter_on else 'OFF'} | align: {'OFF' if args.no_align else 'ON'}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # show
            cv2.imshow(win_c, color)
            cv2.imshow(win_l, ir1_disp)
            cv2.imshow(win_r, ir2_disp)
            cv2.imshow(win_d, depth_color)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

            if key == ord('e'):
                emitter_on = not emitter_on
                if not try_set_emitter(dev, emitter_on):
                    print("Emitter toggle failed (not supported).")

            if key == ord('s'):
                ts = int(time.time() * 1000)
                p_color = os.path.join(args.out, f"{ts}_color.png")
                p_ir1 = os.path.join(args.out, f"{ts}_ir_left.png")
                p_ir2 = os.path.join(args.out, f"{ts}_ir_right.png")
                p_depth = os.path.join(args.out, f"{ts}_depth.png")
                p_depth_z16 = os.path.join(args.out, f"{ts}_depth_z16.png")

                cv2.imwrite(p_color, color)
                cv2.imwrite(p_ir1, ir1)
                cv2.imwrite(p_ir2, ir2)

                # pretty depth (colormap)
                cv2.imwrite(p_depth, depth_color)

                # raw depth z16 (PNG 16-bit)
                cv2.imwrite(p_depth_z16, depth_z16)

                print("Saved:", p_color, p_ir1, p_ir2, p_depth, p_depth_z16)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
