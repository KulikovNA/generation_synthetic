#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyrealsense2 as rs
import numpy as np


def print_intrinsics(name: str, intr: rs.intrinsics):
    print(f"\n[{name}] intrinsics")
    print(f"  width,height : {intr.width} x {intr.height}")
    print(f"  fx, fy       : {intr.fx:.6f}, {intr.fy:.6f}")
    print(f"  ppx, ppy     : {intr.ppx:.6f}, {intr.ppy:.6f}")
    print(f"  model        : {intr.model}")
    print(f"  coeffs       : {list(intr.coeffs)}")


def print_extrinsics(src_name: str, dst_name: str, ext: rs.extrinsics):
    R = np.array(ext.rotation, dtype=np.float64).reshape(3, 3)
    t = np.array(ext.translation, dtype=np.float64).reshape(3, 1)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:] = t

    print(f"\n[{src_name} -> {dst_name}] extrinsics (4x4, meters)")
    print(T)
    print(f"  baseline / translation = {t.ravel().tolist()}")


def main():
    W, H, FPS = 1280, 800, 15

    pipeline = rs.pipeline()
    config = rs.config()

    # Только IR-стримы, потому что именно они использовались при записи
    config.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, FPS)
    config.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8, FPS)

    profile = pipeline.start(config)

    try:
        sp_ir1 = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        sp_ir2 = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()

        print("IR1 actual format:", sp_ir1.format(), "fps:", sp_ir1.fps())
        print("IR2 actual format:", sp_ir2.format(), "fps:", sp_ir2.fps())

        intr_ir1 = sp_ir1.get_intrinsics()
        intr_ir2 = sp_ir2.get_intrinsics()

        print_intrinsics("IR1", intr_ir1)
        print_intrinsics("IR2", intr_ir2)

        ext_ir2_to_ir1 = sp_ir2.get_extrinsics_to(sp_ir1)
        ext_ir1_to_ir2 = sp_ir1.get_extrinsics_to(sp_ir2)

        print_extrinsics("IR2", "IR1", ext_ir2_to_ir1)
        print_extrinsics("IR1", "IR2", ext_ir1_to_ir2)

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()