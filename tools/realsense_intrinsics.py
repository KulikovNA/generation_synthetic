import pyrealsense2 as rs
import numpy as np

def print_intrinsics(name: str, intr: rs.intrinsics):
    print(f"\n[{name}] intrinsics")
    print(f"  width,height : {intr.width} x {intr.height}")
    print(f"  fx, fy       : {intr.fx:.6f}, {intr.fy:.6f}")
    print(f"  ppx, ppy     : {intr.ppx:.6f}, {intr.ppy:.6f}")
    print(f"  model        : {intr.model}")  # distortion model enum
    print(f"  coeffs       : {list(intr.coeffs)}")

def print_extrinsics(src_name: str, dst_name: str, ext: rs.extrinsics):
    R = np.array(ext.rotation, dtype=np.float64).reshape(3, 3)
    t = np.array(ext.translation, dtype=np.float64).reshape(3, 1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:] = t
    print(f"\n[{src_name} -> {dst_name}] extrinsics (4x4, meters)")
    print(T)

def main():
    W, H, FPS = 640, 480, 30

    pipeline = rs.pipeline()
    config = rs.config()

    # Форматы:
    # - color обычно RGB8/BGR8
    # - infrared обычно Y8 (8-bit mono)
    # - depth обычно Z16
    config.enable_stream(rs.stream.color,    W, H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8,  FPS)
    config.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8,  FPS)

    # Если хочешь ещё и depth intrinsics/extrinsics:
    config.enable_stream(rs.stream.depth,    W, H, rs.format.z16, FPS)

    profile = pipeline.start(config)

    try:
        # Важно: intrinsics берём из VideoStreamProfile активного стрима
        sp_color = profile.get_stream(rs.stream.color).as_video_stream_profile()
        print("COLOR actual format:", sp_color.format(), "fps:", sp_color.fps())
        sp_ir1   = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        sp_ir2   = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        sp_depth = profile.get_stream(rs.stream.depth).as_video_stream_profile()

        intr_color = sp_color.get_intrinsics()
        intr_ir1   = sp_ir1.get_intrinsics()
        intr_ir2   = sp_ir2.get_intrinsics()
        intr_depth = sp_depth.get_intrinsics()

        print_intrinsics("COLOR", intr_color)
        print_intrinsics("IR1",   intr_ir1)
        print_intrinsics("IR2",   intr_ir2)
        print_intrinsics("DEPTH", intr_depth)

        # Экструзики между сенсорами/стримами
        # (обычно удобно всё приводить к COLOR)
        ext_ir1_to_color   = sp_ir1.get_extrinsics_to(sp_color)
        ext_ir2_to_color   = sp_ir2.get_extrinsics_to(sp_color)
        ext_depth_to_color = sp_depth.get_extrinsics_to(sp_color)

        print_extrinsics("IR1",   "COLOR", ext_ir1_to_color)
        print_extrinsics("IR2",   "COLOR", ext_ir2_to_color)
        print_extrinsics("DEPTH", "COLOR", ext_depth_to_color)

    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()
