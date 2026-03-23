#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import json
import time
import queue
import threading
import argparse
import numpy as np
import pyrealsense2 as rs


# ============================================================
# Fixed capture protocol for projector-pattern acquisition
# Patched version:
#   - OFF phase forces BOTH emitter_enabled=0 and laser_power=0
#   - ON phase forces emitter_enabled=1 and laser_power=fixed target
#   - phase switch is verified by readback
#   - extra flush frames after toggle to avoid stale ON frames in OFF phase
#   - only minimal metadata is stored
# ============================================================

IR_W = 1280
IR_H = 800
IR_FORMAT = "y8"
STREAM_FPS = 15
PREVIEW_SCALE = 1.0
PREVIEW_FPS = 10.0
RECORD_FPS = 10.0

N_FRAMES_PER_PHASE = 300
PNG_COMPRESSION = 1
WRITER_QUEUE = 256

LOCK_IR_AFTER_WARMUP = True
WARMUP_FRAMES = 30
FLUSH_FRAMES_AFTER_TOGGLE = 30
TIMEOUT_MS = 15000

# Single fixed photometric mode per run.
# If MANUAL_EXPOSURE is None, auto exposure is warmed up and then locked.
MANUAL_EXPOSURE = 7000     # e.g. 12000.0
MANUAL_GAIN = 16         # e.g. 16.0

# Laser power used in ON phase only.
USE_MAX_LASER_POWER = True
MANUAL_LASER_POWER = None  # used only if USE_MAX_LASER_POWER = False

WINDOW_LEFT = "IR LEFT"
WINDOW_RIGHT = "IR RIGHT"


# ---------------------------- utils ----------------------------

def make_dir(path: str):
    os.makedirs(path, exist_ok=True)


def now_str():
    return time.strftime("%Y%m%d_%H%M%S")


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_get_option(sensor, option):
    try:
        if sensor.supports(option):
            return float(sensor.get_option(option))
    except Exception:
        pass
    return None


def safe_set_option(sensor, option, value) -> bool:
    try:
        if sensor.supports(option):
            sensor.set_option(option, value)
            return True
    except Exception:
        pass
    return False


def get_option_range(sensor, option):
    try:
        if sensor.supports(option):
            r = sensor.get_option_range(option)
            return {
                "min": float(r.min),
                "max": float(r.max),
                "step": float(r.step),
                "default": float(getattr(r, "def")),
            }
    except Exception:
        pass
    return None


def get_rs_option(name: str):
    return getattr(rs.option, name, None)


def safe_get_option_by_name(sensor, name: str):
    opt = get_rs_option(name)
    if opt is None:
        return None
    return safe_get_option(sensor, opt)


def safe_set_option_by_name(sensor, name: str, value) -> bool:
    opt = get_rs_option(name)
    if opt is None:
        return False
    return safe_set_option(sensor, opt, value)


def intrinsics_to_dict(intr):
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "model": str(intr.model),
        "coeffs": [float(x) for x in intr.coeffs],
    }


def extrinsics_to_dict(ext):
    R = np.array(ext.rotation, dtype=np.float64).reshape(3, 3)
    t = np.array(ext.translation, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return {
        "rotation_row_major_3x3": R.tolist(),
        "translation_m": t.tolist(),
        "T_target_from_source_4x4": T.tolist(),
    }


def resize_for_preview(img: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (nw, nh), interpolation=interp)


def preview_uint8(ir: np.ndarray) -> np.ndarray:
    if ir.dtype == np.uint8:
        out = ir
    else:
        out = cv2.normalize(ir, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return out


def frame_timestamp_ms(frame) -> float:
    try:
        return float(frame.get_timestamp())
    except Exception:
        return float(time.time() * 1000.0)


def frame_number(frame) -> int:
    try:
        return int(frame.get_frame_number())
    except Exception:
        return -1


# ---------------------------- writer ----------------------------

class AsyncPairWriter:
    def __init__(self, session_dir: str, png_compression: int = 1, max_queue: int = 256):
        self.session_dir = session_dir
        self.png_compression = int(png_compression)
        self.q = queue.Queue(maxsize=max_queue)
        self.jsonl_path = os.path.join(session_dir, "frames.jsonl")

        self.phase_dirs = {
            "on": {
                "left": os.path.join(session_dir, "left_on"),
                "right": os.path.join(session_dir, "right_on"),
            },
            "off": {
                "left": os.path.join(session_dir, "left_off"),
                "right": os.path.join(session_dir, "right_off"),
            },
        }
        for phase in self.phase_dirs.values():
            make_dir(phase["left"])
            make_dir(phase["right"])

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def put(self, item: dict):
        self.q.put(item)

    def close(self):
        self.q.join()
        self.q.put(None)
        self.thread.join()

    def _worker(self):
        with open(self.jsonl_path, "a", encoding="utf-8") as jf:
            while True:
                item = self.q.get()
                try:
                    if item is None:
                        return
                    self._save_item(item, jf)
                finally:
                    self.q.task_done()

    def _save_item(self, item: dict, jf):
        phase = item["phase"]
        phase_idx = item["phase_index"]
        stem = f"{phase_idx:06d}_{int(item['timestamp_ms_left'])}"

        left_path = os.path.join(self.phase_dirs[phase]["left"], f"{stem}.png")
        right_path = os.path.join(self.phase_dirs[phase]["right"], f"{stem}.png")

        cv2.imwrite(left_path, item["ir_left"], [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression])
        cv2.imwrite(right_path, item["ir_right"], [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression])

        rec = item["meta"].copy()
        rec["files"] = {
            "left": os.path.relpath(left_path, self.session_dir),
            "right": os.path.relpath(right_path, self.session_dir),
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()


# ---------------------------- pipeline / sensor setup ----------------------------

def start_pipeline_ir_only(serial: str):
    pipe = rs.pipeline()
    cfg = rs.config()

    if serial:
        cfg.enable_device(serial)

    ir_format = rs.format.y8 if IR_FORMAT == "y8" else rs.format.y16
    cfg.enable_stream(rs.stream.infrared, 1, IR_W, IR_H, ir_format, STREAM_FPS)
    cfg.enable_stream(rs.stream.infrared, 2, IR_W, IR_H, ir_format, STREAM_FPS)

    profile = pipe.start(cfg)
    return pipe, profile


def current_sensor_state(depth_sensor):
    return {
        "emitter_enabled": safe_get_option(depth_sensor, rs.option.emitter_enabled),
        "laser_power": safe_get_option(depth_sensor, rs.option.laser_power),
        "exposure": safe_get_option(depth_sensor, rs.option.exposure),
        "gain": safe_get_option(depth_sensor, rs.option.gain),
        "auto_exposure": safe_get_option(depth_sensor, rs.option.enable_auto_exposure),
        "emitter_on_off": safe_get_option_by_name(depth_sensor, "emitter_on_off"),
    }


def choose_on_laser_power(depth_sensor):
    if not depth_sensor.supports(rs.option.laser_power):
        return None

    rng = depth_sensor.get_option_range(rs.option.laser_power)
    if USE_MAX_LASER_POWER:
        return float(rng.max)

    if MANUAL_LASER_POWER is None:
        return float(getattr(rng, "def"))

    return float(np.clip(MANUAL_LASER_POWER, rng.min, rng.max))


def projector_state_matches(readback: dict, enable: bool, laser_power_on):
    em = readback.get("emitter_enabled")
    lp = readback.get("laser_power")
    eoo = readback.get("emitter_on_off")

    # Important practical note:
    # on some D435/D435i stacks emitter_enabled readback can stay at 1.0 even when
    # laser_power has already dropped to 0.0. For OFF validation we therefore trust
    # laser_power first and treat emitter_enabled as advisory only.
    if enable:
        ok_em = True if em is None else (em >= 0.5)
        ok_lp = True if (lp is None or laser_power_on is None) else abs(lp - laser_power_on) <= 1e-3
    else:
        ok_em = True  # advisory only in OFF mode
        ok_lp = True if lp is None else lp <= 1e-3

    # If emitter_on_off exists, force it to 0 in both phases to avoid frame-wise strobing modes.
    ok_eoo = True if eoo is None else abs(eoo - 0.0) <= 1e-3
    return ok_em and ok_lp and ok_eoo


def apply_projector_state(depth_sensor, enable: bool, laser_power_on, max_attempts: int = 3):
    requested = {
        "enable": bool(enable),
        "laser_power_target_on": laser_power_on,
        "requested_emitter_enabled": 1.0 if enable else 0.0,
        "requested_laser_power": laser_power_on if enable else 0.0,
        "requested_emitter_on_off": 0.0,
    }

    last_readback = current_sensor_state(depth_sensor)

    for attempt in range(1, max_attempts + 1):
        # Force continuous mode if such option exists.
        safe_set_option_by_name(depth_sensor, "emitter_on_off", 0.0)

        if enable:
            if laser_power_on is not None:
                safe_set_option(depth_sensor, rs.option.laser_power, float(laser_power_on))
            safe_set_option(depth_sensor, rs.option.emitter_enabled, 1.0)
            if laser_power_on is not None:
                safe_set_option(depth_sensor, rs.option.laser_power, float(laser_power_on))
        else:
            # OFF must explicitly set BOTH fields.
            if depth_sensor.supports(rs.option.laser_power):
                safe_set_option(depth_sensor, rs.option.laser_power, 0.0)
            safe_set_option(depth_sensor, rs.option.emitter_enabled, 0.0)
            if depth_sensor.supports(rs.option.laser_power):
                safe_set_option(depth_sensor, rs.option.laser_power, 0.0)

        time.sleep(0.05)
        last_readback = current_sensor_state(depth_sensor)

        if projector_state_matches(last_readback, enable=enable, laser_power_on=laser_power_on):
            return {
                "success": True,
                "attempts": attempt,
                "requested": requested,
                "readback": last_readback,
            }

    return {
        "success": False,
        "attempts": max_attempts,
        "requested": requested,
        "readback": last_readback,
    }


def warmup_and_lock(depth_sensor, pipeline):
    result = {
        "warmup_frames": int(WARMUP_FRAMES),
        "lock_requested": bool(LOCK_IR_AFTER_WARMUP),
        "manual_exposure": MANUAL_EXPOSURE,
        "manual_gain": MANUAL_GAIN,
        "applied": False,
        "exposure": None,
        "gain": None,
    }

    for _ in range(max(0, WARMUP_FRAMES)):
        try:
            pipeline.wait_for_frames(TIMEOUT_MS)
        except Exception:
            pass

    exp_supported = depth_sensor.supports(rs.option.exposure)
    gain_supported = depth_sensor.supports(rs.option.gain)
    ae_supported = depth_sensor.supports(rs.option.enable_auto_exposure)

    if MANUAL_EXPOSURE is not None:
        if ae_supported:
            safe_set_option(depth_sensor, rs.option.enable_auto_exposure, 0.0)
        if exp_supported:
            safe_set_option(depth_sensor, rs.option.exposure, float(MANUAL_EXPOSURE))
        if gain_supported and MANUAL_GAIN is not None:
            safe_set_option(depth_sensor, rs.option.gain, float(MANUAL_GAIN))

        result["applied"] = True
        result["exposure"] = safe_get_option(depth_sensor, rs.option.exposure)
        result["gain"] = safe_get_option(depth_sensor, rs.option.gain)
        return result

    if LOCK_IR_AFTER_WARMUP and ae_supported and exp_supported and gain_supported:
        exposure = depth_sensor.get_option(rs.option.exposure)
        gain = depth_sensor.get_option(rs.option.gain)

        safe_set_option(depth_sensor, rs.option.enable_auto_exposure, 0.0)
        safe_set_option(depth_sensor, rs.option.exposure, exposure)
        safe_set_option(depth_sensor, rs.option.gain, gain)

        result["applied"] = True
        result["exposure"] = safe_get_option(depth_sensor, rs.option.exposure)
        result["gain"] = safe_get_option(depth_sensor, rs.option.gain)

    return result


def build_capture_meta(profile, depth_sensor, lock_info, session_dir, laser_power_on):
    ir1_vsp = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    ir2_vsp = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()

    meta = {
        "capture_time_local": now_str(),
        "session_dir": session_dir,
        "purpose": "projector_pattern_capture_for_on_off_reconstruction",
        "protocol": {
            "ir_w": IR_W,
            "ir_h": IR_H,
            "ir_format": IR_FORMAT,
            "stream_fps": STREAM_FPS,
            "preview_scale": PREVIEW_SCALE,
            "preview_fps": PREVIEW_FPS,
            "record_fps": RECORD_FPS,
            "n_frames_per_phase": N_FRAMES_PER_PHASE,
            "phases": ["on", "off"],
            "warmup_frames": WARMUP_FRAMES,
            "flush_frames_after_toggle": FLUSH_FRAMES_AFTER_TOGGLE,
            "lock_ir_after_warmup": LOCK_IR_AFTER_WARMUP,
            "use_max_laser_power": USE_MAX_LASER_POWER,
            "manual_laser_power": MANUAL_LASER_POWER,
            "manual_exposure": MANUAL_EXPOSURE,
            "manual_gain": MANUAL_GAIN,
        },
        "selected_laser_power_on": laser_power_on,
        "device": {
            "name": str(profile.get_device()),
        },
        "sensor_option_ranges": {
            "laser_power": get_option_range(depth_sensor, rs.option.laser_power),
            "exposure": get_option_range(depth_sensor, rs.option.exposure),
            "gain": get_option_range(depth_sensor, rs.option.gain),
        },
        "locked_ir": lock_info,
        "sensor_state_before_capture": current_sensor_state(depth_sensor),
        "streams": {
            "IR_LEFT": intrinsics_to_dict(ir1_vsp.get_intrinsics()),
            "IR_RIGHT": intrinsics_to_dict(ir2_vsp.get_intrinsics()),
        },
        "extrinsics": {
            "IR_RIGHT_to_IR_LEFT": extrinsics_to_dict(ir2_vsp.get_extrinsics_to(ir1_vsp)),
        },
        "notes": [
            "Raw IR frames only.",
            "Single fixed exposure/gain across ON and OFF phases.",
            "OFF phase explicitly forces emitter_enabled=0 and laser_power=0.",
            "OFF validation trusts laser_power==0 more than emitter_enabled readback.",
            "Use flat matte wall and stable ambient illumination.",
        ],
    }
    return meta


# ---------------------------- live capture ----------------------------

def latest_frames(pipeline):
    frames = pipeline.wait_for_frames(TIMEOUT_MS)
    while True:
        more = pipeline.poll_for_frames()
        if not more:
            break
        frames = more
    return frames


def draw_preview(ir_left, ir_right, phase_name, saved_idx, target_count, sensor_state, preview_fps_est):
    left = preview_uint8(ir_left).copy()
    right = preview_uint8(ir_right).copy()

    # For preview, prefer actual projector power over emitter_enabled readback because
    # some stacks keep emitter_enabled=1.0 even when the projector is effectively off.
    lp = sensor_state.get("laser_power")
    if lp is not None:
        emitter_text = "ON" if lp > 1e-3 else "OFF"
    else:
        emitter_flag = sensor_state.get("emitter_enabled")
        emitter_text = "ON" if (emitter_flag is not None and emitter_flag >= 0.5) else "OFF"

    text1 = f"phase: {phase_name.upper()}  saved: {saved_idx}/{target_count}"
    text2 = f"emitter: {emitter_text}  fps: {preview_fps_est:4.1f}"
    text3 = f"exp: {sensor_state['exposure']}  gain: {sensor_state['gain']}  laser: {sensor_state['laser_power']}"

    for img, side in ((left, "LEFT"), (right, "RIGHT")):
        cv2.putText(img, f"{side}  {IR_W}x{IR_H} {IR_FORMAT}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2, cv2.LINE_AA)
        cv2.putText(img, text1, (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, 255, 2, cv2.LINE_AA)
        cv2.putText(img, text2, (10, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, 255, 2, cv2.LINE_AA)
        cv2.putText(img, text3, (10, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 1, cv2.LINE_AA)

    left = resize_for_preview(left, PREVIEW_SCALE)
    right = resize_for_preview(right, PREVIEW_SCALE)

    cv2.imshow(WINDOW_LEFT, left)
    cv2.imshow(WINDOW_RIGHT, right)


def flush_after_toggle(pipeline, depth_sensor, phase_name):
    preview_last = 0.0
    preview_dt = 1.0 / PREVIEW_FPS if PREVIEW_FPS > 0 else 0.0

    for i in range(FLUSH_FRAMES_AFTER_TOGGLE):
        frames = latest_frames(pipeline)
        f_left = frames.get_infrared_frame(1)
        f_right = frames.get_infrared_frame(2)
        if not (f_left and f_right):
            continue

        ir_left = np.asanyarray(f_left.get_data())
        ir_right = np.asanyarray(f_right.get_data())

        now = time.time()
        if PREVIEW_FPS <= 0 or now - preview_last >= preview_dt:
            sensor_state = current_sensor_state(depth_sensor)
            draw_preview(
                ir_left=ir_left,
                ir_right=ir_right,
                phase_name=f"{phase_name}_flush",
                saved_idx=i + 1,
                target_count=FLUSH_FRAMES_AFTER_TOGGLE,
                sensor_state=sensor_state,
                preview_fps_est=PREVIEW_FPS,
            )
            preview_last = now

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            raise KeyboardInterrupt


def capture_phase(phase_name, enable_projector, pipeline, depth_sensor, writer, laser_power_on):
    switch_info = apply_projector_state(
        depth_sensor=depth_sensor,
        enable=enable_projector,
        laser_power_on=laser_power_on,
        max_attempts=3,
    )
    if not switch_info["success"]:
        raise RuntimeError(
            f"Failed to apply projector state for phase '{phase_name}'. "
            f"Readback: {switch_info['readback']}"
        )

    flush_after_toggle(pipeline, depth_sensor, phase_name)

    saved = 0
    phase_start_wall = time.time()
    last_preview_t = 0.0
    last_saved_t = 0.0
    preview_interval = 0.0 if PREVIEW_FPS <= 0 else (1.0 / PREVIEW_FPS)
    record_interval = 0.0 if RECORD_FPS <= 0 else (1.0 / RECORD_FPS)

    while saved < N_FRAMES_PER_PHASE:
        frames = latest_frames(pipeline)
        f_left = frames.get_infrared_frame(1)
        f_right = frames.get_infrared_frame(2)
        if not (f_left and f_right):
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                raise KeyboardInterrupt
            continue

        ir_left = np.asanyarray(f_left.get_data())
        ir_right = np.asanyarray(f_right.get_data())

        now = time.time()

        if PREVIEW_FPS <= 0 or (now - last_preview_t) >= preview_interval:
            dt = max(now - last_preview_t, 1e-6) if last_preview_t > 0 else 0.0
            fps_est = (1.0 / dt) if dt > 0 else 0.0
            sensor_state = current_sensor_state(depth_sensor)
            draw_preview(
                ir_left=ir_left,
                ir_right=ir_right,
                phase_name=phase_name,
                saved_idx=saved,
                target_count=N_FRAMES_PER_PHASE,
                sensor_state=sensor_state,
                preview_fps_est=fps_est,
            )
            last_preview_t = now

        due = (RECORD_FPS <= 0.0) or (saved == 0) or ((now - last_saved_t) >= record_interval)
        if due:
            ts_left = frame_timestamp_ms(f_left)
            ts_right = frame_timestamp_ms(f_right)
            sensor_state = current_sensor_state(depth_sensor)

            writer.put({
                "phase": phase_name,
                "phase_index": saved,
                "timestamp_ms_left": ts_left,
                "timestamp_ms_right": ts_right,
                "ir_left": ir_left.copy(),
                "ir_right": ir_right.copy(),
                "meta": {
                    "phase": phase_name,
                    "phase_index": saved,
                    "global_elapsed_sec": round(now - phase_start_wall, 6),
                    "timestamp_ms_left": ts_left,
                    "timestamp_ms_right": ts_right,
                    "frame_number_left": frame_number(f_left),
                    "frame_number_right": frame_number(f_right),
                    "timestamp_delta_ms": float(ts_right - ts_left),
                    "sensor_state": sensor_state,
                    "projector_switch_readback_at_phase_start": switch_info["readback"],
                },
            })
            saved += 1
            last_saved_t = now

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            raise KeyboardInterrupt

    return {
        "phase": phase_name,
        "saved_pairs": saved,
        "projector_switch": switch_info,
        "sensor_state_after_phase": current_sensor_state(depth_sensor),
    }


# ---------------------------- main ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fixed-protocol RealSense D435 IR capture for projector pattern reconstruction"
    )
    parser.add_argument("--serial", type=str, default="", help="Optional camera serial")
    parser.add_argument("--out", type=str, default="wall_capture", help="Root output directory")
    args = parser.parse_args()

    make_dir(args.out)

    pipeline = None
    writer = None

    try:
        pipeline, profile = start_pipeline_ir_only(args.serial)
        depth_sensor = profile.get_device().first_depth_sensor()

        laser_power_on = choose_on_laser_power(depth_sensor)

        # Warm up in ON state because this is the photometric regime that dominates saturation/AE.
        warmup_switch = apply_projector_state(
            depth_sensor=depth_sensor,
            enable=True,
            laser_power_on=laser_power_on,
            max_attempts=3,
        )
        if not warmup_switch["success"]:
            raise RuntimeError(
                f"Failed to enter warmup ON state. Readback: {warmup_switch['readback']}"
            )

        lock_info = warmup_and_lock(depth_sensor, pipeline)

        # Re-assert ON state after lock so the run starts from a verified state.
        on_reassert = apply_projector_state(
            depth_sensor=depth_sensor,
            enable=True,
            laser_power_on=laser_power_on,
            max_attempts=3,
        )
        if not on_reassert["success"]:
            raise RuntimeError(
                f"Failed to re-assert ON state after lock. Readback: {on_reassert['readback']}"
            )

        session_name = f"session_{now_str()}"
        session_dir = os.path.join(args.out, session_name)
        make_dir(session_dir)

        capture_meta = build_capture_meta(
            profile=profile,
            depth_sensor=depth_sensor,
            lock_info=lock_info,
            session_dir=session_dir,
            laser_power_on=laser_power_on,
        )
        capture_meta["warmup_switch"] = warmup_switch
        capture_meta["on_reassert_after_lock"] = on_reassert
        capture_meta["sensor_state_after_lock"] = current_sensor_state(depth_sensor)
        save_json(os.path.join(session_dir, "capture_meta.json"), capture_meta)

        writer = AsyncPairWriter(
            session_dir=session_dir,
            png_compression=PNG_COMPRESSION,
            max_queue=WRITER_QUEUE,
        )

        cv2.namedWindow(WINDOW_LEFT, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WINDOW_RIGHT, cv2.WINDOW_NORMAL)

        print("Fixed capture protocol (patched)")
        print(f"  IR stream: {IR_W}x{IR_H} {IR_FORMAT} @ {STREAM_FPS}")
        print(f"  Preview  : scale={PREVIEW_SCALE}, fps={PREVIEW_FPS}")
        print(f"  Record   : {N_FRAMES_PER_PHASE} pairs ON + {N_FRAMES_PER_PHASE} pairs OFF @ {RECORD_FPS} fps")
        print(f"  Laser ON : {laser_power_on}")
        print(f"  Lock     : {lock_info}")
        print(f"  Output   : {session_dir}")
        print("Press q / ESC to abort")

        summary = {
            "session_dir": session_dir,
            "on": capture_phase("on", True, pipeline, depth_sensor, writer, laser_power_on),
            "off": capture_phase("off", False, pipeline, depth_sensor, writer, laser_power_on),
            "final_sensor_state": current_sensor_state(depth_sensor),
        }
        save_json(os.path.join(session_dir, "capture_summary.json"), summary)
        print("Capture completed successfully")

    except KeyboardInterrupt:
        print("Capture aborted by user")

    finally:
        if writer is not None:
            writer.close()
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
