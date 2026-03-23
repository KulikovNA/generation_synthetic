#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import collections
import pyrealsense2 as rs


def get_info(obj, info):
    try:
        if obj.supports(info):
            return obj.get_info(info)
    except Exception:
        pass
    return "N/A"


def enum_name(x):
    s = str(x)
    # pyrealsense2 обычно возвращает строки вида "stream.infrared"
    if "." in s:
        return s.split(".")[-1]
    return s


def choose_device(ctx, serial: str):
    devices = list(ctx.query_devices())
    if not devices:
        raise RuntimeError("Камеры RealSense не найдены.")

    if serial:
        for dev in devices:
            if get_info(dev, rs.camera_info.serial_number) == serial:
                return dev
        raise RuntimeError(f"Камера с serial={serial} не найдена.")

    if len(devices) > 1:
        print("Найдено несколько устройств. Будет выбрано первое:")
        for i, dev in enumerate(devices):
            print(
                f"  [{i}] "
                f"name={get_info(dev, rs.camera_info.name)}, "
                f"serial={get_info(dev, rs.camera_info.serial_number)}"
            )
    return devices[0]


def main():
    ap = argparse.ArgumentParser(
        description="Показать доступные разрешения / fps / форматы для RealSense камеры"
    )
    ap.add_argument("--serial", type=str, default="", help="Serial number камеры")
    ap.add_argument("--only_video", action="store_true", help="Показывать только видео-профили")
    args = ap.parse_args()

    ctx = rs.context()
    dev = choose_device(ctx, args.serial)

    dev_name = get_info(dev, rs.camera_info.name)
    serial = get_info(dev, rs.camera_info.serial_number)
    product_line = get_info(dev, rs.camera_info.product_line)
    fw = get_info(dev, rs.camera_info.firmware_version)

    print("=== DEVICE ===")
    print(f"Name         : {dev_name}")
    print(f"Serial       : {serial}")
    print(f"Product line : {product_line}")
    print(f"Firmware     : {fw}")
    print()

    for sensor in dev.query_sensors():
        sensor_name = get_info(sensor, rs.camera_info.name)
        print(f"=== SENSOR: {sensor_name} ===")

        grouped = collections.defaultdict(set)
        non_video = []

        for prof in sensor.get_stream_profiles():
            try:
                vsp = prof.as_video_stream_profile()
                stream = enum_name(prof.stream_type())
                idx = prof.stream_index()
                fmt = enum_name(prof.format())
                fps = prof.fps()
                w = vsp.width()
                h = vsp.height()

                grouped[(stream, idx, fmt, w, h)].add(fps)
            except Exception:
                if not args.only_video:
                    try:
                        stream = enum_name(prof.stream_type())
                        fmt = enum_name(prof.format())
                        fps = prof.fps()
                        non_video.append((stream, fmt, fps))
                    except Exception:
                        pass

        if not grouped and (args.only_video or not non_video):
            print("  Нет доступных профилей.")
            print()
            continue

        if grouped:
            # Сортируем: stream -> index -> format -> width -> height
            items = sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3], x[0][4]))
            last_stream = None

            for (stream, idx, fmt, w, h), fps_set in items:
                if stream != last_stream:
                    print(f"  [{stream}]")
                    last_stream = stream

                fps_list = ", ".join(str(x) for x in sorted(fps_set))
                print(
                    f"    index={idx:<2} "
                    f"format={fmt:<8} "
                    f"size={w}x{h:<5} "
                    f"fps=[{fps_list}]"
                )

        if non_video and not args.only_video:
            print("  [non-video profiles]")
            for stream, fmt, fps in sorted(non_video):
                print(f"    stream={stream:<12} format={fmt:<10} fps={fps}")

        print()


if __name__ == "__main__":
    main()