# blendforge/blender_runtime/LolWriterUtility.py

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from datetime import datetime

import numpy as np
from filelock import FileLock


def write_lol_annotations(
    output_root: str,
    split: str,
    input_colors: Sequence[np.ndarray],
    target_colors: Sequence[np.ndarray],
    *,
    jpeg_input_quality: Optional[int] = None,   # None => input PNG, иначе JPEG quality
    target_format: str = "PNG",                 # PNG/JPEG
    append_to_existing_output: bool = True,
    id_width: int = 6,                          # 000000
    file_prefix: str = "",                      # если пусто — глобальный счётчик по всему split
    lock_path: Optional[str] = None,            # рекомендую: os.path.join(output_root, ".lock")
    timestamp_utc: Optional[str] = None,
    domain: str = "sRGB",
    ev_mode: Optional[str] = None,
    ev_input: Optional[Union[float, int]] = None,
    ev_target: Optional[Union[float, int]] = None,
    camera: Optional[Dict[str, Any]] = None,
    render: Optional[Dict[str, Any]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    output_root/<split>/input/<id>.jpg|png
    output_root/<split>/target/<id>.png|jpg
    output_root/<split>/pairs.txt
    output_root/<split>/meta.jsonl
    """

    output_root = str(output_root)
    split = str(split)

    split_dir = os.path.join(output_root, split)
    input_dir = os.path.join(split_dir, "input")
    target_dir = os.path.join(split_dir, "target")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    pairs_path = os.path.join(split_dir, "pairs.txt")
    meta_path = os.path.join(split_dir, "meta.jsonl")

    if len(input_colors) != len(target_colors):
        raise RuntimeError(f"Length mismatch: input={len(input_colors)} target={len(target_colors)}")
    n = len(input_colors)
    if n == 0:
        return {"count": 0, "offset": 0, "split_dir": split_dir}

    if timestamp_utc is None:
        timestamp_utc = datetime.utcnow().isoformat()

    input_ext = "jpg" if jpeg_input_quality is not None else "png"
    target_fmt = str(target_format).upper()
    if target_fmt not in ("PNG", "JPEG"):
        raise ValueError(f"target_format must be 'PNG' or 'JPEG', got: {target_format}")
    target_ext = "png" if target_fmt == "PNG" else "jpg"

    # --- lock (желательно всегда) ---
    lock_ctx = FileLock(lock_path) if lock_path else _NullContext()

    with lock_ctx:
        # --- offset = "сколько уже было сохранено" (по max numeric id + 1) ---
        if append_to_existing_output:
            offset = _LolWriterUtility.compute_global_offset(
                input_dir=input_dir,
                target_dir=target_dir,
                file_prefix=file_prefix,
            )
        else:
            offset = 0

        # --- ids: просто счётчик + i, формат {i:06d} ---
        ids = [f"{file_prefix}{(offset + i):0{id_width}d}" for i in range(n)]

        with open(pairs_path, "a", encoding="utf-8") as pairs_f, \
             open(meta_path, "a", encoding="utf-8") as meta_f:

            for i in range(n):
                stem = ids[i]
                in_name = f"{stem}.{input_ext}"
                tg_name = f"{stem}.{target_ext}"

                in_path = os.path.join(input_dir, in_name)
                tg_path = os.path.join(target_dir, tg_name)

                img_in = _LolWriterUtility.to_uint8_rgb(input_colors[i])
                img_tg = _LolWriterUtility.to_uint8_rgb(target_colors[i])

                if jpeg_input_quality is not None:
                    _LolWriterUtility.save_jpeg(in_path, img_in, quality=int(jpeg_input_quality))
                else:
                    _LolWriterUtility.save_png(in_path, img_in)

                if target_fmt == "PNG":
                    _LolWriterUtility.save_png(tg_path, img_tg)
                else:
                    _LolWriterUtility.save_jpeg(tg_path, img_tg, quality=95)

                pairs_f.write(f"input/{in_name} target/{tg_name}\n")

                meta: Dict[str, Any] = {
                    "id": stem,
                    "timestamp_utc": timestamp_utc,
                    "domain": domain,
                }
                if ev_mode is not None:
                    meta["EV_mode"] = str(ev_mode)
                if ev_input is not None:
                    meta["EV_input"] = float(ev_input)
                if ev_target is not None:
                    meta["EV_target"] = float(ev_target)
                if camera is not None:
                    meta["camera"] = dict(camera)
                if render is not None:
                    meta["render"] = dict(render)
                if extra_meta is not None:
                    meta.update(dict(extra_meta))

                meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    return {"count": n, "offset": int(offset), "split_dir": split_dir}


class _LolWriterUtility:
    @staticmethod
    def compute_global_offset(input_dir: str, target_dir: str, file_prefix: str) -> int:
        """
        Ищем max numeric id среди:
          input/<prefix><digits>.{png,jpg}
          target/<prefix><digits>.{png,jpg}
        Возвращаем max+1.
        Это соответствует "сколько уже было сохранено" при монотонной нумерации.
        """
        max_id = -1
        for d in (input_dir, target_dir):
            p = Path(d)
            for ext in ("png", "jpg", "jpeg"):
                for fp in p.glob(f"{file_prefix}*.{ext}"):
                    stem = fp.stem
                    if file_prefix and not stem.startswith(file_prefix):
                        continue
                    num = stem[len(file_prefix):] if file_prefix else stem
                    if num.isdigit():
                        max_id = max(max_id, int(num))
        return max_id + 1

    @staticmethod
    def to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(f"Expected HxWx3 or HxWx4 image, got shape={arr.shape}")
        if arr.shape[2] == 4:
            arr = arr[..., :3]

        if np.issubdtype(arr.dtype, np.floating):
            mx = float(np.nanmax(arr)) if arr.size else 0.0
            if mx <= 1.5:
                arr = arr * 255.0
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
            return np.ascontiguousarray(arr)

        if arr.dtype == np.uint8:
            return np.ascontiguousarray(arr)

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    @staticmethod
    def save_png(path: str, rgb_uint8: np.ndarray) -> None:
        from PIL import Image
        Image.init()
        Image.fromarray(rgb_uint8, mode="RGB").save(str(path), format="PNG", compress_level=4)

    @staticmethod
    def save_jpeg(path: str, rgb_uint8: np.ndarray, quality: int = 85) -> None:
        from PIL import Image
        Image.init()
        Image.fromarray(rgb_uint8, mode="RGB").save(
            str(path),
            format="JPEG",
            quality=int(quality),
            subsampling=1,
            optimize=True,
        )


class _NullContext:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False
