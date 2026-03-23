#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
BLENDFORGE_SRC = REPO_ROOT / "blendforge" / "src"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BLENDFORGE_SRC) not in sys.path:
    sys.path.insert(0, str(BLENDFORGE_SRC))

from blendforge.host.FiletoDict import Config


def parse_args(argv):
    p = argparse.ArgumentParser(description="Host-side launcher for texture sweep with one BlenderProc process per texture.")
    p.add_argument("--config_file", type=str, default=str(TOOLS_DIR / "config_effective_projector_texture_sweep_fixed.py"))
    p.add_argument("--script", type=str, default=str(TOOLS_DIR / "sweep_effective_projector_textures.py"))
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--start_from", type=str, default="Tiles079")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip_existing", action="store_true", default=True)
    return p.parse_args(argv)


def _load_bproc_env() -> dict:
    env = os.environ.copy()
    proc = subprocess.run(
        [str(REPO_ROOT / "scripts" / "bproc"), "env"],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        payload = line[len("export "):]
        if "=" not in payload:
            continue
        key, value = payload.split("=", 1)
        value = value.strip()
        if value[:1] == value[-1:] and value[:1] in ("'", '"'):
            value = value[1:-1]
        if "$PATH" in value:
            value = value.replace("$PATH", env.get("PATH", ""))
        env[key] = value
    return env


def _iter_texture_names(texture_root: Path):
    return sorted([p.name for p in texture_root.iterdir() if p.is_dir()])


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = Config(args.config_file)

    output_dir = Path(args.output_dir or getattr(cfg, "output_dir", str(TOOLS_DIR / "output" / "texture_sweep_fixed"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    texture_root = Path(cfg.cc_textures.cc_textures_path).resolve()
    texture_names = _iter_texture_names(texture_root)
    if args.start_from:
        if args.start_from not in texture_names:
            raise ValueError(f"Texture '{args.start_from}' was not found in {texture_root}")
        texture_names = texture_names[texture_names.index(args.start_from):]
    if args.limit is not None:
        texture_names = texture_names[: int(args.limit)]

    env = _load_bproc_env()
    completed = []
    failed = []

    for idx, texture_name in enumerate(texture_names, start=1):
        texture_dir = output_dir / texture_name
        done_marker = texture_dir / "000000_rgb.png"
        if args.skip_existing and done_marker.exists():
            print(f"[{idx}/{len(texture_names)}] {texture_name} | skip_existing", flush=True)
            continue

        cmd = [
            "blenderproc",
            "run",
            str(Path(args.script).resolve()),
            "--config_file",
            str(Path(args.config_file).resolve()),
            "--output_dir",
            str(output_dir),
            "--only_texture",
            texture_name,
        ]
        print(f"[{idx}/{len(texture_names)}] START {texture_name}", flush=True)
        try:
            subprocess.run(cmd, env=env, check=True)
            completed.append(texture_name)
        except subprocess.CalledProcessError as exc:
            print(f"[{idx}/{len(texture_names)}] FAIL {texture_name} | returncode={exc.returncode}", flush=True)
            failed.append(texture_name)
            break

    if completed:
        (output_dir / "_launcher_completed.txt").write_text("".join(f"{name}\n" for name in completed), encoding="utf-8")
    if failed:
        (output_dir / "_launcher_failed.txt").write_text("".join(f"{name}\n" for name in failed), encoding="utf-8")

    print(f"[Launcher] output_dir={output_dir}", flush=True)
    print(f"[Launcher] completed={len(completed)} failed={len(failed)}", flush=True)


if __name__ == "__main__":
    main()
