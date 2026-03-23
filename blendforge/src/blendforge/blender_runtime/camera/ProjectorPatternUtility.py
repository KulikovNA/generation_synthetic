from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def _resolve_pattern_path(path: str, *, base_dir: Optional[str] = None) -> Path:
    p = Path(path).expanduser()
    if base_dir not in (None, ""):
        base = Path(base_dir).expanduser()
        if p.is_absolute():
            if p.exists():
                return p.resolve()

            tail_candidates = []
            if len(p.parts) >= 2:
                tail_candidates.append(Path(*p.parts[-2:]))
            if len(p.parts) >= 3:
                tail_candidates.append(Path(*p.parts[-3:]))
            tail_candidates.append(Path(p.name))

            for rel_tail in tail_candidates:
                direct = base / rel_tail
                if direct.exists():
                    return direct.resolve()
                for root in [*list(base.parents[:4])]:
                    candidate = root / rel_tail
                    if candidate.exists():
                        return candidate.resolve()
        else:
            candidate = base / p
            if candidate.exists():
                return candidate.resolve()

            matches = [m for m in base.rglob(str(p)) if m.is_file()]
            if len(matches) == 1:
                return matches[0].resolve()

            for root in [*list(base.parents[:4])]:
                candidate = root / p
                if candidate.exists():
                    return candidate.resolve()
            p = base / p
    elif not p.is_absolute():
        # Backward-compatible fallback for old call sites that only passed a relative path.
        p = Path.cwd() / p
    return p.resolve()


def _ensure_rgba_u8(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.ndim == 2:
        gray = x.astype(np.uint8, copy=False)
        rgba = np.zeros((gray.shape[0], gray.shape[1], 4), dtype=np.uint8)
        rgba[:, :, 0] = gray
        rgba[:, :, 1] = gray
        rgba[:, :, 2] = gray
        rgba[:, :, 3] = 255
        return rgba

    if x.ndim != 3:
        raise ValueError(f"Projector pattern must be 2D or 3D image, got shape {x.shape}")

    if x.shape[2] == 4:
        return x.astype(np.uint8, copy=False)

    if x.shape[2] == 3:
        rgba = np.zeros((x.shape[0], x.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = x.astype(np.uint8, copy=False)
        rgba[:, :, 3] = 255
        return rgba

    if x.shape[2] == 1:
        return _ensure_rgba_u8(x[:, :, 0])

    raise ValueError(f"Unsupported projector pattern channels: {x.shape[2]}")


def load_projector_pattern(path: str, *, base_dir: Optional[str] = None) -> np.ndarray:
    resolved = _resolve_pattern_path(path, base_dir=base_dir)
    if not resolved.is_file():
        raise FileNotFoundError(f"Projector pattern image not found: {resolved}")

    img = cv2.imread(str(resolved), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read projector pattern image: {resolved}")

    if img.ndim == 3 and img.shape[2] in (3, 4):
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA if img.shape[2] == 4 else cv2.COLOR_BGR2RGB)

    return _ensure_rgba_u8(img)


def apply_pattern_orientation(
    pattern_rgba: np.ndarray,
    *,
    flip_u: bool = False,
    flip_v: bool = False,
) -> np.ndarray:
    return apply_pattern_postprocess(pattern_rgba, flip_u=flip_u, flip_v=flip_v, dot_sigma_px=None)


def apply_pattern_postprocess(
    pattern_rgba: np.ndarray,
    *,
    flip_u: bool = False,
    flip_v: bool = False,
    dot_sigma_px: Optional[float] = None,
) -> np.ndarray:
    out = np.asarray(pattern_rgba).copy()
    if flip_u:
        out = np.ascontiguousarray(np.flip(out, axis=1))
    if flip_v:
        out = np.ascontiguousarray(np.flip(out, axis=0))
    if dot_sigma_px is not None and float(dot_sigma_px) > 1e-8:
        sigma = float(dot_sigma_px)
        k = max(3, int(math.ceil(sigma * 6.0)))
        if k % 2 == 0:
            k += 1
        out = cv2.GaussianBlur(out, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    return out.astype(np.uint8, copy=False)


def generate_seeded_projector_pattern(
    width: int,
    height: int,
    dot_count: int,
    *,
    min_sep_px: Optional[float] = None,
    dot_radius_px: Optional[float] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    width = int(width)
    height = int(height)
    dot_count = int(dot_count)
    radius = max(1, int(round(1.0 if dot_radius_px is None else float(dot_radius_px))))
    min_sep = max(0.0, 0.0 if min_sep_px is None else float(min_sep_px))
    rng = np.random.default_rng(0 if seed is None else int(seed))

    if width <= 0 or height <= 0:
        raise ValueError(f"Pattern width/height must be > 0, got {width}x{height}")
    if dot_count < 0:
        raise ValueError(f"dot_count must be >= 0, got {dot_count}")

    x_min = radius
    y_min = radius
    x_max = width - radius - 1
    y_max = height - radius - 1
    if x_min > x_max or y_min > y_max:
        raise ValueError(
            f"Pattern resolution {width}x{height} is too small for dot_radius_px={radius}"
        )

    pattern = np.zeros((height, width, 4), dtype=np.uint8)
    if dot_count == 0:
        return pattern

    accepted: list[tuple[float, float]] = []

    if min_sep > 0.0:
        cell = max(min_sep, 1.0)
        grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
        max_attempts = max(dot_count * 100, 2000)
        attempts = 0
        min_sep_sq = min_sep * min_sep

        while len(accepted) < dot_count and attempts < max_attempts:
            attempts += 1
            x = float(rng.uniform(x_min, x_max + 1.0))
            y = float(rng.uniform(y_min, y_max + 1.0))

            gx = int(x // cell)
            gy = int(y // cell)
            ok = True

            for ny in range(gy - 2, gy + 3):
                for nx in range(gx - 2, gx + 3):
                    for px, py in grid.get((nx, ny), ()):
                        dx = x - px
                        dy = y - py
                        if (dx * dx + dy * dy) < min_sep_sq:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break

            if not ok:
                continue

            accepted.append((x, y))
            grid.setdefault((gx, gy), []).append((x, y))

        if len(accepted) < dot_count:
            raise RuntimeError(
                "Could not place requested number of projector dots with the given "
                f"constraints: width={width}, height={height}, dot_count={dot_count}, "
                f"min_sep_px={min_sep}, dot_radius_px={radius}, seed={0 if seed is None else int(seed)}. "
                f"Placed only {len(accepted)} dots."
            )
    else:
        for _ in range(dot_count):
            x = float(rng.uniform(x_min, x_max + 1.0))
            y = float(rng.uniform(y_min, y_max + 1.0))
            accepted.append((x, y))

    for x, y in accepted:
        center = (int(round(x)), int(round(y)))
        cv2.circle(pattern, center, radius, (255, 255, 255, 255), -1, lineType=cv2.LINE_8)

    return pattern


def load_or_generate_projector_pattern(
    *,
    path: Optional[str],
    width: int,
    height: int,
    dot_count: int,
    min_sep_px: Optional[float],
    dot_radius_px: Optional[float],
    seed: Optional[int],
    base_dir: Optional[str] = None,
) -> np.ndarray:
    if path not in (None, ""):
        return load_projector_pattern(str(path), base_dir=base_dir)

    return generate_seeded_projector_pattern(
        width=width,
        height=height,
        dot_count=dot_count,
        min_sep_px=min_sep_px,
        dot_radius_px=dot_radius_px,
        seed=seed,
    )
