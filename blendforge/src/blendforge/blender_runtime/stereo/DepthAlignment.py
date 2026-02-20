# blendforge/blender_runtime/stereo/DepthAlignment.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, List, Literal

import numpy as np
import cv2


# ----------------------------
# Helpers
# ----------------------------

RectifyMode = Literal["auto", "on", "off"]
DepthValueMode = Literal["source_z", "target_z"]


def _K_as_params(K: np.ndarray) -> Tuple[float, float, float, float]:
    K = np.asarray(K, dtype=np.float64)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return fx, fy, cx, cy


def _np5(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    out = np.zeros(5, dtype=np.float64)
    if a.size:
        out[: min(5, a.size)] = a[: min(5, a.size)]
    return out


@dataclass(frozen=True)
class RectifyMeta:
    """
    Если depth/disp были получены в rectified координатах OpenCV stereoRectify:

      - P1: 3x4 (или 4x4) проекционная матрица левой rectified камеры
      - R1: 3x3 rectification rotation для левой камеры

    Тогда:
      p_rect -> p_left  через  R1.T
      K_rect берём из P1[:3,:3]
    """
    R1: np.ndarray
    P1: np.ndarray


def _build_rectify_meta_from_rs(
    rs: Any,
    left: str = "IR_LEFT",
    right: str = "IR_RIGHT",
    *,
    use_distortion: bool = False,
    alpha: float = 0.0,
) -> RectifyMeta:
    """
    Строит RectifyMeta из RealSenseProfile (intrinsics + extrinsics) через cv2.stereoRectify.

    Важно: мы считаем, что rs.get_T_cv(a,b) возвращает T_b_from_a (CV convention).
    Тогда:
      T_r_from_l = inv(T_c_from_r) @ T_c_from_l   (color как общий якорь)
    """
    sL = rs.get_stream(left)
    sR = rs.get_stream(right)

    K1 = np.asarray(sL.K, dtype=np.float64)
    K2 = np.asarray(sR.K, dtype=np.float64)

    if use_distortion:
        D1 = _np5(getattr(sL, "distortion_coeffs", [0, 0, 0, 0, 0]))
        D2 = _np5(getattr(sR, "distortion_coeffs", [0, 0, 0, 0, 0]))
    else:
        D1 = np.zeros(5, dtype=np.float64)
        D2 = np.zeros(5, dtype=np.float64)

    T_c_l = rs.get_T_cv(left, "COLOR")
    T_c_r = rs.get_T_cv(right, "COLOR")
    T_r_l = np.linalg.inv(np.asarray(T_c_r, dtype=np.float64)) @ np.asarray(T_c_l, dtype=np.float64)

    R = T_r_l[:3, :3].copy()
    t = T_r_l[:3, 3].copy()

    image_size = (int(sL.width), int(sL.height))  # (W,H)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2,
        image_size,
        R, t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=float(alpha),
    )
    return RectifyMeta(R1=np.asarray(R1, dtype=np.float64), P1=np.asarray(P1, dtype=np.float64))


def _select_rectify_mode_by_keep(
    *,
    keep_off: int,
    keep_on: int,
    rel_margin: float = 0.01,
) -> RectifyMode:
    """
    Простая эвристика:
      - если "on" даёт заметно больше валидных пикселей -> выбираем "on"
      - иначе -> "off"
    rel_margin=0.01 означает "на 1% больше" (и плюс 1 пиксель, чтобы не дрожать на нуле).
    """
    if keep_on > int((1.0 + float(rel_margin)) * keep_off) + 1:
        return "on"
    return "off"


# ----------------------------
# Core warp
# ----------------------------

def align_depth_to_target_grid(
    depth_src_m: np.ndarray,
    *,
    K_src: np.ndarray,
    K_tgt: np.ndarray,
    T_tgt_from_src: np.ndarray,
    tgt_wh: Tuple[int, int],
    invalid_value: float = 0.0,
    depth_value_mode: DepthValueMode = "source_z",  # "source_z" | "target_z"
    splat_2x2: bool = True,
    rectify: Optional[RectifyMeta] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp depth map из source-пикселей в target-пиксели.

    depth_value_mode:
      - "source_z": сохраняем исходную глубину (в source системе), но окклюзии разрешаем по Z_target
      - "target_z": сохраняем Z в target системе (геометрически консистентно)

    Возвращает:
      depth_tgt_m: (H_tgt, W_tgt) float32, invalid заполнен invalid_value
      mask_valid: (H_tgt, W_tgt) bool
    """
    depth = np.asarray(depth_src_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth_src_m must be 2D, got {depth.shape}")

    Wt, Ht = int(tgt_wh[0]), int(tgt_wh[1])

    # --- choose intrinsics for source (rectified or original) ---
    if rectify is not None:
        P1 = np.asarray(rectify.P1, dtype=np.float64)
        if P1.shape == (3, 4) or P1.shape == (4, 4):
            K_src_use = P1[:3, :3]
        else:
            raise ValueError(f"RectifyMeta.P1 must be 3x4 or 4x4, got {P1.shape}")
        R_rect_to_src = np.asarray(rectify.R1, dtype=np.float64).T  # p_left = R1.T @ p_rect
    else:
        K_src_use = np.asarray(K_src, dtype=np.float64)
        R_rect_to_src = None

    fx_s, fy_s, cx_s, cy_s = _K_as_params(K_src_use)
    fx_t, fy_t, cx_t, cy_t = _K_as_params(np.asarray(K_tgt, dtype=np.float64))

    T = np.asarray(T_tgt_from_src, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_tgt_from_src must be 4x4, got {T.shape}")
    R = T[:3, :3]
    t = T[:3, 3]

    Hs, Ws = depth.shape

    # --- build pixel grid ---
    uu, vv = np.meshgrid(
        np.arange(Ws, dtype=np.float32),
        np.arange(Hs, dtype=np.float32),
    )  # (Hs,Ws)

    Zs = depth
    valid = np.isfinite(Zs) & (Zs > 0.0)
    if not np.any(valid):
        out = np.full((Ht, Wt), float(invalid_value), dtype=np.float32)
        return out, np.zeros((Ht, Wt), dtype=bool)

    uu = uu[valid]
    vv = vv[valid]
    Zs = Zs[valid]

    # --- backproject to 3D in source (or rectified-source) camera coords ---
    X = (uu - cx_s) / fx_s * Zs
    Y = (vv - cy_s) / fy_s * Zs
    P = np.stack([X, Y, Zs], axis=0)  # (3,N)

    # --- if rectified, rotate back to original source camera coords ---
    if R_rect_to_src is not None:
        P = R_rect_to_src @ P  # (3,N)

    # --- transform to target camera coords ---
    Pc = (R @ P) + t.reshape(3, 1)  # (3,N)
    Xc, Yc, Zc = Pc[0], Pc[1], Pc[2]

    ok = np.isfinite(Zc) & (Zc > 1e-6)
    if not np.any(ok):
        out = np.full((Ht, Wt), float(invalid_value), dtype=np.float32)
        return out, np.zeros((Ht, Wt), dtype=bool)

    Xc = Xc[ok]
    Yc = Yc[ok]
    Zc = Zc[ok]
    Zs_ok = Zs[ok]

    # --- project to target pixels ---
    ut = fx_t * (Xc / Zc) + cx_t
    vt = fy_t * (Yc / Zc) + cy_t

    ui = np.rint(ut).astype(np.int32)
    vi = np.rint(vt).astype(np.int32)

    # value to store
    if depth_value_mode == "target_z":
        val = Zc.astype(np.float32)
    elif depth_value_mode == "source_z":
        val = Zs_ok.astype(np.float32)
    else:
        raise ValueError("depth_value_mode must be 'source_z' or 'target_z'")

    # z-buffer for occlusion must be in TARGET coords
    zbuf = Zc.astype(np.float32)

    # --- splat with z-buffer keeping corresponding val ---
    offsets = [(0, 0)]
    if splat_2x2:
        offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]

    idx_all: List[np.ndarray] = []
    z_all: List[np.ndarray] = []
    v_all: List[np.ndarray] = []

    for du, dv in offsets:
        u2 = ui + int(du)
        v2 = vi + int(dv)
        inside = (u2 >= 0) & (u2 < Wt) & (v2 >= 0) & (v2 < Ht)
        if not np.any(inside):
            continue
        idx = (v2[inside] * Wt + u2[inside]).astype(np.int64)
        idx_all.append(idx)
        z_all.append(zbuf[inside])
        v_all.append(val[inside])

    if not idx_all:
        out = np.full((Ht, Wt), float(invalid_value), dtype=np.float32)
        return out, np.zeros((Ht, Wt), dtype=bool)

    idx_all = np.concatenate(idx_all, axis=0)
    z_all = np.concatenate(z_all, axis=0)
    v_all = np.concatenate(v_all, axis=0)

    # sort by (idx, z) => first occurrence per idx is minimal z
    order = np.lexsort((z_all, idx_all))
    idx_s = idx_all[order]
    z_s = z_all[order]
    v_s = v_all[order]

    uniq_idx, first_pos = np.unique(idx_s, return_index=True)

    out_flat = np.full((Ht * Wt,), float(invalid_value), dtype=np.float32)
    out_flat[uniq_idx] = v_s[first_pos]

    mask_flat = np.zeros((Ht * Wt,), dtype=bool)
    mask_flat[uniq_idx] = True

    return out_flat.reshape(Ht, Wt), mask_flat.reshape(Ht, Wt)


# ----------------------------
# Public wrappers (auto rectify)
# ----------------------------

def align_depth_ir_left_to_color(
    rs: Any,
    depth_ir_m: np.ndarray,
    *,
    depth_value_mode: DepthValueMode = "source_z",
    splat_2x2: bool = True,
    rectify: Optional[RectifyMeta] = None,
    rectify_mode: RectifyMode = "auto",
    # если когда-нибудь захочешь включить дисторсию — сделай параметром;
    # сейчас оставляем False, как договорились
    rectify_use_distortion: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper под RealSenseProfile.

    rectify_mode:
      - "off": считаем depth в оригинальной IR_LEFT сетке
      - "on" : считаем depth в rectified IR_LEFT сетке (используем rectify/или строим из rs)
      - "auto": пробуем оба варианта и выбираем тот, где больше валидных пикселей после warp
    """
    s_src = rs.get_stream("IR_LEFT")
    s_tgt = rs.get_stream("COLOR")

    K_src = np.asarray(s_src.K, dtype=np.float64)
    K_tgt = np.asarray(s_tgt.K, dtype=np.float64)
    T_c_from_l = np.asarray(rs.get_T_cv("IR_LEFT", "COLOR"), dtype=np.float64)

    tgt_wh = (int(s_tgt.width), int(s_tgt.height))

    if rectify_mode == "off":
        return align_depth_to_target_grid(
            depth_ir_m,
            K_src=K_src,
            K_tgt=K_tgt,
            T_tgt_from_src=T_c_from_l,
            tgt_wh=tgt_wh,
            invalid_value=0.0,
            depth_value_mode=depth_value_mode,
            splat_2x2=splat_2x2,
            rectify=None,
        )

    # "on" или "auto" => нужен RectifyMeta
    meta = rectify
    if meta is None:
        meta = _build_rectify_meta_from_rs(
            rs,
            left="IR_LEFT",
            right="IR_RIGHT",
            use_distortion=bool(rectify_use_distortion),
            alpha=0.0,
        )

    if rectify_mode == "on":
        return align_depth_to_target_grid(
            depth_ir_m,
            K_src=K_src,
            K_tgt=K_tgt,
            T_tgt_from_src=T_c_from_l,
            tgt_wh=tgt_wh,
            invalid_value=0.0,
            depth_value_mode=depth_value_mode,
            splat_2x2=splat_2x2,
            rectify=meta,
        )

    if rectify_mode != "auto":
        raise ValueError("rectify_mode must be one of: 'auto' | 'on' | 'off'")

    # auto: считаем оба и выбираем, где больше валидных пикселей
    d_off, m_off = align_depth_to_target_grid(
        depth_ir_m,
        K_src=K_src,
        K_tgt=K_tgt,
        T_tgt_from_src=T_c_from_l,
        tgt_wh=tgt_wh,
        invalid_value=0.0,
        depth_value_mode=depth_value_mode,
        splat_2x2=splat_2x2,
        rectify=None,
    )
    d_on, m_on = align_depth_to_target_grid(
        depth_ir_m,
        K_src=K_src,
        K_tgt=K_tgt,
        T_tgt_from_src=T_c_from_l,
        tgt_wh=tgt_wh,
        invalid_value=0.0,
        depth_value_mode=depth_value_mode,
        splat_2x2=splat_2x2,
        rectify=meta,
    )

    keep_off = int(m_off.sum())
    keep_on = int(m_on.sum())

    mode = _select_rectify_mode_by_keep(keep_off=keep_off, keep_on=keep_on, rel_margin=0.01)
    return (d_on, m_on) if mode == "on" else (d_off, m_off)


def align_depth_series_ir_left_to_color(
    rs: Any,
    depths_ir_m: List[np.ndarray],
    *,
    depth_value_mode: DepthValueMode = "source_z",
    splat_2x2: bool = True,
    rectify: Optional[RectifyMeta] = None,
    rectify_mode: RectifyMode = "auto",
    rectify_use_distortion: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Серийный align с авто-выбором режима (rectified или нет) *один раз* по первому кадру,
    чтобы на всех кадрах было единообразно.

    Если rectify_mode:
      - "off": всегда без rectify
      - "on" : всегда с rectify
      - "auto": выбираем по первому кадру, затем фиксируем режим для остальных
    """
    if not depths_ir_m:
        return [], []

    # подготовим meta заранее (чтобы не пересчитывать)
    meta: Optional[RectifyMeta] = rectify
    if rectify_mode in ("auto", "on") and meta is None:
        meta = _build_rectify_meta_from_rs(
            rs,
            left="IR_LEFT",
            right="IR_RIGHT",
            use_distortion=bool(rectify_use_distortion),
            alpha=0.0,
        )

    out_d: List[np.ndarray] = []
    out_m: List[np.ndarray] = []

    if rectify_mode == "off":
        for d in depths_ir_m:
            dd, mm = align_depth_ir_left_to_color(
                rs,
                d,
                depth_value_mode=depth_value_mode,
                splat_2x2=splat_2x2,
                rectify=None,
                rectify_mode="off",
            )
            out_d.append(dd)
            out_m.append(mm)
        return out_d, out_m

    if rectify_mode == "on":
        for d in depths_ir_m:
            dd, mm = align_depth_ir_left_to_color(
                rs,
                d,
                depth_value_mode=depth_value_mode,
                splat_2x2=splat_2x2,
                rectify=meta,
                rectify_mode="on",
            )
            out_d.append(dd)
            out_m.append(mm)
        return out_d, out_m

    if rectify_mode != "auto":
        raise ValueError("rectify_mode must be one of: 'auto' | 'on' | 'off'")

    # AUTO: решаем по первому кадру и фиксируем
    d0 = depths_ir_m[0]

    dd_off, mm_off = align_depth_ir_left_to_color(
        rs,
        d0,
        depth_value_mode=depth_value_mode,
        splat_2x2=splat_2x2,
        rectify=None,
        rectify_mode="off",
    )
    dd_on, mm_on = align_depth_ir_left_to_color(
        rs,
        d0,
        depth_value_mode=depth_value_mode,
        splat_2x2=splat_2x2,
        rectify=meta,
        rectify_mode="on",
    )

    keep_off = int(mm_off.sum())
    keep_on = int(mm_on.sum())
    chosen = _select_rectify_mode_by_keep(keep_off=keep_off, keep_on=keep_on, rel_margin=0.01)

    if chosen == "on":
        out_d.append(dd_on)
        out_m.append(mm_on)
        for d in depths_ir_m[1:]:
            dd, mm = align_depth_ir_left_to_color(
                rs,
                d,
                depth_value_mode=depth_value_mode,
                splat_2x2=splat_2x2,
                rectify=meta,
                rectify_mode="on",
            )
            out_d.append(dd)
            out_m.append(mm)
    else:
        out_d.append(dd_off)
        out_m.append(mm_off)
        for d in depths_ir_m[1:]:
            dd, mm = align_depth_ir_left_to_color(
                rs,
                d,
                depth_value_mode=depth_value_mode,
                splat_2x2=splat_2x2,
                rectify=None,
                rectify_mode="off",
            )
            out_d.append(dd)
            out_m.append(mm)

    return out_d, out_m
