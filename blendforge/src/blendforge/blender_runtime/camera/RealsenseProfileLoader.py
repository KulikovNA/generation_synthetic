# blendforge/blender_runtime/camera/RealsenseProfileLoader.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import blenderproc as bproc


# ---------------------------------------------------------------------------
# Coordinate conventions
# ---------------------------------------------------------------------------
# RealSense / OpenCV optical frame (CV):
#   x right, y down, z forward
# Blender camera frame:
#   x right, y up,   z backward  (camera looks along -Z)
#
# Homogeneous vector conversion:
#   p_bl = C * p_cv
#
# Transform conversion (source->target):
#   p_t_cv = T_cv * p_s_cv
#   p_t_bl = T_bl * p_s_bl
#   T_bl = C * T_cv * C^{-1}
#
# Here C^{-1} == C (diagonal with +/-1), so:
#   T_bl = C @ T_cv @ C
_CV_TO_BLENDER = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _k_from_intrinsics_dict(intr: Dict[str, Any]) -> np.ndarray:
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["ppx"])
    cy = float(intr["ppy"])
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _as_T44(m: Any) -> np.ndarray:
    T = np.asarray(m, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {T.shape}")
    return T


def _cv_T_to_blender(T_cv: np.ndarray) -> np.ndarray:
    C = _CV_TO_BLENDER
    return C @ T_cv @ C


def _deg2rad(x: float) -> float:
    return float(np.deg2rad(float(x)))


def _np5(x: Any) -> Tuple[float, float, float, float, float]:
    """
    Force distortion coeffs to 5-tuple (k1,k2,p1,p2,k3) style.
    Pads with zeros / truncates if needed.
    """
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    out = np.zeros(5, dtype=np.float64)
    if a.size:
        out[: min(5, a.size)] = a[: min(5, a.size)]
    return tuple(float(v) for v in out)


def _ceil_to_16(x: float) -> int:
    return int(np.ceil(float(x) / 16.0) * 16.0)


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(int(lo), min(int(hi), int(x)))


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StreamCalib:
    width: int
    height: int
    K: np.ndarray
    distortion_model: str
    distortion_coeffs: Tuple[float, float, float, float, float]


@dataclass(frozen=True)
class ProjectorCalib:
    """
    Projector != camera.

    fov_h/v describe projector optics cone,
    pattern_image describes cookie/texture resolution (independent from render res).
    """
    wavelength_nm: float
    fov_h_deg: float
    fov_v_deg: float
    pattern_w: int
    pattern_h: int
    dot_count: int
    energy: float
    mount_frame: str
    type: str = "vcsel_dot"

    @property
    def fov_h_rad(self) -> float:
        return _deg2rad(self.fov_h_deg)

    @property
    def fov_v_rad(self) -> float:
        return _deg2rad(self.fov_v_deg)


@dataclass(frozen=True)
class DepthRange:
    min_m: float = 0.2
    max_m: float = 10.0


@dataclass(frozen=True)
class StereoConfig:
    """
    Heuristic config for OpenCV SGBM:
      d_max ~= fx * B / z_min
      numDisp ~= ceil_to_16( (d_max - minDisp) * margin )
    """
    num_disparities_margin: float = 1.15
    num_disparities_clamp: Tuple[int, int] = (64, 256)  # must be multiples of 16 ideally

    def clamp_num_disparities(self, n: int) -> int:
        lo, hi = self.num_disparities_clamp
        # force to >=16 and /16
        n = max(16, int(n))
        n = _ceil_to_16(n)
        # clamp
        n = _clamp_int(n, lo, hi)
        # ensure /16 after clamp too
        n = _ceil_to_16(n)
        return n


@dataclass(frozen=True)
class DeviceInfo:
    family: str = "Intel RealSense D400"
    model: str = "D435"
    profile_name: str = ""
    units_translation: str = "meters"
    units_intrinsics: str = "pixels"


@dataclass(frozen=True)
class RealSenseProfile:
    streams: Dict[str, StreamCalib]
    extrinsics_cv: Dict[str, np.ndarray]         # key example: "IR_LEFT_to_COLOR"
    baseline_m: float
    stream_index_map: Dict[str, int]

    depth_range: DepthRange = DepthRange()
    stereo: StereoConfig = StereoConfig()
    device: DeviceInfo = DeviceInfo()

    projector: Optional[ProjectorCalib] = None


    # -----------------------------------------------------------------------
    # Parsing
    # -----------------------------------------------------------------------
    @staticmethod
    def from_json(path: str) -> "RealSenseProfile":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"RealSense profile JSON not found: {path}")

        with p.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        # --- device (optional) ---
        dev = cfg.get("device", {}) or {}
        units = dev.get("units", {}) or {}
        device = DeviceInfo(
            family=str(dev.get("family", "Intel RealSense D400")),
            model=str(dev.get("model", "D435")),
            profile_name=str(dev.get("profile_name", "")),
            units_translation=str(units.get("translation", "meters")),
            units_intrinsics=str(units.get("intrinsics", "pixels")),
        )

        # --- streams ---
        streams: Dict[str, StreamCalib] = {}
        streams_cfg = cfg.get("streams", {}) or {}
        if not isinstance(streams_cfg, dict) or not streams_cfg:
            raise ValueError("JSON must contain non-empty 'streams' dict")

        for name, s in streams_cfg.items():
            if not isinstance(s, dict) or "intrinsics" not in s:
                raise ValueError(f"Bad stream entry for '{name}'")
            intr = s["intrinsics"]
            K = _k_from_intrinsics_dict(intr)

            streams[str(name)] = StreamCalib(
                width=int(s["width"]),
                height=int(s["height"]),
                K=K,
                distortion_model=str(intr.get("distortion_model", "")),
                distortion_coeffs=_np5(intr.get("distortion_coeffs", [0, 0, 0, 0, 0])),
            )

        # --- stream index map (optional) ---
        sim = cfg.get("stream_index_map", {}) or {}
        stream_index_map: Dict[str, int] = {}
        if isinstance(sim, dict):
            for k, v in sim.items():
                if isinstance(v, (int, float)):
                    stream_index_map[str(k)] = int(v)

        # --- extrinsics (CV convention) ---
        extr_cv: Dict[str, np.ndarray] = {}
        extr_cfg = cfg.get("extrinsics", {}) or {}
        if isinstance(extr_cfg, dict):
            for key, e in extr_cfg.items():
                if not isinstance(e, dict):
                    continue
                if "T_target_from_source_4x4" not in e:
                    continue
                T = _as_T44(e["T_target_from_source_4x4"])

                if "source_frame" in e and "target_frame" in e:
                    k2 = f"{e['source_frame']}_to_{e['target_frame']}"
                    extr_cv[str(k2)] = T
                else:
                    extr_cv[str(key)] = T

        # --- baseline (meters) ---
        baseline_m: Optional[float] = None
        derived = cfg.get("derived", {}) or {}
        if isinstance(derived, dict) and "baseline_m" in derived:
            baseline_m = float(derived["baseline_m"])
        else:
            # try compute from IR_LEFT_to_COLOR and IR_RIGHT_to_COLOR translations (x) in COLOR frame
            T_c_l = extr_cv.get("IR_LEFT_to_COLOR", None)
            T_c_r = extr_cv.get("IR_RIGHT_to_COLOR", None)
            if T_c_l is not None and T_c_r is not None:
                baseline_m = float(T_c_r[0, 3] - T_c_l[0, 3])

        if baseline_m is None:
            raise ValueError("baseline_m not found and could not be computed from extrinsics")

        baseline_m = float(abs(baseline_m))

        # --- depth range (optional) ---
        dr = cfg.get("depth_range_m", {}) or {}
        depth_range = DepthRange(
            min_m=float(dr.get("min", 0.2)),
            max_m=float(dr.get("max", 10.0)),
        )
        if depth_range.min_m <= 0 or depth_range.max_m <= 0 or depth_range.max_m <= depth_range.min_m:
            raise ValueError(f"Invalid depth_range_m: min={depth_range.min_m}, max={depth_range.max_m}")

        # --- stereo config (optional) ---
        st = cfg.get("stereo", {}) or {}
        clamp = st.get("num_disparities_clamp", [64, 256])
        if not (isinstance(clamp, (list, tuple)) and len(clamp) == 2):
            clamp = [64, 256]
        stereo = StereoConfig(
            num_disparities_margin=float(st.get("num_disparities_margin", 1.15)),
            num_disparities_clamp=(int(clamp[0]), int(clamp[1])),
        )

        # --- projector (optional) ---
        proj_cfg = cfg.get("projector", None)
        projector: Optional[ProjectorCalib] = None
        if isinstance(proj_cfg, dict):
            pat = proj_cfg.get("pattern_image", {}) or {}
            projector = ProjectorCalib(
                type=str(proj_cfg.get("type", "vcsel_dot")),
                wavelength_nm=float(proj_cfg.get("wavelength_nm", 850.0)),
                fov_h_deg=float(proj_cfg.get("fov_h_deg", 91.0)),
                fov_v_deg=float(proj_cfg.get("fov_v_deg", 65.0)),
                pattern_w=int(pat.get("width", 640)),
                pattern_h=int(pat.get("height", 480)),
                dot_count=int(proj_cfg.get("dot_count", 5000)),
                energy=float(proj_cfg.get("energy", 3000.0)),
                mount_frame=str(proj_cfg.get("mount_frame", "IR_LEFT")),
            )

        return RealSenseProfile(
            streams=streams,
            extrinsics_cv=extr_cv,
            baseline_m=baseline_m,
            stream_index_map=stream_index_map,
            depth_range=depth_range,
            stereo=stereo,
            device=device,
            projector=projector,
        )

    # -----------------------------------------------------------------------
    # Streams
    # -----------------------------------------------------------------------
    def get_stream(self, stream: str) -> StreamCalib:
        if stream not in self.streams:
            raise KeyError(f"Unknown stream '{stream}'. Available: {list(self.streams.keys())}")
        return self.streams[stream]

    def set_bproc_intrinsics(self, stream: str) -> Tuple[np.ndarray, int, int]:
        """
        Configure BlenderProc active camera intrinsics/resolution for the given stream.
        Returns (K, W, H).
        """
        s = self.get_stream(stream)
        bproc.camera.set_intrinsics_from_K_matrix(s.K, s.width, s.height)
        return s.K.copy(), int(s.width), int(s.height)

    # BlenderProc 2.7.1 signature:
    #   set_stereo_parameters(convergence_mode: str, convergence_distance: float, interocular_distance: float)
    def set_bproc_stereo_from_ir(
        self,
        baseline_m: Optional[float] = None,
        convergence_distance: float = 1.0
    ) -> float:
        """
        Enable stereo baseline (interocular distance) for IR pair.
        """
        b = float(self.baseline_m if baseline_m is None else baseline_m)
        bproc.camera.set_stereo_parameters("PARALLEL", float(convergence_distance), b)
        return b

    # -----------------------------------------------------------------------
    # Depth range
    # -----------------------------------------------------------------------
    @property
    def depth_min_m(self) -> float:
        return float(self.depth_range.min_m)

    @property
    def depth_max_m(self) -> float:
        return float(self.depth_range.max_m)

    def clamp_depth(self, depth_m: np.ndarray, invalid_value: float = 0.0) -> np.ndarray:
        """
        Utility: clamp/validate depth map using profile range.
        """
        d = np.asarray(depth_m, dtype=np.float32)
        out = d.copy()
        out[~np.isfinite(out)] = float(invalid_value)
        out[(out < self.depth_min_m) | (out > self.depth_max_m)] = float(invalid_value)
        return out

    # -----------------------------------------------------------------------
    # Stereo heuristic: numDisparities
    # -----------------------------------------------------------------------
    def recommend_num_disparities(
        self,
        *,
        z_min_m: Optional[float] = None,
        min_disparity: int = 0,
        stream: str = "IR_LEFT",
        margin: Optional[float] = None,
        clamp: Optional[Tuple[int, int]] = None,
    ) -> int:
        """
        Heuristic for OpenCV StereoSGBM numDisparities.

        Uses:
          d_max ~= fx * baseline / z_min
          need  ~= (d_max - minDisp) * margin
          numDisp = ceil_to_16(need) clamped to stereo.num_disparities_clamp

        IMPORTANT:
          - z_min is a design/assumption (your chosen minimum working depth).
          - It is NOT computed from disparities.
        """
        zmin = float(self.depth_min_m if z_min_m is None else z_min_m)
        zmin = max(zmin, 1e-6)

        s = self.get_stream(stream)
        fx = float(np.asarray(s.K, dtype=np.float64)[0, 0])
        B = float(self.baseline_m)

        dmax = fx * B / zmin
        md = float(int(min_disparity))

        m = float(self.stereo.num_disparities_margin if margin is None else margin)

        # cover [minDisp .. minDisp+numDisp)
        need = max(16.0, (dmax - md) * m)
        num = _ceil_to_16(need)

        if clamp is None:
            num = self.stereo.clamp_num_disparities(num)
        else:
            lo, hi = int(clamp[0]), int(clamp[1])
            num = _ceil_to_16(_clamp_int(num, lo, hi))

        return int(num)

    # -----------------------------------------------------------------------
    # Extrinsics
    # -----------------------------------------------------------------------
    def get_T_cv(self, source: str, target: str) -> np.ndarray:
        """
        Return T_target_from_source in CV convention.
        Example: get_T_cv("IR_LEFT","COLOR") -> key "IR_LEFT_to_COLOR".
        """
        key = f"{source}_to_{target}"
        if key not in self.extrinsics_cv:
            raise KeyError(
                f"Extrinsic '{key}' not found. Available: {list(self.extrinsics_cv.keys())}"
            )
        return self.extrinsics_cv[key].copy()

    def get_T_blender(self, source: str, target: str) -> np.ndarray:
        """
        Same as get_T_cv, but converted into Blender camera-axis convention.
        """
        return _cv_T_to_blender(self.get_T_cv(source, target))

    def get_T_inv_cv(self, source: str, target: str) -> np.ndarray:
        """
        Inverse transform in CV convention: T_source_from_target.
        """
        return np.linalg.inv(self.get_T_cv(source, target))

    def get_T_inv_blender(self, source: str, target: str) -> np.ndarray:
        """
        Inverse transform in Blender convention: T_source_from_target.
        """
        return np.linalg.inv(self.get_T_blender(source, target))

    # -----------------------------------------------------------------------
    # Projector
    # -----------------------------------------------------------------------
    def has_projector(self) -> bool:
        return self.projector is not None

    def get_projector(self) -> ProjectorCalib:
        if self.projector is None:
            raise ValueError("Projector section is missing in JSON ('projector': {...})")
        return self.projector

    def get_projector_fov_rad(self) -> Tuple[float, float]:
        pr = self.get_projector()
        return pr.fov_h_rad, pr.fov_v_rad

    def get_projector_pattern_wh(self) -> Tuple[int, int]:
        pr = self.get_projector()
        return int(pr.pattern_w), int(pr.pattern_h)

    def get_projector_dot_count(self) -> int:
        return int(self.get_projector().dot_count)

    def get_projector_energy(self) -> float:
        return float(self.get_projector().energy)

    def get_projector_mount_frame(self) -> str:
        return str(self.get_projector().mount_frame)

    def get_projector_wavelength_nm(self) -> float:
        return float(self.get_projector().wavelength_nm)
