#stereo/types/StereoTypes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


FillMode = Literal["none", "mean", "dilate_max"]
DepthRangePolicy = Literal["zero", "clamp"]  # outside [min,max]: invalid(0) OR clamp


@dataclass
class StereoMatcherParams:
    # depth / disparity core
    depth_range_policy: DepthRangePolicy = "zero"
    block_size: int = 7
    num_disparities: int = 128
    min_disparity: int = 0
    preprocess: str = "clahe"

    # consistency / filtering
    use_wls: bool = True
    lr_check: bool = True
    lr_thresh_px: float = 1.0
    lr_min_keep_ratio: float = 0.02
    speckle_filter: bool = True
    fill_mode: FillMode = "none"
    fill_iters: int = 0
    depth_completion: bool = False

    # left-border artifact mitigation
    border_pad: bool = True
    pad_left: Optional[int] = None

    # SGBM advanced defaults
    sgbm_mode: int = 1  # cv2.STEREO_SGBM_MODE_HH (avoid cv2 import in types module)
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1
    pre_filter_cap: int = 63

    # WLS defaults
    wls_lambda: float = 80000.0
    wls_sigma_color: float = 1.2

    def validate(self) -> None:
        if int(self.block_size) <= 0 or int(self.block_size) % 2 == 0:
            raise ValueError(f"block_size must be positive odd, got {self.block_size}")
        if int(self.num_disparities) <= 0 or int(self.num_disparities) % 16 != 0:
            raise ValueError(
                f"num_disparities must be >0 and divisible by 16, got {self.num_disparities}"
            )
        if float(self.lr_thresh_px) < 0:
            raise ValueError("lr_thresh_px must be >= 0")
        if float(self.lr_min_keep_ratio) < 0:
            raise ValueError("lr_min_keep_ratio must be >= 0")
        if int(self.fill_iters) < 0:
            raise ValueError("fill_iters must be >= 0")
        if self.fill_mode not in ("none", "mean", "dilate_max"):
            raise ValueError(f"Unknown fill_mode: {self.fill_mode}")
        if self.depth_range_policy not in ("zero", "clamp"):
            raise ValueError(f"Unknown depth_range_policy: {self.depth_range_policy}")


@dataclass
class StereoRectifiedCalib:
    fx_rect: float
    baseline_rect_m: float

    def validate(self) -> None:
        if not (self.fx_rect > 0):
            raise ValueError(f"fx_rect must be > 0, got {self.fx_rect}")
        if not (self.baseline_rect_m > 0):
            raise ValueError(f"baseline_rect_m must be > 0, got {self.baseline_rect_m}")

    @property
    def fx_times_baseline(self) -> float:
        return float(self.fx_rect) * float(self.baseline_rect_m)