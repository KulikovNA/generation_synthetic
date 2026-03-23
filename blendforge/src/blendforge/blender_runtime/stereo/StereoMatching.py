# stereo/StereoMatching.py
from __future__ import annotations

"""
Legacy compatibility facade for stereo matching.

Current effective/random paths should import the public API directly from
StereoPipline.py. This module stays only to avoid breaking older callers such as
the old single_depth_gen branch and calibration/debug scripts.
"""

from typing import Optional

import numpy as np

# Public API (new orchestration)
from blendforge.blender_runtime.stereo.StereoPipline import (  # noqa: F401
    stereo_depth_from_rectified_pair,
    stereo_global_matching_rectified,
)

# Backward-compatible helper aliases (если где-то импортировали напрямую)
from blendforge.blender_runtime.stereo.utils.PadCropUtility import (  # noqa: F401
    compute_pad_left as _compute_pad_left,
    crop_w as _crop_w,
)

# Types re-export
from blendforge.blender_runtime.stereo.types.StereoTypes import (  # noqa: F401
    FillMode,
    DepthRangePolicy,
    StereoMatcherParams,
    StereoRectifiedCalib,
)


# Optional compatibility wrapper:
# раньше у тебя был stereo_global_matching(..., rectify=...)
# теперь pipeline strictly rectified-only. Для совместимости можно оставить алиас.
def stereo_global_matching(*args, **kwargs):
    """
    Backward-compatible alias to rectified-only pipeline.
    """
    return stereo_global_matching_rectified(*args, **kwargs)


__all__ = [
    "stereo_depth_from_rectified_pair",
    "stereo_global_matching_rectified",
    "stereo_global_matching",
    "_compute_pad_left",
    "_crop_w",
    "FillMode",
    "DepthRangePolicy",
    "StereoMatcherParams",
    "StereoRectifiedCalib",
]
