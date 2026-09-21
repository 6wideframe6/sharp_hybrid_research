"""TinyViM differential-detail extraction for K5-A.

No metric calibration is performed here. TinyViM raw values remain arbitrary
relative-depth/disparity coordinates.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .types import GradientField


def extract_gradient_field(
    raw_relative_depth: np.ndarray,
    *,
    sigma_px: float = 1.0,
    min_magnitude: float = 0.0,
) -> GradientField:
    """Extract a native-pixel TinyViM gradient direction field."""
    raw = np.asarray(raw_relative_depth, dtype=np.float64)

    if raw.ndim != 2:
        raise ValueError("raw_relative_depth must be a 2-D array")
    if not np.isfinite(raw).all():
        raise ValueError("K5-A requires a dense finite TinyViM prediction")
    if sigma_px < 0:
        raise ValueError("sigma_px must be non-negative")
    if min_magnitude < 0:
        raise ValueError("min_magnitude must be non-negative")

    if sigma_px > 0:
        filtered = ndi.gaussian_filter(raw, sigma=float(sigma_px), mode="nearest")
    else:
        filtered = raw.copy()

    gy, gx = np.gradient(filtered)
    magnitude = np.hypot(gx, gy)

    numerical_floor = np.finfo(np.float64).eps * max(
        1.0, float(np.max(np.abs(filtered)))
    )
    threshold = max(float(min_magnitude), numerical_floor)
    valid = magnitude > threshold

    direction_x = np.zeros_like(gx)
    direction_y = np.zeros_like(gy)
    direction_x[valid] = gx[valid] / magnitude[valid]
    direction_y[valid] = gy[valid] / magnitude[valid]

    return GradientField(
        gx=gx,
        gy=gy,
        magnitude=magnitude,
        direction_x=direction_x,
        direction_y=direction_y,
        valid=valid,
        sigma_px=float(sigma_px),
    )
