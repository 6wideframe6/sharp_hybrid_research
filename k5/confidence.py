"""K5-A.1 context-consistent high-frequency detail confidence.

This module does NOT assign metric inverse-depth amplitude.

TinyViM gradient magnitude is used only as a within-context *detector* of
high-frequency structure. The detector is robustly normalized independently
for every context before multi-context consensus.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .detail_signal import extract_gradient_field


def multiscale_detail_strength(
    raw_relative_depth: np.ndarray,
    *,
    fine_sigma_px: float = 1.0,
    coarse_sigma_px: float = 4.0,
) -> np.ndarray:
    """Return non-metric high-frequency gradient strength.

    The detector is

        max(|grad G_fine(r)| - |grad G_coarse(r)|, 0)

    where ``r`` is the TinyViM raw relative-depth prediction.

    This quantity is deliberately NOT interpreted as metric depth or metric
    gradient amplitude. It only answers whether local structure survives at
    the fine scale but is suppressed at the coarse scale.
    """
    if fine_sigma_px < 0:
        raise ValueError("fine_sigma_px must be non-negative")
    if coarse_sigma_px <= fine_sigma_px:
        raise ValueError("coarse_sigma_px must be greater than fine_sigma_px")

    fine = extract_gradient_field(
        raw_relative_depth,
        sigma_px=float(fine_sigma_px),
    )
    coarse = extract_gradient_field(
        raw_relative_depth,
        sigma_px=float(coarse_sigma_px),
    )

    strength = fine.magnitude - coarse.magnitude
    return np.maximum(strength, 0.0)


def robust_normalize_detail(
    strength: np.ndarray,
    *,
    low_percentile: float = 50.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """Normalize one context's detector to [0, 1] using robust percentiles.

    Normalization is context-local. Therefore different TinyViM raw scales do
    not directly determine final confidence amplitude.

    Values <= low percentile map to zero. Values >= high percentile map to one.
    Degenerate/constant inputs return zeros.
    """
    value = np.asarray(strength, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("strength must be a 2-D array")
    if not np.isfinite(value).all():
        raise ValueError("strength must be finite")
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError("Require 0 <= low_percentile < high_percentile <= 100")

    lo, hi = np.percentile(value, [low_percentile, high_percentile])
    span = float(hi - lo)

    scale = max(1.0, abs(float(lo)), abs(float(hi)))
    floor = np.finfo(np.float64).eps * scale
    if span <= floor:
        return np.zeros_like(value, dtype=np.float64)

    normalized = (value - lo) / span
    return np.clip(normalized, 0.0, 1.0)


def detail_consensus_min(
    normalized_strengths: Sequence[np.ndarray],
) -> np.ndarray:
    """Require detail evidence to be present in every context.

    ``min`` is intentionally conservative: a feature that is strong in only one
    TinyViM context is treated as context-specific evidence and suppressed.
    """
    strengths = tuple(np.asarray(v, dtype=np.float64) for v in normalized_strengths)
    if len(strengths) < 2:
        raise ValueError("At least two normalized detail maps are required")

    shape = strengths[0].shape
    if strengths[0].ndim != 2 or any(v.shape != shape for v in strengths):
        raise ValueError("All normalized detail maps must share one 2-D shape")
    if any(not np.isfinite(v).all() for v in strengths):
        raise ValueError("Normalized detail maps must be finite")

    stacked = np.stack(
        [np.clip(v, 0.0, 1.0) for v in strengths],
        axis=0,
    )
    return np.min(stacked, axis=0)


def combine_direction_and_detail(
    direction_confidence: np.ndarray,
    detail_confidence: np.ndarray,
    *,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Multiply orientation agreement by high-frequency detail evidence.

    Both inputs are treated as dimensionless confidence terms in [0, 1].
    This output remains a diagnostic weight, not a replacement mask and not a
    metric correction amplitude.
    """
    direction = np.asarray(direction_confidence, dtype=np.float64)
    detail = np.asarray(detail_confidence, dtype=np.float64)

    if direction.ndim != 2 or direction.shape != detail.shape:
        raise ValueError("direction_confidence and detail_confidence must share shape")
    if not np.isfinite(direction).all() or not np.isfinite(detail).all():
        raise ValueError("confidence inputs must be finite")

    result = np.clip(direction, 0.0, 1.0) * np.clip(detail, 0.0, 1.0)

    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != result.shape:
            raise ValueError("valid mask must match confidence shape")
        result = np.where(valid, result, 0.0)

    return result
