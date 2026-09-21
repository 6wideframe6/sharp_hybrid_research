"""Generic TinyViM polarity reliability for K5 final composition.

This module uses only TinyViM raw predictions and native-coordinate context
geometry.  Known wire masks are never inputs.

For each context:

    dog = G_fine(raw) - G_coarse(raw)

The signed DoG is invariant to additive offsets and preserves sign under
positive affine rescaling.  Its absolute magnitude is independently normalized
per context over the same halo-safe physical overlap.

The final reliability is

    sign_agree * smoothstep(min(norm_abs_dog_a, norm_abs_dog_b), low, high)

and remains dimensionless.  It is NOT metric depth amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage as ndi

from .overlap import crop_to_native_box, native_halo_support_mask
from .types import NativeBox


@dataclass(frozen=True)
class PolarityReliability:
    dog_a: np.ndarray
    dog_b: np.ndarray
    strength_a: np.ndarray
    strength_b: np.ndarray
    strength_consensus: np.ndarray
    sign_agreement: np.ndarray
    reliability: np.ndarray
    valid: np.ndarray
    low_a: float
    high_a: float
    low_b: float
    high_b: float
    strict_halo_px: int


def _validate_raw(raw: np.ndarray, box: NativeBox, name: str) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float64)
    if value.shape != box.shape:
        raise ValueError(f"{name} shape {value.shape} does not match {box.shape}")
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a dense finite 2-D array")
    return value


def _normalize_abs(
    value: np.ndarray,
    valid: np.ndarray,
    *,
    low_percentile: float,
    high_percentile: float,
) -> tuple[np.ndarray, float, float]:
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError("Require 0 <= low_percentile < high_percentile <= 100")

    magnitude = np.abs(np.asarray(value, dtype=np.float64))
    mask = np.asarray(valid, dtype=bool)
    if magnitude.shape != mask.shape:
        raise ValueError("value/valid shape mismatch")

    samples = magnitude[mask]
    if not samples.size:
        return np.zeros_like(magnitude), 0.0, 0.0

    lo, hi = np.percentile(samples, [low_percentile, high_percentile])
    span = float(hi - lo)
    scale = max(1.0, abs(float(lo)), abs(float(hi)))
    floor = np.finfo(np.float64).eps * scale

    if span <= floor:
        return np.zeros_like(magnitude), float(lo), float(hi)

    normalized = np.clip((magnitude - lo) / span, 0.0, 1.0)
    normalized[~mask] = 0.0
    return normalized, float(lo), float(hi)


def _smoothstep(value: np.ndarray, low: float, high: float) -> np.ndarray:
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("reliability thresholds must be finite")
    if low < 0.0 or high <= low:
        raise ValueError("require 0 <= reliability_low < reliability_high")

    t = np.clip((value - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def compute_polarity_reliability(
    raw_a: np.ndarray,
    box_a: NativeBox,
    raw_b: np.ndarray,
    box_b: NativeBox,
    target_box: NativeBox,
    *,
    fine_sigma_px: float = 1.0,
    coarse_sigma_px: float = 2.0,
    normalize_low_percentile: float = 50.0,
    normalize_high_percentile: float = 99.0,
    reliability_low: float = 0.025,
    reliability_high: float = 0.10,
) -> PolarityReliability:
    """Return generic context-consistent TinyViM polarity reliability."""
    if fine_sigma_px < 0:
        raise ValueError("fine_sigma_px must be non-negative")
    if coarse_sigma_px <= fine_sigma_px:
        raise ValueError("coarse_sigma_px must be greater than fine_sigma_px")
    if not box_a.contains(target_box) or not box_b.contains(target_box):
        raise ValueError("target_box must be contained in both source boxes")

    a = _validate_raw(raw_a, box_a, "raw_a")
    b = _validate_raw(raw_b, box_b, "raw_b")

    fine_a = ndi.gaussian_filter(
        a, sigma=float(fine_sigma_px), mode="nearest"
    )
    coarse_a = ndi.gaussian_filter(
        a, sigma=float(coarse_sigma_px), mode="nearest"
    )
    fine_b = ndi.gaussian_filter(
        b, sigma=float(fine_sigma_px), mode="nearest"
    )
    coarse_b = ndi.gaussian_filter(
        b, sigma=float(coarse_sigma_px), mode="nearest"
    )

    dog_a = np.array(
        crop_to_native_box(fine_a - coarse_a, box_a, target_box),
        copy=True,
    )
    dog_b = np.array(
        crop_to_native_box(fine_b - coarse_b, box_b, target_box),
        copy=True,
    )

    # DoG contains Gaussian filtering but no subsequent spatial derivative.
    strict_halo_px = int(math.ceil(4.0 * float(coarse_sigma_px)))
    valid = (
        native_halo_support_mask(
            box_a, target_box, halo_px=strict_halo_px
        )
        & native_halo_support_mask(
            box_b, target_box, halo_px=strict_halo_px
        )
    )

    strength_a, low_a, high_a = _normalize_abs(
        dog_a,
        valid,
        low_percentile=normalize_low_percentile,
        high_percentile=normalize_high_percentile,
    )
    strength_b, low_b, high_b = _normalize_abs(
        dog_b,
        valid,
        low_percentile=normalize_low_percentile,
        high_percentile=normalize_high_percentile,
    )
    strength_consensus = np.minimum(strength_a, strength_b)

    sign_agreement = (
        valid
        & (dog_a != 0.0)
        & (dog_b != 0.0)
        & (np.signbit(dog_a) == np.signbit(dog_b))
    )

    reliability = _smoothstep(
        strength_consensus,
        reliability_low,
        reliability_high,
    )
    reliability = np.where(sign_agreement, reliability, 0.0)
    reliability = np.where(valid, reliability, 0.0)

    return PolarityReliability(
        dog_a=dog_a,
        dog_b=dog_b,
        strength_a=strength_a,
        strength_b=strength_b,
        strength_consensus=strength_consensus,
        sign_agreement=sign_agreement,
        reliability=reliability,
        valid=valid,
        low_a=low_a,
        high_a=high_a,
        low_b=low_b,
        high_b=high_b,
        strict_halo_px=strict_halo_px,
    )
