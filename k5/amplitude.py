"""K5-B amplitude diagnostics.

Contains vector-scale diagnostics from B.1 and scalar magnitude-scale
diagnostics for B.2. None of these functions integrate a depth correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScaleFit:
    scale: float
    iterations: int
    samples: int
    positive_projection_fraction: float


@dataclass(frozen=True)
class ScalarScaleFit:
    scale: float
    iterations: int
    samples: int


def _prepare_vector(source_x, source_y, target_x, target_y, weights):
    sx = np.asarray(source_x, dtype=np.float64).ravel()
    sy = np.asarray(source_y, dtype=np.float64).ravel()
    tx = np.asarray(target_x, dtype=np.float64).ravel()
    ty = np.asarray(target_y, dtype=np.float64).ravel()
    if not (sx.shape == sy.shape == tx.shape == ty.shape):
        raise ValueError("All vector components must share shape")

    finite = np.isfinite(sx) & np.isfinite(sy) & np.isfinite(tx) & np.isfinite(ty)
    sx, sy, tx, ty = sx[finite], sy[finite], tx[finite], ty[finite]

    if weights is None:
        w = np.ones_like(sx)
    else:
        raw_w = np.asarray(weights, dtype=np.float64).ravel()
        if raw_w.shape != finite.shape:
            raise ValueError("weights must match the unfiltered input shape")
        w = raw_w[finite]
        good = np.isfinite(w) & (w > 0)
        sx, sy, tx, ty, w = sx[good], sy[good], tx[good], ty[good], w[good]

    if len(sx) < 8:
        raise ValueError("Need at least 8 finite positive-weight samples")
    return sx, sy, tx, ty, w


def _fit_vector_scale(
    source_x,
    source_y,
    target_x,
    target_y,
    *,
    weights=None,
    iterations=12,
    huber_k=1.5,
    require_positive=True,
):
    sx, sy, tx, ty, base_w = _prepare_vector(
        source_x, source_y, target_x, target_y, weights
    )
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if huber_k <= 0:
        raise ValueError("huber_k must be positive")

    energy = sx * sx + sy * sy
    if float(np.sum(base_w * energy)) <= np.finfo(float).tiny:
        raise ValueError("Source vector field is degenerate")

    projection = sx * tx + sy * ty
    positive_fraction = float(np.mean(projection > 0))
    robust_w = np.ones_like(base_w)
    scale = 0.0

    for step in range(iterations):
        w = base_w * robust_w
        denom = float(np.sum(w * energy))
        if denom <= np.finfo(float).tiny:
            raise ValueError("Weighted source vector field is degenerate")
        candidate = float(np.sum(w * projection) / denom)
        if not np.isfinite(candidate):
            raise ValueError(f"Non-finite diagnostic scale: {candidate}")
        if require_positive and candidate <= 0:
            raise ValueError(f"Non-positive diagnostic scale: {candidate}")

        residual = np.hypot(candidate * sx - tx, candidate * sy - ty)
        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med)))
        sigma = max(1.4826 * mad, np.finfo(float).eps)
        cutoff = huber_k * sigma
        new_robust = np.ones_like(residual)
        large = residual > cutoff
        new_robust[large] = cutoff / residual[large]

        scale = candidate
        if np.allclose(new_robust, robust_w, rtol=1e-5, atol=1e-8):
            return ScaleFit(scale, step + 1, int(len(sx)), positive_fraction)
        robust_w = new_robust

    return ScaleFit(scale, iterations, int(len(sx)), positive_fraction)


def fit_positive_vector_scale(
    source_x,
    source_y,
    target_x,
    target_y,
    *,
    weights=None,
    iterations=12,
    huber_k=1.5,
):
    return _fit_vector_scale(
        source_x, source_y, target_x, target_y,
        weights=weights, iterations=iterations, huber_k=huber_k,
        require_positive=True,
    )


def fit_signed_vector_scale(
    source_x,
    source_y,
    target_x,
    target_y,
    *,
    weights=None,
    iterations=12,
    huber_k=1.5,
):
    return _fit_vector_scale(
        source_x, source_y, target_x, target_y,
        weights=weights, iterations=iterations, huber_k=huber_k,
        require_positive=False,
    )


def signed_vector_residuals(source_x, source_y, target_x, target_y, scale):
    if not np.isfinite(scale):
        raise ValueError("scale must be finite")
    sx = np.asarray(source_x, dtype=np.float64)
    sy = np.asarray(source_y, dtype=np.float64)
    tx = np.asarray(target_x, dtype=np.float64)
    ty = np.asarray(target_y, dtype=np.float64)
    if not (sx.shape == sy.shape == tx.shape == ty.shape):
        raise ValueError("All vector components must share shape")
    return np.hypot(scale * sx - tx, scale * sy - ty)


def vector_residuals(source_x, source_y, target_x, target_y, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    return signed_vector_residuals(
        source_x, source_y, target_x, target_y, scale
    )


def fit_positive_scalar_scale(
    source,
    target,
    *,
    weights=None,
    iterations=12,
    huber_k=1.5,
) -> ScalarScaleFit:
    """Robustly fit target ~= scale * source for non-negative magnitudes."""
    x = np.asarray(source, dtype=np.float64).ravel()
    y = np.asarray(target, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError("source and target must share shape")

    finite = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
    x, y = x[finite], y[finite]

    if weights is None:
        base_w = np.ones_like(x)
    else:
        raw_w = np.asarray(weights, dtype=np.float64).ravel()
        if raw_w.shape != finite.shape:
            raise ValueError("weights must match the unfiltered input shape")
        base_w = raw_w[finite]
        good = np.isfinite(base_w) & (base_w > 0)
        x, y, base_w = x[good], y[good], base_w[good]

    if len(x) < 8:
        raise ValueError("Need at least 8 finite positive-weight samples")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if huber_k <= 0:
        raise ValueError("huber_k must be positive")

    energy = x * x
    if float(np.sum(base_w * energy)) <= np.finfo(float).tiny:
        raise ValueError("Scalar source is degenerate")

    robust_w = np.ones_like(base_w)
    scale = 0.0

    for step in range(iterations):
        w = base_w * robust_w
        denom = float(np.sum(w * energy))
        candidate = float(np.sum(w * x * y) / denom)
        if not np.isfinite(candidate) or candidate <= 0:
            raise ValueError(f"Non-positive scalar diagnostic scale: {candidate}")

        residual = np.abs(candidate * x - y)
        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med)))
        sigma = max(1.4826 * mad, np.finfo(float).eps)
        cutoff = huber_k * sigma
        new_robust = np.ones_like(residual)
        large = residual > cutoff
        new_robust[large] = cutoff / residual[large]

        scale = candidate
        if np.allclose(new_robust, robust_w, rtol=1e-5, atol=1e-8):
            return ScalarScaleFit(scale, step + 1, int(len(x)))
        robust_w = new_robust

    return ScalarScaleFit(scale, iterations, int(len(x)))


def scalar_residuals(source, target, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("source and target must share shape")
    return np.abs(scale * x - y)
