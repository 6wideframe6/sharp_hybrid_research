"""K5-B metric-amplitude diagnostics.

K5-A establishes context-consistent TinyViM detail direction/confidence.
This module estimates only a *scalar diagnostic conversion* from a
dimensionless K5 detail-vector template to SHARP inverse-depth gradient units.

No depth correction is integrated here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScaleFit:
    """Robust positive scalar fit for target ~= scale * source."""

    scale: float
    iterations: int
    samples: int
    positive_projection_fraction: float


def fit_positive_vector_scale(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    iterations: int = 12,
    huber_k: float = 1.5,
) -> ScaleFit:
    """Fit a positive scalar between two 2-D vector fields.

    Minimizes vector residuals without an intercept. A Huber-style IRLS update
    limits the influence of mismatched/disconnected edges.

    This is intended for K5-B diagnostics on trusted shared SHARP geometry.
    It must not be interpreted as proof that TinyViM magnitude is metric.
    """
    sx = np.asarray(source_x, dtype=np.float64).ravel()
    sy = np.asarray(source_y, dtype=np.float64).ravel()
    tx = np.asarray(target_x, dtype=np.float64).ravel()
    ty = np.asarray(target_y, dtype=np.float64).ravel()

    if not (sx.shape == sy.shape == tx.shape == ty.shape):
        raise ValueError("All vector components must share shape")

    finite = (
        np.isfinite(sx)
        & np.isfinite(sy)
        & np.isfinite(tx)
        & np.isfinite(ty)
    )
    sx, sy, tx, ty = sx[finite], sy[finite], tx[finite], ty[finite]

    if weights is None:
        base_w = np.ones_like(sx)
    else:
        raw_w = np.asarray(weights, dtype=np.float64).ravel()
        if raw_w.shape != finite.shape:
            raise ValueError("weights must match the unfiltered input shape")
        base_w = raw_w[finite]
        good_w = np.isfinite(base_w) & (base_w > 0)
        sx, sy, tx, ty, base_w = (
            sx[good_w], sy[good_w], tx[good_w], ty[good_w], base_w[good_w]
        )

    if len(sx) < 8:
        raise ValueError("Need at least 8 finite positive-weight samples")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if huber_k <= 0:
        raise ValueError("huber_k must be positive")

    source_energy = sx * sx + sy * sy
    if float(np.sum(base_w * source_energy)) <= np.finfo(float).tiny:
        raise ValueError("Source vector field is degenerate")

    projection = sx * tx + sy * ty
    positive_projection_fraction = float(np.mean(projection > 0))

    robust_w = np.ones_like(base_w)
    scale = 0.0

    for step in range(iterations):
        w = base_w * robust_w
        denom = float(np.sum(w * source_energy))
        if denom <= np.finfo(float).tiny:
            raise ValueError("Weighted source vector field is degenerate")

        candidate = float(np.sum(w * projection) / denom)
        if not np.isfinite(candidate) or candidate <= 0:
            raise ValueError(f"Non-positive diagnostic scale: {candidate}")

        residual = np.hypot(
            candidate * sx - tx,
            candidate * sy - ty,
        )
        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med)))
        sigma = max(1.4826 * mad, np.finfo(float).eps)

        cutoff = huber_k * sigma
        new_robust_w = np.ones_like(residual)
        large = residual > cutoff
        new_robust_w[large] = cutoff / residual[large]

        scale = candidate
        if np.allclose(new_robust_w, robust_w, rtol=1e-5, atol=1e-8):
            return ScaleFit(
                scale=scale,
                iterations=step + 1,
                samples=int(len(sx)),
                positive_projection_fraction=positive_projection_fraction,
            )
        robust_w = new_robust_w

    return ScaleFit(
        scale=scale,
        iterations=iterations,
        samples=int(len(sx)),
        positive_projection_fraction=positive_projection_fraction,
    )


def vector_residuals(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Euclidean vector residual for target - scale*source."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")

    sx = np.asarray(source_x, dtype=np.float64)
    sy = np.asarray(source_y, dtype=np.float64)
    tx = np.asarray(target_x, dtype=np.float64)
    ty = np.asarray(target_y, dtype=np.float64)

    if not (sx.shape == sy.shape == tx.shape == ty.shape):
        raise ValueError("All vector components must share shape")

    return np.hypot(scale * sx - tx, scale * sy - ty)
