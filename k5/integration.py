"""Bounded diagnostic integration for K5 detail gradients.

This module does not touch SHARP inference or Gaussian generation.

It solves a screened Poisson problem on a rectangular native-pixel domain with
zero Dirichlet boundary:

    argmin_delta ||grad(delta) - g||^2 + screening * ||delta||^2

The implementation uses the corresponding interior linear system

    (-Laplacian + screening I) delta = -div(g)

and keeps the outermost pixel boundary at exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


@dataclass(frozen=True)
class IntegrationResult:
    delta_q: np.ndarray
    converged: bool
    cg_info: int


def saturated_gate(
    confidence: np.ndarray,
    *,
    low: float,
    high: float,
) -> np.ndarray:
    """Linear ramp from 0 to 1, saturated above ``high``."""
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("gate thresholds must be finite")
    if low < 0 or high <= low:
        raise ValueError("require 0 <= low < high")

    c = np.asarray(confidence, dtype=np.float64)
    gate = (c - low) / (high - low)
    gate = np.clip(gate, 0.0, 1.0)
    gate[~np.isfinite(c)] = 0.0
    return gate


def solve_screened_gradient_field(
    gx: np.ndarray,
    gy: np.ndarray,
    *,
    screening: float,
    rtol: float = 1e-7,
    maxiter: int = 5000,
) -> IntegrationResult:
    """Integrate a desired 2-D gradient field into a scalar correction.

    Parameters
    ----------
    gx, gy
        Desired inverse-depth gradient components in metric units per native
        pixel. Both arrays are ``[H,W]``.
    screening
        Non-negative zero-order screening coefficient.
    rtol, maxiter
        Conjugate-gradient controls.

    Returns
    -------
    IntegrationResult
        ``delta_q`` has exactly zero outer boundary.
    """
    gx = np.asarray(gx, dtype=np.float64)
    gy = np.asarray(gy, dtype=np.float64)

    if gx.shape != gy.shape or gx.ndim != 2:
        raise ValueError("gx and gy must be same-shape 2-D arrays")
    if min(gx.shape) < 3:
        raise ValueError("gradient field must be at least 3x3")
    if not np.all(np.isfinite(gx)) or not np.all(np.isfinite(gy)):
        raise ValueError("gradient field must be finite")
    if not np.isfinite(screening) or screening < 0:
        raise ValueError("screening must be finite and non-negative")
    if rtol <= 0 or maxiter < 1:
        raise ValueError("invalid CG controls")

    h, w = gx.shape
    ih, iw = h - 2, w - 2
    n = ih * iw

    div = np.gradient(gx, axis=1) + np.gradient(gy, axis=0)
    rhs = (-div[1:-1, 1:-1]).reshape(-1)

    def matvec(v):
        u = np.asarray(v, dtype=np.float64).reshape(ih, iw)
        out = (4.0 + screening) * u.copy()

        out[:, 1:] -= u[:, :-1]
        out[:, :-1] -= u[:, 1:]
        out[1:, :] -= u[:-1, :]
        out[:-1, :] -= u[1:, :]

        return out.reshape(-1)

    operator = LinearOperator(
        shape=(n, n),
        matvec=matvec,
        dtype=np.float64,
    )

    solution, info = cg(
        operator,
        rhs,
        rtol=rtol,
        atol=0.0,
        maxiter=maxiter,
    )

    delta = np.zeros((h, w), dtype=np.float64)
    delta[1:-1, 1:-1] = solution.reshape(ih, iw)

    return IntegrationResult(
        delta_q=delta,
        converged=(info == 0),
        cg_info=int(info),
    )
