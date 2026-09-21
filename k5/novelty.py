"""SHARP-resolved-detail suppression for K5 application support.

K5/TinyViM determines candidate support and direction.  This module removes
candidate support where SHARP already has a strong fine-scale metric gradient.

The threshold is derived outside this module from SHARP diagnostics; known wire
masks are never inputs to the novelty gate.
"""

from __future__ import annotations

import numpy as np


def sharp_novelty_gate(
    sharp_fine_magnitude: np.ndarray,
    *,
    suppress_threshold: float,
    transition_fraction: float = 0.5,
) -> np.ndarray:
    """Return 1 for SHARP-weak detail and 0 for SHARP-resolved detail.

    The transition is a cubic smoothstep from
    ``transition_fraction * suppress_threshold`` to ``suppress_threshold``.

    Parameters
    ----------
    sharp_fine_magnitude
        Non-negative SHARP fine-gradient magnitude.
    suppress_threshold
        Metric magnitude at and above which K5 application is fully suppressed.
    transition_fraction
        Lower transition edge as a fraction of ``suppress_threshold``.
    """
    m = np.asarray(sharp_fine_magnitude, dtype=np.float64)
    if not np.isfinite(suppress_threshold) or suppress_threshold <= 0:
        raise ValueError("suppress_threshold must be finite and positive")
    if not np.isfinite(transition_fraction) or not (0.0 <= transition_fraction < 1.0):
        raise ValueError("transition_fraction must satisfy 0 <= value < 1")

    low = transition_fraction * suppress_threshold
    high = suppress_threshold

    if high == low:
        resolved = (m >= high).astype(np.float64)
    else:
        t = np.clip((m - low) / (high - low), 0.0, 1.0)
        resolved = t * t * (3.0 - 2.0 * t)

    novelty = 1.0 - resolved
    novelty[~np.isfinite(m)] = 0.0
    novelty[m < 0] = 0.0
    return novelty
