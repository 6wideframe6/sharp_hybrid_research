"""Local SHARP-derived metric-amplitude helpers for K5-B diagnostics.

TinyViM/K5 strength is not treated as metric amplitude here.
The helper estimates a local scalar only from trusted SHARP calibration
samples using k-nearest-neighbour robust medians.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class KNNMedianResult:
    values: np.ndarray
    nearest_distance_px: np.ndarray
    kth_distance_px: np.ndarray


def knn_median_predict(
    train_yx: np.ndarray,
    train_values: np.ndarray,
    query_yx: np.ndarray,
    *,
    k: int,
) -> KNNMedianResult:
    """Predict local amplitude with the median of k nearest SHARP samples.

    Coordinates are native-pixel ``(y, x)`` pairs.
    Values must be finite and strictly positive metric amplitudes.
    """
    train_yx = np.asarray(train_yx, dtype=np.float64)
    query_yx = np.asarray(query_yx, dtype=np.float64)
    values = np.asarray(train_values, dtype=np.float64).reshape(-1)

    if train_yx.ndim != 2 or train_yx.shape[1] != 2:
        raise ValueError("train_yx must have shape [N,2]")
    if query_yx.ndim != 2 or query_yx.shape[1] != 2:
        raise ValueError("query_yx must have shape [M,2]")
    if len(train_yx) != len(values):
        raise ValueError("train_yx and train_values length mismatch")
    if k < 1:
        raise ValueError("k must be positive")
    if len(train_yx) < k:
        raise ValueError(f"Need at least k={k} training samples")
    if not np.all(np.isfinite(train_yx)) or not np.all(np.isfinite(query_yx)):
        raise ValueError("coordinates must be finite")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("train_values must be finite and strictly positive")

    tree = cKDTree(train_yx)
    distances, indices = tree.query(query_yx, k=k)

    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    prediction = np.median(values[indices], axis=1)

    return KNNMedianResult(
        values=np.asarray(prediction, dtype=np.float64),
        nearest_distance_px=np.asarray(distances[:, 0], dtype=np.float64),
        kth_distance_px=np.asarray(distances[:, -1], dtype=np.float64),
    )
