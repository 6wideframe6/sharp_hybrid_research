"""Multi-context TinyViM gradient-orientation consensus for K5-A."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .types import ConsensusField, GradientField


def consensus_gradient_direction(
    fields: list[GradientField] | tuple[GradientField, ...],
    *,
    min_contexts: int = 2,
    min_agreement: float = 0.0,
) -> ConsensusField:
    """Combine TinyViM directions without using gradient magnitudes."""
    fields = tuple(fields)
    if len(fields) < 2:
        raise ValueError("At least two gradient fields are required")
    if not 2 <= min_contexts <= len(fields):
        raise ValueError("min_contexts must be between 2 and the field count")
    if not -1.0 <= min_agreement <= 1.0:
        raise ValueError("min_agreement must be in [-1, 1]")

    shape = fields[0].shape
    if any(field.shape != shape for field in fields):
        raise ValueError("All gradient fields must share the same shape")

    valid_stack = np.stack([field.valid for field in fields], axis=0)
    dx = np.stack([field.direction_x for field in fields], axis=0)
    dy = np.stack([field.direction_y for field in fields], axis=0)
    context_count = valid_stack.sum(axis=0).astype(np.int16)

    sum_x = np.sum(np.where(valid_stack, dx, 0.0), axis=0)
    sum_y = np.sum(np.where(valid_stack, dy, 0.0), axis=0)
    norm = np.hypot(sum_x, sum_y)

    pair_sum = np.zeros(shape, dtype=np.float64)
    pair_count = np.zeros(shape, dtype=np.int16)
    for i, j in combinations(range(len(fields)), 2):
        pair_valid = valid_stack[i] & valid_stack[j]
        dot = dx[i] * dx[j] + dy[i] * dy[j]
        pair_sum[pair_valid] += np.clip(dot[pair_valid], -1.0, 1.0)
        pair_count[pair_valid] += 1

    agreement = np.full(shape, np.nan, dtype=np.float64)
    has_pair = pair_count > 0
    agreement[has_pair] = pair_sum[has_pair] / pair_count[has_pair]

    numerical_floor = np.finfo(np.float64).eps * len(fields)
    valid = (
        (context_count >= min_contexts)
        & has_pair
        & (agreement >= min_agreement)
        & (norm > numerical_floor)
    )

    direction_x = np.zeros(shape, dtype=np.float64)
    direction_y = np.zeros(shape, dtype=np.float64)
    direction_x[valid] = sum_x[valid] / norm[valid]
    direction_y[valid] = sum_y[valid] / norm[valid]

    confidence = np.zeros(shape, dtype=np.float64)
    confidence[valid] = np.clip(agreement[valid], 0.0, 1.0)

    return ConsensusField(
        direction_x=direction_x,
        direction_y=direction_y,
        agreement=agreement,
        confidence=confidence,
        context_count=context_count,
        valid=valid,
    )
