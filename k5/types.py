"""Typed containers for K5 native-coordinate differential detail processing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NativeBox:
    """Half-open native-image rectangle [x0, y0, x1, y1]."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"Invalid NativeBox: {self}")

    @classmethod
    def from_sequence(cls, values) -> "NativeBox":
        if len(values) != 4:
            raise ValueError("NativeBox requires exactly four coordinates")
        return cls(*(int(v) for v in values))

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def intersect(self, other: "NativeBox") -> "NativeBox":
        x0 = max(self.x0, other.x0)
        y0 = max(self.y0, other.y0)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        if x0 >= x1 or y0 >= y1:
            raise ValueError(f"Native boxes do not overlap: {self}, {other}")
        return NativeBox(x0, y0, x1, y1)

    def contains(self, other: "NativeBox") -> bool:
        return (
            self.x0 <= other.x0
            and self.y0 <= other.y0
            and self.x1 >= other.x1
            and self.y1 >= other.y1
        )

    def local_slices(self, child: "NativeBox") -> tuple[slice, slice]:
        if not self.contains(child):
            raise ValueError(f"{child} is not contained in {self}")
        return (
            slice(child.y0 - self.y0, child.y1 - self.y0),
            slice(child.x0 - self.x0, child.x1 - self.x0),
        )


@dataclass(frozen=True)
class GradientField:
    """TinyViM differential field in native pixel coordinates."""

    gx: np.ndarray
    gy: np.ndarray
    magnitude: np.ndarray
    direction_x: np.ndarray
    direction_y: np.ndarray
    valid: np.ndarray
    sigma_px: float

    def __post_init__(self) -> None:
        arrays = (
            self.gx,
            self.gy,
            self.magnitude,
            self.direction_x,
            self.direction_y,
            self.valid,
        )
        shape = self.gx.shape
        if self.gx.ndim != 2 or any(a.shape != shape for a in arrays):
            raise ValueError("GradientField arrays must share one 2-D shape")
        if self.valid.dtype != np.bool_:
            raise ValueError("GradientField.valid must be boolean")
        if self.sigma_px < 0:
            raise ValueError("sigma_px must be non-negative")

    @property
    def shape(self) -> tuple[int, int]:
        return self.gx.shape


@dataclass(frozen=True)
class ConsensusField:
    """Context-consensus TinyViM gradient orientation."""

    direction_x: np.ndarray
    direction_y: np.ndarray
    agreement: np.ndarray
    confidence: np.ndarray
    context_count: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            self.direction_x,
            self.direction_y,
            self.agreement,
            self.confidence,
            self.context_count,
            self.valid,
        )
        shape = self.direction_x.shape
        if self.direction_x.ndim != 2 or any(a.shape != shape for a in arrays):
            raise ValueError("ConsensusField arrays must share one 2-D shape")
        if self.valid.dtype != np.bool_:
            raise ValueError("ConsensusField.valid must be boolean")

    @property
    def shape(self) -> tuple[int, int]:
        return self.direction_x.shape
