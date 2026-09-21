"""Exact native-coordinate overlap utilities for K5."""

from __future__ import annotations

import numpy as np

from .types import NativeBox


def native_overlap(box_a: NativeBox, box_b: NativeBox) -> NativeBox:
    """Return the half-open native-coordinate intersection."""
    return box_a.intersect(box_b)


def crop_to_native_box(
    array: np.ndarray,
    source_box: NativeBox,
    target_box: NativeBox,
) -> np.ndarray:
    """Return a view of array corresponding to target_box, with no resizing."""
    if array.ndim < 2 or tuple(array.shape[:2]) != source_box.shape:
        raise ValueError(
            f"Array shape {array.shape[:2]} does not match source box {source_box.shape}"
        )
    ys, xs = source_box.local_slices(target_box)
    return array[ys, xs, ...]


def extract_native_overlap(
    array_a: np.ndarray,
    box_a: NativeBox,
    array_b: np.ndarray,
    box_b: NativeBox,
) -> tuple[NativeBox, np.ndarray, np.ndarray]:
    """Return common native box and aligned array views for two contexts."""
    common = native_overlap(box_a, box_b)
    return (
        common,
        crop_to_native_box(array_a, box_a, common),
        crop_to_native_box(array_b, box_b, common),
    )


def native_halo_support_mask(
    source_box: NativeBox,
    target_box: NativeBox,
    *,
    halo_px: int,
) -> np.ndarray:
    """Mask target pixels whose raw-data footprint stays inside source_box.

    ``halo_px`` is measured in native pixels. A target pixel is valid only if
    at least ``halo_px`` source pixels exist on all four sides before reaching
    the source-context boundary.
    """
    if halo_px < 0:
        raise ValueError("halo_px must be non-negative")
    if not source_box.contains(target_box):
        raise ValueError("target_box must be contained in source_box")

    yy, xx = np.mgrid[target_box.y0:target_box.y1, target_box.x0:target_box.x1]
    return (
        (xx >= source_box.x0 + halo_px)
        & (xx < source_box.x1 - halo_px)
        & (yy >= source_box.y0 + halo_px)
        & (yy < source_box.y1 - halo_px)
    )


def map_mask_to_box(
    mask: np.ndarray,
    mask_box: NativeBox,
    target_box: NativeBox,
) -> np.ndarray:
    """Map a boolean native-coordinate mask into target_box coordinates."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.shape != mask_box.shape:
        raise ValueError("Mask shape does not match mask_box")

    result = np.zeros(target_box.shape, dtype=bool)
    try:
        common = mask_box.intersect(target_box)
    except ValueError:
        return result

    source_y, source_x = mask_box.local_slices(common)
    target_y, target_x = target_box.local_slices(common)
    result[target_y, target_x] = mask[source_y, source_x]
    return result
