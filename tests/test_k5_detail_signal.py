"""Checkpoint-independent unit tests for K5 TinyViM differential extraction."""

from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from k5.detail_signal import extract_gradient_field
from k5.overlap import extract_native_overlap, map_mask_to_box
from k5.types import NativeBox


class DetailSignalTests(unittest.TestCase):
    def test_linear_plane_exact_direction_without_smoothing(self):
        y, x = np.mgrid[:31, :37]
        raw = 2.0 * x + 3.0 * y + 11.0
        field = extract_gradient_field(raw, sigma_px=0)

        np.testing.assert_allclose(field.gx, 2.0, atol=0, rtol=0)
        np.testing.assert_allclose(field.gy, 3.0, atol=0, rtol=0)
        np.testing.assert_allclose(field.magnitude, np.sqrt(13.0), atol=1e-14)
        np.testing.assert_allclose(
            field.direction_x, 2.0 / np.sqrt(13.0), atol=1e-14
        )
        np.testing.assert_allclose(
            field.direction_y, 3.0 / np.sqrt(13.0), atol=1e-14
        )
        self.assertTrue(field.valid.all())

    def test_positive_affine_raw_remap_preserves_direction(self):
        rng = np.random.default_rng(5)
        raw = rng.normal(size=(64, 80)).cumsum(axis=1)
        remapped = 4.7 * raw - 13.2

        a = extract_gradient_field(raw, sigma_px=1)
        b = extract_gradient_field(remapped, sigma_px=1)
        valid = a.valid & b.valid
        dot = (
            a.direction_x[valid] * b.direction_x[valid]
            + a.direction_y[valid] * b.direction_y[valid]
        )
        np.testing.assert_allclose(dot, 1.0, atol=1e-12, rtol=0)

    def test_constant_field_has_no_valid_direction(self):
        field = extract_gradient_field(np.ones((17, 23)), sigma_px=2)
        self.assertFalse(field.valid.any())
        self.assertTrue((field.direction_x == 0).all())
        self.assertTrue((field.direction_y == 0).all())

    def test_nonfinite_input_is_rejected(self):
        raw = np.ones((8, 8), dtype=np.float64)
        raw[3, 4] = np.nan
        with self.assertRaisesRegex(ValueError, "dense finite"):
            extract_gradient_field(raw)

    def test_native_overlap_has_no_resize_or_axis_flip(self):
        box_a = NativeBox(100, 200, 106, 205)
        box_b = NativeBox(103, 198, 109, 204)
        a = np.arange(30).reshape(5, 6)
        b = np.arange(36).reshape(6, 6) + 1000

        common, av, bv = extract_native_overlap(a, box_a, b, box_b)

        self.assertEqual(common, NativeBox(103, 200, 106, 204))
        np.testing.assert_array_equal(av, a[0:4, 3:6])
        np.testing.assert_array_equal(bv, b[2:6, 0:3])

    def test_mask_mapping_uses_half_open_native_coordinates(self):
        source_box = NativeBox(10, 20, 14, 23)
        target_box = NativeBox(12, 18, 17, 24)
        mask = np.zeros(source_box.shape, dtype=bool)
        mask[1, 2] = True  # native (12, 21)

        mapped = map_mask_to_box(mask, source_box, target_box)

        self.assertEqual(mapped.shape, target_box.shape)
        self.assertTrue(mapped[21 - 18, 12 - 12])
        self.assertEqual(mapped.sum(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
