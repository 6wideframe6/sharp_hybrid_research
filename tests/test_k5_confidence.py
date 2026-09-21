"""Checkpoint-independent tests for K5-A.1 detail confidence."""

from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from k5.confidence import (
    combine_direction_and_detail,
    detail_consensus_min,
    multiscale_detail_strength,
    robust_normalize_detail,
)


class ConfidenceTests(unittest.TestCase):
    def test_constant_linear_slope_has_negligible_multiscale_detail(self):
        y, x = np.mgrid[:96, :128]
        raw = 0.01 * x + 0.02 * y + 3.0

        detail = multiscale_detail_strength(
            raw,
            fine_sigma_px=1.0,
            coarse_sigma_px=4.0,
        )

        # Ignore Gaussian boundary effects. A true plane carries no
        # high-frequency detail in the interior.
        interior = detail[16:-16, 16:-16]
        plane_gradient = float(np.hypot(0.01, 0.02))
        self.assertLess(
            float(interior.max()),
            1e-4 * plane_gradient,
        )

    def test_thin_structure_is_detected_more_strongly_than_smooth_background(self):
        y, x = np.mgrid[:128, :160]
        raw = 0.002 * x + 0.001 * y
        raw[:, 80] += 1.0

        detail = multiscale_detail_strength(
            raw,
            fine_sigma_px=1.0,
            coarse_sigma_px=4.0,
        )

        wire_band = detail[:, 76:85]
        background = np.concatenate(
            [detail[:, 20:50].ravel(), detail[:, 110:140].ravel()]
        )
        self.assertGreater(
            float(np.percentile(wire_band, 90)),
            20.0 * max(float(np.percentile(background, 99)), 1e-15),
        )

    def test_context_local_normalization_removes_positive_scale_difference(self):
        y, x = np.mgrid[:80, :100]
        strength = np.exp(-((x - 50) ** 2 + (y - 40) ** 2) / 180.0)
        a = robust_normalize_detail(strength)
        b = robust_normalize_detail(17.0 * strength)

        np.testing.assert_allclose(a, b, atol=1e-14, rtol=0)

    def test_degenerate_normalization_returns_zero(self):
        result = robust_normalize_detail(np.full((20, 30), 7.0))
        self.assertTrue((result == 0).all())

    def test_min_consensus_rejects_context_only_detail(self):
        a = np.zeros((10, 12))
        b = np.zeros((10, 12))
        a[4, 5] = 1.0
        b[7, 8] = 0.8
        a[2, 3] = 0.7
        b[2, 3] = 0.6

        result = detail_consensus_min([a, b])

        self.assertEqual(result[4, 5], 0.0)
        self.assertEqual(result[7, 8], 0.0)
        self.assertAlmostEqual(result[2, 3], 0.6)

    def test_final_confidence_is_product_and_respects_valid_mask(self):
        direction = np.array([[1.0, 0.5], [0.8, 0.2]])
        detail = np.array([[0.4, 0.6], [0.5, 1.0]])
        valid = np.array([[True, True], [False, True]])

        result = combine_direction_and_detail(direction, detail, valid=valid)

        expected = np.array([[0.4, 0.3], [0.0, 0.2]])
        np.testing.assert_allclose(result, expected, atol=0, rtol=0)

    def test_invalid_sigma_order_is_rejected(self):
        raw = np.ones((8, 8))
        with self.assertRaisesRegex(ValueError, "greater"):
            multiscale_detail_strength(
                raw,
                fine_sigma_px=4.0,
                coarse_sigma_px=1.0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
