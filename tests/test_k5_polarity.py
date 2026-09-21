from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from k5.polarity import compute_polarity_reliability
from k5.types import NativeBox


class PolarityTests(unittest.TestCase):
    def test_positive_affine_remap_preserves_reliability(self):
        y, x = np.mgrid[:96, :112]
        raw = (
            0.01 * x
            + 0.005 * y
            + 0.7 * np.exp(-((x - 56.0) ** 2) / 8.0)
        )
        box = NativeBox(100, 200, 212, 296)

        a = compute_polarity_reliability(
            raw,
            box,
            5.7 * raw - 13.0,
            box,
            box,
            fine_sigma_px=1.0,
            coarse_sigma_px=2.0,
        )

        safe = a.valid
        np.testing.assert_array_equal(
            np.signbit(a.dog_a[safe]),
            np.signbit(a.dog_b[safe]),
        )
        np.testing.assert_allclose(
            a.strength_a[safe],
            a.strength_b[safe],
            atol=2e-12,
            rtol=0,
        )
        np.testing.assert_allclose(
            a.reliability[safe],
            np.where(a.strength_consensus[safe] >= 0, a.reliability[safe], 0),
            atol=0,
            rtol=0,
        )

    def test_opposite_polarity_is_rejected(self):
        y, x = np.mgrid[:80, :90]
        ridge = np.exp(-((x - 45.0) ** 2) / 5.0)
        box = NativeBox(0, 0, 90, 80)

        result = compute_polarity_reliability(
            ridge,
            box,
            -ridge,
            box,
            box,
            fine_sigma_px=1.0,
            coarse_sigma_px=2.0,
            reliability_low=0.0,
            reliability_high=0.1,
        )

        self.assertFalse(result.sign_agreement[result.valid].any())
        self.assertTrue((result.reliability == 0).all())

    def test_halo_is_eight_pixels_for_coarse_sigma_two(self):
        raw = np.zeros((64, 70), dtype=np.float64)
        box = NativeBox(10, 20, 80, 84)

        result = compute_polarity_reliability(
            raw,
            box,
            raw,
            box,
            box,
            fine_sigma_px=1.0,
            coarse_sigma_px=2.0,
        )

        self.assertEqual(result.strict_halo_px, 8)
        self.assertFalse(result.valid[:8].any())
        self.assertFalse(result.valid[-8:].any())
        self.assertFalse(result.valid[:, :8].any())
        self.assertFalse(result.valid[:, -8:].any())
        self.assertTrue(result.valid[8:-8, 8:-8].all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
