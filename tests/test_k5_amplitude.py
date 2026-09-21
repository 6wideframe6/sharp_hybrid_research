"""Tests for K5-B amplitude scalar/vector diagnostics."""

from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from k5.amplitude import (
    fit_positive_scalar_scale,
    fit_positive_vector_scale,
    fit_signed_vector_scale,
    scalar_residuals,
)


class AmplitudeTests(unittest.TestCase):
    def test_exact_positive_vector_scale(self):
        rng = np.random.default_rng(1)
        sx = rng.normal(size=1000)
        sy = rng.normal(size=1000)
        fit = fit_positive_vector_scale(sx, sy, 0.004*sx, 0.004*sy)
        self.assertAlmostEqual(fit.scale, 0.004, places=14)

    def test_signed_vector_negative_scale(self):
        rng = np.random.default_rng(2)
        sx = rng.normal(size=1000)
        sy = rng.normal(size=1000)
        fit = fit_signed_vector_scale(sx, sy, -0.003*sx, -0.003*sy)
        self.assertAlmostEqual(fit.scale, -0.003, places=14)

    def test_exact_positive_scalar_scale(self):
        x = np.linspace(0.01, 1.0, 1000)
        y = 0.0075 * x
        fit = fit_positive_scalar_scale(x, y)
        self.assertAlmostEqual(fit.scale, 0.0075, places=14)
        np.testing.assert_allclose(
            scalar_residuals(x, y, fit.scale), 0.0, atol=1e-14, rtol=0
        )

    def test_scalar_fit_robust_to_outliers(self):
        rng = np.random.default_rng(5)
        x = rng.uniform(0.01, 1.0, 4000)
        y = 0.006 * x
        idx = rng.choice(len(x), 300, replace=False)
        y[idx] += rng.uniform(0.03, 0.08, len(idx))
        fit = fit_positive_scalar_scale(x, y)
        self.assertAlmostEqual(fit.scale, 0.006, delta=2e-4)

    def test_scalar_negative_target_rejected_from_support(self):
        x = np.ones(100)
        y = np.ones(100)
        y[:20] = -1
        fit = fit_positive_scalar_scale(x, y)
        self.assertAlmostEqual(fit.scale, 1.0, places=14)
        self.assertEqual(fit.samples, 80)


if __name__ == "__main__":
    unittest.main(verbosity=2)
