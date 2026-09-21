"""Checkpoint-independent tests for K5-B amplitude diagnostics."""

from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from k5.amplitude import (
    fit_positive_vector_scale,
    fit_signed_vector_scale,
    signed_vector_residuals,
    vector_residuals,
)


class AmplitudeTests(unittest.TestCase):
    def test_exact_positive_scale_recovery(self):
        rng = np.random.default_rng(11)
        sx = rng.normal(size=(40, 50))
        sy = rng.normal(size=(40, 50))
        scale = 0.0037
        tx = scale * sx
        ty = scale * sy

        fit = fit_positive_vector_scale(sx, sy, tx, ty)

        self.assertAlmostEqual(fit.scale, scale, places=14)
        self.assertEqual(fit.samples, sx.size)
        self.assertAlmostEqual(fit.positive_projection_fraction, 1.0)
        np.testing.assert_allclose(
            vector_residuals(sx, sy, tx, ty, fit.scale),
            0.0,
            atol=1e-14,
            rtol=0,
        )

    def test_signed_fit_recovers_negative_relation(self):
        rng = np.random.default_rng(2)
        sx = rng.normal(size=2000)
        sy = rng.normal(size=2000)
        scale = -0.0042
        tx = scale * sx
        ty = scale * sy

        fit = fit_signed_vector_scale(sx, sy, tx, ty)

        self.assertAlmostEqual(fit.scale, scale, places=14)
        self.assertAlmostEqual(fit.positive_projection_fraction, 0.0)
        np.testing.assert_allclose(
            signed_vector_residuals(sx, sy, tx, ty, fit.scale),
            0.0,
            atol=1e-14,
            rtol=0,
        )

    def test_positive_fit_still_rejects_negative_relation(self):
        sx = np.ones(100)
        sy = np.zeros(100)
        tx = -2 * sx
        ty = np.zeros(100)

        with self.assertRaisesRegex(ValueError, "Non-positive"):
            fit_positive_vector_scale(sx, sy, tx, ty)

    def test_signed_fit_robust_with_outliers(self):
        rng = np.random.default_rng(3)
        sx = rng.normal(size=4000)
        sy = rng.normal(size=4000)
        true_scale = -0.005

        tx = true_scale * sx
        ty = true_scale * sy

        outliers = rng.choice(len(sx), 300, replace=False)
        tx[outliers] += rng.normal(0, 0.08, len(outliers))
        ty[outliers] += rng.normal(0, 0.08, len(outliers))

        fit = fit_signed_vector_scale(sx, sy, tx, ty)
        self.assertAlmostEqual(fit.scale, true_scale, delta=2e-4)

    def test_weights_shape_checked(self):
        sx = np.ones((4, 5))
        sy = np.ones((4, 5))
        tx = sx.copy()
        ty = sy.copy()

        with self.assertRaisesRegex(ValueError, "weights"):
            fit_signed_vector_scale(
                sx,
                sy,
                tx,
                ty,
                weights=np.ones(7),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
