from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from k5.integration import saturated_gate, solve_screened_gradient_field


class IntegrationTests(unittest.TestCase):
    def test_saturated_gate(self):
        c = np.array([0.0, 0.025, 0.0625, 0.1, 1.0])
        g = saturated_gate(c, low=0.025, high=0.1)
        np.testing.assert_allclose(g, [0, 0, 0.5, 1, 1])

    def test_zero_gradient_returns_zero(self):
        gx = np.zeros((20, 21))
        gy = np.zeros((20, 21))
        result = solve_screened_gradient_field(
            gx, gy, screening=0.05
        )
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.delta_q, 0.0)

    def test_recovers_smooth_zero_boundary_potential_direction(self):
        h, w = 32, 35
        y = np.linspace(0.0, 1.0, h)
        x = np.linspace(0.0, 1.0, w)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        truth = np.sin(np.pi * yy) * np.sin(np.pi * xx)

        gy, gx = np.gradient(truth)
        result = solve_screened_gradient_field(
            gx,
            gy,
            screening=0.0,
            rtol=1e-9,
        )
        self.assertTrue(result.converged)

        pred = result.delta_q[1:-1, 1:-1].ravel()
        ref = truth[1:-1, 1:-1].ravel()
        corr = np.corrcoef(pred, ref)[0, 1]
        self.assertGreater(corr, 0.98)

        self.assertTrue(np.all(result.delta_q[0] == 0))
        self.assertTrue(np.all(result.delta_q[-1] == 0))
        self.assertTrue(np.all(result.delta_q[:, 0] == 0))
        self.assertTrue(np.all(result.delta_q[:, -1] == 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
