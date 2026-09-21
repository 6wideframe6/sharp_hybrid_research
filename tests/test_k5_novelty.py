from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from k5.novelty import sharp_novelty_gate


class NoveltyTests(unittest.TestCase):
    def test_gate_endpoints(self):
        m = np.array([0.0, 0.5, 0.75, 1.0, 2.0])
        g = sharp_novelty_gate(
            m,
            suppress_threshold=1.0,
            transition_fraction=0.5,
        )
        self.assertAlmostEqual(g[0], 1.0)
        self.assertAlmostEqual(g[1], 1.0)
        self.assertAlmostEqual(g[2], 0.5)
        self.assertAlmostEqual(g[3], 0.0)
        self.assertAlmostEqual(g[4], 0.0)

    def test_nonfinite_is_rejected_from_support(self):
        m = np.array([0.0, np.nan, np.inf])
        g = sharp_novelty_gate(m, suppress_threshold=1.0)
        np.testing.assert_allclose(g, [1.0, 0.0, 0.0])

    def test_stronger_sharp_gradient_never_increases_novelty(self):
        m = np.linspace(0.0, 2.0, 1000)
        g = sharp_novelty_gate(m, suppress_threshold=1.0)
        self.assertTrue(np.all(np.diff(g) <= 1e-15))


if __name__ == "__main__":
    unittest.main(verbosity=2)
