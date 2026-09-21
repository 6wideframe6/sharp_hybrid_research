from pathlib import Path
import sys
import unittest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from k5.amplitude import fit_positive_scalar_scale

class TestAmplitudeB3(unittest.TestCase):
    def test_constant_source_fits_positive_target_scale(self):
        source = np.ones(100)
        target = np.full(100, 0.003)
        fit = fit_positive_scalar_scale(source, target)
        self.assertAlmostEqual(fit.scale, 0.003, places=14)

if __name__ == "__main__":
    unittest.main()
