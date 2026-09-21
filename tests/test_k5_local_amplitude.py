from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from k5.local_amplitude import knn_median_predict


class LocalAmplitudeTests(unittest.TestCase):
    def test_exact_local_clusters(self):
        coords = np.array([
            [0, 0], [0, 1], [1, 0],
            [100, 100], [100, 101], [101, 100],
        ], dtype=float)
        values = np.array([1, 1, 1, 3, 3, 3], dtype=float)
        queries = np.array([[0.5, 0.5], [100.5, 100.5]], dtype=float)

        result = knn_median_predict(coords, values, queries, k=3)
        np.testing.assert_allclose(result.values, [1, 3])

    def test_k_one_shape(self):
        coords = np.array([[0, 0], [10, 10]], dtype=float)
        values = np.array([2, 4], dtype=float)
        queries = np.array([[0.1, 0.1]], dtype=float)

        result = knn_median_predict(coords, values, queries, k=1)
        self.assertEqual(result.values.shape, (1,))
        self.assertEqual(result.nearest_distance_px.shape, (1,))
        self.assertEqual(result.kth_distance_px.shape, (1,))

    def test_reject_nonpositive_values(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            knn_median_predict(
                np.array([[0, 0], [1, 1]], dtype=float),
                np.array([1, 0], dtype=float),
                np.array([[0, 0]], dtype=float),
                k=1,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
