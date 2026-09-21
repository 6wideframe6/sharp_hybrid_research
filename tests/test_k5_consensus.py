"""Checkpoint-independent unit tests for K5 multi-context direction consensus."""

from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from k5.consensus import consensus_gradient_direction
from k5.types import GradientField


def field_from_direction(dx, dy, shape=(11, 13), valid=True):
    norm = float(np.hypot(dx, dy))
    if norm == 0:
        raise ValueError("test direction must be nonzero")
    nx, ny = dx / norm, dy / norm
    valid_mask = np.full(shape, bool(valid), dtype=bool)
    magnitude = np.ones(shape, dtype=np.float64)
    return GradientField(
        gx=np.full(shape, nx),
        gy=np.full(shape, ny),
        magnitude=magnitude,
        direction_x=np.full(shape, nx),
        direction_y=np.full(shape, ny),
        valid=valid_mask,
        sigma_px=1.0,
    )


class ConsensusTests(unittest.TestCase):
    def test_identical_context_directions_give_unit_confidence(self):
        a = field_from_direction(1, 2)
        b = field_from_direction(1, 2)
        result = consensus_gradient_direction([a, b])

        self.assertTrue(result.valid.all())
        np.testing.assert_allclose(result.agreement, 1.0, atol=1e-15)
        np.testing.assert_allclose(result.confidence, 1.0, atol=1e-15)
        expected = np.array([1.0, 2.0]) / np.sqrt(5.0)
        np.testing.assert_allclose(result.direction_x, expected[0], atol=1e-15)
        np.testing.assert_allclose(result.direction_y, expected[1], atol=1e-15)

    def test_opposing_contexts_are_rejected_by_default(self):
        a = field_from_direction(1, 0)
        b = field_from_direction(-1, 0)
        result = consensus_gradient_direction([a, b])

        self.assertFalse(result.valid.any())
        np.testing.assert_allclose(result.agreement, -1.0)
        np.testing.assert_allclose(result.confidence, 0.0)

    def test_orthogonal_directions_threshold_behavior(self):
        a = field_from_direction(1, 0)
        b = field_from_direction(0, 1)
        loose = consensus_gradient_direction([a, b], min_agreement=0.0)
        strict = consensus_gradient_direction([a, b], min_agreement=0.1)

        self.assertTrue(loose.valid.all())
        self.assertFalse(strict.valid.any())
        expected = 1.0 / np.sqrt(2.0)
        np.testing.assert_allclose(loose.direction_x, expected, atol=1e-15)
        np.testing.assert_allclose(loose.direction_y, expected, atol=1e-15)
        np.testing.assert_allclose(loose.confidence, 0.0, atol=1e-15)

    def test_three_contexts_average_pairwise_cosine(self):
        a = field_from_direction(1, 0)
        b = field_from_direction(1, 0)
        c = field_from_direction(0, 1)
        result = consensus_gradient_direction([a, b, c], min_contexts=2)

        np.testing.assert_allclose(result.agreement, 1.0 / 3.0, atol=1e-15)
        self.assertTrue(result.valid.all())
        np.testing.assert_allclose(result.confidence, 1.0 / 3.0, atol=1e-15)

    def test_missing_context_can_still_use_two_valid_contexts(self):
        a = field_from_direction(1, 0)
        b = field_from_direction(1, 0)
        c = field_from_direction(0, 1, valid=False)
        result = consensus_gradient_direction([a, b, c], min_contexts=2)

        self.assertTrue(result.valid.all())
        self.assertTrue((result.context_count == 2).all())
        np.testing.assert_allclose(result.agreement, 1.0, atol=1e-15)

    def test_shape_mismatch_is_rejected(self):
        a = field_from_direction(1, 0, shape=(8, 9))
        b = field_from_direction(1, 0, shape=(8, 10))
        with self.assertRaisesRegex(ValueError, "same shape"):
            consensus_gradient_direction([a, b])


if __name__ == "__main__":
    unittest.main(verbosity=2)
