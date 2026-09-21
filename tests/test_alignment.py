"""Checkpoint-independent K4 tests: units, fitting, exclusions and holdout."""
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from experiments.sharp_hybrid_research.alignment import (
    AlignmentConfig, align_affine, make_fitting_maps, restricted_smoothing,
    sample_sharp_inverse_native, stratify_anchors,
)


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.y, self.x = np.mgrid[:128, :160]
        self.raw = 1. + self.x / 80 + self.y / 110 + .15 * np.sin(self.x / 9)
        self.target = .003 * self.raw + .0004
        self.mask = np.ones(self.raw.shape, bool)
        self.config = AlignmentConfig(sigma=0, bucket_cap=64, trim_fraction=.05)

    def fit(self, raw=None, target=None, mask=None, **kwargs):
        return align_affine(self.raw if raw is None else raw,
                            self.target if target is None else target,
                            self.mask if mask is None else mask, config=self.config, **kwargs)

    def test_exact_affine_recovery(self):
        result = self.fit()
        report = result["report"]
        self.assertTrue(report["accepted"], report)
        self.assertAlmostEqual(report["a"], .003, places=12)
        self.assertAlmostEqual(report["b"], .0004, places=12)
        np.testing.assert_allclose(result["aligned"], self.target, atol=1e-14, rtol=0)
        self.assertLess(report["heldout_normalized_p90"], 1e-12)

    def test_positive_slope_enforced(self):
        result = self.fit(target=.03 - self.target)
        self.assertFalse(result["report"]["accepted"])
        self.assertFalse(result["report"]["positive_slope"])
        self.assertFalse(result["valid_transformed"].any())
        self.assertTrue(np.isnan(result["aligned"]).all())

    def test_robust_coefficients_with_outliers(self):
        rng = np.random.default_rng(21)
        target = self.target.copy()
        mask = rng.random(target.shape) < .15
        target[mask] += rng.uniform(-.04, .04, mask.sum())
        # Negative physical target values are excluded before fitting.
        result = self.fit(target=target)
        self.assertAlmostEqual(result["report"]["a"], .003, delta=.00006)
        self.assertAlmostEqual(result["report"]["b"], .0004, delta=.00008)
        # Good coefficients do not excuse poor validation p90 from gross outliers.
        self.assertGreater(result["report"]["heldout_normalized_p90"], .25)
        self.assertFalse(result["report"]["accepted"])

    def test_nearly_constant_predictor_diagnostic(self):
        result = self.fit(raw=1. + 1e-15 * self.x / 160)
        self.assertEqual(result["report"]["status"], "underdetermined")
        self.assertIsNone(result["report"]["a"])
        self.assertIn("nearly constant", result["report"]["reasons"][0])

    def test_spatial_holdouts_are_whole_disjoint_blocks(self):
        report = self.fit()["report"]
        all_validation = []
        for fold in report["folds"]:
            self.assertFalse(set(fold["training_blocks"]) & set(fold["validation_blocks"]))
            all_validation.extend(fold["validation_blocks"])
        self.assertEqual(len(all_validation), len(set(all_validation)))
        self.assertEqual(len(all_validation), report["spatial_blocks"])

    def test_smoothed_holdout_has_disjoint_training_validation_kernel_samples(self):
        config = replace(self.config, sigma=8)
        groups = np.ones(self.raw.shape, np.int32)
        raw_fit, _ = restricted_smoothing(self.raw, self.mask, groups, config)
        target_fit, _ = restricted_smoothing(self.target, self.mask, groups, config)
        from experiments.sharp_hybrid_research import alignment
        calls = []

        def record(values, valid, regions, cfg):
            calls.append(valid.copy())
            return restricted_smoothing(values, valid, regions, cfg)

        with patch.object(alignment, "restricted_smoothing", side_effect=record):
            result = align_affine(self.raw, self.target, self.mask, raw_fit=raw_fit,
                                  target_fit=target_fit, regions=groups, config=config)
        self.assertTrue(result["report"]["accepted"], result["report"])
        self.assertEqual(len(calls), config.folds * 4)
        for start in range(0, len(calls), 4):
            self.assertFalse((calls[start] & calls[start + 2]).any())

    def test_spatially_inconsistent_relation_fails_holdout(self):
        target = self.target.copy()
        target[self.x >= 80] += .02
        report = self.fit(target=target)["report"]
        self.assertFalse(report["accepted"])
        self.assertTrue(report["folds"])

    def test_invalid_nonfinite_masks(self):
        raw, target = self.raw.copy(), self.target.copy()
        raw[0, 0], raw[1, 1] = np.nan, np.inf
        target[2, 2], target[3, 3], target[4, 4] = np.nan, 0., -1.
        result = self.fit(raw=raw, target=target)
        self.assertTrue(result["report"]["accepted"])
        for y, x in ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4)):
            self.assertFalse(result["selected_anchor_mask"][y, x])
        self.assertEqual(result["report"]["invalid_transformed_pixels"], 2)

    def test_bucket_cap_and_hierarchical_balance(self):
        groups = np.where(self.x < 144, 1, 2)
        config = replace(self.config, bucket_cap=7)
        sample = stratify_anchors(self.target, self.mask, groups, config)
        self.assertTrue(all(row["selected"] <= 7 for row in sample["buckets"]))
        self.assertAlmostEqual(sample["weights"].sum(), 1.)
        masses = {}
        for row in sample["buckets"]:
            pair = (row["region"], row["depth_bin"])
            masses[pair] = masses.get(pair, 0.) + row["weight_mass"]
        np.testing.assert_allclose(list(masses.values()), 1 / len(masses))

    def test_one_block_is_underdetermined_despite_many_pixels(self):
        mask = (self.y < 30) & (self.x < 30)
        result = self.fit(mask=mask)
        self.assertEqual(result["report"]["status"], "underdetermined")
        self.assertEqual(result["report"]["spatial_blocks"], 1)

    def test_sky_only_anchor_range_is_underdetermined(self):
        raw = self.raw.copy()
        raw[:, :120] = 1. + (raw[:, :120] - 1) * 1e-4
        target = .003 * raw + .0004
        mask = self.x < 120
        result = self.fit(raw=raw, target=target, mask=mask, foreground_mask=~mask)
        self.assertEqual(result["report"]["status"], "underdetermined")
        self.assertGreater(result["report"]["foreground_extrapolation"]["max_distance_in_anchor_spreads"], 100)

    def test_units_and_extreme_cross_model_slopes(self):
        raw = self.raw * 1e6 + 17.
        target = self.target * 1e-9
        result = self.fit(raw=raw, target=target)
        self.assertTrue(result["report"]["accepted"], result["report"])
        self.assertAlmostEqual(result["report"]["a"] / 3e-18, 1., places=10)
        self.assertEqual(result["report"]["target_units"], "1/metre")
        self.assertEqual(result["report"]["input_units"], "arbitrary relative disparity")
        self.assertEqual(result["report"]["output_units"], "1/metre")

    def test_region_restricted_smoothing_never_blends_the_step(self):
        values = np.where(self.x < 80, 1., 100.)
        groups = np.where(self.x < 80, 1, 2)
        values[10, 10] = np.nan
        filtered, mass = restricted_smoothing(values, self.mask, groups, AlignmentConfig())
        np.testing.assert_allclose(filtered[:, :80][np.isfinite(filtered[:, :80])], 1., atol=1e-13)
        np.testing.assert_allclose(filtered[:, 80:], 100., atol=1e-12)
        self.assertTrue(np.isnan(filtered[10, 10]))
        self.assertGreater(mass[64, 79], 0)

    def test_wire_border_padding_and_boundary_shoulder_exclusion(self):
        rgb = np.full((*self.raw.shape, 3), 180, np.uint8)
        rgb[:, 80:] = 30
        wires = np.zeros_like(self.mask)
        wires[:, 40] = True
        unpadded = self.mask.copy()
        unpadded[:, 140:] = False
        config = replace(AlignmentConfig(), border=8, shoulder=3)
        maps = make_fitting_maps(self.raw, self.target, rgb, wires, unpadded=unpadded, config=config)
        mask = maps["anchor_mask"]
        self.assertFalse(mask[:, 37:44].any())
        self.assertFalse(mask[:, 76:84].any())
        self.assertFalse(mask[:, 140:].any())
        self.assertFalse(mask[:8].any())
        self.assertTrue(mask.any())

    def test_apply_transform_to_sharp_raw_not_smoothed_field(self):
        raw = self.raw.copy()
        raw[60, 75] += 3
        mask = self.mask.copy()
        mask[60, 75] = False
        result = self.fit(raw=raw, mask=mask, raw_fit=self.raw)
        self.assertTrue(result["report"]["accepted"])
        self.assertAlmostEqual(result["aligned"][60, 75], self.target[60, 75] + .009)

    def test_align_corners_native_registration_and_inverse_first(self):
        y, x = np.mgrid[:9, :13]
        q = .01 + x * .002 + y * .003
        sampled = sample_sharp_inverse_native(1 / q, (25, 37), [4, 7, 17, 18])
        yy, xx = np.mgrid[7:18, 4:17]
        np.testing.assert_allclose(sampled, .01 + xx * 12 / 36 * .002 + yy * 8 / 24 * .003,
                                   atol=1e-16)

    def test_historical_alignment_unchanged(self):
        self.assertEqual(hashlib.sha256((ROOT / "src/alignment.py").read_bytes()).hexdigest(),
                         "60a13e72fc5d60e6c7132202d28855bfed8d10ea8d87fa8aebe30678dcb34f6b")


if __name__ == "__main__":
    unittest.main(verbosity=2)
