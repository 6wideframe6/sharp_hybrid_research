"""K3 source-contract tests, runnable with stdlib unittest and the existing venv.

Actual upstream predictor, initializer, Gaussian decoder, head, composer,
unprojection and PLY I/O execute. Only the expensive monodepth encoder is replaced
with deterministic synthetic features/disparities. Optional checkpoint test:
SHARP_K3_CHECKPOINT=/absolute/path/sharp_2572gikvuh.pt (no automatic downloads).
"""
from __future__ import annotations

import ast
import copy
from dataclasses import replace
import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "third_party/ml-sharp"
sys.path.insert(0, str(SOURCE / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from sharp.models import PredictorParams, create_predictor
from sharp.models.composer import GaussianComposer
from sharp.models.gaussian_decoder import create_gaussian_decoder
from sharp.models.heads import DirectPredictionHead
from sharp.models.initializer import MultiLayerInitializer, _rescale_depth
from sharp.models.monodepth import MonodepthOutput
from sharp.models.params import DeltaFactor, GaussianDecoderParams
from sharp.models.predictor import RGBGaussianPredictor
from sharp.utils.gaussians import load_ply, save_ply, unproject_gaussians

from experiments.sharp_hybrid_research import sharp_adapter as adapter_module
from experiments.sharp_hybrid_research.sharp_adapter import (
    SharpAdapter, export_ply, flattened_layer_indices, load_capture, prepare_rgb,
    save_capture, verify_source_contract, world_gaussians,
)

torch.set_num_threads(2)


class SyntheticMonodepth(nn.Module):
    """Known metric surfaces and deterministic features; not a SHARP depth claim."""
    def __init__(self, depth, factor):
        super().__init__()
        self.calls = 0
        self.register_buffer("disparity", factor[:, None, None, None] / depth)
        batch, _, height, width = depth.shape
        for level in range(5):
            self.register_buffer(f"feature_{level}", torch.randn(
                batch, 16, height // (2 ** (level + 1)), width // (2 ** (level + 1))))

    def forward(self, image):
        self.calls += 1
        encodings = [getattr(self, f"feature_{i}") for i in range(5)]
        return MonodepthOutput(self.disparity, encodings, encodings[0], list(encodings), [])

    def internal_resolution(self):
        return self.disparity.shape[-1]


def make_predictor(batch=2, *, custom_depth=None, seed=41):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        height, width = 32, 64
        factor = torch.linspace(.7, 1.1, batch)
        if custom_depth is None:
            depth = torch.full((batch, 2, height, width), 20.)
            depth[:, 0, 0, 0] = 10.
            depth[:, 1] = 1500.  # Exercises normalization's far clamp.
        else:
            depth = custom_depth
        image = torch.rand(batch, 3, height, width)
        params = GaussianDecoderParams(dim_in=5, dim_out=8,
                                       dims_decoder=(16, 16, 16, 16, 16), norm_num_groups=4)
        decoder = create_gaussian_decoder(params, [16] * 5)
        head = DirectPredictionHead(feature_dim=8, num_layers=2)
        # Upstream initializes heads to zero. Nonzero weights are essential to
        # expose depth-input coupling and test secondary-delta selection.
        for p in head.parameters():
            nn.init.normal_(p, std=.02)
        model = RGBGaussianPredictor(
            init_model=MultiLayerInitializer(num_layers=2, stride=2, base_depth=10.,
                                             scale_factor=1.3, disparity_factor=1.2,
                                             color_option="all_layers", normalize_depth=True,
                                             feature_input_stop_grad=False),
            monodepth_model=SyntheticMonodepth(depth, factor),
            feature_model=decoder, prediction_head=head,
            gaussian_composer=GaussianComposer(
                delta_factor=DeltaFactor(), min_scale=0., max_scale=10.,
                color_activation_type="sigmoid", opacity_activation_type="sigmoid",
                color_space="linearRGB", base_scale_on_predicted_mean=True),
            scale_map_estimator=None,
        ).eval()
    return model, image, factor


def assert_tree_equal(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertEqual(left.dtype, right.dtype)
        test.assertEqual(left.shape, right.shape)
        test.assertTrue(torch.equal(left, right), f"Unequal tensor of shape {left.shape}")
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_tree_equal(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree_equal(test, a, b)
    else:
        test.assertEqual(left, right)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.model, self.image, self.factor = make_predictor()
        self.adapter = SharpAdapter(self.model)
        self.capture = self.adapter.capture(self.image, self.factor,
                                            provenance={"kind": "synthetic contract fixture"})

    def inputs(self):
        q = self.capture.metric_depth[:, :1].reciprocal()
        accept = torch.zeros_like(q, dtype=torch.bool)
        labels = torch.full_like(q, -1, dtype=torch.int64)
        return q, accept, labels

    def edit_one(self):
        q, accept, labels = self.inputs()
        q[0, 0, 8, 12] = 1 / 12.
        accept[0, 0, 8, 12] = True
        labels[0, 0, 8, 12] = 7
        return q, accept, labels

    def test_capture_is_exact_upstream_forward_and_replay_skips_encoder(self):
        original_forward = self.model(self.image, self.factor)
        assert_tree_equal(self, self.capture.gaussians_ndc, original_forward)
        calls = self.model.monodepth_model.calls
        original_mono = self.model.monodepth_model
        assert_tree_equal(self, self.adapter.replay(self.capture), original_forward)
        self.assertEqual(self.model.monodepth_model.calls, calls)
        self.assertIs(self.model.monodepth_model, original_mono)
        self.assertTrue(self.model.init_model.normalize_depth)
        self.assertFalse(self.adapter.normalized_initializer.normalize_depth)

    def test_capture_snapshots_have_no_input_alias_and_keep_feature_aliases(self):
        self.assertNotEqual(self.image.data_ptr(), self.capture.image.data_ptr())
        self.assertNotEqual(self.model.monodepth_model.disparity.data_ptr(),
                            self.capture.monodepth_output.disparity.data_ptr())
        self.assertIs(self.capture.monodepth_output.encoder_features[0],
                      self.capture.monodepth_output.output_features[0])
        old = self.capture.image.clone()
        self.image.zero_()
        self.assertTrue(torch.equal(old, self.capture.image))
        self.assertFalse(self.capture.delta.requires_grad)

    def test_empty_and_identity_corrections_are_bit_exact_noops(self):
        q, accept, labels = self.inputs()
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                accept.fill_(enabled)
                labels.fill_(1)
                result = self.adapter.replay_corrected(self.capture, q, accept, labels)
                self.assertFalse(result.edit.owned_cells.any())
                self.assertFalse(result.edit.diagnostics["out_of_support"].any())
                assert_tree_equal(self, result.edit.initializer, self.capture.initializer)
                assert_tree_equal(self, result.delta, self.capture.delta)
                assert_tree_equal(self, result.gaussians_ndc, self.capture.gaussians_ndc)
                assert_tree_equal(self, result.edit.metric_depth, self.capture.metric_depth)

    def test_original_normalization_reused_without_reciprocal_roundtrip(self):
        # Exact source oracle: evaluate the same upstream initializer with its
        # normalization disabled on upstream _rescale_depth's exact output.
        normalized, factor = _rescale_depth(self.capture.metric_depth)
        self.assertTrue(torch.equal(factor, self.capture.depth_factor))
        init = copy.copy(self.model.init_model)
        init.normalize_depth = False
        reference = init(self.capture.image, normalized)
        assert_tree_equal(self, reference.gaussian_base_values,
                          self.capture.initializer.gaussian_base_values)
        assert_tree_equal(self, reference.feature_input, self.capture.initializer.feature_input)
        # Exercise non-default disparity/scale factors as well as the far clamp.
        q, accept, labels = self.edit_one()
        result = self.adapter.edit_initializer(self.capture, q, accept, labels)
        normalized[0, 0, 8, 12] = q[0, 0, 8, 12].reciprocal() * factor[0]
        oracle = init(self.capture.image, normalized)
        assert_tree_equal(self, result.initializer.gaussian_base_values, oracle.gaussian_base_values)
        assert_tree_equal(self, result.initializer.feature_input, oracle.feature_input)
        self.assertTrue(torch.equal(result.initializer.global_scale, self.capture.initializer.global_scale))

    def test_shared_normalization_regression_removing_global_minimum(self):
        q, accept, labels = self.inputs()
        q[0, 0, 0, 0] = 1 / 30.
        accept[0, 0, 0, 0] = True
        labels[0, 0, 0, 0] = 0  # Valid caller-identified background surface.
        result = self.adapter.replay_corrected(self.capture, q, accept, labels)
        self.assertTrue(result.edit.owned_cells[0, 0, 0, 0])
        naive = self.model.init_model(self.capture.image, result.edit.metric_depth)
        self.assertFalse(torch.equal(naive.global_scale, self.capture.initializer.global_scale))
        self.assertFalse(torch.equal(naive.gaussian_base_values.mean_inverse_z_ndc[:, :, 1],
                                     self.capture.initializer.gaussian_base_values.mean_inverse_z_ndc[:, :, 1]))
        assert_tree_equal(self, result.edit.initializer.global_scale, self.capture.initializer.global_scale)
        assert_tree_equal(self, result.edit.initializer.feature_input[:, 4:],
                          self.capture.initializer.feature_input[:, 4:])
        assert_tree_equal(self, result.delta[:, :, 1], self.capture.delta[:, :, 1])

    def test_new_nearest_sample_is_rejected_not_clamped(self):
        q, accept, labels = self.inputs()
        q[0, 0, 8, 12] = 1.  # 1m, outside original normalized support.
        accept[0, 0, 8, 12] = True
        labels[0, 0, 8, 12] = 4
        result = self.adapter.replay_corrected(self.capture, q, accept, labels)
        self.assertTrue(result.edit.diagnostics["out_of_support"][0, 0, 8, 12])
        assert_tree_equal(self, result.gaussians_ndc, self.capture.gaussians_ndc)
        self.assertEqual(float(result.edit.metric_depth[0, 0, 8, 12]),
                         float(self.capture.metric_depth[0, 0, 8, 12]))

    def test_all_invalid_and_out_of_range_are_exact_fallbacks(self):
        for value in (float("nan"), float("inf"), -float("inf"), 0., -1., 1e-40, 1e30, 1e-8):
            with self.subTest(value=value):
                q, accept, labels = self.inputs()
                q.fill_(value)
                accept.fill_(True)
                labels.fill_(1)
                result = self.adapter.replay_corrected(self.capture, q, accept, labels)
                self.assertFalse(result.edit.accepted_pixels.any())
                assert_tree_equal(self, result.edit.initializer, self.capture.initializer)
                assert_tree_equal(self, result.gaussians_ndc, self.capture.gaussians_ndc)
                self.assertTrue(torch.isfinite(result.edit.metric_depth).all())

    def test_invalid_holes_and_unknown_labels_do_not_contaminate_valid_edits(self):
        q, accept, labels = self.edit_one()
        accept[0, 0, 8, 13] = True
        labels[0, 0, 8, 13] = 99
        q[0, 0, 8, 13] = float("nan")
        accept[0, 0, 9, 12] = True  # Unknown label remains -1.
        q[0, 0, 9, 12] = 1 / 13.
        q[1].fill_(float("nan"))  # Unselected invalid values are harmless.
        result = self.adapter.replay_corrected(self.capture, q, accept, labels)
        self.assertEqual(result.edit.accepted_pixels.sum().item(), 1)
        mask = result.edit.accepted_pixels
        self.assertTrue(torch.equal(result.edit.metric_depth[:, :1][~mask],
                                    self.capture.metric_depth[:, :1][~mask]))
        self.assertTrue(torch.equal(result.edit.metric_depth[:, 1:], self.capture.metric_depth[:, 1:]))
        self.assertTrue(all(torch.isfinite(t).all() for t in result.gaussians_ndc))

    def test_secondary_and_unowned_base_delta_gaussians_preserved(self):
        before = copy.deepcopy(self.capture)
        q, accept, labels = self.edit_one()
        result = self.adapter.replay_corrected(self.capture, q, accept, labels)
        base0 = self.capture.initializer.gaussian_base_values
        base1 = result.edit.initializer.gaussian_base_values
        for name in base0._fields:
            if name in ("mean_inverse_z_ndc", "scales"):
                self.assertTrue(torch.equal(getattr(base0, name)[:, :, 1], getattr(base1, name)[:, :, 1]))
                mask = result.edit.owned_cells
                self.assertTrue(torch.equal(getattr(base0, name)[:, :, 0][~mask],
                                            getattr(base1, name)[:, :, 0][~mask]))
            else:
                assert_tree_equal(self, getattr(base0, name), getattr(base1, name))
        self.assertTrue(torch.equal(result.delta[:, :, 1], self.capture.delta[:, :, 1]))
        mask = result.edit.owned_cells.expand(-1, 14, -1, -1)
        self.assertTrue(torch.equal(result.delta[:, :, 0][~mask], self.capture.delta[:, :, 0][~mask]))
        self.assertFalse(torch.equal(result.delta[:, :, 0][mask], self.capture.delta[:, :, 0][mask]))
        # The real decoder DOES change secondary deltas before selection.
        with torch.no_grad():
            all_new = self.model.prediction_head(self.model.feature_model(
                result.edit.initializer.feature_input,
                encodings=self.capture.monodepth_output.output_features))
        self.assertFalse(torch.equal(all_new[:, :, 1], self.capture.delta[:, :, 1]))
        flat_mask = torch.cat((result.edit.owned_cells.flatten(1),
                               torch.zeros_like(result.edit.owned_cells.flatten(1))), dim=1)
        for old, new in zip(self.capture.gaussians_ndc, result.gaussians_ndc):
            self.assertTrue(torch.equal(old[~flat_mask], new[~flat_mask]))
        for field in before.__dataclass_fields__:
            assert_tree_equal(self, getattr(before, field), getattr(self.capture, field))

    def test_cell_collision_reverts_features_and_geometry_for_whole_cell(self):
        q, accept, labels = self.edit_one()
        q[0, 0, 9, 13] = 1 / 14.
        accept[0, 0, 9, 13] = True
        labels[0, 0, 9, 13] = 8
        result = self.adapter.replay_corrected(self.capture, q, accept, labels)
        self.assertTrue(result.edit.diagnostics["collision_cells"][0, 0, 4, 6])
        self.assertFalse(result.edit.owned_cells.any())
        assert_tree_equal(self, result.edit.initializer, self.capture.initializer)
        assert_tree_equal(self, result.gaussians_ndc, self.capture.gaussians_ndc)

    def test_same_surface_multiple_pixels_one_cell_and_batch_isolation(self):
        q, accept, labels = self.edit_one()
        q[0, 0, 9, 13] = 1 / 14.
        accept[0, 0, 9, 13] = True
        labels[0, 0, 9, 13] = 7
        edit = self.adapter.edit_initializer(self.capture, q, accept, labels)
        self.assertEqual(edit.owned_cells.sum().item(), 1)
        self.assertEqual(edit.accepted_pixels.sum().item(), 2)
        self.assertFalse(edit.owned_cells[1].any())
        self.assertEqual(edit.diagnostics["applied_winner_indices"][0, 0, 4, 6].item(), 8 * 64 + 12)

    def test_pixel_straddling_cell_boundary_produces_two_owned_cells(self):
        q, accept, labels = self.inputs()
        for col in (13, 14):
            q[0, 0, 8, col] = 1 / 12.
            accept[0, 0, 8, col] = True
            labels[0, 0, 8, col] = 7
        edit = self.adapter.edit_initializer(self.capture, q, accept, labels)
        self.assertEqual(edit.owned_cells.sum().item(), 2)
        self.assertTrue(edit.owned_cells[0, 0, 4, 6:8].all())

    def test_losing_sample_cannot_own_cell_and_explicit_veto_reverts(self):
        q, accept, labels = self.edit_one()
        q[0, 0, 8, 12] = 1 / 25.  # Deeper than other unedited20m pixels.
        edit = self.adapter.edit_initializer(self.capture, q, accept, labels)
        self.assertTrue(edit.diagnostics["pool_unchanged_cells"][0, 0, 4, 6])
        self.assertFalse(edit.accepted_pixels.any())
        assert_tree_equal(self, edit.initializer, self.capture.initializer)
        q, accept, labels = self.edit_one()
        veto = torch.zeros_like(edit.owned_cells)
        edited = self.adapter.edit_initializer(self.capture, q, accept, labels, cell_allow=veto)
        assert_tree_equal(self, edited.initializer, self.capture.initializer)

    def test_withdrawing_old_winner_tracks_revealed_background(self):
        q, accept, labels = self.inputs()
        q[0, 0, 0, 0] = 1 / 30.
        accept[0, 0, 0, 0] = True
        labels[0, 0, 0, 0] = 0
        edit = self.adapter.edit_initializer(self.capture, q, accept, labels)
        self.assertEqual(edit.diagnostics["old_winner_indices"][0, 0, 0, 0].item(), 0)
        self.assertEqual(edit.diagnostics["applied_winner_indices"][0, 0, 0, 0].item(), 1)
        self.assertTrue(edit.owned_cells[0, 0, 0, 0])

    def test_frozen_delta_mode_keeps_all_deltas_but_updates_owned_base(self):
        result = self.adapter.replay_corrected(self.capture, *self.edit_one(), frozen_delta=True)
        assert_tree_equal(self, result.delta, self.capture.delta)
        self.assertFalse(torch.equal(result.gaussians_ndc.mean_vectors,
                                     self.capture.gaussians_ndc.mean_vectors))

    def test_source_composer_flattening_is_layer_major_for_every_field(self):
        composer = self.model.gaussian_composer
        with torch.no_grad():
            unflattened = composer(self.capture.delta, self.capture.initializer.gaussian_base_values,
                                   global_scale=None, flatten_output=False)
            flat = composer(self.capture.delta, self.capture.initializer.gaussian_base_values,
                            global_scale=None, flatten_output=True)
        height, width = self.capture.delta.shape[-2:]
        for name in flat._fields:
            unflat_value, flat_value = getattr(unflattened, name), getattr(flat, name)
            for layer in (0, 1):
                indices = flattened_layer_indices(height, width, layer)
                if name == "opacities":
                    expected = unflat_value[:, layer].flatten(1)
                else:
                    expected = unflat_value[:, :, layer].permute(0, 2, 3, 1).flatten(1, 2)
                self.assertTrue(torch.equal(flat_value[:, indices], expected), name)
        # Distinct per-layer/cell opacity deltas make an interleaved mapping fail.
        delta = torch.zeros_like(self.capture.delta)
        for layer in (0, 1):
            delta[:, 13, layer] = (torch.arange(height * width).reshape(height, width)
                                    + layer * height * width) / 1000
        ordered = composer(delta, self.capture.initializer.gaussian_base_values, global_scale=None)
        self.assertTrue((torch.diff(ordered.opacities, dim=1) > 0).all())
        with self.assertRaises(ValueError):
            flattened_layer_indices(height, width, 2)

    def test_cache_roundtrip_and_wrong_weights_rejected(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            path = Path(directory) / "capture.pt"
            save_capture(self.capture, path)
            loaded = load_capture(path)
            self.assertIs(loaded.monodepth_output.encoder_features[0],
                          loaded.monodepth_output.output_features[0])
            for field in self.capture.__dataclass_fields__:
                assert_tree_equal(self, getattr(loaded, field), getattr(self.capture, field))
            assert_tree_equal(self, self.adapter.replay(loaded), self.capture.gaussians_ndc)
            assert_tree_equal(self, self.adapter.replay_corrected(loaded, *self.edit_one()).gaussians_ndc,
                              self.adapter.replay_corrected(self.capture, *self.edit_one()).gaussians_ndc)
            with self.assertRaises(FileExistsError):
                save_capture(self.capture, path)
            other, _, _ = make_predictor(seed=42)
            with self.assertRaisesRegex(ValueError, "weights mismatch"):
                SharpAdapter(other).replay(loaded)

    def test_invalid_shapes_dtypes_mode_and_model_mutation_rejected(self):
        q, accept, labels = self.edit_one()
        for args in ((q[:, 0], accept, labels), (q.double(), accept, labels),
                     (q, accept.float(), labels), (q, accept, labels.float())):
            with self.subTest(shapes=[tuple(x.shape) for x in args]):
                with self.assertRaises(ValueError):
                    self.adapter.edit_initializer(self.capture, *args)
        self.model.train()
        with self.assertRaisesRegex(ValueError, "eval mode"):
            self.adapter.replay(self.capture)
        self.model.eval()
        with torch.no_grad():
            next(self.model.prediction_head.parameters()).add_(1)
        with self.assertRaisesRegex(ValueError, "tensors changed"):
            self.adapter.replay(self.capture)

    def test_hook_cleanup_on_upstream_exception(self):
        counts = {name: len(getattr(self.model, name)._forward_hooks)
                  for name in ("monodepth_model", "depth_alignment", "init_model", "prediction_head")}
        with patch.object(self.model.prediction_head, "forward", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                self.adapter.capture(self.image, self.factor)
        for name, count in counts.items():
            self.assertEqual(len(getattr(self.model, name)._forward_hooks), count)
        assert_tree_equal(self, self.adapter.replay(self.capture), self.capture.gaussians_ndc)

    def test_cache_source_mismatch_rejected(self):
        metadata = copy.deepcopy(self.capture.metadata)
        metadata["source"]["commit"] = "wrong"
        with self.assertRaisesRegex(ValueError, "source contract mismatch"):
            self.adapter.replay(replace(self.capture, metadata=metadata))


class CameraAndContractTests(unittest.TestCase):
    def test_pinned_checkout_and_wrong_pin(self):
        self.assertEqual(verify_source_contract()["commit"], adapter_module.PINNED_COMMIT)
        with patch.object(adapter_module, "PINNED_COMMIT", "not-the-pinned-source"):
            with self.assertRaisesRegex(RuntimeError, "revision mismatch"):
                verify_source_contract()

    def test_preprocessing_unprojection_and_export_match_actual_cli_function(self):
        # Importing the CLI transitively imports a CUDA renderer. Execute the
        # unmodified predict_image AST only, with its real torch/F/unprojection
        # dependencies; no mathematical operation is replaced in this oracle.
        source_path = SOURCE / "src/sharp/cli/predict.py"
        tree = ast.parse(source_path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "predict_image")
        namespace = {"torch": torch, "np": np, "F": F,
                     "unproject_gaussians": unproject_gaussians, "LOGGER": logging.getLogger(__name__)}
        module = ast.Module(body=ast.parse("from __future__ import annotations").body + [function], type_ignores=[])
        exec(compile(module, str(source_path), "exec"), namespace)
        model, image, factor = make_predictor(batch=1)
        adapter = SharpAdapter(model)
        capture = adapter.capture(image, factor)

        class RecordingPredictor:
            def __call__(self, image, factor):
                self.image, self.factor = image, factor
                return capture.gaussians_ndc

        recorder = RecordingPredictor()
        rgb = np.random.default_rng(12).integers(0, 256, (21, 35, 3), dtype=np.uint8)
        expected_world = namespace["predict_image"](recorder, rgb, 26.5, torch.device("cpu"))
        prepared, k, camera = prepare_rgb(rgb, 26.5)
        self.assertTrue(torch.equal(recorder.image, prepared))
        self.assertTrue(torch.equal(recorder.factor, k))
        assert_tree_equal(self, world_gaussians(capture.gaussians_ndc, camera), expected_world)
        replayed = adapter.replay(capture)
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            direct_path, adapter_path = Path(directory) / "direct.ply", Path(directory) / "replay.ply"
            save_ply(expected_world, 26.5, (21, 35), direct_path)
            export_ply(replayed, camera, adapter_path)
            self.assertEqual(direct_path.read_bytes(), adapter_path.read_bytes())
            assert_tree_equal(self, load_ply(direct_path), load_ply(adapter_path))
            capture_with_camera = replace(capture, camera=camera)
            cache_path = Path(directory) / "with_camera.pt"
            save_capture(capture_with_camera, cache_path)
            loaded = load_capture(cache_path)
            assert_tree_equal(self, loaded.camera, camera)
            with self.assertRaises(FileExistsError):
                export_ply(replayed, camera, adapter_path)


@unittest.skipUnless(os.environ.get("SHARP_K3_CHECKPOINT"), "No SHARP checkpoint supplied; source tests run above")
class ReleasedCheckpointTests(unittest.TestCase):
    def test_released_model_capture_and_noop_replay(self):
        device = os.environ.get("SHARP_K3_DEVICE", "cpu")
        model = create_predictor(PredictorParams())
        state = torch.load(os.environ["SHARP_K3_CHECKPOINT"], map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        model.eval().to(device)
        rgb = np.array(Image.open(ROOT / "bridge.png").convert("RGB"))
        # Explicit focal value for numerical testing only, not camera accuracy.
        adapter = SharpAdapter(model)
        capture = adapter.capture_rgb(rgb, float(rgb.shape[1]), device=device,
                                      provenance={"kind": "released checkpoint no-op numerical gate"})
        assert_tree_equal(self, adapter.replay(capture), capture.gaussians_ndc)
        q = capture.metric_depth[:, :1].reciprocal()
        result = adapter.replay_corrected(capture, q, torch.ones_like(q, dtype=torch.bool),
                                          torch.zeros_like(q, dtype=torch.int64))
        assert_tree_equal(self, result.gaussians_ndc, capture.gaussians_ndc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
