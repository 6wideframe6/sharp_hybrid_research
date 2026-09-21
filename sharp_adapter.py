"""K3: exact pinned-SHARP capture/replay and preserved-secondary depth injection.

No model construction, checkpoint download, alignment, segmentation or fusion.
Import with the pinned checkout's ``src`` on PYTHONPATH. The caller supplies an
eval-mode RGBGaussianPredictor and already registered metric inverse depth.

Capture executes the upstream predictor itself. Replay substitutes only its
monodepth module on a shallow copy, so the expensive encoder is not rerun.
Fixed-normalization initialization invokes the upstream initializer on depth
normalized with the ORIGINAL upstream depth factor (not its rounded reciprocal).
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import sharp.models.initializer as initializer_source
from sharp.models.initializer import GaussianBaseValues, InitializerOutput, MultiLayerInitializer
from sharp.models.monodepth import MonodepthOutput
from sharp.models.predictor import RGBGaussianPredictor
from sharp.utils.gaussians import Gaussians3D, save_ply, unproject_gaussians

PINNED_COMMIT = "aed6527499ef91cba3b54c18d49a870f25947190"
SOURCE_ROOT = Path(__file__).resolve().parents[2] / "third_party/ml-sharp"
SCHEMA_VERSION = 1
_RECORDS = {c.__name__: c for c in
            (GaussianBaseValues, InitializerOutput, MonodepthOutput, Gaussians3D)}


def verify_source_contract(root: Path = SOURCE_ROOT) -> dict[str, str]:
    """Fail closed on another checkout, changed source, or shadowed imports."""
    root = Path(root).resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"SHARP revision mismatch: {commit}")
    changed = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--", "src/sharp"], text=True)
    if changed:
        raise RuntimeError(f"SHARP source differs from pinned contract:\n{changed}")
    # Validate every currently imported SHARP source module, not just the initializer.
    import sys
    expected = root / "src/sharp"
    for name, module in tuple(sys.modules.items()):
        if name == "sharp" or name.startswith("sharp."):
            location = getattr(module, "__file__", None)
            if location and not Path(location).resolve().is_relative_to(expected):
                raise RuntimeError(f"Shadowed SHARP module: {name}: {location}")
    names = ["models/predictor.py", "models/initializer.py", "models/composer.py",
             "models/heads.py", "models/monodepth.py", "models/gaussian_decoder.py",
             "utils/gaussians.py", "cli/predict.py"]
    return {"commit": commit, **{name: hashlib.sha256((expected / name).read_bytes()).hexdigest()
                                  for name in names}}


def _snapshot(value, memo=None):
    """Detach/copy tensors while retaining aliases between feature lists."""
    if memo is None:
        memo = {}
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        result = value.detach().clone()
    elif isinstance(value, tuple) and hasattr(value, "_fields"):
        result = type(value)(*(_snapshot(v, memo) for v in value))
    elif isinstance(value, (tuple, list)):
        result = type(value)(_snapshot(v, memo) for v in value)
    elif isinstance(value, dict):
        result = {k: _snapshot(v, memo) for k, v in value.items()}
    else:
        result = copy.deepcopy(value)
    memo[id(value)] = result
    return result


def _pack(value, memo=None):
    if memo is None:
        memo = {}
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu()
    elif isinstance(value, tuple) and hasattr(value, "_fields"):
        result = {"record": type(value).__name__, "items": [_pack(v, memo) for v in value]}
    elif isinstance(value, tuple):
        result = {"tuple": [_pack(v, memo) for v in value]}
    elif isinstance(value, list):
        result = [_pack(v, memo) for v in value]
    elif isinstance(value, dict):
        result = {k: _pack(v, memo) for k, v in value.items()}
    else:
        result = value
    memo[id(value)] = result
    return result


def _unpack(value):
    if isinstance(value, dict):
        if set(value) == {"record", "items"}:
            return _RECORDS[value["record"]](*(_unpack(v) for v in value["items"]))
        if set(value) == {"tuple"}:
            return tuple(_unpack(v) for v in value["tuple"])
        return {k: _unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unpack(v) for v in value]
    return value


@dataclass(frozen=True)
class SharpCapture:
    image: torch.Tensor
    disparity_factor: torch.Tensor
    monodepth_output: MonodepthOutput
    metric_depth: torch.Tensor
    normalized_depth: torch.Tensor
    depth_factor: torch.Tensor
    initializer: InitializerOutput
    delta: torch.Tensor
    gaussians_ndc: Gaussians3D
    metadata: dict[str, Any]
    camera: dict[str, Any] | None = None


@dataclass(frozen=True)
class InitializerEdit:
    initializer: InitializerOutput
    metric_depth: torch.Tensor
    normalized_depth: torch.Tensor
    accepted_pixels: torch.Tensor
    owned_cells: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


@dataclass(frozen=True)
class HybridReplay:
    edit: InitializerEdit
    delta: torch.Tensor
    gaussians_ndc: Gaussians3D


def save_capture(capture: SharpCapture, path: str | Path) -> None:
    """Portable tensor/dict-only payload, loadable with weights_only=True.

    Refuses overwrite; checkpoint-sized feature caches deserve an explicit new
    destination. File is removed on a failed write. No upstream files are touched.
    """
    path = Path(path)
    memo = {}
    payload = {"schema": SCHEMA_VERSION,
               "capture": {f.name: _pack(getattr(capture, f.name), memo) for f in fields(capture)}}
    with path.open("xb") as stream:
        try:
            torch.save(payload, stream)
        except BaseException:
            path.unlink(missing_ok=True)
            raise


def load_capture(path: str | Path, device: str | torch.device = "cpu") -> SharpCapture:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != SCHEMA_VERSION:
        raise ValueError("Unsupported SHARP cache schema")
    return SharpCapture(**{k: _unpack(v) for k, v in payload["capture"].items()})


def _tensor_sha(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(data).hexdigest()


def _model_sha(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(_tensor_sha(tensor).encode())
    return digest.hexdigest()


def _versions(model):
    return tuple((name, id(t), t._version) for name, t in
                 list(model.named_parameters()) + list(model.named_buffers()))


def _configuration(model):
    init = model.init_model
    composer = model.gaussian_composer
    return {"structure": repr(model),
            "initializer": {k: getattr(init, k) for k in (
                "num_layers", "stride", "base_depth", "scale_factor", "disparity_factor",
                "color_option", "first_layer_depth_option", "rest_layer_depth_option",
                "normalize_depth", "feature_input_stop_grad")},
            "composer": {k: getattr(composer, k) for k in (
                "min_scale", "max_scale", "color_activation_type", "opacity_activation_type",
                "color_space", "scale_factor", "base_scale_on_predicted_mean")},
            "monodepth": {k: getattr(model.monodepth_model, k, None) for k in (
                "num_monodepth_layers", "sorting_monodepth", "return_encoder_features",
                "return_decoder_features")},
            "decoder": {k: getattr(model.feature_model, k, None) for k in (
                "use_depth_input", "stride_out", "image_encoder_type", "grad_checkpointing")},
            "delta_factor": asdict(composer.delta_factor)}


class _CachedMonodepth(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, image):
        return self.output


def flattened_layer_indices(height: int, width: int, layer: int, *, device="cpu"):
    """Composer order is layer-major, then row-major; never interleaved."""
    if height <= 0 or width <= 0 or layer not in (0, 1):
        raise ValueError("Expected positive grid dimensions and layer 0 or 1")
    size = height * width
    return torch.arange(layer * size, (layer + 1) * size, device=device)


def _cells(tensor: torch.Tensor, stride: int):
    b, c, h, w = tensor.shape
    return tensor.reshape(b, c, h // stride, stride, w // stride, stride).permute(
        0, 1, 2, 4, 3, 5).reshape(b, c, h // stride, w // stride, stride * stride)


def _expand_cells(cells, stride):
    return cells.repeat_interleave(stride, -2).repeat_interleave(stride, -1)


class SharpAdapter:
    """Inference-only adapter bound to one immutable eval-mode predictor.

    FP32/default two-layer surface-min, stride-2 contract only. Small spatial
    grids are allowed for source-contract tests; capture_rgb uses CLI1536².
    Do not mutate model weights (including .data) or captured tensors. Recreate
    the adapter after loading weights/moving the model. One-time state hashing
    binds disk caches to actual weights, not an unchecked checkpoint filename.
    """

    def __init__(self, predictor: RGBGaussianPredictor, source_root: Path = SOURCE_ROOT):
        self.source_root = Path(source_root)
        self.contract = verify_source_contract(self.source_root)
        if type(predictor) is not RGBGaussianPredictor:
            raise TypeError("Expected the pinned upstream RGBGaussianPredictor")
        init = predictor.init_model
        if type(init) is not MultiLayerInitializer or not (
            init.normalize_depth and init.num_layers == 2 and init.stride == 2
            and init.first_layer_depth_option == "surface_min"
            and init.rest_layer_depth_option == "surface_min"
            and predictor.gaussian_composer.scale_factor == 1
        ):
            raise ValueError("Unsupported initializer/composer configuration for K3")
        self.predictor = predictor
        self.configuration = _configuration(predictor)
        self.weights_sha256 = _model_sha(predictor)
        self.versions = _versions(predictor)
        self._check_model()
        # The upstream module has no trainable initializer state. Copy the module
        # object, not its implementation, and leave the caller's module untouched.
        self.normalized_initializer = copy.copy(init)
        self.normalized_initializer.normalize_depth = False

    def _check_model(self):
        if any(m.training for m in self.predictor.modules()):
            raise ValueError("K3 requires every predictor module in eval mode")
        if _versions(self.predictor) != self.versions:
            raise ValueError("Predictor tensors changed; recreate adapter after loading/moving weights")
        if _configuration(self.predictor) != self.configuration:
            raise ValueError("Predictor configuration changed")

    def _check_capture(self, capture):
        self._check_model()
        if capture.metadata["source"] != self.contract:
            raise ValueError("Capture source contract mismatch")
        if capture.metadata["weights_sha256"] != self.weights_sha256:
            raise ValueError("Capture weights mismatch")
        if capture.metadata["configuration"] != self.configuration:
            raise ValueError("Capture configuration mismatch")
        self._validate_inputs(capture.image, capture.disparity_factor)
        b, _, h, w = capture.image.shape
        if capture.metric_depth.shape != (b, 2, h, w):
            raise ValueError("Invalid capture depth shape")
        model_tensors = list(self.predictor.parameters()) + list(self.predictor.buffers())
        if any(t.device != capture.image.device for t in model_tensors):
            raise ValueError("Capture and predictor must be on the same device")

    @staticmethod
    def _validate_inputs(image, disparity_factor):
        if (image.ndim != 4 or image.shape[1] != 3 or image.dtype != torch.float32
                or min(image.shape[-2:]) < 2 or any(n % 2 for n in image.shape[-2:])):
            raise ValueError("Expected FP32 [B,3,H,W], positive even H/W")
        if (disparity_factor.shape != (image.shape[0],) or disparity_factor.dtype != image.dtype
                or disparity_factor.device != image.device):
            raise ValueError("Expected same-device FP32 disparity_factor [B]")
        if not (torch.isfinite(image).all() and ((image >= 0) & (image <= 1)).all()
                and torch.isfinite(disparity_factor).all() and (disparity_factor > 0).all()):
            raise ValueError("Invalid image or camera disparity factor")
        if torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
            raise ValueError("K3 FP32 contract does not support autocast")

    @torch.no_grad()
    def capture(self, image, disparity_factor, *, camera=None, provenance=None) -> SharpCapture:
        """Run the original forward once; capture its actual intermediate outputs."""
        self._check_model()
        self._validate_inputs(image, disparity_factor)
        observed = {}
        handles = []

        def hook(name):
            def capture_output(module, args, output):
                if name in observed:
                    raise RuntimeError(f"Unexpected repeated {name} call")
                observed[name] = _snapshot(output)
            return capture_output

        try:
            for name in ("monodepth_model", "depth_alignment", "init_model", "prediction_head"):
                handles.append(getattr(self.predictor, name).register_forward_hook(hook(name)))
            gaussians = self.predictor(image, disparity_factor, depth=None)
        finally:
            for handle in handles:
                handle.remove()
        depth = observed["depth_alignment"][0]
        if depth.shape != (image.shape[0], 2, *image.shape[-2:]) or not (
            torch.isfinite(depth).all() and (depth > 0).all()
        ):
            raise ValueError("Upstream produced invalid two-layer metric depth")
        normalized, factor = initializer_source._rescale_depth(depth.contiguous())
        init = observed["init_model"]
        if not torch.equal(init.global_scale, 1.0 / factor):
            raise RuntimeError("Unexpected upstream normalization contract")
        metadata = {"source": copy.deepcopy(self.contract), "weights_sha256": self.weights_sha256,
                    "configuration": copy.deepcopy(self.configuration), "torch": str(torch.__version__),
                    "image_sha256": _tensor_sha(image), "image_shape": list(image.shape),
                    "depth_units": "metric z; corrections are metric inverse z",
                    "layer_order": "layer-major,row-major", "dtype": "float32",
                    "provenance": copy.deepcopy(provenance or {})}
        # Check that provenance is data, not an arbitrary pickle object.
        json.dumps(metadata)
        return SharpCapture(_snapshot(image), _snapshot(disparity_factor),
                            observed["monodepth_model"], depth, normalized, factor, init,
                            observed["prediction_head"], _snapshot(gaussians), metadata, _snapshot(camera))

    @torch.no_grad()
    def replay(self, capture: SharpCapture) -> Gaussians3D:
        """Exact upstream tail, including disparity conversion/alignment/initializer.

        Neither the caller's module registry nor its monodepth forward is changed.
        """
        self._check_capture(capture)
        replay_model = copy.copy(self.predictor)
        replay_model._modules = self.predictor._modules.copy()
        replay_model.monodepth_model = _CachedMonodepth(capture.monodepth_output).eval()
        return replay_model(capture.image, capture.disparity_factor, depth=None)

    @torch.no_grad()
    def edit_initializer(self, capture: SharpCapture, inverse_depth: torch.Tensor,
                         accept: torch.Tensor, surface_ids: torch.Tensor,
                         *, cell_allow: torch.Tensor | None = None) -> InitializerEdit:
        """Inject registered visible depth using caller-validated surface ownership.

        All three inputs are [B,1,H,W]. surface_ids is int64: nonnegative IDs
        identify accepted surfaces, -1 is unknown. IDs are compared only within
        a cell, never spatially inferred. Multiple different corrected surfaces
        within one 2x2 cell are rejected conservatively. Optional cell_allow is
        Boolean [B,1,H/2,W/2] for caller vetoes (e.g. known subcell collisions).

        A changed cell must affect its pooled base depth; edits to losing samples
        alone are rejected. Both adding a nearer winner and withdrawing a former
        winner are supported. Diagnostics retain proposed old/new pooling indices.
        Rejected cells revert ALL their pixels before decoder feature preparation.
        """
        self._check_capture(capture)
        reference = capture.metric_depth[:, :1]
        for name, tensor, dtype in (("inverse_depth", inverse_depth, reference.dtype),
                                    ("accept", accept, torch.bool), ("surface_ids", surface_ids, torch.int64)):
            if tensor.shape != reference.shape or tensor.device != reference.device or tensor.dtype != dtype:
                raise ValueError(f"{name}: expected {reference.shape}, {dtype}, {reference.device}")
        stride = self.predictor.init_model.stride
        grid_shape = (reference.shape[0], 1, reference.shape[2] // stride, reference.shape[3] // stride)
        if cell_allow is not None and (cell_allow.shape != grid_shape or cell_allow.dtype != torch.bool
                                       or cell_allow.device != reference.device):
            raise ValueError("cell_allow has wrong shape/dtype/device")

        finite_positive = torch.isfinite(inverse_depth) & (inverse_depth > 0)
        identity = inverse_depth == reference.reciprocal()
        eligible = accept & finite_positive & (surface_ids >= 0) & ~identity
        # Only valid selected values are reciprocated. Exact original tensors are
        # retained elsewhere, including the no-op reciprocal round-trip case.
        candidate_z = reference.clone()
        candidate_z[eligible] = inverse_depth[eligible].reciprocal()
        raw_normalized = candidate_z * capture.depth_factor[:, None, None, None]
        bounds_ok = torch.isfinite(candidate_z) & torch.isfinite(raw_normalized)
        bounds_ok &= (raw_normalized >= 1.0) & (raw_normalized <= 100.0)
        eligible &= bounds_ok
        changed = eligible & (candidate_z != reference)

        proposed_normalized = capture.normalized_depth.clone()
        proposed_normalized[:, :1] = torch.where(changed, raw_normalized,
                                                 capture.normalized_depth[:, :1])
        # Execute the original pooling, scale initialization and feature path.
        proposed = self.normalized_initializer(capture.image, proposed_normalized)
        old_base = capture.initializer.gaussian_base_values
        new_base = proposed.gaussian_base_values
        old_pool, old_winner = F.max_pool2d(1.0 / capture.normalized_depth[:, :1], stride, stride,
                                          return_indices=True)
        new_pool, new_winner = F.max_pool2d(1.0 / proposed_normalized[:, :1], stride, stride,
                                          return_indices=True)
        if not (torch.equal(old_pool, old_base.mean_inverse_z_ndc[:, :, 0]) and
                torch.equal(new_pool, new_base.mean_inverse_z_ndc[:, :, 0])):
            raise RuntimeError("Pooling ownership no longer matches upstream initializer")
        present = _cells(changed, stride)
        labels = _cells(surface_ids, stride)
        high = torch.iinfo(torch.int64).max
        low_label = torch.where(present, labels, high).amin(-1)
        high_label = torch.where(present, labels, -1).amax(-1)
        touched = present.any(-1)
        collision = touched & (low_label != high_label)
        owned = touched & ~collision & (old_pool != new_pool)
        if cell_allow is not None:
            owned &= cell_allow
        applied = changed & _expand_cells(owned, stride)
        normalized = capture.normalized_depth.clone()
        normalized[:, :1] = torch.where(applied, raw_normalized, normalized[:, :1])
        metric = capture.metric_depth.clone()
        metric[:, :1] = torch.where(applied, candidate_z, reference)

        # Selection is exact: do not reconstruct unchanged values through inverses.
        feature_input = capture.initializer.feature_input.clone()
        feature_input[:, 3:4] = torch.where(applied, proposed.feature_input[:, 3:4],
                                           feature_input[:, 3:4])
        replacements = {}
        for name in ("mean_inverse_z_ndc", "scales"):
            value = getattr(old_base, name).clone()
            value[:, :, 0] = torch.where(owned, getattr(new_base, name)[:, :, 0], value[:, :, 0])
            replacements[name] = value
        init = capture.initializer._replace(
            gaussian_base_values=old_base._replace(**replacements), feature_input=feature_input)
        diagnostics = {"invalid_value": accept & ~finite_positive,
                       "out_of_support": accept & finite_positive & ~identity & ~bounds_ok,
                       "unknown_surface": accept & (surface_ids < 0),
                       "candidate_pixels": changed, "touched_cells": touched,
                       "collision_cells": collision, "pool_unchanged_cells": touched & (old_pool == new_pool),
                       "old_winner_indices": old_winner, "proposed_winner_indices": new_winner,
                       "applied_winner_indices": torch.where(owned, new_winner, old_winner),
                       "layer_order_reversal": applied &
                       ((metric[:, :1] < metric[:, 1:]) !=
                        (capture.metric_depth[:, :1] < capture.metric_depth[:, 1:]))}
        return InitializerEdit(init, metric, normalized, applied, owned, diagnostics)

    @torch.no_grad()
    def replay_corrected(self, capture: SharpCapture, inverse_depth, accept, surface_ids,
                         *, cell_allow=None, frozen_delta=False) -> HybridReplay:
        edit = self.edit_initializer(capture, inverse_depth, accept, surface_ids, cell_allow=cell_allow)
        delta = capture.delta.clone()
        if edit.owned_cells.any() and not frozen_delta:
            features = self.predictor.feature_model(
                edit.initializer.feature_input, encodings=capture.monodepth_output.output_features)
            candidate = self.predictor.prediction_head(features)
            if candidate.shape != delta.shape or delta.shape[1:3] != (14, 2):
                raise RuntimeError("Unexpected Gaussian delta shape")
            owned = edit.owned_cells.expand(-1, delta.shape[1], -1, -1)
            if not torch.isfinite(candidate[:, :, 0][owned]).all():
                raise ValueError("Nonfinite corrected deltas; no hybrid scene produced")
            delta[:, :, 0] = torch.where(owned, candidate[:, :, 0], delta[:, :, 0])
        gaussians = self.predictor.gaussian_composer(
            delta=delta, base_values=edit.initializer.gaussian_base_values,
            global_scale=edit.initializer.global_scale)
        if not all(torch.isfinite(t).all() for t in gaussians):
            raise ValueError("Nonfinite composed Gaussians")
        return HybridReplay(edit, delta, gaussians)

    def capture_rgb(self, rgb: np.ndarray, f_px: float, *, device="cpu", provenance=None):
        image, factor, camera = prepare_rgb(rgb, f_px, device=device)
        return self.capture(image, factor, camera=camera, provenance=provenance)


@torch.no_grad()
def prepare_rgb(rgb: np.ndarray, f_px: float, *, device="cpu"):
    """Literal preprocessing/camera operations from pinned cli.predict_image.

    This small wrapper avoids importing CUDA rendering/CLI-only dependencies.
    A test executes the actual upstream predict_image function as the oracle.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 2:
        raise ValueError("Expected uint8 RGB [H,W,3]")
    if not np.isfinite(f_px) or f_px <= 0:
        raise ValueError("f_px must be positive and finite")
    internal_shape = (1536, 1536)
    image_pt = torch.from_numpy(rgb.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    factor = torch.tensor([f_px / width]).float().to(device)
    resized = F.interpolate(image_pt[None], size=(internal_shape[1], internal_shape[0]),
                            mode="bilinear", align_corners=True)
    intrinsics = torch.tensor([[f_px, 0, width / 2, 0], [0, f_px, height / 2, 0],
                               [0, 0, 1, 0], [0, 0, 0, 1]]).float().to(device)
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height
    camera = {"f_px": float(f_px), "native_hw": (height, width),
              "internal_hw": (internal_shape[1], internal_shape[0]),
              "intrinsics_resized": intrinsics_resized, "extrinsics": torch.eye(4).to(device),
              "resize": {"mode": "bilinear", "align_corners": True, "aspect": "square_stretch"}}
    return resized, factor, camera


def world_gaussians(gaussians_ndc: Gaussians3D, camera: dict) -> Gaussians3D:
    height, width = camera["internal_hw"]
    return unproject_gaussians(gaussians_ndc, camera["extrinsics"],
                              camera["intrinsics_resized"], (width, height))


def export_ply(gaussians_ndc: Gaussians3D, camera: dict, path: str | Path):
    """Original camera unprojection and PLY writer, without an alternate renderer."""
    if gaussians_ndc.mean_vectors.shape[0] != 1:
        raise ValueError("Export one image/scene at a time; batch export would merge scenes")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    return save_ply(world_gaussians(gaussians_ndc, camera), camera["f_px"],
                    tuple(camera["native_hw"]), path)
