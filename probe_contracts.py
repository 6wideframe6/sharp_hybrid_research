"""Small CPU research probes; no checkpoint inference or production fusion claims.

Executes the unmodified initializer definitions extracted from the pinned SHARP
source, avoiding importing the full model and its unavailable dependencies.
The 1D fusion probe uses oracle boundaries to isolate the integration mathematics.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "probes"


def initializer_definitions():
    path = ROOT / "third_party/ml-sharp/src/sharp/models/initializer.py"
    source = path.read_text()
    names = {"GaussianBaseValues", "InitializerOutput", "MultiLayerInitializer",
             "_create_base_xy", "_create_base_scale", "_rescale_depth"}
    nodes = [n for n in ast.parse(source).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == names
    future = ast.parse("from __future__ import annotations").body
    namespace = {"torch": torch, "nn": nn, "NamedTuple": NamedTuple}
    exec(compile(ast.Module(body=future + nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, hashlib.sha256(source.encode()).hexdigest()


def profile_panel(profiles):
    width, height = 960, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 12), "SYNTHETIC oracle-boundary probe; inverse depth (larger = nearer)", fill="black")
    colors = ["black", "gray", "red", "blue", "green"]
    for i, ((name, values), color) in enumerate(zip(profiles.items(), colors)):
        draw.text((25, 35 + i * 18), name, fill=color)
        points = [(30 + j * 900 / (len(values) - 1), 580 - float(v) * 400)
                  for j, v in enumerate(values)]
        draw.line(points, fill=color, width=2)
    image.save(OUT / "synthetic_profiles.png")


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    definitions, sha = initializer_definitions()
    initializer = definitions["MultiLayerInitializer"](
        num_layers=2, stride=2, base_depth=10., scale_factor=1.,
        disparity_factor=1., color_option="all_layers")
    image = torch.zeros(1, 3, 8, 8)
    depth = torch.full((1, 2, 8, 8), 10.)
    depth[:, 1] = 300.  # Exercises the shared normalization's far clamp.
    base = initializer(image, depth)
    changed = depth.clone()
    changed[0, 0, 3, 3] = 1.
    naive = initializer(image, changed)
    assert torch.equal(changed[:, 1], depth[:, 1])
    assert not torch.equal(base.feature_input[:, 4], naive.feature_input[:, 4])
    secondary_base_z = base.global_scale / base.gaussian_base_values.mean_inverse_z_ndc[0, 0, 1, 0, 0]
    secondary_naive_z = naive.global_scale / naive.gaussian_base_values.mean_inverse_z_ndc[0, 0, 1, 0, 0]

    # The integration contract: validate BEFORE reciprocal, preserve exact fallback.
    fused_q = torch.full((1, 1, 8, 8), .2)
    fused_q[0, 0, 0, :4] = torch.tensor([float("nan"), 0., -1., float("inf")])
    valid = torch.isfinite(fused_q) & (fused_q > 0)
    safe_q = torch.where(valid, fused_q, depth[:, :1].reciprocal())
    hybrid = depth.clone()
    hybrid[:, :1] = torch.where(valid, safe_q.reciprocal(), depth[:, :1])
    assert torch.equal(hybrid[:, 1:], depth[:, 1:])
    assert torch.equal(hybrid[:, :1][~valid], depth[:, :1][~valid])
    assert torch.isfinite(hybrid).all() and (hybrid > 0).all()

    # 1D analogue of a foreground component with background fixed exactly.
    n = 129
    truth = np.full(n, .1)
    truth[62:65] = 1.
    sharp = cv2.GaussianBlur(truth[None], (0, 0), 4)[0]
    residual = truth - sharp
    hp = sharp + residual - cv2.GaussianBlur(residual[None], (0, 0), 8)[0]
    derivative = np.diff(truth)
    incidence = np.zeros((n - 1, n))
    for e in range(n - 1):
        incidence[e, e:e + 2] = [-1., 1.]
    # Ordinary screened Poisson with a global base tether loses the correct plateau.
    lam = .3
    ordinary = np.linalg.solve(incidence.T @ incidence + lam * np.eye(n),
                               incidence.T @ derivative + lam * sharp)
    # Permit correction of the known contaminated shoulder as well as the wire.
    support = np.arange(42, 85)
    fixed = np.setdiff1d(np.arange(n), support)
    matrix = incidence[:, support]
    rhs = derivative - incidence[:, fixed] @ sharp[fixed]
    # Consistent oracle jumps and a hard background boundary suffice in this
    # special case. The proposed real-data interval constraints are not tested.
    constrained = sharp.copy()
    solution = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    constrained[support] = solution
    assert np.array_equal(constrained[fixed], sharp[fixed])
    background = (truth == .1) & (np.abs(np.arange(n) - 63) <= 15)
    assert np.max(np.abs(constrained[background] - .1)) < 1e-12
    profile_panel({"truth": truth, "blurred base": sharp, "high-pass residual": hp,
                   "ordinary screened Poisson": ordinary, "surface-constrained": constrained})

    # Sampling-phase losses before the unchanged 2x2 min-depth pooling.
    sampling = []
    for wire_width in range(1, 6):
        peaks = []
        for start in range(2400, 2420):
            q = torch.full((1, 1, 1, 5388), .1)
            q[..., start:start + wire_width] = 1.
            resized = F.interpolate(q, size=(1, 1536), mode="bilinear", align_corners=True)
            pooled = F.max_pool2d(resized, (1, 2), (1, 2))
            peaks.append(float(pooled.max()))
        sampling.append({"native_width": wire_width, "phases": len(peaks),
                         "peak_min": min(peaks), "peak_max": max(peaks),
                         "fully_missed_phases": sum(v < .10001 for v in peaks)})

    stats = {
        "scope": "source initializer and synthetic mathematics only; no learned inference",
        "initializer_sha256": sha,
        "secondary_input_unchanged": True,
        "global_scale_before_after": [float(base.global_scale), float(naive.global_scale)],
        "secondary_metric_base_depth_before_after": [float(secondary_base_z), float(secondary_naive_z)],
        "secondary_decoder_input_max_change": float((base.feature_input[:, 4] - naive.feature_input[:, 4]).abs().max()),
        "invalid_depth_fallback_and_layer1_guards": "PASS",
        "background_max_abs_inverse_depth_error": {
            "highpass": float(np.max(np.abs(hp[background] - .1))),
            "screened_poisson": float(np.max(np.abs(ordinary[background] - .1))),
            "oracle_surface_constrained": float(np.max(np.abs(constrained[background] - .1)))},
        "sampling_phase_probe": sampling,
        "caveat": "Oracle support and consistent jumps explain near-zero synthetic halo; automatic labels and real-data interval constraints remain unvalidated."}
    (OUT / "contracts.json").write_text(json.dumps(stats, indent=2) + "\n")
    np.savez(OUT / "synthetic_profiles.npz", truth=truth, base=sharp, highpass=hp,
             screened_poisson=ordinary, constrained=constrained)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
