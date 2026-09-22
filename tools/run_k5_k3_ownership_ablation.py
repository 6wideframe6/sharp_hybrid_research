#!/usr/bin/env python3
"""K5-D.2a.1 ownership ablation on the real K3 frozen-delta path.

Purpose:
isolate why D.2a retained only part of the transferred metric correction before
we run the live Gaussian decoder.

The exact same candidate inverse-depth grid and exact same pre-K3 accept pixels
from D.2a are evaluated under four ownership policies:

  component_veto
      current D.2a policy: nearest K5-support component IDs + layer-order veto.

  component_no_veto
      same component IDs, but only reports layer-order reversals instead of
      vetoing those cells.

  single_veto
      all accepted pixels share one local owner ID; retains the layer-order veto.
      This is an upper-bound diagnostic for false component-fragment collisions,
      NOT a production surface-label policy.

  single_no_veto
      one owner ID and no layer-order veto. This is a diagnostic ownership upper
      bound only.

Every replay uses frozen_delta=True: the Gaussian decoder is never re-run.
No PLY is written. Wire proxy, if supplied, is evaluation-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party/ml-sharp/src"))
sys.path.insert(0, str(ROOT))

from sharp.models import PredictorParams, create_predictor
from experiments.sharp_hybrid_research.alignment import sample_sharp_inverse_native
from experiments.sharp_hybrid_research.sharp_adapter import SharpAdapter, load_capture
from experiments.sharp_hybrid_research.k5.types import NativeBox


OFFSETS = np.arange(-8, 9, dtype=int)
OUTER = np.array([-8, -7, -6, 6, 7, 8], dtype=int)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0}
    return {
        "n": int(len(a)),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def abs_stats(values):
    return stats(np.abs(np.asarray(values, dtype=np.float64)))


def map_wire_proxy(folder, target_box):
    meta = json.loads((folder / "metadata.json").read_text())
    source_box = NativeBox.from_sequence(meta["box"])
    mask = np.asarray(Image.open(folder / "rgb_proxy_mask.png").convert("L")) > 0
    if mask.shape != source_box.shape:
        raise ValueError("wire proxy shape mismatch")
    out = np.zeros(target_box.shape, dtype=bool)
    try:
        common = source_box.intersect(target_box)
    except ValueError:
        return out
    sy, sx = source_box.local_slices(common)
    ty, tx = target_box.local_slices(common)
    out[ty, tx] = mask[sy, sx]
    return out


def row_runs(row):
    x = np.flatnonzero(row)
    if not len(x):
        return []
    split = np.where(np.diff(x) > 1)[0] + 1
    return [chunk for chunk in np.split(x, split) if len(chunk)]


def select_vertical_wire_samples(wire):
    h, w = wire.shape
    tolerant = ndi.maximum_filter1d(
        wire.astype(np.uint8), size=3, axis=1, mode="constant", cval=0
    ).astype(bool)
    out = []
    for y in range(9, h - 9):
        if y % 2:
            continue
        for run in row_runs(wire[y]):
            if int(run[-1] - run[0] + 1) > 3:
                continue
            x = int(round((int(run[0]) + int(run[-1])) * 0.5))
            if 9 <= x < w - 9 and float(np.mean(tolerant[y-4:y+5, x])) >= 0.75:
                out.append((y, x))
    return out


def aligned_wire_profile(field, samples):
    outer_idx = [int(np.where(OFFSETS == o)[0][0]) for o in OUTER]
    center_idx = int(np.where(OFFSETS == 0)[0][0])
    profiles, contrasts = [], []
    for y, x in samples:
        p = np.asarray(field[y, x + OFFSETS], dtype=np.float64)
        local = p - float(np.median(p[outer_idx]))
        center = float(local[center_idx])
        if abs(center) <= np.finfo(float).tiny:
            continue
        profiles.append((1.0 if center > 0 else -1.0) * local)
        contrasts.append(abs(center))
    if not profiles:
        return {"samples": 0}
    p = np.asarray(profiles)
    med = np.median(p, axis=0)
    peak_i = int(np.argmax(med))
    peak = float(med[peak_i])
    above = np.flatnonzero(med >= 0.5 * peak)
    return {
        "samples": int(len(p)),
        "center_abs_contrast": stats(contrasts),
        "peak_offset_px": int(OFFSETS[peak_i]),
        "peak_value": peak,
        "fwhm_px": (
            float(OFFSETS[above[-1]] - OFFSETS[above[0]] + 1)
            if len(above) else None
        ),
        "median_profile": [float(v) for v in med],
    }


def count(t):
    return int(t.sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--transfer", type=Path, required=True)
    ap.add_argument("--d2a-output", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--wire-crop", type=Path)
    args = ap.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    capture = load_capture(args.capture.resolve(), device="cpu")
    if capture.camera is None:
        raise ValueError("capture needs camera metadata")

    native_hw = tuple(int(v) for v in capture.camera["native_hw"])
    grid_hw = tuple(int(v) for v in capture.metric_depth.shape[-2:])

    d2a = args.d2a_output.resolve()
    d2a_summary = json.loads((d2a / "summary.json").read_text())
    box = NativeBox.from_sequence(d2a_summary["overlap_box_native"])

    pre_accept = np.load(d2a / "pre_k3_accept_mask.npy").astype(bool)
    component_ids = np.load(d2a / "surface_ids_grid.npy").astype(np.int64)
    old_applied = np.load(d2a / "k3_applied_pixels.npy").astype(bool)

    transfer = args.transfer.resolve()
    q_candidate = np.load(transfer / "q_candidate_sharp_grid.npy").astype(np.float64)
    delta_transfer = np.load(transfer / "delta_sharp_grid.npy").astype(np.float64)

    if any(a.shape != grid_hw for a in (pre_accept, component_ids, old_applied, q_candidate, delta_transfer)):
        raise ValueError("grid shape mismatch")

    baseline_q_t = capture.metric_depth[:, :1].reciprocal()
    baseline_q = baseline_q_t[0, 0].detach().cpu().numpy()
    q32 = q_candidate.astype(np.float32)

    if not np.array_equal(pre_accept, component_ids >= 0):
        # D.2a stores -1 outside accept and non-negative IDs inside accept.
        raise ValueError("D2a accept/surface-id contract mismatch")

    z0 = capture.metric_depth[0, 0].detach().cpu().numpy()
    z1 = capture.metric_depth[0, 1].detach().cpu().numpy()
    candidate_z = z0.copy()
    candidate_z[pre_accept] = 1.0 / q32[pre_accept]
    reversal_pixels = pre_accept & ((candidate_z < z1) != (z0 < z1))
    reversal_cells = F.max_pool2d(
        torch.from_numpy(reversal_pixels)[None, None].float(),
        2, 2,
    ).bool()
    veto = ~reversal_cells

    inverse_depth = baseline_q_t.clone()
    q_t = torch.from_numpy(q32)[None, None]
    accept_t = torch.from_numpy(pre_accept)[None, None]
    inverse_depth[accept_t] = q_t[accept_t]

    labels_component = torch.from_numpy(component_ids)[None, None]
    labels_single_np = np.full(grid_hw, -1, dtype=np.int64)
    labels_single_np[pre_accept] = 0
    labels_single = torch.from_numpy(labels_single_np)[None, None]

    torch.set_num_threads(2)
    model = create_predictor(PredictorParams())
    state = torch.load(
        args.checkpoint.resolve(),
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    adapter = SharpAdapter(model)

    policies = [
        ("component_veto", labels_component, veto),
        ("component_no_veto", labels_component, None),
        ("single_veto", labels_single, veto),
        ("single_no_veto", labels_single, None),
    ]

    wire = None
    wire_samples = None
    if args.wire_crop is not None:
        wire = map_wire_proxy(args.wire_crop.resolve(), box)
        wire_samples = select_vertical_wire_samples(wire)

    base_native_q = sample_sharp_inverse_native(
        capture.metric_depth[0, 0].detach().cpu().numpy(),
        native_hw,
        [box.x0, box.y0, box.x1, box.y1],
    )

    transfer_l1 = float(np.sum(np.abs(delta_transfer)))
    results = {}
    saved = {}

    for name, surface_ids, cell_allow in policies:
        replay = adapter.replay_corrected(
            capture,
            inverse_depth,
            accept_t,
            surface_ids,
            cell_allow=cell_allow,
            frozen_delta=True,
        )
        edit = replay.edit
        d = edit.diagnostics

        q_applied_grid = (
            edit.metric_depth[:, :1].reciprocal()[0, 0].detach().cpu().numpy()
        )
        delta_applied_grid = q_applied_grid - baseline_q
        q_native = sample_sharp_inverse_native(
            edit.metric_depth[0, 0].detach().cpu().numpy(),
            native_hw,
            [box.x0, box.y0, box.x1, box.y1],
        )
        delta_native = q_native - base_native_q

        result = {
            "applied_pixels": count(edit.accepted_pixels),
            "owned_cells": count(edit.owned_cells),
            "touched_cells": count(d["touched_cells"]),
            "collision_cells": count(d["collision_cells"]),
            "pool_unchanged_cells": count(d["pool_unchanged_cells"]),
            "out_of_support_pixels": count(d["out_of_support"]),
            "layer_order_reversal_pixels_after_edit": count(
                d["layer_order_reversal"]
            ),
            "delta_bit_exact": bool(torch.equal(replay.delta, capture.delta)),
            "secondary_metric_depth_bit_exact": bool(
                torch.equal(
                    edit.metric_depth[:, 1:],
                    capture.metric_depth[:, 1:],
                )
            ),
            "applied_fraction_of_pre_accept": float(
                count(edit.accepted_pixels) / int(pre_accept.sum())
            ),
            "l1_mass_retention_vs_D1_transfer": float(
                np.sum(np.abs(delta_applied_grid)) / transfer_l1
            ) if transfer_l1 else None,
            "native_applied_abs": abs_stats(delta_native),
        }

        if wire is not None:
            result["evaluation_only_wire"] = {
                "wire_pixels": int(wire.sum()),
                "native_applied_abs": abs_stats(delta_native[wire]),
                "profile": aligned_wire_profile(delta_native, wire_samples),
            }

        results[name] = result
        saved[name] = (
            edit.accepted_pixels[0, 0].detach().cpu().numpy(),
            delta_native,
        )

    if not np.array_equal(saved["component_veto"][0], old_applied):
        raise RuntimeError(
            "component_veto did not reproduce the existing D2a applied mask"
        )

    report = {
        "purpose": "K5-D.2a.1 K3 ownership-policy ablation",
        "capture_sha256": sha256(args.capture.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "overlap_box_native": [
            box.x0, box.y0, box.x1, box.y1
        ],
        "pre_k3_accept_pixels": int(pre_accept.sum()),
        "pre_k3_order_reversal_pixels": int(reversal_pixels.sum()),
        "pre_k3_order_reversal_cells": count(reversal_cells),
        "policies": {
            "component_veto": (
                "current D2a: local K5 support-component IDs + reversal veto"
            ),
            "component_no_veto": (
                "same component IDs, no reversal veto; diagnostic only"
            ),
            "single_veto": (
                "one global local-owner ID + reversal veto; collision upper bound only"
            ),
            "single_no_veto": (
                "one local-owner ID, no reversal veto; diagnostic upper bound only"
            ),
        },
        "results": results,
        "guardrails": [
            "All policies use exactly the same candidate inverse-depth grid.",
            "All policies use exactly the same pre-K3 accept pixels.",
            "Every replay uses frozen_delta=True.",
            "single_* policies are diagnostics, not production ownership labels.",
            "Wire proxy is evaluation-only.",
        ],
    }

    output.mkdir(parents=True, exist_ok=False)
    for name, (mask, delta_native) in saved.items():
        np.save(output / f"applied_pixels_{name}.npy", mask)
        np.save(output / f"delta_native_{name}.npy", delta_native)
        Image.fromarray(np.uint8(mask) * 255).save(
            output / f"applied_pixels_{name}.png"
        )

    (output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print("K5-D.2a.1 ownership ablation complete")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
