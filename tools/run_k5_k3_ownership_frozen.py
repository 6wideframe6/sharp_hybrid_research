#!/usr/bin/env python3
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

ROOT = Path(__file__).resolve().parents[2] if "experiments" not in str(Path(__file__).resolve()) else Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party/ml-sharp/src"))
sys.path.insert(0, str(ROOT))

from sharp.models import PredictorParams, create_predictor
from experiments.sharp_hybrid_research.alignment import sample_sharp_inverse_native
from experiments.sharp_hybrid_research.sharp_adapter import (
    SharpAdapter,
    flattened_layer_indices,
    load_capture,
)
from experiments.sharp_hybrid_research.k5.types import NativeBox

OFFSETS = np.arange(-8, 9, dtype=int)
OUTER = np.array([-8, -7, -6, 6, 7, 8], dtype=int)

def sha256(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()

def stats(v):
    a = np.asarray(v, dtype=np.float64)
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

def abs_stats(v):
    return stats(np.abs(np.asarray(v, dtype=np.float64)))

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

def grid_native_coordinates(native_hw, grid_hw):
    nh, nw = native_hw
    gh, gw = grid_hw
    gy, gx = np.mgrid[:gh, :gw].astype(np.float64)
    return gy * (nh - 1) / (gh - 1), gx * (nw - 1) / (gw - 1)

def sample_native_field_to_grid(values, box, *, native_hw, grid_hw, order, outside_value):
    values = np.asarray(values)
    ny, nx = grid_native_coordinates(native_hw, grid_hw)
    inside = (
        (nx >= box.x0) & (nx <= box.x1 - 1)
        & (ny >= box.y0) & (ny <= box.y1 - 1)
    )
    dtype = np.float64 if order else values.dtype
    out = np.full(grid_hw, outside_value, dtype=dtype)
    if inside.any():
        out[inside] = ndi.map_coordinates(
            values,
            [ny[inside] - box.y0, nx[inside] - box.x0],
            order=order,
            mode="nearest",
            prefilter=False,
        )
    return out, inside

def row_runs(row):
    x = np.flatnonzero(row)
    if not len(x):
        return []
    split = np.where(np.diff(x) > 1)[0] + 1
    return [c for c in np.split(x, split) if len(c)]

def select_vertical_wire_samples(wire):
    h, w = wire.shape
    tol = ndi.maximum_filter1d(
        wire.astype(np.uint8), size=3, axis=1, mode="constant", cval=0
    ).astype(bool)
    samples = []
    for y in range(9, h - 9):
        if y % 2:
            continue
        for run in row_runs(wire[y]):
            if int(run[-1] - run[0] + 1) > 3:
                continue
            x = int(round((int(run[0]) + int(run[-1])) * 0.5))
            if 9 <= x < w - 9 and float(np.mean(tol[y-4:y+5, x])) >= 0.75:
                samples.append((y, x))
    return samples

def aligned_wire_profile(field, samples):
    outer_idx = [int(np.where(OFFSETS == o)[0][0]) for o in OUTER]
    center_i = int(np.where(OFFSETS == 0)[0][0])
    profiles, contrasts = [], []
    for y, x in samples:
        p = np.asarray(field[y, x + OFFSETS], dtype=np.float64)
        local = p - float(np.median(p[outer_idx]))
        center = float(local[center_i])
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
        "fwhm_px": float(OFFSETS[above[-1]] - OFFSETS[above[0]] + 1) if len(above) else None,
        "median_profile": [float(v) for v in med],
    }

def save_mask(mask, path):
    Image.fromarray(np.uint8(mask) * 255).save(path)

def count(t):
    return int(t.sum().item())

def gaussian_preservation(capture, replay):
    h, w = capture.delta.shape[-2:]
    vis = flattened_layer_indices(h, w, 0)
    sec = flattened_layer_indices(h, w, 1)
    out = {}
    for name in capture.gaussians_ndc._fields:
        old = getattr(capture.gaussians_ndc, name)
        new = getattr(replay.gaussians_ndc, name)
        diff = (new[:, vis] - old[:, vis]).abs()
        out[name] = {
            "secondary_bit_exact": bool(torch.equal(old[:, sec], new[:, sec])),
            "visible_changed_values": int(torch.count_nonzero(diff).item()),
            "visible_max_abs_change": float(diff.max().item()),
        }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--composition", type=Path, required=True)
    ap.add_argument("--transfer", type=Path, required=True)
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--wire-crop", type=Path)
    ap.add_argument("--accept-radius-native-px", type=float, default=8.0)
    args = ap.parse_args()

    if args.accept_radius_native_px < 0 or not np.isfinite(args.accept_radius_native_px):
        raise ValueError("accept radius must be finite and non-negative")

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    capture = load_capture(args.capture.resolve(), device="cpu")
    if capture.camera is None:
        raise ValueError("capture needs camera metadata")

    native_hw = tuple(int(v) for v in capture.camera["native_hw"])
    grid_hw = tuple(int(v) for v in capture.metric_depth.shape[-2:])

    comp = args.composition.resolve()
    comp_summary = json.loads((comp / "summary.json").read_text())
    box = NativeBox.from_sequence(comp_summary["overlap_box_native"])
    gate_native = np.load(comp / f"application_gate_{args.variant}.npy").astype(np.float64)
    if gate_native.shape != box.shape:
        raise ValueError("composition gate mismatch")

    transfer = args.transfer.resolve()
    delta_grid = np.load(transfer / "delta_sharp_grid.npy").astype(np.float64)
    q_candidate_grid = np.load(transfer / "q_candidate_sharp_grid.npy").astype(np.float64)
    if delta_grid.shape != grid_hw or q_candidate_grid.shape != grid_hw:
        raise ValueError("D1 grid mismatch")

    support_native = gate_native > 0
    labels_native, component_count = ndi.label(
        support_native, structure=np.ones((3, 3), dtype=np.uint8)
    )
    if component_count < 1:
        raise RuntimeError("empty support")

    distance_native, nearest = ndi.distance_transform_edt(
        ~support_native, return_indices=True
    )
    owner_native = labels_native[tuple(nearest)]

    distance_grid, inside = sample_native_field_to_grid(
        distance_native, box, native_hw=native_hw, grid_hw=grid_hw,
        order=1, outside_value=np.inf
    )
    owner_grid, inside2 = sample_native_field_to_grid(
        owner_native.astype(np.int32), box, native_hw=native_hw, grid_hw=grid_hw,
        order=0, outside_value=0
    )
    owner_grid = owner_grid.astype(np.int64)
    if not np.array_equal(inside, inside2):
        raise RuntimeError("mapping mismatch")

    baseline_q_t = capture.metric_depth[:, :1].reciprocal()
    baseline_q = baseline_q_t[0, 0].detach().cpu().numpy()
    q32 = q_candidate_grid.astype(np.float32)
    changed32 = q32 != baseline_q
    accept_np = (
        inside
        & (distance_grid <= args.accept_radius_native_px)
        & (owner_grid > 0)
        & changed32
    )

    surface_np = np.full(grid_hw, -1, dtype=np.int64)
    surface_np[accept_np] = owner_grid[accept_np] - 1

    z0 = capture.metric_depth[0, 0].detach().cpu().numpy()
    z1 = capture.metric_depth[0, 1].detach().cpu().numpy()
    candidate_z = z0.copy()
    candidate_z[accept_np] = 1.0 / q32[accept_np]
    reversal_np = accept_np & ((candidate_z < z1) != (z0 < z1))
    reversal_t = torch.from_numpy(reversal_np)[None, None]
    reversal_cells = F.max_pool2d(reversal_t.float(), 2, 2).bool()
    cell_allow = ~reversal_cells

    inverse_depth = baseline_q_t.clone()
    q_candidate_t = torch.from_numpy(q32)[None, None]
    accept = torch.from_numpy(accept_np)[None, None]
    surface_ids = torch.from_numpy(surface_np)[None, None]
    inverse_depth[accept] = q_candidate_t[accept]

    torch.set_num_threads(2)
    model = create_predictor(PredictorParams())
    state = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval()

    adapter = SharpAdapter(model)
    replay = adapter.replay_corrected(
        capture, inverse_depth, accept, surface_ids,
        cell_allow=cell_allow, frozen_delta=True
    )
    edit = replay.edit

    q_applied_grid = edit.metric_depth[:, :1].reciprocal()[0, 0].detach().cpu().numpy()
    delta_applied_grid = q_applied_grid - baseline_q

    q_applied_native = sample_sharp_inverse_native(
        edit.metric_depth[0, 0].detach().cpu().numpy(),
        native_hw,
        [box.x0, box.y0, box.x1, box.y1],
    )
    q_baseline_native = sample_sharp_inverse_native(
        capture.metric_depth[0, 0].detach().cpu().numpy(),
        native_hw,
        [box.x0, box.y0, box.x1, box.y1],
    )
    delta_applied_native = q_applied_native - q_baseline_native

    d = edit.diagnostics
    pre = int(accept_np.sum())
    l1_before = float(np.sum(np.abs(delta_grid)))
    l1_after = float(np.sum(np.abs(delta_applied_grid)))

    report = {
        "purpose": "K5-D.2a real K3 ownership + frozen-delta replay",
        "variant": args.variant,
        "capture_sha256": sha256(args.capture.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "overlap_box_native": [box.x0, box.y0, box.x1, box.y1],
        "native_hw": list(native_hw),
        "sharp_grid_hw": list(grid_hw),
        "acceptance_policy": {
            "native_support_components": int(component_count),
            "accept_radius_native_px": float(args.accept_radius_native_px),
            "pre_k3_accept_pixels": pre,
            "pre_k3_order_reversal_pixels": int(reversal_np.sum()),
            "vetoed_2x2_cells_for_order_reversal": count(reversal_cells),
            "surface_id_semantics": "nearest 8-connected K5 application-support component; local ownership, not semantic class",
        },
        "k3_ownership": {
            "candidate_pixels_after_value_support_checks": count(d["candidate_pixels"]),
            "applied_pixels": count(edit.accepted_pixels),
            "touched_cells": count(d["touched_cells"]),
            "owned_cells": count(edit.owned_cells),
            "collision_cells": count(d["collision_cells"]),
            "pool_unchanged_cells": count(d["pool_unchanged_cells"]),
            "out_of_support_pixels": count(d["out_of_support"]),
            "layer_order_reversal_pixels_after_veto": count(d["layer_order_reversal"]),
            "applied_fraction_of_pre_k3_accept": float(count(edit.accepted_pixels) / pre) if pre else 0.0,
        },
        "metric_correction": {
            "transferred_delta_abs_1_per_m": abs_stats(delta_grid),
            "applied_delta_abs_1_per_m": abs_stats(delta_applied_grid),
            "l1_mass_retention_vs_D1_transfer": l1_after / l1_before if l1_before else None,
            "native_roundtrip_applied_delta_abs_1_per_m": abs_stats(delta_applied_native),
        },
        "frozen_delta_replay": {
            "delta_bit_exact": bool(torch.equal(replay.delta, capture.delta)),
            "secondary_metric_depth_bit_exact": bool(torch.equal(edit.metric_depth[:, 1:], capture.metric_depth[:, 1:])),
            "gaussian_preservation": gaussian_preservation(capture, replay),
        },
        "guardrails": [
            "Real pinned K3 adapter is used.",
            "Gaussian decoder is not re-run; frozen_delta=True.",
            "Secondary metric depth is never edited.",
            "Cells with proposed visible/secondary order reversal are vetoed.",
            "Mixed nearest-support owners in one 2x2 cell are rejected by K3.",
            "Wire proxy is evaluation-only.",
        ],
    }

    if args.wire_crop:
        wire = map_wire_proxy(args.wire_crop.resolve(), box)
        samples = select_vertical_wire_samples(wire)
        report["evaluation_only_wire"] = {
            "wire_pixels": int(wire.sum()),
            "vertical_profile_samples": int(len(samples)),
            "applied_native_abs": abs_stats(delta_applied_native[wire]),
            "applied_native_profile": aligned_wire_profile(delta_applied_native, samples),
        }

    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "pre_k3_accept_mask.npy", accept_np)
    np.save(output / "surface_ids_grid.npy", surface_np)
    np.save(output / "k3_applied_pixels.npy", edit.accepted_pixels[0,0].detach().cpu().numpy())
    np.save(output / "k3_owned_cells.npy", edit.owned_cells[0,0].detach().cpu().numpy())
    np.save(output / "q_applied_sharp_grid.npy", q_applied_grid)
    np.save(output / "delta_applied_sharp_grid.npy", delta_applied_grid)
    np.save(output / "delta_applied_native.npy", delta_applied_native)
    save_mask(accept_np, output / "pre_k3_accept_mask.png")
    save_mask(edit.accepted_pixels[0,0].detach().cpu().numpy(), output / "k3_applied_pixels.png")
    save_mask(edit.owned_cells[0,0].detach().cpu().numpy(), output / "k3_owned_cells.png")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    print("K5-D.2a K3 ownership/frozen-delta replay complete")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
