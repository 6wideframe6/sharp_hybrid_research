#!/usr/bin/env python3
"""K5-D.2a.2 order-safe pixel clipping sweep on the real K3 frozen-delta path.

This version performs the safety clamp in inverse-depth space and then verifies
the actual float32 reciprocal used by K3.  If reciprocal rounding lands exactly
on / across the secondary layer, the inverse-depth value is advanced by further
float32 ULPs toward the original visible layer until the original layer order is
strictly restored.

The experiment keeps:
- exactly the same D.2a pre-K3 accept mask;
- exactly the same D.2a component surface IDs;
- K3 collision/support/pooling rules;
- frozen_delta=True (no Gaussian decoder rerun).

Only originally order-reversing pixels are clipped.

gap_fraction is the fraction of the ORIGINAL visible-secondary inverse-depth
gap retained on the original visible side:

    q_safe = q_secondary
           + gap_fraction * (q_visible_baseline - q_secondary)

0.0 means the largest possible correction that is still strictly order-safe
(after any required ULP repair); 1.0 means an exact visible-depth no-op for
crossing pixels.

Wire proxy, if supplied, is evaluation-only.
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
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


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
    return [c for c in np.split(x, split) if len(c)]


def select_vertical_wire_samples(wire):
    h, w = wire.shape
    tolerant = ndi.maximum_filter1d(
        wire.astype(np.uint8),
        size=3,
        axis=1,
        mode="constant",
        cval=0,
    ).astype(bool)

    samples = []
    for y in range(9, h - 9):
        if y % 2:
            continue
        for run in row_runs(wire[y]):
            if int(run[-1] - run[0] + 1) > 3:
                continue
            x = int(round((int(run[0]) + int(run[-1])) * 0.5))
            if 9 <= x < w - 9 and float(np.mean(tolerant[y - 4:y + 5, x])) >= 0.75:
                samples.append((y, x))
    return samples


def aligned_wire_profile(field, samples):
    outer_idx = [int(np.where(OFFSETS == o)[0][0]) for o in OUTER]
    center_idx = int(np.where(OFFSETS == 0)[0][0])

    profiles = []
    contrasts = []

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


def layer_reversal_mask(q_candidate, z0, z1, mask):
    q = np.asarray(q_candidate, dtype=np.float32)
    z_candidate = np.asarray(z0, dtype=np.float32).copy()

    selected_z = np.empty(np.count_nonzero(mask), dtype=np.float32)
    np.divide(
        np.float32(1.0),
        q[mask],
        out=selected_z,
    )
    z_candidate[mask] = selected_z

    return mask & ((z_candidate < z1) != (z0 < z1))


def make_safe_candidate(
    q_requested,
    q0,
    q1,
    z0,
    z1,
    reversal,
    gap_fraction,
    *,
    max_ulp_steps=1024,
):
    """Clip only reversal pixels and repair float32 reciprocal rounding.

    Returns:
      q_safe
      ulp_steps_per_pixel
      initial_bad_after_formula
    """
    q_requested = np.asarray(q_requested, dtype=np.float32)
    q0 = np.asarray(q0, dtype=np.float32)
    q1 = np.asarray(q1, dtype=np.float32)

    q_safe = q_requested.copy()

    # Define the requested safety point in inverse-depth space.
    safe64 = (
        q1.astype(np.float64)
        + float(gap_fraction)
        * (q0.astype(np.float64) - q1.astype(np.float64))
    )
    safe = safe64.astype(np.float32)
    q_safe[reversal] = safe[reversal]

    bad = layer_reversal_mask(q_safe, z0, z1, reversal)
    initial_bad = bad.copy()

    ulp_steps = np.zeros(q_safe.shape, dtype=np.int32)

    for _ in range(max_ulp_steps):
        if not bad.any():
            break

        # Move the inverse depth by one float32 ULP toward the original
        # visible-layer inverse depth. This monotonically restores the original
        # layer side without inventing a metric scale.
        q_safe[bad] = np.nextafter(
            q_safe[bad],
            q0[bad],
            dtype=np.float32,
        )
        ulp_steps[bad] += 1
        bad = layer_reversal_mask(q_safe, z0, z1, reversal)

    if bad.any():
        raise RuntimeError(
            "Could not restore strict original layer order for "
            f"{int(bad.sum())} pixels after {max_ulp_steps} float32 ULP steps"
        )

    return q_safe, ulp_steps, initial_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--transfer", type=Path, required=True)
    ap.add_argument("--d2a-output", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--wire-crop", type=Path)
    ap.add_argument(
        "--gap-fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0],
    )
    args = ap.parse_args()

    if any(
        (not np.isfinite(v)) or v < 0.0 or v > 1.0
        for v in args.gap_fractions
    ):
        raise ValueError("gap fractions must be finite in [0,1]")

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
    surface_ids_np = np.load(d2a / "surface_ids_grid.npy").astype(np.int64)

    transfer = args.transfer.resolve()
    q_requested = np.load(
        transfer / "q_candidate_sharp_grid.npy"
    ).astype(np.float32)
    delta_transfer = np.load(
        transfer / "delta_sharp_grid.npy"
    ).astype(np.float64)
    d1_native = np.load(
        transfer / "delta_roundtrip_native.npy"
    ).astype(np.float64)

    if any(
        a.shape != grid_hw
        for a in (
            pre_accept,
            surface_ids_np,
            q_requested,
            delta_transfer,
        )
    ):
        raise ValueError("grid shape mismatch")

    if not np.array_equal(surface_ids_np >= 0, pre_accept):
        raise ValueError("D2a pre-accept/surface-id contract mismatch")

    z0 = capture.metric_depth[0, 0].detach().cpu().numpy().astype(np.float32)
    z1 = capture.metric_depth[0, 1].detach().cpu().numpy().astype(np.float32)
    q0 = (np.float32(1.0) / z0).astype(np.float32)
    q1 = (np.float32(1.0) / z1).astype(np.float32)

    original_reversal = layer_reversal_mask(
        q_requested,
        z0,
        z1,
        pre_accept,
    )

    expected_reversal = int(
        d2a_summary["acceptance_policy"]["pre_k3_order_reversal_pixels"]
    )
    if int(original_reversal.sum()) != expected_reversal:
        raise RuntimeError(
            "reversal count does not reproduce D2a: "
            f"{int(original_reversal.sum())} != {expected_reversal}"
        )

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

    baseline_q_t = capture.metric_depth[:, :1].reciprocal()
    surface_ids = torch.from_numpy(surface_ids_np)[None, None]
    accept_t = torch.from_numpy(pre_accept)[None, None]

    baseline_native = sample_sharp_inverse_native(
        capture.metric_depth[0, 0].detach().cpu().numpy(),
        native_hw,
        [box.x0, box.y0, box.x1, box.y1],
    )

    transfer_l1 = float(np.sum(np.abs(delta_transfer)))

    wire = None
    wire_samples = None
    d1_wire_p90 = None

    if args.wire_crop is not None:
        wire = map_wire_proxy(args.wire_crop.resolve(), box)
        wire_samples = select_vertical_wire_samples(wire)
        d1_wire_p90 = float(
            np.percentile(np.abs(d1_native[wire]), 90)
        )

    results = {}
    saved = {}

    for gap_fraction in args.gap_fractions:
        q_safe, ulp_steps, initial_bad = make_safe_candidate(
            q_requested,
            q0,
            q1,
            z0,
            z1,
            original_reversal,
            float(gap_fraction),
        )

        safe_reversal = layer_reversal_mask(
            q_safe,
            z0,
            z1,
            pre_accept,
        )
        if safe_reversal.any():
            raise RuntimeError(
                f"gap_fraction={gap_fraction:g} still reverses "
                f"{int(safe_reversal.sum())} pixels after ULP repair"
            )

        inverse_depth = baseline_q_t.clone()
        q_t = torch.from_numpy(q_safe)[None, None]
        inverse_depth[accept_t] = q_t[accept_t]

        replay = adapter.replay_corrected(
            capture,
            inverse_depth,
            accept_t,
            surface_ids,
            cell_allow=None,
            frozen_delta=True,
        )
        edit = replay.edit
        diagnostics = edit.diagnostics

        if diagnostics["layer_order_reversal"].any():
            raise RuntimeError(
                f"gap_fraction={gap_fraction:g}: "
                "K3 reports post-edit reversal"
            )
        if not torch.equal(replay.delta, capture.delta):
            raise RuntimeError("frozen_delta replay changed learned deltas")
        if not torch.equal(
            edit.metric_depth[:, 1:],
            capture.metric_depth[:, 1:],
        ):
            raise RuntimeError("secondary metric depth changed")

        q_applied_grid = (
            edit.metric_depth[:, :1]
            .reciprocal()[0, 0]
            .detach()
            .cpu()
            .numpy()
        )
        delta_applied_grid = q_applied_grid - q0

        q_applied_native = sample_sharp_inverse_native(
            edit.metric_depth[0, 0].detach().cpu().numpy(),
            native_hw,
            [box.x0, box.y0, box.x1, box.y1],
        )
        delta_native = q_applied_native - baseline_native

        applied_np = (
            edit.accepted_pixels[0, 0]
            .detach()
            .cpu()
            .numpy()
        )
        clipped_applied = applied_np & original_reversal

        q_gap_before = np.abs(
            q0.astype(np.float64) - q1.astype(np.float64)
        )
        q_gap_requested = np.abs(
            q_safe.astype(np.float64) - q1.astype(np.float64)
        )

        requested_ratio = np.divide(
            q_gap_requested[original_reversal],
            q_gap_before[original_reversal],
            out=np.zeros(
                int(original_reversal.sum()),
                dtype=np.float64,
            ),
            where=q_gap_before[original_reversal] > 0,
        )

        q_after = q_applied_grid.astype(np.float32)
        q_gap_after = np.abs(
            q_after.astype(np.float64) - q1.astype(np.float64)
        )
        applied_ratio = np.divide(
            q_gap_after[clipped_applied],
            q_gap_before[clipped_applied],
            out=np.zeros(
                int(clipped_applied.sum()),
                dtype=np.float64,
            ),
            where=q_gap_before[clipped_applied] > 0,
        )

        repair_steps = ulp_steps[original_reversal]
        repaired_pixels = int(np.count_nonzero(repair_steps))

        result = {
            "gap_fraction": float(gap_fraction),
            "original_reversal_pixels": int(original_reversal.sum()),
            "float32_reciprocal_repair": {
                "pixels_bad_after_nominal_formula": int(
                    initial_bad.sum()
                ),
                "pixels_requiring_ulp_nudge": repaired_pixels,
                "ulp_steps": stats(repair_steps),
                "max_ulp_steps": int(
                    repair_steps.max()
                    if len(repair_steps)
                    else 0
                ),
            },
            "requested_inverse_depth_gap_ratio_on_original_reversals": (
                stats(requested_ratio)
            ),
            "applied_clipped_pixels": int(
                clipped_applied.sum()
            ),
            "applied_inverse_depth_gap_ratio_on_clipped_pixels": (
                stats(applied_ratio)
            ),
            "applied_pixels": count(edit.accepted_pixels),
            "owned_cells": count(edit.owned_cells),
            "collision_cells": count(
                diagnostics["collision_cells"]
            ),
            "pool_unchanged_cells": count(
                diagnostics["pool_unchanged_cells"]
            ),
            "out_of_support_pixels": count(
                diagnostics["out_of_support"]
            ),
            "layer_order_reversal_pixels": count(
                diagnostics["layer_order_reversal"]
            ),
            "applied_fraction_of_pre_accept": float(
                count(edit.accepted_pixels)
                / int(pre_accept.sum())
            ),
            "l1_mass_retention_vs_D1_transfer": float(
                np.sum(np.abs(delta_applied_grid))
                / transfer_l1
            ) if transfer_l1 else None,
            "native_applied_abs": abs_stats(
                delta_native
            ),
        }

        if wire is not None:
            wire_p90 = float(
                np.percentile(
                    np.abs(delta_native[wire]),
                    90,
                )
            )
            result["evaluation_only_wire"] = {
                "wire_pixels": int(wire.sum()),
                "native_applied_abs": abs_stats(
                    delta_native[wire]
                ),
                "wire_p90_retention_vs_D1_roundtrip": (
                    wire_p90 / d1_wire_p90
                    if d1_wire_p90
                    and d1_wire_p90 > 0
                    else None
                ),
                "profile": aligned_wire_profile(
                    delta_native,
                    wire_samples,
                ),
            }

        key = f"gap_{gap_fraction:g}"
        results[key] = result
        saved[key] = (
            applied_np,
            delta_native,
            q_safe,
        )

    report = {
        "purpose": (
            "K5-D.2a.2 order-safe pixel clipping sweep "
            "with float32 reciprocal ULP repair"
        ),
        "capture_sha256": sha256(
            args.capture.resolve()
        ),
        "checkpoint_sha256": sha256(
            args.checkpoint.resolve()
        ),
        "overlap_box_native": [
            box.x0,
            box.y0,
            box.x1,
            box.y1,
        ],
        "pre_k3_accept_pixels": int(
            pre_accept.sum()
        ),
        "original_order_reversal_pixels": int(
            original_reversal.sum()
        ),
        "surface_ids": (
            "unchanged D2a nearest connected-support-component ownership"
        ),
        "gap_fractions": [
            float(v) for v in args.gap_fractions
        ],
        "results": results,
        "guardrails": [
            "Same D2a pre-K3 accept mask for every variant.",
            "Same D2a component surface IDs for every variant.",
            "Only originally order-reversing pixels are clipped.",
            "No cell-level reversal veto is used.",
            "Float32 reciprocal order is explicitly rechecked after clipping.",
            "ULP repair moves only toward the original visible layer.",
            "All variants require zero post-edit layer-order reversals.",
            "Every replay uses frozen_delta=True.",
            "Wire proxy is evaluation-only.",
        ],
    }

    output.mkdir(parents=True, exist_ok=False)

    np.save(
        output / "original_reversal_pixels.npy",
        original_reversal,
    )

    for key, (mask, delta_native, q_safe) in saved.items():
        np.save(
            output / f"applied_pixels_{key}.npy",
            mask,
        )
        np.save(
            output / f"delta_native_{key}.npy",
            delta_native,
        )
        np.save(
            output / f"q_safe_grid_{key}.npy",
            q_safe,
        )

        Image.fromarray(
            np.uint8(mask) * 255
        ).save(
            output / f"applied_pixels_{key}.png"
        )

    (output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print(
        "K5-D.2a.2 order-safe clipping sweep complete"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
