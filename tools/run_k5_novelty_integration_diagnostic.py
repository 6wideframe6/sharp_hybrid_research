#!/usr/bin/env python3
"""K5-C.2 SHARP-novelty application-support diagnostic.

No corrected depth is sent into K3 / Gaussian generation.

C.1 showed that a confidence-only gate applies K5 correction strongly to broad
structures SHARP already resolves.  C.2 keeps the K5 consensus direction and
median B.5 metric amplitude, but multiplies the confidence gate by a SHARP
novelty gate:

    application_gate =
        saturated_confidence_gate
        * (1 - resolved_by_sharp)

``resolved_by_sharp`` is derived only from SHARP fine-gradient magnitude.
Known wire masks are evaluation-only and never affect the correction.

The run compares:
- baseline: no SHARP suppression;
- threshold multipliers 0.5, 1.0, 2.0 relative to the B.5 SHARP fine-gradient
  calibration threshold.

All signed PNG previews share ONE common numeric scale so their brightness is
directly comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from k5.integration import saturated_gate, solve_screened_gradient_field
from k5.novelty import sharp_novelty_gate
from k5.overlap import crop_to_native_box
from k5.types import NativeBox


def load_report(folder):
    return json.loads((folder / "alignment.json").read_text())


def load_context_array(folder, name, box):
    a = np.load(folder / name)
    if a.shape != box.shape:
        raise ValueError(f"{folder / name}: {a.shape} != {box.shape}")
    return a


def map_wire_proxy(folder, target_box):
    if folder is None:
        return None
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


def stats(a):
    a = np.asarray(a, dtype=np.float64)
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


def abs_stats(a):
    return stats(np.abs(np.asarray(a, dtype=np.float64)))


def save_gray01(a, path):
    image = np.round(np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(image).save(path)


def save_signed_common_scale(a, path, scale):
    if scale <= np.finfo(float).tiny:
        image = np.full(a.shape, 128, dtype=np.uint8)
    else:
        normalized = 0.5 + 0.5 * np.clip(a / scale, -1.0, 1.0)
        image = np.round(normalized * 255.0).astype(np.uint8)
    Image.fromarray(image).save(path)


def gate_mass(mask, gate):
    values = np.asarray(gate, dtype=np.float64)[mask]
    return {
        "pixels": int(np.count_nonzero(mask)),
        "nonzero_pixels": int(np.count_nonzero(values > 0)),
        "sum": float(np.sum(values)),
        "mean": float(np.mean(values)) if len(values) else None,
        "p90": float(np.percentile(values, 90)) if len(values) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-a", type=Path, required=True)
    ap.add_argument("--context-b", type=Path, required=True)
    ap.add_argument("--k5-output", type=Path, required=True)
    ap.add_argument("--amplitude-summary", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--wire-crop", type=Path)

    ap.add_argument("--fine-sigma", type=float, default=1.0)
    ap.add_argument("--coarse-sigma", type=float, default=2.0)
    ap.add_argument("--gate-low", type=float, default=0.025)
    ap.add_argument("--gate-high", type=float, default=0.10)
    ap.add_argument("--screening", type=float, default=0.05)
    ap.add_argument("--transition-fraction", type=float, default=0.5)
    ap.add_argument(
        "--threshold-multipliers",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
    )
    ap.add_argument("--cg-rtol", type=float, default=1e-7)
    ap.add_argument("--cg-maxiter", type=int, default=5000)
    ap.add_argument("--lowpass-sigma", type=float, default=16.0)
    args = ap.parse_args()

    if any(v <= 0 for v in args.threshold_multipliers):
        raise ValueError("threshold multipliers must be positive")

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    ca, cb = args.context_a.resolve(), args.context_b.resolve()
    ra, rb = load_report(ca), load_report(cb)
    ba = NativeBox.from_sequence(ra["provenance"]["box_native_half_open"])
    bb = NativeBox.from_sequence(rb["provenance"]["box_native_half_open"])
    common = ba.intersect(bb)

    k5_root = args.k5_output.resolve()
    k5_summary = json.loads((k5_root / "summary.json").read_text())
    if NativeBox.from_sequence(k5_summary["overlap_box_native"]) != common:
        raise ValueError("K5/context overlap mismatch")

    variant = k5_root / f"fine_{args.fine_sigma:g}_coarse_{args.coarse_sigma:g}"
    dx = np.load(variant / "direction_x.npy").astype(np.float64)
    dy = np.load(variant / "direction_y.npy").astype(np.float64)
    confidence = np.load(variant / "final_confidence.npy").astype(np.float64)

    qa = crop_to_native_box(
        load_context_array(ca, "sharp_visible_inverse_m.npy", ba), ba, common
    )
    qb = crop_to_native_box(
        load_context_array(cb, "sharp_visible_inverse_m.npy", bb), bb, common
    )
    q_sharp = 0.5 * (qa + qb)

    fine = ndi.gaussian_filter(
        q_sharp,
        sigma=args.fine_sigma,
        mode="nearest",
    )
    fy, fx = np.gradient(fine)
    sharp_fine_mag = np.hypot(fx, fy)

    amp_summary = json.loads(args.amplitude_summary.resolve().read_text())
    amplitude = float(
        amp_summary["support"]["target_positive"]["median"]
    )
    base_sharp_threshold = float(
        amp_summary["config"]["sharp_fine_threshold"]
    )

    if amplitude <= 0 or base_sharp_threshold <= 0:
        raise ValueError("invalid B.5 amplitude/SHARP threshold")

    base_gate = saturated_gate(
        confidence,
        low=args.gate_low,
        high=args.gate_high,
    )

    direction_norm = np.hypot(dx, dy)
    valid_direction = np.isfinite(direction_norm) & (direction_norm > 0)
    base_gate = np.where(valid_direction, base_gate, 0.0)

    variants = [("baseline", None)]
    for multiplier in args.threshold_multipliers:
        variants.append(
            (
                f"novelty_x{multiplier:g}",
                base_sharp_threshold * multiplier,
            )
        )

    wire = map_wire_proxy(
        args.wire_crop.resolve() if args.wire_crop else None,
        common,
    )
    near2 = near4 = near8 = None
    if wire is not None:
        near2 = ndi.binary_dilation(wire, iterations=2)
        near4 = ndi.binary_dilation(wire, iterations=4)
        near8 = ndi.binary_dilation(wire, iterations=8)

    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "sharp_fine_magnitude.npy", sharp_fine_mag)
    np.save(output / "base_confidence_gate.npy", base_gate)
    save_gray01(base_gate, output / "base_confidence_gate.png")

    results = {}
    deltas = {}

    base_gate_sum = float(np.sum(base_gate))
    if base_gate_sum <= np.finfo(float).tiny:
        raise RuntimeError("base confidence gate is empty")

    for label, threshold in variants:
        if threshold is None:
            novelty = np.ones(common.shape, dtype=np.float64)
        else:
            novelty = sharp_novelty_gate(
                sharp_fine_mag,
                suppress_threshold=threshold,
                transition_fraction=args.transition_fraction,
            )

        gate = base_gate * novelty
        gx = amplitude * dx * gate
        gy = amplitude * dy * gate

        solved = solve_screened_gradient_field(
            gx,
            gy,
            screening=args.screening,
            rtol=args.cg_rtol,
            maxiter=args.cg_maxiter,
        )
        if not solved.converged:
            raise RuntimeError(f"{label}: CG failed info={solved.cg_info}")

        delta = solved.delta_q
        corrected = q_sharp + delta
        dgy, dgx = np.gradient(delta)

        active = gate > 0
        desired_mag = np.hypot(gx, gy)
        residual = np.hypot(dgx - gx, dgy - gy)
        lowpass = ndi.gaussian_filter(
            delta,
            sigma=args.lowpass_sigma,
            mode="nearest",
        )

        report = {
            "sharp_suppress_threshold_1_per_m_per_px": threshold,
            "gate": {
                "nonzero_pixels": int(np.count_nonzero(active)),
                "sum": float(np.sum(gate)),
                "mean": float(np.mean(gate)),
                "mass_retention_vs_baseline": float(
                    np.sum(gate) / base_gate_sum
                ),
            },
            "delta_q_abs_1_per_m": abs_stats(delta),
            "gradient_fit_residual_1_per_m_per_px": (
                stats(residual[active]) if active.any() else {"n": 0}
            ),
            "lowpass_delta_abs_1_per_m": abs_stats(lowpass),
            "q_corrected": {
                "min": float(np.min(corrected)),
                "nonpositive_pixels": int(
                    np.count_nonzero(corrected <= 0)
                ),
                "nonfinite_pixels": int(
                    np.count_nonzero(~np.isfinite(corrected))
                ),
            },
        }

        if wire is not None:
            evaluation = {
                "wire": gate_mass(wire, gate),
                "near_wire_2px": gate_mass(near2, gate),
                "near_wire_4px": gate_mass(near4, gate),
                "near_wire_8px": gate_mass(near8, gate),
                "delta_q_abs_wire": abs_stats(delta[wire]),
                "delta_q_abs_near_wire_2px": abs_stats(delta[near2]),
            }

            for key, mask in (
                ("wire", wire),
                ("near_wire_2px", near2),
                ("near_wire_4px", near4),
                ("near_wire_8px", near8),
            ):
                base_mass = float(np.sum(base_gate[mask]))
                new_mass = float(np.sum(gate[mask]))
                evaluation[key]["mass_retention_vs_baseline"] = (
                    float(new_mass / base_mass)
                    if base_mass > np.finfo(float).tiny
                    else None
                )

            report["evaluation_only_wire"] = evaluation

        results[label] = report
        deltas[label] = delta

        np.save(output / f"novelty_gate_{label}.npy", novelty)
        np.save(output / f"application_gate_{label}.npy", gate)
        np.save(output / f"delta_q_{label}.npy", delta)
        save_gray01(
            novelty,
            output / f"novelty_gate_{label}.png",
        )
        save_gray01(
            gate,
            output / f"application_gate_{label}.png",
        )

    # ONE common preview scale for all variants.
    preview_scale = max(
        float(np.percentile(np.abs(delta), 99))
        for delta in deltas.values()
    )
    for label, delta in deltas.items():
        save_signed_common_scale(
            delta,
            output / f"delta_q_{label}_signed_common.png",
            preview_scale,
        )

    summary = {
        "purpose": "K5-C.2 SHARP-novelty support diagnostic only",
        "variant": variant.name,
        "overlap_box_native": [
            common.x0, common.y0, common.x1, common.y1
        ],
        "metric_amplitude_1_per_m_per_px": amplitude,
        "base_sharp_fine_threshold_1_per_m_per_px": base_sharp_threshold,
        "config": {
            "gate_low": args.gate_low,
            "gate_high": args.gate_high,
            "screening": args.screening,
            "transition_fraction": args.transition_fraction,
            "threshold_multipliers": args.threshold_multipliers,
            "common_signed_preview_scale_1_per_m": preview_scale,
        },
        "sharp_fine_magnitude": stats(sharp_fine_mag),
        "results": results,
        "guardrails": [
            "SHARP novelty suppression uses no wire mask.",
            "Wire proxy is evaluation-only.",
            "K5/TinyViM magnitude is not metric amplitude.",
            "All signed previews use the same numeric scale.",
            "No corrected depth is sent into K3 or Gaussian generation.",
        ],
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("K5-C.2 novelty integration diagnostic complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
