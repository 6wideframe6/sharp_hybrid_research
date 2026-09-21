#!/usr/bin/env python3
"""K5-C.8 final-composition diagnostic with generic polarity reliability.

This is the first diagnostic that explicitly writes

    q_final = q_sharp + delta_q

for candidate final gating variants.

Fixed upstream choices:
- K5-A.1b direction/confidence
- K5-B.5b SHARP-derived metric amplitude
- K5-C.2b novelty multiplier = 1 by default
- K5-C.3 screening = 0.05 by default
- discrete-consistent k5.integration solver

The new factor is generic TinyViM DoG polarity reliability.  The wire proxy is
optional and evaluation-only.

No corrected depth is fed into K3 or Gaussian generation.
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
from k5.polarity import compute_polarity_reliability
from k5.sharp_fields import shared_full_context_gradients
from k5.types import NativeBox


def load_report(folder):
    return json.loads((folder / "alignment.json").read_text())


def load_context_array(folder, name, box):
    value = np.load(folder / name)
    if value.shape != box.shape:
        raise ValueError(f"{folder / name}: {value.shape} != {box.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{folder / name}: array must be finite")
    return value.astype(np.float64)


def map_wire_proxy(folder, target_box):
    if folder is None:
        return None
    meta = json.loads((folder / "metadata.json").read_text())
    source_box = NativeBox.from_sequence(meta["box"])
    mask = np.asarray(
        Image.open(folder / "rgb_proxy_mask.png").convert("L")
    ) > 0
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


def stats(values):
    value = np.asarray(values, dtype=np.float64)
    value = value[np.isfinite(value)]
    if not len(value):
        return {"n": 0}
    return {
        "n": int(len(value)),
        "mean": float(np.mean(value)),
        "median": float(np.median(value)),
        "p10": float(np.percentile(value, 10)),
        "p25": float(np.percentile(value, 25)),
        "p75": float(np.percentile(value, 75)),
        "p90": float(np.percentile(value, 90)),
        "p99": float(np.percentile(value, 99)),
        "min": float(np.min(value)),
        "max": float(np.max(value)),
    }


def abs_stats(values):
    return stats(np.abs(np.asarray(values, dtype=np.float64)))


def smoothstep(value, low, high):
    if low < 0 or high <= low:
        raise ValueError("require 0 <= low < high")
    t = np.clip((value - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def leakage_report(delta, support):
    abs_delta = np.abs(delta)
    total_l1 = float(np.sum(abs_delta))
    total_l2 = float(np.sum(delta * delta))
    out = {}

    for radius in (0, 2, 4, 8, 16):
        if radius == 0:
            allowed = support
            key = "outside_active"
        else:
            allowed = ndi.binary_dilation(support, iterations=radius)
            key = f"outside_active_plus_{radius}px"

        outside = ~allowed
        l1 = float(np.sum(abs_delta[outside]))
        l2 = float(np.sum(delta[outside] ** 2))
        out[key] = {
            "abs_mass_fraction": (
                l1 / total_l1 if total_l1 > np.finfo(float).tiny else 0.0
            ),
            "energy_fraction": (
                l2 / total_l2 if total_l2 > np.finfo(float).tiny else 0.0
            ),
            "abs_delta": abs_stats(delta[outside]),
        }

    return out


def mass_retention(mask, new_gate, reference_gate):
    denom = float(np.sum(reference_gate[mask]))
    numer = float(np.sum(new_gate[mask]))
    return {
        "pixels": int(np.count_nonzero(mask)),
        "reference_gate_mass": denom,
        "gate_mass": numer,
        "retention": (
            numer / denom if denom > np.finfo(float).tiny else None
        ),
    }


def save_gray01(array, path):
    image = np.round(
        np.clip(array, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    Image.fromarray(image).save(path)


def save_signed_common(array, path, scale):
    if scale <= np.finfo(float).tiny:
        image = np.full(array.shape, 128, dtype=np.uint8)
    else:
        mapped = 0.5 + 0.5 * np.clip(array / scale, -1.0, 1.0)
        image = np.round(mapped * 255.0).astype(np.uint8)
    Image.fromarray(image).save(path)


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
    ap.add_argument("--novelty-multiplier", type=float, default=1.0)
    ap.add_argument("--transition-fraction", type=float, default=0.5)
    ap.add_argument("--screening", type=float, default=0.05)

    ap.add_argument(
        "--polarity-variants",
        nargs="+",
        default=[
            "baseline",
            "sign_only",
            "soft_0_0.10",
            "soft_0.025_0.10",
        ],
    )
    ap.add_argument("--dog-low-percentile", type=float, default=50.0)
    ap.add_argument("--dog-high-percentile", type=float, default=99.0)
    ap.add_argument("--lowpass-sigma", type=float, default=16.0)
    args = ap.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    ca, cb = args.context_a.resolve(), args.context_b.resolve()
    ra, rb = load_report(ca), load_report(cb)
    ba = NativeBox.from_sequence(
        ra["provenance"]["box_native_half_open"]
    )
    bb = NativeBox.from_sequence(
        rb["provenance"]["box_native_half_open"]
    )
    common = ba.intersect(bb)

    k5_root = args.k5_output.resolve()
    k5_summary = json.loads((k5_root / "summary.json").read_text())
    if NativeBox.from_sequence(k5_summary["overlap_box_native"]) != common:
        raise ValueError("K5/context overlap mismatch")

    variant = (
        k5_root
        / f"fine_{args.fine_sigma:g}_coarse_{args.coarse_sigma:g}"
    )
    dx = np.load(variant / "direction_x.npy").astype(np.float64)
    dy = np.load(variant / "direction_y.npy").astype(np.float64)
    confidence = np.load(
        variant / "final_confidence.npy"
    ).astype(np.float64)
    k5_valid = np.load(variant / "valid.npy").astype(bool)

    tiny_a = load_context_array(ca, "raw_tinyvim_relative.npy", ba)
    tiny_b = load_context_array(cb, "raw_tinyvim_relative.npy", bb)

    qa_full = load_context_array(ca, "sharp_visible_inverse_m.npy", ba)
    qb_full = load_context_array(cb, "sharp_visible_inverse_m.npy", bb)
    qa = crop_to_native_box(qa_full, ba, common)
    qb = crop_to_native_box(qb_full, bb, common)
    q_sharp = 0.5 * (qa + qb)

    sharp = shared_full_context_gradients(
        qa_full,
        ba,
        qb_full,
        bb,
        common,
        fine_sigma_px=args.fine_sigma,
    )
    sharp_fine_mag = np.hypot(
        np.asarray(sharp["fine_x"], dtype=np.float64),
        np.asarray(sharp["fine_y"], dtype=np.float64),
    )
    sharp_safe = np.asarray(sharp["safe_mask"], dtype=bool)

    amp_summary = json.loads(
        args.amplitude_summary.resolve().read_text()
    )
    amplitude = float(
        amp_summary["support"]["target_positive"]["median"]
    )
    sharp_threshold = float(
        amp_summary["config"]["sharp_fine_threshold"]
    )
    suppress_threshold = sharp_threshold * args.novelty_multiplier

    confidence_gate = saturated_gate(
        confidence,
        low=args.gate_low,
        high=args.gate_high,
    )
    direction_norm = np.hypot(dx, dy)
    base_valid = (
        k5_valid
        & sharp_safe
        & np.isfinite(direction_norm)
        & (direction_norm > 0)
    )
    confidence_gate = np.where(
        base_valid, confidence_gate, 0.0
    )

    novelty_gate = sharp_novelty_gate(
        sharp_fine_mag,
        suppress_threshold=suppress_threshold,
        transition_fraction=args.transition_fraction,
    )
    novelty_gate = np.where(sharp_safe, novelty_gate, 0.0)

    base_application_gate = confidence_gate * novelty_gate

    polarity = compute_polarity_reliability(
        tiny_a,
        ba,
        tiny_b,
        bb,
        common,
        fine_sigma_px=args.fine_sigma,
        coarse_sigma_px=args.coarse_sigma,
        normalize_low_percentile=args.dog_low_percentile,
        normalize_high_percentile=args.dog_high_percentile,
        reliability_low=0.025,
        reliability_high=0.10,
    )

    variant_gates = {}
    for name in args.polarity_variants:
        if name == "baseline":
            factor = np.ones(common.shape, dtype=np.float64)
        elif name == "sign_only":
            factor = polarity.sign_agreement.astype(np.float64)
        elif name == "soft_0_0.10":
            factor = (
                polarity.sign_agreement.astype(np.float64)
                * smoothstep(
                    polarity.strength_consensus, 0.0, 0.10
                )
            )
        elif name == "soft_0.025_0.10":
            factor = (
                polarity.sign_agreement.astype(np.float64)
                * smoothstep(
                    polarity.strength_consensus, 0.025, 0.10
                )
            )
        else:
            raise ValueError(f"Unknown polarity variant: {name}")

        variant_gates[name] = base_application_gate * factor

    wire = map_wire_proxy(
        args.wire_crop.resolve() if args.wire_crop else None,
        common,
    )
    near2 = near4 = near8 = None
    if wire is not None:
        near2 = ndi.binary_dilation(wire, iterations=2)
        near4 = ndi.binary_dilation(wire, iterations=4)
        near8 = ndi.binary_dilation(wire, iterations=8)

    resolved_mask = sharp_safe & (
        sharp_fine_mag >= suppress_threshold
    )
    original_candidate_mask = base_application_gate > 0

    output.mkdir(parents=True, exist_ok=False)

    np.save(output / "q_sharp.npy", q_sharp)
    np.save(
        output / "polarity_strength_consensus.npy",
        polarity.strength_consensus,
    )
    np.save(
        output / "polarity_sign_agreement.npy",
        polarity.sign_agreement,
    )
    np.save(
        output / "polarity_reliability_reference.npy",
        polarity.reliability,
    )
    save_gray01(
        polarity.strength_consensus,
        output / "polarity_strength_consensus.png",
    )
    save_gray01(
        polarity.sign_agreement.astype(np.float64),
        output / "polarity_sign_agreement.png",
    )
    save_gray01(
        polarity.reliability,
        output / "polarity_reliability_reference.png",
    )

    results = {}
    deltas = {}

    for name, gate in variant_gates.items():
        active = gate > 0
        desired_gx = amplitude * dx * gate
        desired_gy = amplitude * dy * gate

        solved = solve_screened_gradient_field(
            desired_gx,
            desired_gy,
            screening=args.screening,
            rtol=1e-8,
            maxiter=8000,
        )
        if not solved.converged:
            raise RuntimeError(
                f"{name}: integration failed info={solved.cg_info}"
            )

        delta = solved.delta_q
        q_final = q_sharp + delta
        lowpass = ndi.gaussian_filter(
            delta,
            sigma=args.lowpass_sigma,
            mode="nearest",
        )

        np.save(output / f"application_gate_{name}.npy", gate)
        np.save(output / f"delta_q_{name}.npy", delta)
        np.save(output / f"q_final_{name}.npy", q_final)
        save_gray01(gate, output / f"application_gate_{name}.png")

        report = {
            "application_gate": {
                "active_pixels": int(active.sum()),
                "active_fraction": float(active.mean()),
                "mass": float(np.sum(gate)),
                "mass_retention_vs_base": (
                    float(np.sum(gate) / np.sum(base_application_gate))
                    if np.sum(base_application_gate) > 0
                    else None
                ),
            },
            "delta_q_abs_1_per_m": abs_stats(delta),
            "lowpass_delta_abs_1_per_m": abs_stats(lowpass),
            "leakage_relative_to_final_gate": leakage_report(
                delta, active
            ),
            "preservation": {
                "sharp_resolved_pixels": int(resolved_mask.sum()),
                "delta_abs_on_sharp_resolved": abs_stats(
                    delta[resolved_mask]
                ),
                "correction_abs_mass_fraction_on_sharp_resolved": (
                    float(np.sum(np.abs(delta[resolved_mask]))
                          / np.sum(np.abs(delta)))
                    if np.sum(np.abs(delta)) > np.finfo(float).tiny
                    else 0.0
                ),
                "delta_abs_outside_original_candidate": abs_stats(
                    delta[~original_candidate_mask]
                ),
            },
            "q_final": {
                "min": float(np.min(q_final)),
                "max": float(np.max(q_final)),
                "nonpositive_pixels": int(
                    np.count_nonzero(q_final <= 0)
                ),
                "nonfinite_pixels": int(
                    np.count_nonzero(~np.isfinite(q_final))
                ),
            },
        }

        if wire is not None:
            report["evaluation_only_wire"] = {
                "wire_gate_mass_retention": mass_retention(
                    wire, gate, base_application_gate
                ),
                "near2_gate_mass_retention": mass_retention(
                    near2, gate, base_application_gate
                ),
                "near4_gate_mass_retention": mass_retention(
                    near4, gate, base_application_gate
                ),
                "near8_gate_mass_retention": mass_retention(
                    near8, gate, base_application_gate
                ),
                "delta_abs_wire": abs_stats(delta[wire]),
                "delta_abs_near2": abs_stats(delta[near2]),
                "delta_abs_near4": abs_stats(delta[near4]),
                "delta_abs_near8": abs_stats(delta[near8]),
            }

        results[name] = report
        deltas[name] = delta

    preview_scale = max(
        float(np.percentile(np.abs(delta), 99))
        for delta in deltas.values()
    )
    for name, delta in deltas.items():
        save_signed_common(
            delta,
            output / f"delta_q_{name}_signed_common.png",
            preview_scale,
        )

    summary = {
        "purpose": "K5-C.8 final q composition with polarity reliability",
        "overlap_box_native": [
            common.x0, common.y0, common.x1, common.y1
        ],
        "fixed_design": {
            "metric_amplitude_1_per_m_per_px": amplitude,
            "screening": args.screening,
            "novelty_multiplier": args.novelty_multiplier,
            "gate_low": args.gate_low,
            "gate_high": args.gate_high,
            "transition_fraction": args.transition_fraction,
        },
        "polarity": {
            "fine_sigma_px": args.fine_sigma,
            "coarse_sigma_px": args.coarse_sigma,
            "normalize_percentiles": [
                args.dog_low_percentile,
                args.dog_high_percentile,
            ],
            "strict_halo_px": polarity.strict_halo_px,
            "valid_fraction": float(polarity.valid.mean()),
            "sign_agreement_fraction_on_valid": float(
                np.mean(
                    polarity.sign_agreement[polarity.valid]
                )
            ),
            "normalization": {
                "context_a_low": polarity.low_a,
                "context_a_high": polarity.high_a,
                "context_b_low": polarity.low_b,
                "context_b_high": polarity.high_b,
            },
        },
        "variants": args.polarity_variants,
        "common_signed_preview_scale_1_per_m": preview_scale,
        "results": results,
        "guardrails": [
            "TinyViM DoG magnitude is non-metric.",
            "Polarity only gates confidence; it does not set metric amplitude.",
            "Wire proxy is evaluation-only.",
            "q_final is written only for the overlap diagnostic.",
            "No corrected depth is fed into K3 or Gaussian generation.",
        ],
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("K5-C.8 final-composition diagnostic complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
