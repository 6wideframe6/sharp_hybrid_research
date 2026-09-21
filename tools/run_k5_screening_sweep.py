#!/usr/bin/env python3
"""K5-C.3 screening / integrability sensitivity sweep.

No corrected depth is sent into K3, SHARP prediction, or Gaussian generation.

Fixed design:
- K5-A.1b direction/confidence.
- K5-B.5b median metric amplitude.
- K5-C.2b SHARP novelty support with a fixed novelty multiplier (default 1.0).
- fixed confidence gate.

Only the screened-Poisson regularization is swept.

For each screening value the diagnostic reports:
- how well grad(delta_q) matches the desired K5 metric gradient;
- directional cosine and recovered gradient magnitude;
- low-frequency leakage;
- correction mass leaking outside the application support and its dilations;
- evaluation-only correction statistics on/near the known wire proxy;
- positivity / finiteness of q_sharp + delta_q.

The desired gradient field is identical for every screening value, so the sweep
isolates the integration regularizer itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from k5.sharp_fields import shared_full_context_gradients
from k5.types import NativeBox


def load_report(folder):
    return json.loads((folder / "alignment.json").read_text())


def load_context_array(folder, name, box):
    value = np.load(folder / name)
    if value.shape != box.shape:
        raise ValueError(f"{folder / name}: {value.shape} != {box.shape}")
    return value


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


def save_gray01(array, path):
    image = np.round(
        np.clip(array, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    Image.fromarray(image).save(path)


def save_signed_common_scale(array, path, scale):
    value = np.asarray(array, dtype=np.float64)
    if scale <= np.finfo(float).tiny:
        image = np.full(value.shape, 128, dtype=np.uint8)
    else:
        normalized = 0.5 + 0.5 * np.clip(value / scale, -1.0, 1.0)
        image = np.round(normalized * 255.0).astype(np.uint8)
    Image.fromarray(image).save(path)


def vector_cosine(ax, ay, bx, by, mask):
    am = np.hypot(ax, ay)
    bm = np.hypot(bx, by)
    valid = (
        mask
        & np.isfinite(ax)
        & np.isfinite(ay)
        & np.isfinite(bx)
        & np.isfinite(by)
        & (am > np.finfo(float).tiny)
        & (bm > np.finfo(float).tiny)
    )
    if not valid.any():
        return {"n": 0}
    cosine = (
        ax[valid] * bx[valid] + ay[valid] * by[valid]
    ) / (am[valid] * bm[valid])
    return stats(np.clip(cosine, -1.0, 1.0))


def leakage_report(delta, support):
    abs_delta = np.abs(delta)
    total_l1 = float(np.sum(abs_delta))
    total_l2 = float(np.sum(delta * delta))

    report = {}
    for radius in (0, 2, 4, 8, 16):
        if radius == 0:
            allowed = support
            key = "outside_active"
        else:
            allowed = ndi.binary_dilation(support, iterations=radius)
            key = f"outside_active_plus_{radius}px"

        outside = ~allowed
        outside_l1 = float(np.sum(abs_delta[outside]))
        outside_l2 = float(np.sum(delta[outside] ** 2))

        report[key] = {
            "abs_mass_fraction": (
                outside_l1 / total_l1
                if total_l1 > np.finfo(float).tiny
                else 0.0
            ),
            "energy_fraction": (
                outside_l2 / total_l2
                if total_l2 > np.finfo(float).tiny
                else 0.0
            ),
            "abs_delta": abs_stats(delta[outside]),
        }
    return report


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
    ap.add_argument(
        "--screenings",
        type=float,
        nargs="+",
        default=[0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20],
    )
    ap.add_argument("--cg-rtol", type=float, default=1e-7)
    ap.add_argument("--cg-maxiter", type=int, default=8000)
    ap.add_argument("--lowpass-sigma", type=float, default=16.0)
    args = ap.parse_args()

    if args.novelty_multiplier <= 0:
        raise ValueError("novelty-multiplier must be positive")
    if any((not np.isfinite(v)) or v < 0 for v in args.screenings):
        raise ValueError("screenings must be finite and non-negative")
    if len(set(args.screenings)) != len(args.screenings):
        raise ValueError("screenings must be unique")

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    ca = args.context_a.resolve()
    cb = args.context_b.resolve()
    ra = load_report(ca)
    rb = load_report(cb)
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

    qa_full = load_context_array(
        ca, "sharp_visible_inverse_m.npy", ba
    )
    qb_full = load_context_array(
        cb, "sharp_visible_inverse_m.npy", bb
    )
    qa = crop_to_native_box(qa_full, ba, common)
    qb = crop_to_native_box(qb_full, bb, common)
    q_sharp = 0.5 * (qa + qb)

    fields = shared_full_context_gradients(
        qa_full,
        ba,
        qb_full,
        bb,
        common,
        fine_sigma_px=args.fine_sigma,
    )
    fx = np.asarray(fields["fine_x"], dtype=np.float64)
    fy = np.asarray(fields["fine_y"], dtype=np.float64)
    sharp_safe = np.asarray(fields["safe_mask"], dtype=bool)
    sharp_fine_mag = np.hypot(fx, fy)

    amp_summary = json.loads(
        args.amplitude_summary.resolve().read_text()
    )
    amplitude = float(
        amp_summary["support"]["target_positive"]["median"]
    )
    base_sharp_threshold = float(
        amp_summary["config"]["sharp_fine_threshold"]
    )
    suppress_threshold = (
        base_sharp_threshold * args.novelty_multiplier
    )
    if amplitude <= 0 or suppress_threshold <= 0:
        raise ValueError("invalid amplitude or novelty threshold")

    confidence_gate = saturated_gate(
        confidence,
        low=args.gate_low,
        high=args.gate_high,
    )
    direction_norm = np.hypot(dx, dy)
    valid_direction = (
        np.isfinite(direction_norm)
        & (direction_norm > 0)
        & k5_valid
        & sharp_safe
    )
    confidence_gate = np.where(
        valid_direction, confidence_gate, 0.0
    )

    novelty_gate = sharp_novelty_gate(
        sharp_fine_mag,
        suppress_threshold=suppress_threshold,
        transition_fraction=args.transition_fraction,
    )
    novelty_gate = np.where(
        sharp_safe, novelty_gate, 0.0
    )

    application_gate = confidence_gate * novelty_gate
    active = application_gate > 0
    if not active.any():
        raise RuntimeError("application gate is empty")

    desired_gx = amplitude * dx * application_gate
    desired_gy = amplitude * dy * application_gate
    desired_mag = np.hypot(desired_gx, desired_gy)

    wire = map_wire_proxy(
        args.wire_crop.resolve()
        if args.wire_crop
        else None,
        common,
    )
    near2 = near4 = near8 = None
    if wire is not None:
        near2 = ndi.binary_dilation(wire, iterations=2)
        near4 = ndi.binary_dilation(wire, iterations=4)
        near8 = ndi.binary_dilation(wire, iterations=8)

    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "application_gate.npy", application_gate)
    np.save(output / "desired_gx.npy", desired_gx)
    np.save(output / "desired_gy.npy", desired_gy)
    save_gray01(
        application_gate,
        output / "application_gate.png",
    )

    results = {}
    deltas = {}

    desired_l2 = float(
        np.sum(
            desired_gx[active] ** 2
            + desired_gy[active] ** 2
        )
    )

    for screening in args.screenings:
        solved = solve_screened_gradient_field(
            desired_gx,
            desired_gy,
            screening=float(screening),
            rtol=args.cg_rtol,
            maxiter=args.cg_maxiter,
        )
        if not solved.converged:
            raise RuntimeError(
                f"screening={screening:g}: "
                f"CG failed info={solved.cg_info}"
            )

        delta = solved.delta_q
        dgy, dgx = np.gradient(delta)
        recovered_mag = np.hypot(dgx, dgy)
        residual_x = dgx - desired_gx
        residual_y = dgy - desired_gy
        residual_mag = np.hypot(residual_x, residual_y)

        residual_l2 = float(
            np.sum(
                residual_x[active] ** 2
                + residual_y[active] ** 2
            )
        )
        relative_l2_residual = (
            math.sqrt(residual_l2 / desired_l2)
            if desired_l2 > np.finfo(float).tiny
            else None
        )

        nonzero_desired = (
            active
            & (desired_mag > np.finfo(float).tiny)
        )
        magnitude_ratio = (
            recovered_mag[nonzero_desired]
            / desired_mag[nonzero_desired]
        )

        lowpass = ndi.gaussian_filter(
            delta,
            sigma=args.lowpass_sigma,
            mode="nearest",
        )
        corrected = q_sharp + delta

        delta_abs = abs_stats(delta)
        lowpass_abs = abs_stats(lowpass)
        p99_ratio = (
            lowpass_abs["p99"] / delta_abs["p99"]
            if delta_abs.get("p99", 0.0)
            > np.finfo(float).tiny
            else None
        )

        label = f"{screening:g}".replace(".", "p")
        report = {
            "screening": float(screening),
            "characteristic_length_px": (
                None
                if screening == 0
                else float(1.0 / math.sqrt(screening))
            ),
            "cg_info": solved.cg_info,
            "gradient_fit": {
                "residual_magnitude": stats(
                    residual_mag[active]
                ),
                "relative_l2_residual": relative_l2_residual,
                "direction_cosine": vector_cosine(
                    dgx,
                    dgy,
                    desired_gx,
                    desired_gy,
                    active,
                ),
                "recovered_over_desired_magnitude": stats(
                    magnitude_ratio
                ),
            },
            "delta_q_abs_1_per_m": delta_abs,
            "lowpass_delta_abs_1_per_m": lowpass_abs,
            "lowpass_p99_over_delta_p99": p99_ratio,
            "leakage": leakage_report(delta, active),
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
            report["evaluation_only_wire"] = {
                "delta_q_abs_wire": abs_stats(delta[wire]),
                "delta_q_abs_near_wire_2px": abs_stats(
                    delta[near2]
                ),
                "delta_q_abs_near_wire_4px": abs_stats(
                    delta[near4]
                ),
                "delta_q_abs_near_wire_8px": abs_stats(
                    delta[near8]
                ),
            }

        results[f"screening_{screening:g}"] = report
        deltas[label] = delta
        np.save(output / f"delta_q_screening_{label}.npy", delta)

    preview_scale = max(
        float(np.percentile(np.abs(delta), 99))
        for delta in deltas.values()
    )
    for label, delta in deltas.items():
        save_signed_common_scale(
            delta,
            output / f"delta_q_screening_{label}_signed_common.png",
            preview_scale,
        )

    summary = {
        "purpose": "K5-C.3 screening / integrability sensitivity sweep",
        "variant": variant.name,
        "overlap_box_native": [
            common.x0,
            common.y0,
            common.x1,
            common.y1,
        ],
        "fixed_design": {
            "metric_amplitude_1_per_m_per_px": amplitude,
            "base_sharp_fine_threshold_1_per_m_per_px": (
                base_sharp_threshold
            ),
            "novelty_multiplier": args.novelty_multiplier,
            "novelty_suppress_threshold_1_per_m_per_px": (
                suppress_threshold
            ),
            "gate_low": args.gate_low,
            "gate_high": args.gate_high,
            "transition_fraction": args.transition_fraction,
            "active_pixels": int(active.sum()),
            "application_gate_sum": float(
                np.sum(application_gate)
            ),
            "desired_gradient_magnitude": stats(
                desired_mag[active]
            ),
        },
        "screenings": [float(v) for v in args.screenings],
        "common_signed_preview_scale_1_per_m": preview_scale,
        "results": results,
        "guardrails": [
            "Only screening is swept; support/direction/amplitude are fixed.",
            "SHARP filtering is boundary-safe full-context filtering.",
            "Wire proxy is evaluation-only.",
            "No corrected depth is sent into K3 or Gaussian generation.",
        ],
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    rows = []
    for key, report in results.items():
        grad = report["gradient_fit"]
        leak = report["leakage"]
        wire_report = report.get("evaluation_only_wire", {})
        rows.append({
            "screening": report["screening"],
            "characteristic_length_px": report[
                "characteristic_length_px"
            ],
            "relative_l2_residual": grad[
                "relative_l2_residual"
            ],
            "direction_cosine_median": grad[
                "direction_cosine"
            ].get("median"),
            "recovered_over_desired_median": grad[
                "recovered_over_desired_magnitude"
            ].get("median"),
            "delta_abs_p99": report[
                "delta_q_abs_1_per_m"
            ].get("p99"),
            "lowpass_p99_over_delta_p99": report[
                "lowpass_p99_over_delta_p99"
            ],
            "outside_active_abs_mass_fraction": leak[
                "outside_active"
            ]["abs_mass_fraction"],
            "outside_active_plus_4px_abs_mass_fraction": leak[
                "outside_active_plus_4px"
            ]["abs_mass_fraction"],
            "outside_active_plus_8px_abs_mass_fraction": leak[
                "outside_active_plus_8px"
            ]["abs_mass_fraction"],
            "wire_delta_abs_median": (
                wire_report.get(
                    "delta_q_abs_wire", {}
                ).get("median")
            ),
            "wire_delta_abs_p90": (
                wire_report.get(
                    "delta_q_abs_wire", {}
                ).get("p90")
            ),
        })

    with (output / "screening_sweep.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("K5-C.3 screening sweep complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
