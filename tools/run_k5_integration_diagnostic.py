#!/usr/bin/env python3
"""K5-C.1 bounded screened-integration diagnostic.

No corrected depth is sent into K3/SHARP/Gaussian generation.

Inputs:
- K5-A consensus direction + final confidence;
- SHARP inverse depth only as the baseline for preservation diagnostics;
- K5-B.5 summary only to obtain p25 / median / p75 amplitude priors.

Desired gradient:
    g = amplitude * direction * saturated_gate(confidence)

TinyViM/K5 detail magnitude is NOT used as metric amplitude.

The known wire proxy, when supplied, is evaluation-only. It does not influence
the gate, gradient field, integration domain or amplitude.
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


def signed_preview(array, path):
    a = np.asarray(array, dtype=np.float64)
    finite = np.isfinite(a)
    if not finite.any():
        image = np.zeros(a.shape, dtype=np.uint8)
    else:
        scale = float(np.percentile(np.abs(a[finite]), 99))
        if scale <= np.finfo(float).tiny:
            image = np.full(a.shape, 128, dtype=np.uint8)
        else:
            normalized = 0.5 + 0.5 * np.clip(a / scale, -1.0, 1.0)
            image = np.round(normalized * 255.0).astype(np.uint8)
    Image.fromarray(image, mode="L").save(path)


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
    ap.add_argument("--cg-rtol", type=float, default=1e-7)
    ap.add_argument("--cg-maxiter", type=int, default=5000)
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

    for name, a in (
        ("direction_x", dx),
        ("direction_y", dy),
        ("final_confidence", confidence),
    ):
        if a.shape != common.shape:
            raise ValueError(f"{name} shape mismatch")

    qa = crop_to_native_box(
        load_context_array(
            ca, "sharp_visible_inverse_m.npy", ba
        ),
        ba,
        common,
    )
    qb = crop_to_native_box(
        load_context_array(
            cb, "sharp_visible_inverse_m.npy", bb
        ),
        bb,
        common,
    )
    q_sharp = 0.5 * (qa + qb)

    amp_summary = json.loads(
        args.amplitude_summary.resolve().read_text()
    )
    target_stats = amp_summary["support"]["target_positive"]

    amplitudes = {
        "p25": float(target_stats["p25"]),
        "median": float(target_stats["median"]),
        "p75": float(target_stats["p75"]),
    }
    if not all(np.isfinite(v) and v > 0 for v in amplitudes.values()):
        raise ValueError("invalid amplitude quantiles in B.5 summary")

    gate = saturated_gate(
        confidence,
        low=args.gate_low,
        high=args.gate_high,
    )

    direction_norm = np.hypot(dx, dy)
    direction_valid = np.isfinite(direction_norm) & (direction_norm > 0)
    gate = np.where(direction_valid, gate, 0.0)

    wire = map_wire_proxy(
        args.wire_crop.resolve() if args.wire_crop else None,
        common,
    )

    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "gate.npy", gate)
    Image.fromarray(
        np.round(np.clip(gate, 0, 1) * 255).astype(np.uint8),
        mode="L",
    ).save(output / "gate.png")

    results = {}

    for label, amplitude in amplitudes.items():
        gx = amplitude * dx * gate
        gy = amplitude * dy * gate

        integrated = solve_screened_gradient_field(
            gx,
            gy,
            screening=args.screening,
            rtol=args.cg_rtol,
            maxiter=args.cg_maxiter,
        )
        if not integrated.converged:
            raise RuntimeError(
                f"{label}: CG did not converge; info={integrated.cg_info}"
            )

        delta = integrated.delta_q
        q_corrected = q_sharp + delta

        dgy, dgx = np.gradient(delta)
        gradient_residual = np.hypot(dgx - gx, dgy - gy)
        desired_mag = np.hypot(gx, gy)
        active = gate > 0
        saturated = gate >= 1.0 - 1e-12

        lowpass_delta = ndi.gaussian_filter(
            delta,
            sigma=args.lowpass_sigma,
            mode="nearest",
        )

        report = {
            "amplitude_1_per_m_per_px": amplitude,
            "cg_info": integrated.cg_info,
            "delta_q_signed_1_per_m": stats(delta),
            "delta_q_abs_1_per_m": abs_stats(delta),
            "desired_gradient_magnitude_1_per_m_per_px": stats(
                desired_mag[active]
            ),
            "gradient_fit_residual_1_per_m_per_px": stats(
                gradient_residual[active]
            ),
            "lowpass_delta_abs_1_per_m": abs_stats(lowpass_delta),
            "q_corrected": {
                "min": float(np.min(q_corrected)),
                "nonpositive_pixels": int(
                    np.count_nonzero(q_corrected <= 0)
                ),
                "nonfinite_pixels": int(
                    np.count_nonzero(~np.isfinite(q_corrected))
                ),
            },
            "active_pixels": int(np.count_nonzero(active)),
            "saturated_gate_pixels": int(np.count_nonzero(saturated)),
        }

        if wire is not None and wire.any():
            near_wire = ndi.binary_dilation(wire, iterations=8)
            report["evaluation_only_wire"] = {
                "wire_pixels": int(wire.sum()),
                "delta_q_signed_on_wire": stats(delta[wire]),
                "delta_q_abs_on_wire": abs_stats(delta[wire]),
                "delta_q_abs_near_wire_8px": abs_stats(
                    delta[near_wire]
                ),
                "gate_on_wire": stats(gate[wire]),
            }

        results[label] = report

        np.save(output / f"desired_gx_{label}.npy", gx)
        np.save(output / f"desired_gy_{label}.npy", gy)
        np.save(output / f"delta_q_{label}.npy", delta)
        np.save(output / f"q_corrected_{label}.npy", q_corrected)

        signed_preview(
            delta,
            output / f"delta_q_{label}_signed.png",
        )

    summary = {
        "purpose": "K5-C.1 bounded screened-integration diagnostic only",
        "variant": variant.name,
        "overlap_box_native": [
            common.x0, common.y0, common.x1, common.y1
        ],
        "amplitude_source": (
            "B.5 positive SHARP energy-excess p25/median/p75; "
            "K5 detail magnitude is not metric amplitude"
        ),
        "amplitudes_1_per_m_per_px": amplitudes,
        "config": {
            "gate_low": args.gate_low,
            "gate_high": args.gate_high,
            "screening": args.screening,
            "cg_rtol": args.cg_rtol,
            "cg_maxiter": args.cg_maxiter,
            "lowpass_sigma": args.lowpass_sigma,
        },
        "gate": {
            "nonzero_pixels": int(np.count_nonzero(gate > 0)),
            "saturated_pixels": int(
                np.count_nonzero(gate >= 1.0 - 1e-12)
            ),
            "mean": float(np.mean(gate)),
            "p90": float(np.percentile(gate, 90)),
            "p99": float(np.percentile(gate, 99)),
        },
        "results": results,
        "guardrails": [
            "Wire proxy is evaluation-only and does not affect correction.",
            "TinyViM/K5 magnitude is not used as metric amplitude.",
            "Outer boundary of delta_q is exactly zero.",
            "No corrected depth is sent into K3 or Gaussian generation.",
            "This is a sensitivity diagnostic, not an accepted final correction.",
        ],
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("K5-C.1 screened integration diagnostic complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
