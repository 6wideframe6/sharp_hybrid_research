#!/usr/bin/env python3
"""K5-B.1 signed amplitude diagnostic on shared SHARP geometry.

No depth is modified and no correction is integrated.

This revision intentionally allows a SIGNED scalar during diagnosis. Negative
fits are reported rather than crashing. A negative fit is NOT accepted as a
metric amplitude; it means the current K5 vector template and the chosen SHARP
detail target have opposite orientation on that support.

The report also compares K5 direction against:
- SHARP band-pass detail gradient: grad(G_fine(q) - G_coarse(q))
- SHARP fine gradient: grad(G_fine(q))
- SHARP raw inverse-depth gradient: grad(q)

This separates "K5 direction is globally reversed" from "the band-pass target
has a different local sign than the underlying SHARP gradient".
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

from k5.amplitude import fit_signed_vector_scale, signed_vector_residuals
from k5.overlap import crop_to_native_box
from k5.types import NativeBox


def load_report(folder):
    return json.loads((folder / "alignment.json").read_text())


def load_context_array(folder, name, box):
    value = np.load(folder / name)
    if value.shape != box.shape:
        raise ValueError(f"{folder / name}: {value.shape} != native box {box.shape}")
    return value


def map_wire_proxy(wire_crop, target_box):
    meta = json.loads((wire_crop / "metadata.json").read_text())
    wire_box = NativeBox.from_sequence(meta["box"])
    mask = np.asarray(Image.open(wire_crop / "rgb_proxy_mask.png").convert("L")) > 0
    if mask.shape != wire_box.shape:
        raise ValueError("Wire proxy shape/metadata mismatch")

    out = np.zeros(target_box.shape, dtype=bool)
    try:
        common = wire_box.intersect(target_box)
    except ValueError:
        return out

    sy, sx = wire_box.local_slices(common)
    ty, tx = target_box.local_slices(common)
    out[ty, tx] = mask[sy, sx]
    return out


def gradient_fields(q, fine_sigma, coarse_sigma):
    fine = ndi.gaussian_filter(q, sigma=fine_sigma, mode="nearest")
    coarse = ndi.gaussian_filter(q, sigma=coarse_sigma, mode="nearest")

    fine_y, fine_x = np.gradient(fine)
    coarse_y, coarse_x = np.gradient(coarse)
    raw_y, raw_x = np.gradient(q)

    detail_x = fine_x - coarse_x
    detail_y = fine_y - coarse_y

    return {
        "detail_x": detail_x,
        "detail_y": detail_y,
        "fine_x": fine_x,
        "fine_y": fine_y,
        "raw_x": raw_x,
        "raw_y": raw_y,
    }


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
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(np.max(a)),
    }


def cosine_stats(sx, sy, tx, ty):
    sm = np.hypot(sx, sy)
    tm = np.hypot(tx, ty)
    valid = (
        np.isfinite(sx)
        & np.isfinite(sy)
        & np.isfinite(tx)
        & np.isfinite(ty)
        & (sm > 0)
        & (tm > 0)
    )
    if not valid.any():
        return {"n": 0}

    cos = (sx[valid] * tx[valid] + sy[valid] * ty[valid]) / (
        sm[valid] * tm[valid]
    )
    cos = np.clip(cos, -1, 1)

    result = stats(cos)
    result.update({
        "negative_fraction": float(np.mean(cos < 0)),
        "positive_fraction": float(np.mean(cos > 0)),
        "lt_minus_0.5_fraction": float(np.mean(cos < -0.5)),
        "gt_0.5_fraction": float(np.mean(cos > 0.5)),
        "lt_minus_0.9_fraction": float(np.mean(cos < -0.9)),
        "gt_0.9_fraction": float(np.mean(cos > 0.9)),
    })
    return result


def count(mask):
    return int(np.count_nonzero(mask))


def sign_name(value):
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-a", type=Path, required=True)
    ap.add_argument("--context-b", type=Path, required=True)
    ap.add_argument("--k5-output", type=Path, required=True)
    ap.add_argument("--wire-crop", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)

    ap.add_argument("--fine-sigma", type=float, default=1.0)
    ap.add_argument("--coarse-sigma", type=float, default=2.0)
    ap.add_argument("--min-k5-confidence", type=float, default=0.025)
    ap.add_argument("--sharp-detail-percentile", type=float, default=70.0)
    ap.add_argument("--wire-exclusion-px", type=int, default=8)
    ap.add_argument("--border-px", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()

    context_a = args.context_a.resolve()
    context_b = args.context_b.resolve()
    k5_root = args.k5_output.resolve()
    wire_crop = args.wire_crop.resolve()
    output = args.output.resolve()

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if args.coarse_sigma <= args.fine_sigma:
        raise ValueError("coarse_sigma must be greater than fine_sigma")
    if not 0.0 <= args.sharp_detail_percentile < 100.0:
        raise ValueError("sharp_detail_percentile must be in [0, 100)")
    if args.min_k5_confidence < 0:
        raise ValueError("min_k5_confidence must be non-negative")
    if args.wire_exclusion_px < 0 or args.border_px < 0:
        raise ValueError("wire_exclusion_px and border_px must be non-negative")
    if args.block_size <= 0:
        raise ValueError("block_size must be positive")
    if args.folds < 2:
        raise ValueError("folds must be >= 2")

    report_a = load_report(context_a)
    report_b = load_report(context_b)
    box_a = NativeBox.from_sequence(
        report_a["provenance"]["box_native_half_open"]
    )
    box_b = NativeBox.from_sequence(
        report_b["provenance"]["box_native_half_open"]
    )
    common = box_a.intersect(box_b)

    k5_summary = json.loads((k5_root / "summary.json").read_text())
    k5_box = NativeBox.from_sequence(k5_summary["overlap_box_native"])
    if k5_box != common:
        raise ValueError(f"K5 overlap {k5_box} != context overlap {common}")

    variant = k5_root / f"fine_{args.fine_sigma:g}_coarse_{args.coarse_sigma:g}"
    if not variant.exists():
        raise FileNotFoundError(f"K5 variant missing: {variant}")

    direction_x = np.load(variant / "direction_x.npy")
    direction_y = np.load(variant / "direction_y.npy")
    detail = np.load(variant / "detail_consensus.npy")
    confidence = np.load(variant / "final_confidence.npy")

    for name, value in (
        ("direction_x", direction_x),
        ("direction_y", direction_y),
        ("detail_consensus", detail),
        ("final_confidence", confidence),
    ):
        if value.shape != common.shape:
            raise ValueError(f"{name}: {value.shape} != {common.shape}")

    source_x = detail * direction_x
    source_y = detail * direction_y
    source_mag = np.hypot(source_x, source_y)

    q_a_full = load_context_array(
        context_a, "sharp_visible_inverse_m.npy", box_a
    )
    q_b_full = load_context_array(
        context_b, "sharp_visible_inverse_m.npy", box_b
    )
    q_a = crop_to_native_box(q_a_full, box_a, common)
    q_b = crop_to_native_box(q_b_full, box_b, common)

    finite_q = (
        np.isfinite(q_a)
        & np.isfinite(q_b)
        & (q_a > 0)
        & (q_b > 0)
    )
    q_diff = np.abs(q_a - q_b)
    q = 0.5 * (q_a + q_b)

    g = gradient_fields(q, args.fine_sigma, args.coarse_sigma)
    sharp_x = g["detail_x"]
    sharp_y = g["detail_y"]
    sharp_mag = np.hypot(sharp_x, sharp_y)

    finite_k5 = (
        np.isfinite(source_x)
        & np.isfinite(source_y)
        & np.isfinite(source_mag)
        & np.isfinite(confidence)
    )
    finite_sharp_detail = (
        np.isfinite(sharp_x)
        & np.isfinite(sharp_y)
        & np.isfinite(sharp_mag)
    )

    wire = map_wire_proxy(wire_crop, common)
    wire_excluded = (
        ndi.binary_dilation(wire, iterations=args.wire_exclusion_px)
        if args.wire_exclusion_px > 0
        else wire.copy()
    )

    border = np.ones(common.shape, dtype=bool)
    b = args.border_px
    if b > 0:
        border[:b] = False
        border[-b:] = False
        border[:, :b] = False
        border[:, -b:] = False

    support0 = finite_q
    support1 = support0 & finite_sharp_detail
    support2 = support1 & finite_k5
    support3 = support2 & (~wire_excluded)
    support4 = support3 & border
    base = support4 & (confidence >= args.min_k5_confidence)

    support_stages = {
        "overlap_pixels": int(common.height * common.width),
        "finite_positive_sharp_both_contexts": count(support0),
        "plus_finite_sharp_detail": count(support1),
        "plus_finite_k5": count(support2),
        "plus_wire_exclusion": count(support3),
        "plus_border": count(support4),
        "plus_min_k5_confidence": count(base),
    }

    print("K5-B.1 signed diagnostic support stages:")
    for key, value in support_stages.items():
        print(f"  {key}: {value}")

    if count(base) < 100:
        raise RuntimeError(
            f"Insufficient base support: {count(base)}. See stages above."
        )

    sharp_threshold = float(
        np.percentile(sharp_mag[base], args.sharp_detail_percentile)
    )
    selected = base & (sharp_mag >= sharp_threshold)
    support_stages["selected_after_sharp_detail_percentile"] = count(selected)

    print(
        f"  selected_after_sharp_detail_percentile_"
        f"{args.sharp_detail_percentile:g}: {count(selected)}"
    )

    if count(selected) < 100:
        raise RuntimeError(f"Only {count(selected)} selected pixels")

    selected_orientation = {
        "k5_vs_sharp_bandpass_detail": cosine_stats(
            source_x[selected],
            source_y[selected],
            sharp_x[selected],
            sharp_y[selected],
        ),
        "k5_vs_sharp_fine_gradient": cosine_stats(
            source_x[selected],
            source_y[selected],
            g["fine_x"][selected],
            g["fine_y"][selected],
        ),
        "k5_vs_sharp_raw_gradient": cosine_stats(
            source_x[selected],
            source_y[selected],
            g["raw_x"][selected],
            g["raw_y"][selected],
        ),
    }

    print("selected orientation:")
    for name, value in selected_orientation.items():
        print(
            f"  {name}: median={value.get('median')} "
            f"positive_fraction={value.get('positive_fraction')} "
            f"negative_fraction={value.get('negative_fraction')}"
        )

    yy, xx = np.mgrid[:common.height, :common.width]
    ncols = int(np.ceil(common.width / args.block_size))
    block_map = (
        (yy // args.block_size) * ncols
        + (xx // args.block_size)
    )

    blocks = np.unique(block_map[selected])
    if len(blocks) < args.folds:
        raise RuntimeError(
            f"Only {len(blocks)} spatial blocks for {args.folds} folds"
        )

    rng = np.random.default_rng(0)
    blocks = blocks.copy()
    rng.shuffle(blocks)
    fold_blocks = np.array_split(blocks, args.folds)

    target_scale = max(
        float(np.percentile(sharp_mag[selected], 90)),
        np.finfo(float).tiny,
    )

    folds = []
    fold_scales = []
    pooled_residuals = []

    for fold_id, validation_blocks in enumerate(fold_blocks):
        val = selected & np.isin(block_map, validation_blocks)
        train = selected & (~np.isin(block_map, validation_blocks))

        if count(train) < 50 or count(val) < 20:
            folds.append({
                "fold": fold_id,
                "usable": False,
                "reason": "insufficient train/validation samples",
                "training_samples": count(train),
                "validation_samples": count(val),
            })
            continue

        fit = fit_signed_vector_scale(
            source_x[train],
            source_y[train],
            sharp_x[train],
            sharp_y[train],
            weights=confidence[train],
        )

        residual = signed_vector_residuals(
            source_x[val],
            source_y[val],
            sharp_x[val],
            sharp_y[val],
            fit.scale,
        )
        normalized = residual / target_scale

        fold_scales.append(float(fit.scale))
        pooled_residuals.extend(normalized.tolist())

        predicted_mag = abs(fit.scale) * np.hypot(
            source_x[val], source_y[val]
        )
        target_mag = sharp_mag[val]
        ratio_mask = target_mag > np.finfo(float).tiny
        magnitude_ratio = (
            predicted_mag[ratio_mask] / target_mag[ratio_mask]
        )

        folds.append({
            "fold": fold_id,
            "usable": True,
            "training_samples": count(train),
            "validation_samples": count(val),
            "signed_scale": float(fit.scale),
            "scale_sign": sign_name(fit.scale),
            "metric_amplitude_accepted": bool(fit.scale > 0),
            "fit_iterations": int(fit.iterations),
            "positive_projection_fraction_train": (
                fit.positive_projection_fraction
            ),
            "training_direction_cosine": cosine_stats(
                source_x[train],
                source_y[train],
                sharp_x[train],
                sharp_y[train],
            ),
            "validation_direction_cosine": cosine_stats(
                source_x[val],
                source_y[val],
                sharp_x[val],
                sharp_y[val],
            ),
            "normalized_vector_residual": stats(normalized),
            "absolute_scale_magnitude_ratio_predicted_over_sharp": stats(
                magnitude_ratio
            ),
        })

    if not fold_scales:
        raise RuntimeError("No usable cross-validation folds")

    fold_scales_array = np.asarray(fold_scales, dtype=np.float64)
    signs = np.sign(fold_scales_array)
    nonzero_signs = signs[signs != 0]
    all_same_nonzero_sign = bool(
        len(nonzero_signs)
        and np.all(nonzero_signs == nonzero_signs[0])
    )

    abs_median = float(np.median(np.abs(fold_scales_array)))
    signed_scale_relative_spread = (
        None
        if abs_median <= np.finfo(float).tiny
        else float(np.ptp(fold_scales_array) / abs_median)
    )

    full_fit = fit_signed_vector_scale(
        source_x[selected],
        source_y[selected],
        sharp_x[selected],
        sharp_y[selected],
        weights=confidence[selected],
    )

    positive_folds = int(np.sum(fold_scales_array > 0))
    negative_folds = int(np.sum(fold_scales_array < 0))

    metric_candidate_accepted = bool(
        full_fit.scale > 0
        and negative_folds == 0
        and positive_folds == len(fold_scales_array)
    )

    wire_valid = wire & finite_k5
    wire_template_mag = np.hypot(
        source_x[wire_valid], source_y[wire_valid]
    )

    wire_metric_magnitude = (
        stats(full_fit.scale * wire_template_mag)
        if metric_candidate_accepted
        else None
    )

    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "base_support_mask.npy", base)
    np.save(output / "training_support_mask.npy", selected)
    np.save(output / "sharp_detail_magnitude.npy", sharp_mag)

    summary = {
        "purpose": "K5-B.1 signed diagnostic scalar/orientation test",
        "metric_amplitude_candidate_accepted": metric_candidate_accepted,
        "acceptance_rule": (
            "full signed fit must be positive and every usable spatial fold "
            "must also have positive signed scale"
        ),
        "context_a": str(context_a),
        "context_b": str(context_b),
        "k5_output": str(k5_root),
        "variant": variant.name,
        "overlap_box_native": [
            common.x0, common.y0, common.x1, common.y1
        ],
        "config": {
            "fine_sigma": args.fine_sigma,
            "coarse_sigma": args.coarse_sigma,
            "min_k5_confidence": args.min_k5_confidence,
            "sharp_detail_percentile": args.sharp_detail_percentile,
            "sharp_detail_threshold_1_per_m_per_px": sharp_threshold,
            "wire_exclusion_px": args.wire_exclusion_px,
            "border_px": args.border_px,
            "block_size": args.block_size,
            "folds": args.folds,
        },
        "support_stages": {
            **support_stages,
            "spatial_blocks": int(len(blocks)),
            "wire_pixels": count(wire),
            "wire_excluded_pixels": count(wire_excluded),
        },
        "sharp_overlap_consistency": {
            "finite_pixels": count(finite_q),
            "abs_difference_1_per_m": stats(q_diff[finite_q]),
        },
        "selected_orientation": selected_orientation,
        "crossfit": {
            "folds": folds,
            "signed_fold_scales": [float(v) for v in fold_scales],
            "positive_folds": positive_folds,
            "negative_folds": negative_folds,
            "all_same_nonzero_sign": all_same_nonzero_sign,
            "signed_scale_relative_spread": signed_scale_relative_spread,
            "pooled_normalized_vector_residual": stats(
                np.asarray(pooled_residuals)
            ),
            "normalization_sharp_detail_p90_1_per_m_per_px": target_scale,
        },
        "full_signed_fit": {
            "signed_scale": float(full_fit.scale),
            "scale_sign": sign_name(full_fit.scale),
            "iterations": int(full_fit.iterations),
            "samples": int(full_fit.samples),
            "positive_projection_fraction": (
                full_fit.positive_projection_fraction
            ),
        },
        "wire_diagnostic_only": {
            "wire_template_magnitude": stats(wire_template_mag),
            "accepted_predicted_metric_gradient_magnitude_1_per_m_per_px": (
                wire_metric_magnitude
            ),
            "warning": (
                "Wire amplitude remains unset when the signed scale is not "
                "consistently positive across folds."
            ),
        },
        "guardrails": [
            "Negative signed fits are diagnostic failures, not amplitudes.",
            "No automatic direction flip is performed.",
            "Wire pixels are excluded from fitting.",
            "No depth correction is integrated.",
            "No SHARP depth/Gaussians/PLY are modified.",
        ],
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("K5-B.1 signed amplitude diagnostic complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
