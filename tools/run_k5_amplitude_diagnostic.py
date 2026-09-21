#!/usr/bin/env python3
"""K5-B.1 diagnostic metric-amplitude calibration on shared SHARP geometry.

This script DOES NOT modify SHARP depth and DOES NOT integrate a correction.

It estimates whether one bounded global scalar can map the K5-A.1
dimensionless detail-vector template into SHARP inverse-depth gradient units
on trusted non-wire shared geometry.

Training support:
- intersection of K4 anchor-candidate masks from both TinyViM contexts;
- outside the known wire proxy + exclusion dilation;
- sufficiently strong SHARP high-frequency gradient;
- sufficiently strong K5 final confidence;
- image-border excluded.

Validation:
- spatial block cross-fit;
- reports fold scale variation and vector residuals;
- wire pixels are never used for fitting and are reported only as extrapolated
  diagnostic amplitude.
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

from k5.amplitude import fit_positive_vector_scale, vector_residuals
from k5.overlap import crop_to_native_box
from k5.types import NativeBox


def load_report(folder):
    return json.loads((folder / "alignment.json").read_text())


def load_context_array(folder, name, box):
    value = np.load(folder / name)
    if value.shape != box.shape:
        raise ValueError(f"{folder/name}: {value.shape} != native box {box.shape}")
    return value


def load_candidate(folder, box):
    path = folder / "anchor_candidates.png"
    if not path.exists():
        raise FileNotFoundError(f"Required K4 anchor candidates not found: {path}")
    value = np.asarray(Image.open(path).convert("L")) > 0
    if value.shape != box.shape:
        raise ValueError(f"{path}: shape mismatch")
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


def sharp_detail_vector(q, fine_sigma, coarse_sigma):
    fine = ndi.gaussian_filter(q, sigma=fine_sigma, mode="nearest")
    coarse = ndi.gaussian_filter(q, sigma=coarse_sigma, mode="nearest")
    fy, fx = np.gradient(fine)
    cy, cx = np.gradient(coarse)
    return fx - cx, fy - cy


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
    valid = (sm > 0) & (tm > 0)
    if not valid.any():
        return {"n": 0}

    cos = (sx[valid] * tx[valid] + sy[valid] * ty[valid]) / (
        sm[valid] * tm[valid]
    )
    cos = np.clip(cos, -1, 1)
    return {
        **stats(cos),
        "positive_fraction": float(np.mean(cos > 0)),
        "gt_0.5_fraction": float(np.mean(cos > 0.5)),
        "gt_0.9_fraction": float(np.mean(cos > 0.9)),
    }


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
    if args.folds < 2:
        raise ValueError("folds must be >= 2")

    report_a = load_report(context_a)
    report_b = load_report(context_b)
    box_a = NativeBox.from_sequence(report_a["provenance"]["box_native_half_open"])
    box_b = NativeBox.from_sequence(report_b["provenance"]["box_native_half_open"])
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

    # Dimensionless K5 template: detail strength controls relative spatial
    # amplitude; direction comes from context consensus. Confidence is used only
    # as fitting weight / support gate, not multiplied into the template here.
    source_x = detail * direction_x
    source_y = detail * direction_y

    q_a_full = load_context_array(
        context_a, "sharp_visible_inverse_m.npy", box_a
    )
    q_b_full = load_context_array(
        context_b, "sharp_visible_inverse_m.npy", box_b
    )
    q_a = crop_to_native_box(q_a_full, box_a, common)
    q_b = crop_to_native_box(q_b_full, box_b, common)

    finite_q = np.isfinite(q_a) & np.isfinite(q_b) & (q_a > 0) & (q_b > 0)
    q_diff = np.abs(q_a - q_b)
    q = 0.5 * (q_a + q_b)

    sharp_x, sharp_y = sharp_detail_vector(
        q,
        args.fine_sigma,
        args.coarse_sigma,
    )
    sharp_mag = np.hypot(sharp_x, sharp_y)

    cand_a_full = load_candidate(context_a, box_a)
    cand_b_full = load_candidate(context_b, box_b)
    cand_a = crop_to_native_box(cand_a_full, box_a, common)
    cand_b = crop_to_native_box(cand_b_full, box_b, common)
    candidates = cand_a & cand_b

    wire = map_wire_proxy(wire_crop, common)
    wire_excluded = ndi.binary_dilation(
        wire, iterations=args.wire_exclusion_px
    )

    border = np.ones(common.shape, dtype=bool)
    b = args.border_px
    if b > 0:
        border[:b] = False
        border[-b:] = False
        border[:, :b] = False
        border[:, -b:] = False

    base = (
        candidates
        & finite_q
        & np.isfinite(sharp_mag)
        & np.isfinite(source_x)
        & np.isfinite(source_y)
        & np.isfinite(confidence)
        & (~wire_excluded)
        & border
        & (confidence >= args.min_k5_confidence)
    )

    if int(base.sum()) < 100:
        raise RuntimeError(f"Only {int(base.sum())} base shared-detail samples")

    sharp_threshold = float(
        np.percentile(
            sharp_mag[base],
            args.sharp_detail_percentile,
        )
    )
    selected = base & (sharp_mag >= sharp_threshold)

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

        if int(train.sum()) < 50 or int(val.sum()) < 20:
            folds.append({
                "fold": fold_id,
                "usable": False,
                "reason": "insufficient train/validation samples",
                "training_samples": int(train.sum()),
                "validation_samples": int(val.sum()),
            })
            continue

        fit = fit_positive_vector_scale(
            source_x[train],
            source_y[train],
            sharp_x[train],
            sharp_y[train],
            weights=confidence[train],
        )
        residual = vector_residuals(
            source_x[val],
            source_y[val],
            sharp_x[val],
            sharp_y[val],
            fit.scale,
        )
        normalized = residual / target_scale

        fold_scales.append(fit.scale)
        pooled_residuals.extend(normalized.tolist())

        pred_mag = (
            fit.scale
            * np.hypot(source_x[val], source_y[val])
        )
        target_mag = sharp_mag[val]
        ratio_mask = target_mag > np.finfo(float).tiny
        magnitude_ratio = pred_mag[ratio_mask] / target_mag[ratio_mask]

        folds.append({
            "fold": fold_id,
            "usable": True,
            "training_samples": int(train.sum()),
            "validation_samples": int(val.sum()),
            "scale": float(fit.scale),
            "fit_iterations": int(fit.iterations),
            "positive_projection_fraction_train": (
                fit.positive_projection_fraction
            ),
            "normalized_vector_residual": stats(normalized),
            "magnitude_ratio_predicted_over_sharp": stats(magnitude_ratio),
            "validation_direction_cosine": cosine_stats(
                source_x[val], source_y[val], sharp_x[val], sharp_y[val]
            ),
        })

    if len(fold_scales) >= 2:
        scale_variation = float(
            np.ptp(fold_scales)
            / max(np.median(fold_scales), np.finfo(float).tiny)
        )
    else:
        scale_variation = None

    full_fit = fit_positive_vector_scale(
        source_x[selected],
        source_y[selected],
        sharp_x[selected],
        sharp_y[selected],
        weights=confidence[selected],
    )

    wire_valid = (
        wire
        & np.isfinite(source_x)
        & np.isfinite(source_y)
        & np.isfinite(confidence)
    )
    wire_template_mag = np.hypot(
        source_x[wire_valid], source_y[wire_valid]
    )
    wire_predicted_metric_mag = full_fit.scale * wire_template_mag

    trusted_sharp_mag = sharp_mag[selected]

    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "training_support_mask.npy", selected)
    np.save(output / "sharp_detail_magnitude.npy", sharp_mag)

    summary = {
        "purpose": "K5-B.1 diagnostic global metric amplitude scale only",
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
        "sharp_overlap_consistency": {
            "finite_pixels": int(finite_q.sum()),
            "abs_difference_1_per_m": stats(q_diff[finite_q]),
        },
        "support": {
            "candidate_intersection_pixels": int(candidates.sum()),
            "base_shared_detail_pixels": int(base.sum()),
            "selected_training_support_pixels": int(selected.sum()),
            "spatial_blocks": int(len(blocks)),
            "wire_pixels": int(wire.sum()),
        },
        "crossfit": {
            "folds": folds,
            "fold_scales_1_per_m_per_template_unit": [
                float(v) for v in fold_scales
            ],
            "scale_variation": scale_variation,
            "pooled_normalized_vector_residual": stats(
                np.asarray(pooled_residuals)
            ),
            "normalization_sharp_detail_p90_1_per_m_per_px": target_scale,
        },
        "full_fit": {
            "scale_1_per_m_per_template_unit": float(full_fit.scale),
            "iterations": int(full_fit.iterations),
            "samples": int(full_fit.samples),
            "positive_projection_fraction": (
                full_fit.positive_projection_fraction
            ),
        },
        "trusted_sharp_detail_magnitude_1_per_m_per_px": stats(
            trusted_sharp_mag
        ),
        "wire_diagnostic_only": {
            "wire_template_magnitude": stats(wire_template_mag),
            "predicted_metric_gradient_magnitude_1_per_m_per_px": stats(
                wire_predicted_metric_mag
            ),
            "warning": (
                "Wire values are extrapolated diagnostic amplitudes only. "
                "They are not integrated and do not establish true wire depth."
            ),
        },
        "guardrails": [
            "Wire pixels are excluded from amplitude fitting.",
            "No depth correction is integrated.",
            "No SHARP depth/Gaussians/PLY are modified.",
            "A stable scalar on shared edges is necessary but not sufficient for K5-B.",
        ],
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("K5-B.1 amplitude diagnostic complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
