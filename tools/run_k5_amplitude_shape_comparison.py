#!/usr/bin/env python3
"""K5-B.3 compare amplitude SHAPES on shared SHARP geometry.

No correction is integrated.

B.2 showed that a scalar for SHARP energy-excess is spatially stable, but
TinyViM ``detail_consensus`` has ~zero pixelwise correlation with the metric
target. Therefore B.3 asks a narrower question:

    What should modulate metric amplitude spatially?

TinyViM still determines WHERE and DIRECTION. The tested amplitude shapes are:

1. constant_active
       source = 1 on validated support
   Tests a bounded-global-scalar interpretation.

2. confidence
       source = final_confidence
   Tests whether confidence can also serve as a soft amplitude taper.

3. detail_consensus
       source = detail_consensus
   B.2 baseline, retained for direct comparison.

4. sqrt_detail
       source = sqrt(detail_consensus)
   A deliberately compressed TinyViM-strength dependence.

Metric target is fixed to the most stable B.2 target:
    sqrt(max(|grad G_fine(q)|^2 - |grad G_coarse(q)|^2, 0))

All models use identical spatial folds and support. Wire pixels are excluded
from fitting. Results on wires are diagnostic extrapolations only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from k5.amplitude import fit_positive_scalar_scale, scalar_residuals
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
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0}
    return {
        "n": int(len(a)),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(np.max(a)),
    }


def rank_corr(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    result = spearmanr(x, y)
    return float(result.statistic) if np.isfinite(result.statistic) else None


def cosine(dx, dy, gx, gy):
    gm = np.hypot(gx, gy)
    out = np.full(dx.shape, np.nan, dtype=np.float64)
    valid = gm > np.finfo(float).tiny
    out[valid] = (dx[valid] * gx[valid] + dy[valid] * gy[valid]) / gm[valid]
    return np.clip(out, -1.0, 1.0)


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
    ap.add_argument("--min-direction-cosine", type=float, default=0.7)
    ap.add_argument("--sharp-fine-percentile", type=float, default=70.0)
    ap.add_argument("--wire-exclusion-px", type=int, default=8)
    ap.add_argument("--border-px", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    ca = args.context_a.resolve()
    cb = args.context_b.resolve()
    ra, rb = load_report(ca), load_report(cb)
    ba = NativeBox.from_sequence(ra["provenance"]["box_native_half_open"])
    bb = NativeBox.from_sequence(rb["provenance"]["box_native_half_open"])
    common = ba.intersect(bb)

    k5_root = args.k5_output.resolve()
    ks = json.loads((k5_root / "summary.json").read_text())
    if NativeBox.from_sequence(ks["overlap_box_native"]) != common:
        raise ValueError("K5/context overlap mismatch")

    variant = k5_root / f"fine_{args.fine_sigma:g}_coarse_{args.coarse_sigma:g}"
    dx = np.load(variant / "direction_x.npy")
    dy = np.load(variant / "direction_y.npy")
    detail = np.load(variant / "detail_consensus.npy")
    confidence = np.load(variant / "final_confidence.npy")

    qa = crop_to_native_box(
        load_context_array(ca, "sharp_visible_inverse_m.npy", ba), ba, common
    )
    qb = crop_to_native_box(
        load_context_array(cb, "sharp_visible_inverse_m.npy", bb), bb, common
    )
    q = 0.5 * (qa + qb)

    fine = ndi.gaussian_filter(q, sigma=args.fine_sigma, mode="nearest")
    coarse = ndi.gaussian_filter(q, sigma=args.coarse_sigma, mode="nearest")
    fy, fx = np.gradient(fine)
    cy, cx = np.gradient(coarse)

    fine_mag = np.hypot(fx, fy)
    coarse_mag = np.hypot(cx, cy)

    target = np.sqrt(
        np.maximum(fine_mag * fine_mag - coarse_mag * coarse_mag, 0.0)
    )

    direction_cosine = cosine(dx, dy, fx, fy)

    finite = (
        np.isfinite(q)
        & (q > 0)
        & np.isfinite(detail)
        & np.isfinite(confidence)
        & np.isfinite(direction_cosine)
        & np.isfinite(fine_mag)
        & np.isfinite(target)
    )

    wire = map_wire_proxy(args.wire_crop.resolve(), common)
    excluded = (
        ndi.binary_dilation(wire, iterations=args.wire_exclusion_px)
        if args.wire_exclusion_px > 0
        else wire.copy()
    )

    border = np.ones(common.shape, dtype=bool)
    b = args.border_px
    if b:
        border[:b] = False
        border[-b:] = False
        border[:, :b] = False
        border[:, -b:] = False

    base = (
        finite
        & (~excluded)
        & border
        & (confidence >= args.min_k5_confidence)
        & (detail > 0)
        & (direction_cosine >= args.min_direction_cosine)
    )
    if int(base.sum()) < 100:
        raise RuntimeError(f"Only {int(base.sum())} base pixels")

    fine_threshold = float(
        np.percentile(fine_mag[base], args.sharp_fine_percentile)
    )
    selected = base & (fine_mag >= fine_threshold)
    if int(selected.sum()) < 100:
        raise RuntimeError(f"Only {int(selected.sum())} selected pixels")

    shapes = {
        "constant_active": np.ones_like(detail, dtype=np.float64),
        "confidence": np.clip(confidence, 0.0, 1.0),
        "detail_consensus": np.clip(detail, 0.0, 1.0),
        "sqrt_detail": np.sqrt(np.clip(detail, 0.0, 1.0)),
    }

    yy, xx = np.mgrid[:common.height, :common.width]
    ncols = int(np.ceil(common.width / args.block_size))
    block_map = (yy // args.block_size) * ncols + xx // args.block_size
    blocks = np.unique(block_map[selected])
    if len(blocks) < args.folds:
        raise RuntimeError("Insufficient spatial blocks")

    rng = np.random.default_rng(0)
    blocks = blocks.copy()
    rng.shuffle(blocks)
    fold_blocks = np.array_split(blocks, args.folds)

    target_norm = max(
        float(np.percentile(target[selected], 90)),
        np.finfo(float).tiny,
    )

    reports = {}

    for name, source in shapes.items():
        folds = []
        scales = []
        pooled_residuals = []
        pooled_ratios = []

        for fold_id, validation_blocks in enumerate(fold_blocks):
            val = selected & np.isin(block_map, validation_blocks)
            train = selected & (~np.isin(block_map, validation_blocks))

            fit = fit_positive_scalar_scale(
                source[train],
                target[train],
                weights=confidence[train],
            )
            residual = scalar_residuals(
                source[val], target[val], fit.scale
            )
            normalized = residual / target_norm

            predicted = fit.scale * source[val]
            good = target[val] > np.finfo(float).tiny
            ratio = predicted[good] / target[val][good]

            scales.append(float(fit.scale))
            pooled_residuals.extend(normalized.tolist())
            pooled_ratios.extend(ratio.tolist())

            folds.append({
                "fold": fold_id,
                "training_samples": int(train.sum()),
                "validation_samples": int(val.sum()),
                "scale": float(fit.scale),
                "normalized_abs_residual": stats(normalized),
                "predicted_over_target_ratio": stats(ratio),
                "validation_spearman": rank_corr(source[val], target[val]),
            })

        scales_a = np.asarray(scales, np.float64)
        scale_spread = float(
            np.ptp(scales_a)
            / max(float(np.median(scales_a)), np.finfo(float).tiny)
        )

        full_fit = fit_positive_scalar_scale(
            source[selected],
            target[selected],
            weights=confidence[selected],
        )

        reports[name] = {
            "source_on_selected": stats(source[selected]),
            "selected_spearman": rank_corr(source[selected], target[selected]),
            "folds": folds,
            "fold_scales": scales,
            "scale_relative_spread": scale_spread,
            "pooled_normalized_abs_residual": stats(
                np.asarray(pooled_residuals)
            ),
            "pooled_predicted_over_target_ratio": stats(
                np.asarray(pooled_ratios)
            ),
            "full_scale": float(full_fit.scale),
            "wire_predicted_metric_magnitude": stats(
                full_fit.scale * source[wire]
            ),
        }

    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "selected_support_mask.npy", selected)
    np.save(output / "energy_excess_target.npy", target)

    summary = {
        "purpose": "K5-B.3 amplitude-shape comparison",
        "metric_target": "SHARP energy-excess magnitude",
        "variant": variant.name,
        "overlap_box_native": [common.x0, common.y0, common.x1, common.y1],
        "config": {
            "fine_sigma": args.fine_sigma,
            "coarse_sigma": args.coarse_sigma,
            "min_k5_confidence": args.min_k5_confidence,
            "min_direction_cosine": args.min_direction_cosine,
            "sharp_fine_percentile": args.sharp_fine_percentile,
            "sharp_fine_threshold": fine_threshold,
            "wire_exclusion_px": args.wire_exclusion_px,
            "block_size": args.block_size,
            "folds": args.folds,
        },
        "support": {
            "base_pixels": int(base.sum()),
            "selected_pixels": int(selected.sum()),
            "spatial_blocks": int(len(blocks)),
            "wire_pixels": int(wire.sum()),
            "direction_cosine_selected": stats(direction_cosine[selected]),
            "target_on_selected": stats(target[selected]),
        },
        "amplitude_shapes": reports,
        "guardrails": [
            "TinyViM/K5 strength is not assumed metric.",
            "Constant-active explicitly tests direction-only TinyViM usage.",
            "Wire pixels are excluded from fitting.",
            "Wire predictions are diagnostic extrapolations only.",
            "No correction is integrated or composed.",
        ],
    }

    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("K5-B.3 amplitude-shape comparison complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
