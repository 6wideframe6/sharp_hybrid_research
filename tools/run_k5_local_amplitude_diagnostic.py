#!/usr/bin/env python3
"""K5-B.5 cross-fitted local SHARP amplitude diagnostic.

No depth correction is integrated or composed.

B.4 showed substantial spatial non-stationarity of the SHARP energy-excess
amplitude, while K5 confidence/detail had almost no amplitude dependence.
B.5 therefore compares:

- fold-global robust median amplitude;
- local k-nearest-neighbour median amplitude for k = 8,16,32,64 by default.

Training samples:
- same trusted shared geometry used in B.4;
- positive SHARP energy-excess only;
- wire pixels + exclusion radius are absent from calibration.

Validation:
- whole spatial blocks are held out;
- local predictions may only use samples from the remaining blocks;
- report residuals and the physical distance to the k-th neighbour.

Finally, each local model is extrapolated to known wire pixels for diagnostic
amplitude and neighbour-distance reporting only. No wire value is used for fit.
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

from k5.local_amplitude import knn_median_predict
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
        "max": float(np.max(a)),
    }


def rank_corr(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    r = spearmanr(x, y)
    return float(r.statistic) if np.isfinite(r.statistic) else None


def cosine(dx, dy, gx, gy):
    gm = np.hypot(gx, gy)
    out = np.full(dx.shape, np.nan, dtype=np.float64)
    valid = gm > np.finfo(float).tiny
    out[valid] = (dx[valid] * gx[valid] + dy[valid] * gy[valid]) / gm[valid]
    return np.clip(out, -1.0, 1.0)


def ratio_stats(prediction, target):
    good = target > np.finfo(float).tiny
    return stats(prediction[good] / target[good])


def distance_coverage(distance):
    distance = np.asarray(distance, dtype=np.float64)
    result = {}
    for threshold in (8, 16, 32, 48, 64, 96, 128):
        result[f"le_{threshold}px_fraction"] = float(
            np.mean(distance <= threshold)
        )
    return result


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
    ap.add_argument("--k-values", type=int, nargs="+", default=[8, 16, 32, 64])
    args = ap.parse_args()

    if any(k < 1 for k in args.k_values):
        raise ValueError("All k values must be positive")

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
    positive = selected & (target > 0)

    if int(positive.sum()) < max(args.k_values) + 50:
        raise RuntimeError(
            f"Only {int(positive.sum())} positive calibration pixels"
        )

    yy, xx = np.mgrid[:common.height, :common.width]
    ncols = int(np.ceil(common.width / args.block_size))
    block_map = (yy // args.block_size) * ncols + xx // args.block_size

    blocks = np.unique(block_map[positive])
    if len(blocks) < args.folds:
        raise RuntimeError("Insufficient positive-support spatial blocks")

    rng = np.random.default_rng(0)
    blocks = blocks.copy()
    rng.shuffle(blocks)
    fold_blocks = np.array_split(blocks, args.folds)

    target_norm = max(
        float(np.percentile(target[positive], 90)),
        np.finfo(float).tiny,
    )

    global_fold_reports = []
    global_residuals = []
    global_ratios = []

    candidate_acc = {
        k: {
            "folds": [],
            "residuals": [],
            "ratios": [],
            "predictions": [],
            "targets": [],
            "nearest": [],
            "kth": [],
        }
        for k in args.k_values
    }

    for fold_id, validation_blocks in enumerate(fold_blocks):
        val = positive & np.isin(block_map, validation_blocks)
        train = positive & (~np.isin(block_map, validation_blocks))

        train_coords = np.column_stack(np.nonzero(train)).astype(np.float64)
        train_values = target[train]
        val_coords = np.column_stack(np.nonzero(val)).astype(np.float64)
        val_values = target[val]

        if len(train_values) < max(args.k_values) or len(val_values) < 10:
            raise RuntimeError(
                f"Fold {fold_id} too small: train={len(train_values)} "
                f"val={len(val_values)}"
            )

        global_value = float(np.median(train_values))
        global_pred = np.full(len(val_values), global_value, dtype=np.float64)
        global_abs = np.abs(global_pred - val_values)
        global_norm = global_abs / target_norm

        global_fold_reports.append({
            "fold": fold_id,
            "training_samples": int(len(train_values)),
            "validation_samples": int(len(val_values)),
            "train_global_median": global_value,
            "normalized_abs_residual": stats(global_norm),
            "predicted_over_target_ratio": ratio_stats(
                global_pred, val_values
            ),
        })
        global_residuals.extend(global_norm.tolist())
        global_ratios.extend((global_pred / val_values).tolist())

        for k in args.k_values:
            pred = knn_median_predict(
                train_coords,
                train_values,
                val_coords,
                k=k,
            )
            residual = np.abs(pred.values - val_values)
            normalized = residual / target_norm

            acc = candidate_acc[k]
            acc["residuals"].extend(normalized.tolist())
            acc["ratios"].extend((pred.values / val_values).tolist())
            acc["predictions"].extend(pred.values.tolist())
            acc["targets"].extend(val_values.tolist())
            acc["nearest"].extend(pred.nearest_distance_px.tolist())
            acc["kth"].extend(pred.kth_distance_px.tolist())

            acc["folds"].append({
                "fold": fold_id,
                "training_samples": int(len(train_values)),
                "validation_samples": int(len(val_values)),
                "normalized_abs_residual": stats(normalized),
                "predicted_over_target_ratio": ratio_stats(
                    pred.values, val_values
                ),
                "validation_spearman": rank_corr(
                    pred.values, val_values
                ),
                "nearest_training_distance_px": stats(
                    pred.nearest_distance_px
                ),
                "kth_training_distance_px": stats(
                    pred.kth_distance_px
                ),
                "kth_distance_coverage": distance_coverage(
                    pred.kth_distance_px
                ),
            })

    global_report = {
        "folds": global_fold_reports,
        "pooled_normalized_abs_residual": stats(
            np.asarray(global_residuals)
        ),
        "pooled_predicted_over_target_ratio": stats(
            np.asarray(global_ratios)
        ),
    }

    full_coords = np.column_stack(np.nonzero(positive)).astype(np.float64)
    full_values = target[positive]
    wire_coords = np.column_stack(np.nonzero(wire)).astype(np.float64)

    candidates = {}
    wire_predictions = {}

    for k, acc in candidate_acc.items():
        predictions = np.asarray(acc["predictions"], dtype=np.float64)
        targets = np.asarray(acc["targets"], dtype=np.float64)
        residuals = np.asarray(acc["residuals"], dtype=np.float64)
        ratios = np.asarray(acc["ratios"], dtype=np.float64)
        nearest = np.asarray(acc["nearest"], dtype=np.float64)
        kth = np.asarray(acc["kth"], dtype=np.float64)

        wire_pred = knn_median_predict(
            full_coords,
            full_values,
            wire_coords,
            k=k,
        )
        wire_predictions[int(k)] = np.asarray(
            wire_pred.values,
            dtype=np.float64,
        )

        global_median_resid = global_report[
            "pooled_normalized_abs_residual"
        ]["median"]
        global_p90_resid = global_report[
            "pooled_normalized_abs_residual"
        ]["p90"]

        local_stats = stats(residuals)

        candidates[f"k_{k}"] = {
            "k": int(k),
            "folds": acc["folds"],
            "pooled_normalized_abs_residual": local_stats,
            "pooled_predicted_over_target_ratio": stats(ratios),
            "pooled_prediction_target_spearman": rank_corr(
                predictions, targets
            ),
            "pooled_nearest_training_distance_px": stats(nearest),
            "pooled_kth_training_distance_px": stats(kth),
            "pooled_kth_distance_coverage": distance_coverage(kth),
            "improvement_vs_global": {
                "median_residual_fraction_reduction": float(
                    1.0 - local_stats["median"] / global_median_resid
                ) if global_median_resid > 0 else None,
                "p90_residual_fraction_reduction": float(
                    1.0 - local_stats["p90"] / global_p90_resid
                ) if global_p90_resid > 0 else None,
            },
            "wire_diagnostic": {
                "predicted_metric_amplitude": stats(wire_pred.values),
                "nearest_calibration_distance_px": stats(
                    wire_pred.nearest_distance_px
                ),
                "kth_calibration_distance_px": stats(
                    wire_pred.kth_distance_px
                ),
                "kth_distance_coverage": distance_coverage(
                    wire_pred.kth_distance_px
                ),
            },
        }

    summary = {
        "purpose": "K5-B.5 cross-fitted local SHARP amplitude diagnostic",
        "metric_target": "positive SHARP energy-excess magnitude",
        "variant": variant.name,
        "overlap_box_native": [
            common.x0, common.y0, common.x1, common.y1
        ],
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
            "k_values": [int(k) for k in args.k_values],
        },
        "support": {
            "base_pixels": int(base.sum()),
            "selected_pixels": int(selected.sum()),
            "positive_calibration_pixels": int(positive.sum()),
            "calibration_spatial_blocks": int(len(blocks)),
            "wire_pixels": int(wire.sum()),
            "target_positive": stats(target[positive]),
        },
        "global_median_baseline": global_report,
        "local_knn_candidates": candidates,
        "guardrails": [
            "Only SHARP-derived positive target amplitudes are calibration values.",
            "K5/TinyViM magnitude is not used as metric amplitude.",
            "Whole spatial blocks are held out during cross-validation.",
            "Wire pixels are excluded from all fitting.",
            "Wire amplitudes are diagnostic extrapolations only.",
            "No depth correction is integrated or composed.",
        ],
    }

    # Create output only after every diagnostic calculation has succeeded.
    output.mkdir(parents=True, exist_ok=False)

    np.save(output / "positive_calibration_mask.npy", positive)
    np.save(output / "energy_excess_target.npy", target)

    for k, values in wire_predictions.items():
        np.save(
            output / f"wire_predicted_amplitude_k{k}.npy",
            values,
        )

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("K5-B.5 local SHARP amplitude diagnostic complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
