#!/usr/bin/env python3
"""
K4 model-family ablation ONLY.

Compares four diagnostic mappings on the exact same K4 anchors, spatial folds,
and leakage-safe disjoint-kernel smoothing:

    M0: q = a*r + b
    M1: q = a*r + b + c*y
    M2: q = a*r + b + c*x + d*y
    M3: q = a*r^2 + b*r + c

where:
    r = TinyViM relative inverse-depth output
    q = SHARP metric inverse depth [1/metre]
    x,y = native crop coordinates normalized to [0,1]

This script:
- DOES NOT modify alignment.py
- DOES NOT run SHARP or TinyViM inference
- DOES NOT perform fusion/replacement/K5
- DOES NOT change K4 acceptance gates
- uses the same selected anchors and spatial cross-fit partition as K4
- recomputes smoothing separately inside train/validation domains exactly as K4
- asserts that M0 reproduces the source K4 fold coefficients and pooled metrics

M1/M2/M3 are scientific diagnostics only. A lower residual does not by itself
authorize a new calibration model.

Outputs:
    summary.json
    model_comparison.csv
    fold_metrics.csv
    depth_bin_metrics.csv
    region_metrics.csv
    heldout_points.csv
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.sharp_hybrid_research.alignment import (
    AlignmentConfig,
    DegenerateFit,
    _spread,
    restricted_smoothing,
    robust_affine,
    stratify_anchors,
)

DEFAULT_INPUT = ROOT / "results/sharp_hybrid/k4_context/far_1480_512"
DEFAULT_OUTPUT = ROOT / "results/sharp_hybrid/k4_model_ablation/far_1480_512"


def load_json(path: Path):
    return json.loads(path.read_text())


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "n": 0,
            "median": None,
            "p90": None,
            "mean": None,
            "max": None,
            "gt_0.10": None,
            "gt_0.25": None,
        }
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "gt_0.10": float(np.mean(values > 0.10)),
        "gt_0.25": float(np.mean(values > 0.25)),
    }


def robust_linear(features, target, weights, config):
    """
    Generalization of K4 robust_affine to multiple predictors.

    Each predictor and the target are robustly normalized using median and
    p90-p10 spread on TRAINING DATA ONLY. Huber IRLS and residual trimming match
    the K4 logic. Returned coefficients operate in original feature units.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    if x.ndim != 2 or y.ndim != 1 or w.ndim != 1:
        raise ValueError("Expected X[N,D], y[N], weights[N]")
    if not (len(x) == len(y) == len(w)) or len(x) < max(4, x.shape[1] + 2):
        raise DegenerateFit("insufficient multivariate fit support")
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(w).all()):
        raise DegenerateFit("nonfinite multivariate fit input")
    if np.any(w < 0) or not np.any(w > 0):
        raise DegenerateFit("invalid multivariate weights")

    centers = np.median(x, axis=0)
    scales = np.array([_spread(x[:, j]) for j in range(x.shape[1])], dtype=np.float64)
    target_center = float(np.median(y))
    target_scale = float(_spread(y))

    eps = 64 * np.finfo(np.float64).eps
    feature_ref = np.maximum(np.max(np.abs(x), axis=0), np.finfo(np.float64).tiny)
    if np.any(scales <= eps * feature_ref):
        raise DegenerateFit("unresolvable multivariate predictor variation")
    if target_scale <= eps * max(float(np.max(np.abs(y))), np.finfo(np.float64).tiny):
        raise DegenerateFit("unresolvable target variation")

    xn = (x - centers) / scales
    yn = (y - target_center) / target_scale
    design = np.column_stack((xn, np.ones(len(xn), dtype=np.float64)))

    work = w / w.sum()
    coefficient = np.zeros(design.shape[1], dtype=np.float64)

    for iteration in range(config.iterations + 1):
        previous = coefficient.copy()

        sqrt_w = np.sqrt(work)
        matrix = design * sqrt_w[:, None]
        if np.linalg.matrix_rank(matrix) != design.shape[1]:
            raise DegenerateFit("rank-deficient weighted multivariate fit")

        coefficient = np.linalg.lstsq(
            matrix,
            yn * sqrt_w,
            rcond=None,
        )[0]

        residual = design @ coefficient - yn
        sigma = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        delta = max(1.345 * sigma, eps)
        robust_weight = np.minimum(
            1.0,
            delta / np.maximum(np.abs(residual), np.finfo(np.float64).tiny),
        )

        if config.trim_fraction:
            threshold = np.quantile(
                np.abs(residual),
                1.0 - config.trim_fraction,
            )
            robust_weight[np.abs(residual) > threshold] = 0.0

        work = w * robust_weight
        if work.sum() <= 0:
            raise DegenerateFit("no IRLS weight remains")
        work /= work.sum()

        if iteration and np.linalg.norm(coefficient - previous) < 1e-11:
            break

    feature_coefficients = target_scale * coefficient[:-1] / scales
    intercept = (
        target_center
        + target_scale * coefficient[-1]
        - float(feature_coefficients @ centers)
    )

    return feature_coefficients, float(intercept)


def predict_model(model, raw_values, xs, ys, fit):
    raw_values = np.asarray(raw_values, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    if model == "M0":
        return fit["raw"] * raw_values + fit["intercept"]

    if model == "M1":
        return (
            fit["raw"] * raw_values
            + fit["y"] * ys
            + fit["intercept"]
        )

    if model == "M2":
        return (
            fit["raw"] * raw_values
            + fit["x"] * xs
            + fit["y"] * ys
            + fit["intercept"]
        )

    if model == "M3":
        return (
            fit["raw2"] * raw_values * raw_values
            + fit["raw"] * raw_values
            + fit["intercept"]
        )

    raise KeyError(model)


def fit_model(model, raw_values, target, xs, ys, weights, config):
    raw_values = np.asarray(raw_values, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if model == "M0":
        a, b = robust_affine(raw_values, target, weights, config)
        return {
            "raw": float(a),
            "intercept": float(b),
        }

    if model == "M1":
        coef, intercept = robust_linear(
            np.column_stack((raw_values, ys)),
            target,
            weights,
            config,
        )
        return {
            "raw": float(coef[0]),
            "y": float(coef[1]),
            "intercept": intercept,
        }

    if model == "M2":
        coef, intercept = robust_linear(
            np.column_stack((raw_values, xs, ys)),
            target,
            weights,
            config,
        )
        return {
            "raw": float(coef[0]),
            "x": float(coef[1]),
            "y": float(coef[2]),
            "intercept": intercept,
        }

    if model == "M3":
        coef, intercept = robust_linear(
            np.column_stack((raw_values * raw_values, raw_values)),
            target,
            weights,
            config,
        )
        return {
            "raw2": float(coef[0]),
            "raw": float(coef[1]),
            "intercept": intercept,
        }

    raise KeyError(model)


def physical_diagnostic(model, fit, raw_range):
    """
    Report, but do not enforce, monotonicity in TinyViM relative inverse depth.
    """
    lo, hi = map(float, raw_range)

    if model in ("M0", "M1", "M2"):
        slope = float(fit["raw"])
        return {
            "dq_dr_constant": slope,
            "positive_over_selected_range": bool(slope > 0),
        }

    if model == "M3":
        d_lo = 2.0 * fit["raw2"] * lo + fit["raw"]
        d_hi = 2.0 * fit["raw2"] * hi + fit["raw"]

        if fit["raw2"] == 0:
            vertex = None
        else:
            vertex = -fit["raw"] / (2.0 * fit["raw2"])

        samples = [d_lo, d_hi]
        if vertex is not None and lo <= vertex <= hi:
            samples.append(0.0)

        return {
            "raw_selected_range": [lo, hi],
            "dq_dr_at_raw_min": float(d_lo),
            "dq_dr_at_raw_max": float(d_hi),
            "quadratic_vertex_raw": None if vertex is None else float(vertex),
            "min_dq_dr_over_selected_range": float(min(samples)),
            "max_dq_dr_over_selected_range": float(max(samples)),
            "positive_over_selected_range": bool(min(samples) > 0),
        }

    raise KeyError(model)


def model_features_description():
    return {
        "M0": "q = a*r + b",
        "M1": "q = a*r + b + c*y_norm",
        "M2": "q = a*r + b + c*x_norm + d*y_norm",
        "M3": "q = a*r^2 + b*r + c",
        "coordinate_definition": "x_norm=x/(W-1), y_norm=y/(H-1)",
    }


def grouped_metrics(rows, group_field):
    groups = {}
    for row in rows:
        key = (row["model"], row[group_field])
        groups.setdefault(key, []).append(row["normalized_error"])

    out = []
    for (model, group), values in groups.items():
        item = {
            "model": model,
            group_field: int(group),
        }
        item.update(summarize(values))
        out.append(item)

    out.sort(key=lambda r: (r["model"], r[group_field]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.output.resolve()

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output}"
        )

    report = load_json(source / "alignment.json")
    config = AlignmentConfig(**report["config"])

    if config.sigma <= 0:
        raise ValueError("Expected authoritative sigma>0 K4 context artifacts")

    raw = np.load(source / "raw_tinyvim_relative.npy")
    target = np.load(source / "sharp_visible_inverse_m.npy")
    raw_fit = np.load(source / "tinyvim_common_bandwidth.npy")
    target_fit = np.load(source / "sharp_common_bandwidth.npy")
    regions = np.load(source / "fitting_regions.npy")
    selected_saved = np.load(source / "anchor_mask.npy").astype(bool)
    candidate = (
        np.asarray(Image.open(source / "anchor_candidates.png").convert("L")) > 0
    )

    shapes = {
        arr.shape
        for arr in (
            raw,
            target,
            raw_fit,
            target_fit,
            regions,
            selected_saved,
            candidate,
        )
    }
    if len(shapes) != 1:
        raise ValueError(f"Artifact shape mismatch: {shapes}")

    h, w = raw.shape

    valid = (
        candidate
        & np.isfinite(raw)
        & np.isfinite(target)
        & (target > 0)
        & np.isfinite(raw_fit)
        & np.isfinite(target_fit)
        & (target_fit > 0)
        & (regions > 0)
    )

    sample = stratify_anchors(target_fit, valid, regions, config)
    indices = sample["indices"]
    weights = sample["weights"]
    blocks = sample["blocks"]
    depth_bins = sample["bins"]
    sample_regions = sample["regions"]

    recreated = np.zeros_like(candidate)
    recreated.flat[indices] = True
    if not np.array_equal(recreated, selected_saved):
        mismatch = int(np.count_nonzero(recreated ^ selected_saved))
        raise RuntimeError(
            f"Selected-anchor reproduction failed at {mismatch} pixels"
        )

    if int(candidate.sum()) != int(report["candidate_anchors"]):
        raise RuntimeError(
            f"Candidate count mismatch: artifacts={candidate.sum()}, "
            f"report={report['candidate_anchors']}"
        )

    x_full = raw_fit.flat[indices]
    y_full = target_fit.flat[indices]
    yspan = float(_spread(y_full))

    flat_y = indices // w
    flat_x = indices % w
    x_norm = flat_x.astype(np.float64) / max(w - 1, 1)
    y_norm = flat_y.astype(np.float64) / max(h - 1, 1)

    # Exact full-fit M0 reproduction before any ablation interpretation.
    full_a, full_b = robust_affine(x_full, y_full, weights, config)
    if not np.isclose(full_a, report["a"], rtol=1e-10, atol=1e-12):
        raise RuntimeError(
            f"M0 full slope mismatch: reproduced={full_a}, report={report['a']}"
        )
    if not np.isclose(full_b, report["b"], rtol=1e-10, atol=1e-12):
        raise RuntimeError(
            f"M0 full intercept mismatch: reproduced={full_b}, report={report['b']}"
        )

    yy, xx = np.mgrid[:h, :w]
    ncols = math.ceil(w / config.block_size)
    block_map = (
        (yy // config.block_size) * ncols
        + xx // config.block_size
    )

    unique_blocks = np.unique(blocks)
    np.random.default_rng(config.seed).shuffle(unique_blocks)

    models = ("M0", "M1", "M2", "M3")
    heldout_rows = []
    fold_metrics = []
    model_errors = {model: [] for model in models}
    model_fits = {model: [] for model in models}

    source_fold_by_number = {
        int(item["fold"]): item
        for item in report["folds"]
    }

    for fold_number, validation_blocks in enumerate(
        np.array_split(unique_blocks, config.folds)
    ):
        validation = np.isin(blocks, validation_blocks)
        training = ~validation
        validation_domain = np.isin(block_map, validation_blocks)

        xf = np.full_like(x_full, np.nan)
        yf = np.full_like(y_full, np.nan)

        for subset, domain in (
            (training.copy(), ~validation_domain),
            (validation.copy(), validation_domain),
        ):
            rf, rm = restricted_smoothing(
                raw,
                valid & domain,
                regions,
                config,
            )
            tf, tm = restricted_smoothing(
                target,
                valid & domain,
                regions,
                config,
            )

            supported = (
                (rm.flat[indices] >= config.min_kernel_mass)
                & (tm.flat[indices] >= config.min_kernel_mass)
            )

            use = subset & supported
            xf[use] = rf.flat[indices[use]]
            yf[use] = tf.flat[indices[use]]

        training_use = training & np.isfinite(xf) & np.isfinite(yf)
        validation_use = validation & np.isfinite(xf) & np.isfinite(yf)

        if training_use.sum() < config.min_anchors // 2:
            raise RuntimeError(
                f"Fold {fold_number}: insufficient disjoint-kernel training support"
            )
        if not validation_use.any():
            raise RuntimeError(
                f"Fold {fold_number}: no disjoint-kernel validation support"
            )

        for model in models:
            fit = fit_model(
                model,
                xf[training_use],
                yf[training_use],
                x_norm[training_use],
                y_norm[training_use],
                weights[training_use],
                config,
            )

            prediction = predict_model(
                model,
                xf[validation_use],
                x_norm[validation_use],
                y_norm[validation_use],
                fit,
            )

            error = np.abs(prediction - yf[validation_use]) / yspan
            model_errors[model].extend(error.tolist())

            train_raw = xf[training_use]
            physical = physical_diagnostic(
                model,
                fit,
                [float(np.min(train_raw)), float(np.max(train_raw))],
            )

            fold_item = {
                "model": model,
                "fold": fold_number,
                "training_count": int(training_use.sum()),
                "validation_count": int(validation_use.sum()),
                "fit": fit,
                "physical_diagnostic": physical,
                **summarize(error),
            }
            fold_metrics.append(fold_item)
            model_fits[model].append(fold_item)

            # M0 must reproduce the authoritative K4 fold.
            if model == "M0":
                expected = source_fold_by_number[fold_number]
                for key, reproduced_value in (
                    ("a", fit["raw"]),
                    ("b", fit["intercept"]),
                    ("median", fold_item["median"]),
                    ("p90", fold_item["p90"]),
                ):
                    if not np.isclose(
                        reproduced_value,
                        expected[key],
                        rtol=1e-9,
                        atol=1e-12,
                    ):
                        raise RuntimeError(
                            f"M0 fold {fold_number} {key} mismatch: "
                            f"reproduced={reproduced_value}, report={expected[key]}"
                        )

            selected_positions = np.flatnonzero(validation_use)
            for local_index, sample_index in enumerate(selected_positions):
                heldout_rows.append(
                    {
                        "model": model,
                        "fold": int(fold_number),
                        "x": int(flat_x[sample_index]),
                        "y": int(flat_y[sample_index]),
                        "block": int(blocks[sample_index]),
                        "region": int(sample_regions[sample_index]),
                        "depth_bin": int(depth_bins[sample_index]),
                        "raw_fold_smoothed": float(xf[sample_index]),
                        "sharp_fold_smoothed_inverse_m": float(yf[sample_index]),
                        "predicted_inverse_m": float(prediction[local_index]),
                        "normalized_error": float(error[local_index]),
                    }
                )

    pooled = {
        model: summarize(values)
        for model, values in model_errors.items()
    }

    # Exact pooled M0 reproduction.
    if not np.isclose(
        pooled["M0"]["median"],
        report["heldout_normalized_median"],
        rtol=1e-9,
        atol=1e-12,
    ):
        raise RuntimeError(
            "M0 pooled median does not reproduce authoritative K4"
        )
    if not np.isclose(
        pooled["M0"]["p90"],
        report["heldout_normalized_p90"],
        rtol=1e-9,
        atol=1e-12,
    ):
        raise RuntimeError(
            "M0 pooled p90 does not reproduce authoritative K4"
        )

    baseline = pooled["M0"]
    comparison = []
    for model in models:
        item = {
            "model": model,
            "equation": model_features_description()[model],
            **pooled[model],
            "median_change_vs_M0": float(
                pooled[model]["median"] - baseline["median"]
            ),
            "p90_change_vs_M0": float(
                pooled[model]["p90"] - baseline["p90"]
            ),
            "median_relative_change_vs_M0": float(
                pooled[model]["median"] / baseline["median"] - 1.0
            ),
            "p90_relative_change_vs_M0": float(
                pooled[model]["p90"] / baseline["p90"] - 1.0
            ),
        }
        comparison.append(item)

    depth_metrics = grouped_metrics(
        heldout_rows,
        "depth_bin",
    )
    region_metrics = grouped_metrics(
        heldout_rows,
        "region",
    )

    # Full-context fit for coefficient interpretation only.
    full_fit_diagnostic = {}
    for model in models:
        fit = fit_model(
            model,
            x_full,
            y_full,
            x_norm,
            y_norm,
            weights,
            config,
        )
        full_fit_diagnostic[model] = {
            "fit": fit,
            "physical_diagnostic": physical_diagnostic(
                model,
                fit,
                [float(np.min(x_full)), float(np.max(x_full))],
            ),
            "note": "in-sample coefficient diagnostic only; not acceptance evidence",
        }

    output.mkdir(parents=True, exist_ok=False)

    flat_fold_rows = []
    for item in fold_metrics:
        row = {
            "model": item["model"],
            "fold": item["fold"],
            "training_count": item["training_count"],
            "validation_count": item["validation_count"],
            "n": item["n"],
            "median": item["median"],
            "p90": item["p90"],
            "mean": item["mean"],
            "max": item["max"],
            "gt_0.10": item["gt_0.10"],
            "gt_0.25": item["gt_0.25"],
            "fit_json": json.dumps(item["fit"], sort_keys=True),
            "physical_json": json.dumps(
                item["physical_diagnostic"],
                sort_keys=True,
            ),
        }
        flat_fold_rows.append(row)

    write_csv(
        output / "model_comparison.csv",
        comparison,
        [
            "model",
            "equation",
            "n",
            "median",
            "p90",
            "mean",
            "max",
            "gt_0.10",
            "gt_0.25",
            "median_change_vs_M0",
            "p90_change_vs_M0",
            "median_relative_change_vs_M0",
            "p90_relative_change_vs_M0",
        ],
    )

    write_csv(
        output / "fold_metrics.csv",
        flat_fold_rows,
        [
            "model",
            "fold",
            "training_count",
            "validation_count",
            "n",
            "median",
            "p90",
            "mean",
            "max",
            "gt_0.10",
            "gt_0.25",
            "fit_json",
            "physical_json",
        ],
    )

    write_csv(
        output / "depth_bin_metrics.csv",
        depth_metrics,
        [
            "model",
            "depth_bin",
            "n",
            "median",
            "p90",
            "mean",
            "max",
            "gt_0.10",
            "gt_0.25",
        ],
    )

    write_csv(
        output / "region_metrics.csv",
        region_metrics,
        [
            "model",
            "region",
            "n",
            "median",
            "p90",
            "mean",
            "max",
            "gt_0.10",
            "gt_0.25",
        ],
    )

    write_csv(
        output / "heldout_points.csv",
        heldout_rows,
        [
            "model",
            "fold",
            "x",
            "y",
            "block",
            "region",
            "depth_bin",
            "raw_fold_smoothed",
            "sharp_fold_smoothed_inverse_m",
            "predicted_inverse_m",
            "normalized_error",
        ],
    )

    summary = {
        "source": str(source),
        "output": str(output),
        "purpose": "K4 diagnostic model-family ablation only; no K5/fusion/replacement",
        "models": model_features_description(),
        "authoritative_M0_reference": {
            "a": report["a"],
            "b": report["b"],
            "heldout_median": report["heldout_normalized_median"],
            "heldout_p90": report["heldout_normalized_p90"],
            "status": report["status"],
        },
        "M0_reproduction": {
            "full_a": float(full_a),
            "full_b": float(full_b),
            "pooled": pooled["M0"],
            "exact_metric_assertions_passed": True,
        },
        "pooled_crossfit": pooled,
        "comparison_vs_M0": comparison,
        "fold_metrics": fold_metrics,
        "depth_bin_metrics": depth_metrics,
        "region_metrics": region_metrics,
        "full_fit_coefficient_diagnostic": full_fit_diagnostic,
        "interpretation": {
            "M1": "Tests a vertical spatial/context bias beyond global affine.",
            "M2": "Tests a planar x/y spatial/context bias beyond global affine.",
            "M3": "Tests curvature/nonlinearity in TinyViM-to-SHARP response.",
            "guardrail": (
                "A better diagnostic model does not authorize calibration or K5. "
                "Use held-out behavior and physical monotonicity only to identify "
                "which K4 assumption is failing."
            ),
        },
    }
    save_json(output / "summary.json", summary)

    print("K4 model-family ablation complete")
    print(f"input:  {source}")
    print(f"output: {output}")
    print()
    print("M0 reproduction:")
    print(json.dumps(summary["M0_reproduction"], indent=2))
    print()
    print("Held-out model comparison:")
    for row in comparison:
        print(
            f"{row['model']}  "
            f"median={row['median']:.9f}  "
            f"p90={row['p90']:.9f}  "
            f"dMedian={row['median_change_vs_M0']:+.9f}  "
            f"dP90={row['p90_change_vs_M0']:+.9f}"
        )
    print()
    print("Full-fit physical diagnostics:")
    for model in models:
        print(
            model,
            json.dumps(
                full_fit_diagnostic[model]["physical_diagnostic"],
                sort_keys=True,
            ),
        )


if __name__ == "__main__":
    main()
