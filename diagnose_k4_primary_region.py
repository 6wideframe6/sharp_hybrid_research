#!/usr/bin/env python3
"""
K4 primary-region affine diagnosis ONLY.

Tests whether the dominant connected fitting interior (default region 1) can
support a trustworthy local affine TinyViM -> SHARP inverse-depth calibration
without contamination from unrelated disconnected fitting interiors.

No inference.
No modification of alignment.py.
No fusion/replacement/K5.

The diagnostic:
1. Restricts anchors to one existing fitting region.
2. Re-stratifies those anchors with the original K4 config.
3. Runs the same leakage-safe spatial block cross-fit as K4:
      q = a*r + b
   with train/validation Gaussian smoothing recomputed separately.
4. Reports fold coefficients, residual gates, scale variation, anchor spread.
5. Maps the original far_1480_192 wire proxy into the 512 context and reports
   which fitting region is nearest to those wire pixels and whether the current
   512-context wire raw values extrapolate beyond the selected region-1 anchors.

fitting_regions are connected fitting interiors, NOT semantic surface labels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.sharp_hybrid_research.alignment import (
    AlignmentConfig,
    _spread,
    restricted_smoothing,
    robust_affine,
    stratify_anchors,
)

DEFAULT_CONTEXT = ROOT / "results/sharp_hybrid/k4_context/far_1480_512"
DEFAULT_OLD_CROP = ROOT / "experiments/model_compare/outputs/depthart_tiny_512/far_1480_192"
DEFAULT_OUTPUT = ROOT / "results/sharp_hybrid/k4_primary_region/far_1480_512"


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")


def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
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
        "n": int(len(a)),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "mean": float(np.mean(a)),
        "max": float(np.max(a)),
        "gt_0.10": float(np.mean(a > 0.10)),
        "gt_0.25": float(np.mean(a > 0.25)),
    }


def nearest_region_diagnostic(regions, proxy_mask):
    """
    For every proxy pixel, find the nearest nonzero fitting-region pixel.
    """
    if not proxy_mask.any():
        return {
            "proxy_pixels": 0,
            "nearest_region_counts": {},
            "nearest_region_fractions": {},
            "nearest_distance_px": None,
        }

    # EDT computes distance from True pixels to nearest False pixel.
    # regions==0 -> background/excluded=True, fitting regions=False.
    distance, nearest = ndi.distance_transform_edt(
        regions == 0,
        return_indices=True,
    )

    yy = nearest[0][proxy_mask]
    xx = nearest[1][proxy_mask]
    nearest_ids = regions[yy, xx]
    nearest_dist = distance[proxy_mask]

    ids, counts = np.unique(nearest_ids, return_counts=True)
    total = int(proxy_mask.sum())

    return {
        "proxy_pixels": total,
        "nearest_region_counts": {
            str(int(rid)): int(count)
            for rid, count in zip(ids, counts)
        },
        "nearest_region_fractions": {
            str(int(rid)): float(count / total)
            for rid, count in zip(ids, counts)
        },
        "nearest_distance_px": {
            "median": float(np.median(nearest_dist)),
            "p90": float(np.percentile(nearest_dist, 90)),
            "max": float(np.max(nearest_dist)),
        },
    }


def wire_extrapolation(raw, proxy_mask, nearest_region_mask, anchor_values):
    use = proxy_mask & nearest_region_mask & np.isfinite(raw)
    if not use.any():
        return {
            "pixels": 0,
            "reason": "no proxy pixels assigned to requested nearest region",
        }

    wire_values = raw[use].astype(np.float64)
    anchor_values = np.asarray(anchor_values, dtype=np.float64)
    span = max(_spread(anchor_values), np.finfo(np.float64).tiny)

    lo = float(anchor_values.min())
    hi = float(anchor_values.max())
    distance = np.maximum.reduce(
        (
            lo - wire_values,
            wire_values - hi,
            np.zeros_like(wire_values),
        )
    )

    return {
        "pixels": int(len(wire_values)),
        "wire_raw_range": [
            float(wire_values.min()),
            float(wire_values.max()),
        ],
        "wire_raw_p10_p90": [
            float(np.percentile(wire_values, 10)),
            float(np.percentile(wire_values, 90)),
        ],
        "selected_anchor_raw_range": [lo, hi],
        "selected_anchor_raw_p10_p90": [
            float(np.percentile(anchor_values, 10)),
            float(np.percentile(anchor_values, 90)),
        ],
        "outside_anchor_range_fraction": float(np.mean(distance > 0)),
        "max_distance_in_anchor_spreads": float(distance.max() / span),
        "p90_distance_in_anchor_spreads": float(
            np.percentile(distance, 90) / span
        ),
    }


def make_overlay(rgb, regions, region_id, proxy_mask, path):
    base = Image.fromarray(rgb).convert("RGBA")
    arr = np.zeros((*regions.shape, 4), dtype=np.uint8)

    primary = regions == region_id
    arr[primary] = (40, 220, 80, 65)

    arr[proxy_mask] = (255, 0, 255, 230)

    out = Image.alpha_composite(
        base,
        Image.fromarray(arr, mode="RGBA"),
    )

    draw = ImageDraw.Draw(out)
    draw.text(
        (8, 8),
        f"green = fitting region {region_id}; magenta = far_1480_192 wire proxy",
        fill=(255, 255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 255),
    )
    out.convert("RGB").save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--old-crop", type=Path, default=DEFAULT_OLD_CROP)
    parser.add_argument("--region", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    context = args.context.resolve()
    old_crop = args.old_crop.resolve()
    output = args.output.resolve()
    region_id = int(args.region)

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    source_report = json.loads((context / "alignment.json").read_text())
    config = AlignmentConfig(**source_report["config"])

    raw = np.load(context / "raw_tinyvim_relative.npy")
    target = np.load(context / "sharp_visible_inverse_m.npy")
    raw_fit_saved = np.load(context / "tinyvim_common_bandwidth.npy")
    target_fit_saved = np.load(context / "sharp_common_bandwidth.npy")
    regions = np.load(context / "fitting_regions.npy")
    candidate = (
        np.asarray(Image.open(context / "anchor_candidates.png").convert("L"))
        > 0
    )
    rgb = np.asarray(Image.open(context / "rgb_crop.png").convert("RGB"))

    if not (
        raw.shape
        == target.shape
        == raw_fit_saved.shape
        == target_fit_saved.shape
        == regions.shape
        == candidate.shape
        == rgb.shape[:2]
    ):
        raise ValueError("Context artifact shape mismatch")

    region_mask = regions == region_id
    if not region_mask.any():
        raise ValueError(f"Region {region_id} does not exist")

    valid = (
        candidate
        & region_mask
        & np.isfinite(raw)
        & np.isfinite(target)
        & (target > 0)
        & np.isfinite(raw_fit_saved)
        & np.isfinite(target_fit_saved)
        & (target_fit_saved > 0)
    )

    region_labels = np.where(region_mask, region_id, 0).astype(np.int32)

    sample = stratify_anchors(
        target_fit_saved,
        valid,
        region_labels,
        config,
    )
    indices = sample["indices"]
    weights = sample["weights"]
    blocks = sample["blocks"]
    bins = sample["bins"]

    if len(indices) < config.min_anchors:
        raise RuntimeError(
            f"Region {region_id}: only {len(indices)} selected anchors"
        )
    if len(np.unique(blocks)) < config.min_blocks:
        raise RuntimeError(
            f"Region {region_id}: only {len(np.unique(blocks))} spatial blocks"
        )

    x_full = raw_fit_saved.flat[indices]
    y_full = target_fit_saved.flat[indices]
    yspan = float(_spread(y_full))

    full_a, full_b = robust_affine(
        x_full,
        y_full,
        weights,
        config,
    )

    full_raw_finite = raw[region_mask & np.isfinite(raw)]
    full_target_finite = target[
        region_mask
        & np.isfinite(target)
        & (target > 0)
    ]

    anchor_spread = {
        "raw_p10_p90": float(_spread(x_full)),
        "target_p10_p90": float(_spread(y_full)),
        "raw_fraction_of_region": float(
            _spread(x_full)
            / max(_spread(full_raw_finite), np.finfo(float).tiny)
        ),
        "target_fraction_of_region": float(
            _spread(y_full)
            / max(_spread(full_target_finite), np.finfo(float).tiny)
        ),
    }

    h, w = raw.shape
    yy, xx = np.mgrid[:h, :w]
    ncols = math.ceil(w / config.block_size)
    block_map = (
        (yy // config.block_size) * ncols
        + xx // config.block_size
    )

    unique_blocks = np.unique(blocks)
    np.random.default_rng(config.seed).shuffle(unique_blocks)

    folds = []
    all_errors = []
    positive_scales = []
    attempted_scales = []

    for fold_number, validation_blocks in enumerate(
        np.array_split(unique_blocks, config.folds)
    ):
        validation = np.isin(blocks, validation_blocks)
        training = ~validation

        validation_domain = np.isin(
            block_map,
            validation_blocks,
        )

        xf = np.full_like(x_full, np.nan)
        yf = np.full_like(y_full, np.nan)

        for subset, domain in (
            (training.copy(), ~validation_domain),
            (validation.copy(), validation_domain),
        ):
            rf, rm = restricted_smoothing(
                raw,
                valid & domain,
                region_labels,
                config,
            )
            tf, tm = restricted_smoothing(
                target,
                valid & domain,
                region_labels,
                config,
            )

            supported = (
                (rm.flat[indices] >= config.min_kernel_mass)
                & (tm.flat[indices] >= config.min_kernel_mass)
            )

            use = subset & supported
            xf[use] = rf.flat[indices[use]]
            yf[use] = tf.flat[indices[use]]

        train_use = training & np.isfinite(xf) & np.isfinite(yf)
        val_use = validation & np.isfinite(xf) & np.isfinite(yf)

        fold = {
            "fold": fold_number,
            "training_blocks": np.unique(blocks[train_use]).tolist(),
            "validation_blocks": np.unique(blocks[val_use]).tolist(),
            "training_count": int(train_use.sum()),
            "validation_count": int(val_use.sum()),
        }

        if (
            train_use.sum() < config.min_anchors // 2
            or not val_use.any()
        ):
            fold["usable"] = False
            fold["reason"] = "insufficient disjoint-kernel support"
            folds.append(fold)
            continue

        fa, fb = robust_affine(
            xf[train_use],
            yf[train_use],
            weights[train_use],
            config,
        )

        error = (
            np.abs(fa * xf[val_use] + fb - yf[val_use])
            / yspan
        )

        attempted_scales.append(float(fa))
        if fa > 0:
            positive_scales.append(float(fa))

        fold.update(
            a=float(fa),
            b=float(fb),
            positive_slope=bool(fa > 0),
            usable=bool(fa > 0),
            **summarize(error),
        )

        all_errors.extend(error.tolist())
        folds.append(fold)

    pooled = summarize(all_errors)

    if len(positive_scales) >= 2:
        scale_variation = float(
            np.ptp(positive_scales)
            / np.median(positive_scales)
        )
    else:
        scale_variation = None

    gate_reasons = []

    if not (full_a > 0):
        gate_reasons.append("nonpositive full-fit slope")

    if any(
        not fold.get("usable", False)
        for fold in folds
    ):
        gate_reasons.append(
            "one or more spatial folds are unusable/nonpositive"
        )

    if pooled["median"] is None or pooled["median"] >= config.heldout_median_limit:
        gate_reasons.append("held-out median exceeds K4 development gate")

    if pooled["p90"] is None or pooled["p90"] >= config.heldout_p90_limit:
        gate_reasons.append("held-out p90 exceeds K4 development gate")

    if (
        scale_variation is None
        or scale_variation >= config.scale_variation_limit
    ):
        gate_reasons.append("fold scale variation exceeds K4 development gate")

    if (
        min(
            anchor_spread["raw_fraction_of_region"],
            anchor_spread["target_fraction_of_region"],
        )
        < config.min_anchor_spread_fraction
    ):
        gate_reasons.append(
            "anchor variation covers too little of region predictor/target range"
        )

    # Map the original far_1480_192 proxy into the 512 context.
    context_box = source_report["provenance"]["box_native_half_open"]
    old_meta = json.loads((old_crop / "metadata.json").read_text())
    old_box = old_meta["box"]

    cx0, cy0, cx1, cy1 = map(int, context_box)
    ox0, oy0, ox1, oy1 = map(int, old_box)

    if not (
        cx0 <= ox0 < ox1 <= cx1
        and cy0 <= oy0 < oy1 <= cy1
    ):
        raise RuntimeError(
            f"Old far_1480_192 crop {old_box} is not contained in context {context_box}"
        )

    old_proxy = (
        np.asarray(
            Image.open(old_crop / "rgb_proxy_mask.png").convert("L")
        )
        > 0
    )

    expected_hw = (oy1 - oy0, ox1 - ox0)
    if old_proxy.shape != expected_hw:
        raise RuntimeError(
            f"Old proxy shape {old_proxy.shape} != old crop {expected_hw}"
        )

    proxy_context = np.zeros(raw.shape, dtype=bool)
    lx0 = ox0 - cx0
    ly0 = oy0 - cy0
    lx1 = ox1 - cx0
    ly1 = oy1 - cy0
    proxy_context[ly0:ly1, lx0:lx1] = old_proxy

    nearest = nearest_region_diagnostic(
        regions,
        proxy_context,
    )

    # Nearest region ID at every pixel, for region-specific wire applicability.
    _, nearest_indices = ndi.distance_transform_edt(
        regions == 0,
        return_indices=True,
    )
    nearest_id_map = regions[
        nearest_indices[0],
        nearest_indices[1],
    ]

    nearest_primary = nearest_id_map == region_id

    wire_range = wire_extrapolation(
        raw,
        proxy_context,
        nearest_primary,
        x_full,
    )

    output.mkdir(parents=True, exist_ok=False)

    make_overlay(
        rgb,
        regions,
        region_id,
        proxy_context,
        output / "primary_region_and_wire_proxy.png",
    )

    result = {
        "purpose": (
            "diagnostic local affine identifiability for one existing fitting "
            "interior; no inference/fusion/replacement/K5"
        ),
        "context": str(context),
        "region": region_id,
        "region_pixels": int(region_mask.sum()),
        "candidate_anchor_pixels": int(valid.sum()),
        "selected_anchors": int(len(indices)),
        "spatial_blocks": int(len(np.unique(blocks))),
        "depth_bins": sorted(
            int(v)
            for v in np.unique(bins)
        ),
        "full_fit": {
            "a": float(full_a),
            "b": float(full_b),
            "positive_slope": bool(full_a > 0),
        },
        "anchor_spread": anchor_spread,
        "crossfit": {
            "folds": folds,
            "pooled": pooled,
            "attempted_fold_slopes": attempted_scales,
            "positive_fold_scale_variation": scale_variation,
        },
        "same_numeric_gates_as_K4_for_diagnosis": {
            "heldout_median_limit": config.heldout_median_limit,
            "heldout_p90_limit": config.heldout_p90_limit,
            "scale_variation_limit": config.scale_variation_limit,
            "min_anchor_spread_fraction": config.min_anchor_spread_fraction,
            "passes_all_numeric_diagnostic_gates": len(gate_reasons) == 0,
            "reasons": gate_reasons,
            "warning": (
                "Passing this diagnostic does not change K4's recorded global "
                "rejection and does not authorize K5."
            ),
        },
        "far_1480_192_wire_proxy_in_512_context": {
            "old_native_box": old_box,
            "context_native_box": context_box,
            "local_box_in_context": [lx0, ly0, lx1, ly1],
            "proxy_pixels": int(proxy_context.sum()),
            "nearest_fitting_region": nearest,
            f"wire_pixels_nearest_region_{region_id}_raw_extrapolation": wire_range,
            "warning": (
                "Nearest fitting region is a locality diagnostic only; it is "
                "not surface ownership or a replacement mask."
            ),
        },
    }

    save_json(output / "summary.json", result)

    print("K4 primary-region affine diagnostic complete")
    print(f"context: {context}")
    print(f"region:  {region_id}")
    print(f"output:  {output}")
    print()
    print("FULL FIT")
    print(json.dumps(result["full_fit"], indent=2))
    print()
    print("ANCHOR SUPPORT")
    print(
        json.dumps(
            {
                "selected_anchors": result["selected_anchors"],
                "spatial_blocks": result["spatial_blocks"],
                "depth_bins": result["depth_bins"],
                "anchor_spread": result["anchor_spread"],
            },
            indent=2,
        )
    )
    print()
    print("CROSSFIT")
    print(json.dumps(result["crossfit"], indent=2))
    print()
    print("DIAGNOSTIC GATES")
    print(
        json.dumps(
            result["same_numeric_gates_as_K4_for_diagnosis"],
            indent=2,
        )
    )
    print()
    print("WIRE PROXY LOCALITY")
    print(
        json.dumps(
            result["far_1480_192_wire_proxy_in_512_context"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
