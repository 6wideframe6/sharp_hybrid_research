#!/usr/bin/env python3
"""K5-A.3 wire-edge localization evaluation.

Diagnostic only:
- no TinyViM inference
- no SHARP inference
- no confidence regeneration
- no depth modification
- no fusion/integration

The known wire proxy is used only for evaluation.

This evaluator measures K5-A.1 final confidence as a function of signed
distance to the wire-proxy boundary. This is more appropriate than classifying
the filled wire mask itself because K5-A confidence is a gradient/edge signal.

Definitions:
- signed distance < 0: inside wire proxy
- signed distance > 0: outside wire proxy
- |distance| ~= 0.5 px: pixels directly adjacent to the digital boundary

Reported bands:
- |d| <= 1 px
- 1 < |d| <= 2 px
- 2 < |d| <= 4 px
- 4 < |d| <= 8 px
- 8 < |d| <= 16 px

Also reports:
- inner boundary band
- outer boundary band
- outer sideband 3..8 px
- enrichment of boundary confidence over sideband confidence
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


DEFAULT_THRESHOLDS = (0.01, 0.025, 0.05, 0.10, 0.25, 0.50)


def parse_box(values):
    if len(values) != 4:
        raise ValueError("Expected [x0,y0,x1,y1]")
    return tuple(int(v) for v in values)


def map_wire_proxy(
    wire_crop: Path,
    overlap_box: tuple[int, int, int, int],
    overlap_shape: tuple[int, int],
) -> np.ndarray:
    metadata = json.loads((wire_crop / "metadata.json").read_text())
    wire_box = parse_box(metadata["box"])
    wire = np.asarray(
        Image.open(wire_crop / "rgb_proxy_mask.png").convert("L")
    ) > 0

    wx0, wy0, wx1, wy1 = wire_box
    ox0, oy0, ox1, oy1 = overlap_box

    if wire.shape != (wy1 - wy0, wx1 - wx0):
        raise ValueError("Wire proxy shape does not match metadata box")

    mapped = np.zeros(overlap_shape, dtype=bool)

    ix0 = max(wx0, ox0)
    iy0 = max(wy0, oy0)
    ix1 = min(wx1, ox1)
    iy1 = min(wy1, oy1)

    if ix0 >= ix1 or iy0 >= iy1:
        return mapped

    mapped[
        iy0 - oy0 : iy1 - oy0,
        ix0 - ox0 : ix1 - ox0,
    ] = wire[
        iy0 - wy0 : iy1 - wy0,
        ix0 - wx0 : ix1 - wx0,
    ]

    return mapped


def signed_boundary_distance(mask: np.ndarray) -> np.ndarray:
    """Approximate signed pixel-center distance to a binary boundary.

    Pixel centers immediately inside/outside a one-pixel digital boundary
    evaluate to approximately -0.5 / +0.5 px.

    Negative = inside mask.
    Positive = outside mask.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("Mask is empty")
    if mask.all():
        raise ValueError("Mask fills the whole image")

    inside = ndi.distance_transform_edt(mask)
    outside = ndi.distance_transform_edt(~mask)

    signed = np.empty(mask.shape, dtype=np.float64)
    signed[mask] = -(inside[mask] - 0.5)
    signed[~mask] = outside[~mask] - 0.5
    return signed


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
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(np.max(a)),
    }


def threshold_rates(values, thresholds):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {f"gt_{t:g}": None for t in thresholds}

    return {
        f"gt_{t:g}": float(np.mean(a > t))
        for t in thresholds
    }


def ratio(a, b):
    if a is None or b is None or b <= 0:
        return None
    return float(a / b)


def build_masks(signed_distance):
    d = signed_distance
    ad = np.abs(d)

    return {
        "abs_0_to_1px": ad <= 1.0,
        "abs_1_to_2px": (ad > 1.0) & (ad <= 2.0),
        "abs_2_to_4px": (ad > 2.0) & (ad <= 4.0),
        "abs_4_to_8px": (ad > 4.0) & (ad <= 8.0),
        "abs_8_to_16px": (ad > 8.0) & (ad <= 16.0),

        "inner_boundary_0_to_2px": (d < 0.0) & (d >= -2.0),
        "outer_boundary_0_to_2px": (d >= 0.0) & (d <= 2.0),
        "boundary_both_sides_0_to_2px": ad <= 2.0,

        "outer_sideband_3_to_8px": (d >= 3.0) & (d <= 8.0),
        "outer_far_8_to_16px": (d > 8.0) & (d <= 16.0),
        "outside_gt_16px": d > 16.0,
    }


def variant_evaluation(
    variant_dir: Path,
    masks: dict[str, np.ndarray],
    thresholds,
):
    confidence = np.load(variant_dir / "final_confidence.npy")
    detail = np.load(variant_dir / "detail_consensus.npy")
    direction = np.load(variant_dir / "direction_confidence.npy")

    result = {
        "variant": variant_dir.name,
        "groups": {},
        "boundary_vs_sideband": {},
    }

    for name, mask in masks.items():
        result["groups"][name] = {
            "pixels": int(mask.sum()),
            "final_confidence": stats(confidence[mask]),
            "detail_consensus": stats(detail[mask]),
            "direction_confidence": stats(direction[mask]),
            "threshold_rates": threshold_rates(confidence[mask], thresholds),
        }

    boundary = result["groups"]["boundary_both_sides_0_to_2px"]
    sideband = result["groups"]["outer_sideband_3_to_8px"]

    result["boundary_vs_sideband"] = {
        "median_ratio": ratio(
            boundary["final_confidence"].get("median"),
            sideband["final_confidence"].get("median"),
        ),
        "mean_ratio": ratio(
            boundary["final_confidence"].get("mean"),
            sideband["final_confidence"].get("mean"),
        ),
        "p90_ratio": ratio(
            boundary["final_confidence"].get("p90"),
            sideband["final_confidence"].get("p90"),
        ),
        "threshold_enrichment": {},
    }

    for t in thresholds:
        key = f"gt_{t:g}"
        br = boundary["threshold_rates"][key]
        sr = sideband["threshold_rates"][key]
        result["boundary_vs_sideband"]["threshold_enrichment"][key] = {
            "boundary_rate": br,
            "sideband_rate": sr,
            "ratio": ratio(br, sr),
        }

    return result


def write_profile_csv(path: Path, variants):
    rows = []
    ordered = [
        "abs_0_to_1px",
        "abs_1_to_2px",
        "abs_2_to_4px",
        "abs_4_to_8px",
        "abs_8_to_16px",
    ]

    for variant in variants:
        for band in ordered:
            group = variant["groups"][band]
            fc = group["final_confidence"]
            rows.append({
                "variant": variant["variant"],
                "band": band,
                "pixels": group["pixels"],
                "mean": fc.get("mean"),
                "median": fc.get("median"),
                "p90": fc.get("p90"),
                "p99": fc.get("p99"),
            })

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant", "band", "pixels",
                "mean", "median", "p90", "p99",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k5-output", type=Path, required=True)
    parser.add_argument("--wire-crop", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
    )
    args = parser.parse_args()

    k5_root = args.k5_output.resolve()
    wire_crop = args.wire_crop.resolve()
    output = args.output.resolve()

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    source = json.loads((k5_root / "summary.json").read_text())
    overlap_box = parse_box(source["overlap_box_native"])
    overlap_shape = tuple(int(v) for v in source["overlap_shape"])

    wire = map_wire_proxy(
        wire_crop,
        overlap_box,
        overlap_shape,
    )

    if not wire.any():
        raise RuntimeError("Known wire proxy has no pixels inside overlap")

    signed_distance = signed_boundary_distance(wire)
    masks = build_masks(signed_distance)

    variants = []
    for variant_dir in sorted(k5_root.glob("fine_*_coarse_*")):
        if not variant_dir.is_dir():
            continue

        confidence = np.load(variant_dir / "final_confidence.npy")
        if confidence.shape != overlap_shape:
            raise ValueError(
                f"{variant_dir.name}: shape {confidence.shape} "
                f"!= overlap {overlap_shape}"
            )

        variants.append(
            variant_evaluation(
                variant_dir,
                masks,
                args.thresholds,
            )
        )

    if not variants:
        raise RuntimeError("No fine_*_coarse_* variants found")

    output.mkdir(parents=True, exist_ok=False)

    np.save(output / "wire_signed_boundary_distance.npy", signed_distance)
    write_profile_csv(output / "distance_profile.csv", variants)

    summary = {
        "purpose": "K5-A.3 known-wire edge-localization evaluation only",
        "source_k5_output": str(k5_root),
        "wire_crop": str(wire_crop),
        "overlap_box_native": list(overlap_box),
        "overlap_shape": list(overlap_shape),
        "wire_pixels": int(wire.sum()),
        "thresholds": [float(t) for t in args.thresholds],
        "distance_definition": {
            "negative": "inside wire proxy",
            "positive": "outside wire proxy",
            "units": "native pixels",
            "digital_boundary_adjacent_centers": "approximately +/-0.5 px",
        },
        "variants": variants,
        "guardrails": [
            "Wire proxy is evaluation evidence only.",
            "Wire proxy is not used to generate confidence.",
            "No K5 confidence map is changed.",
            "No TinyViM or SHARP inference is run.",
            "No metric amplitude, integration, fusion or replacement is performed.",
        ],
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("K5-A.3 wire-edge localization evaluation complete")
    print(f"wire pixels: {int(wire.sum())}")
    print()

    profile_order = [
        "abs_0_to_1px",
        "abs_1_to_2px",
        "abs_2_to_4px",
        "abs_4_to_8px",
        "abs_8_to_16px",
    ]

    for variant in variants:
        print(f"===== {variant['variant']} =====")

        print("distance profile:")
        for band in profile_order:
            group = variant["groups"][band]
            fc = group["final_confidence"]
            print(
                f"  {band}: "
                f"n={group['pixels']} "
                f"mean={fc.get('mean')} "
                f"median={fc.get('median')} "
                f"p90={fc.get('p90')} "
                f"p99={fc.get('p99')}"
            )

        print("boundary vs outer sideband:")
        comparison = variant["boundary_vs_sideband"]
        print(
            "  mean_ratio=",
            comparison["mean_ratio"],
            "p90_ratio=",
            comparison["p90_ratio"],
        )

        for key, values in comparison["threshold_enrichment"].items():
            print(
                f"  {key}: "
                f"boundary={values['boundary_rate']} "
                f"sideband={values['sideband_rate']} "
                f"ratio={values['ratio']}"
            )

        print()


if __name__ == "__main__":
    main()
