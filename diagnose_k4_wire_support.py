#!/usr/bin/env python3
"""
K4 wire-centric anchor-support diagnosis ONLY.

No inference.
No modification of alignment.py.
No fusion/replacement/K5.

For the real far_1480_192 wire proxy embedded inside the shifted 512 context,
test increasingly large native-pixel neighborhoods around the wire.

For each radius:
- restrict existing K4 candidate anchors to distance <= radius from wire proxy
- use the original fitting-region labels and original K4 config
- run the existing align_affine() leakage-safe spatial cross-fit
- report anchor/block/depth support
- report TinyViM raw anchor range vs wire raw range
- report wire extrapolation

This tests whether a trustworthy local TinyViM -> SHARP affine calibration is
identifiable near the actual far-wire pixels.

Results are diagnostic only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.sharp_hybrid_research.alignment import (
    AlignmentConfig,
    _spread,
    align_affine,
)

DEFAULT_CONTEXT = ROOT / "results/sharp_hybrid/k4_context/far_1480_512"
DEFAULT_OLD_CROP = ROOT / "experiments/model_compare/outputs/depthart_tiny_512/far_1480_192"
DEFAULT_OUTPUT = ROOT / "results/sharp_hybrid/k4_wire_support/far_1480_512"

RADII = (16, 32, 48, 64, 96, 128, 160, 192, 256)


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")


def map_old_proxy(context_report, old_crop, shape):
    context_box = list(map(int, context_report["provenance"]["box_native_half_open"]))
    old_meta = json.loads((old_crop / "metadata.json").read_text())
    old_box = list(map(int, old_meta["box"]))

    cx0, cy0, cx1, cy1 = context_box
    ox0, oy0, ox1, oy1 = old_box

    if not (
        cx0 <= ox0 < ox1 <= cx1
        and cy0 <= oy0 < oy1 <= cy1
    ):
        raise RuntimeError(
            f"Old crop {old_box} is not contained in context {context_box}"
        )

    proxy = np.asarray(
        Image.open(old_crop / "rgb_proxy_mask.png").convert("L")
    ) > 0

    if proxy.shape != (oy1 - oy0, ox1 - ox0):
        raise RuntimeError("Old proxy/crop shape mismatch")

    mapped = np.zeros(shape, dtype=bool)
    lx0, ly0 = ox0 - cx0, oy0 - cy0
    lx1, ly1 = ox1 - cx0, oy1 - cy0
    mapped[ly0:ly1, lx0:lx1] = proxy

    return mapped, {
        "old_native_box": old_box,
        "context_native_box": context_box,
        "local_box_in_context": [lx0, ly0, lx1, ly1],
    }


def raw_support(raw, selected_mask, wire):
    selected = raw[selected_mask & np.isfinite(raw)].astype(np.float64)
    wire_values = raw[wire & np.isfinite(raw)].astype(np.float64)

    result = {
        "selected_anchor_pixels": int(len(selected)),
        "wire_pixels": int(len(wire_values)),
    }

    if not len(selected) or not len(wire_values):
        result["available"] = False
        return result

    span = max(_spread(selected), np.finfo(np.float64).tiny)
    lo = float(selected.min())
    hi = float(selected.max())

    distance = np.maximum.reduce(
        (
            lo - wire_values,
            wire_values - hi,
            np.zeros_like(wire_values),
        )
    )

    wire_p10, wire_p90 = np.percentile(wire_values, [10, 90])
    anchor_p10, anchor_p90 = np.percentile(selected, [10, 90])

    overlap = max(
        0.0,
        min(float(wire_p90), float(anchor_p90))
        - max(float(wire_p10), float(anchor_p10)),
    )
    wire_band = max(float(wire_p90 - wire_p10), np.finfo(float).tiny)

    result.update(
        available=True,
        anchor_raw_range=[lo, hi],
        anchor_raw_p10_p90=[float(anchor_p10), float(anchor_p90)],
        anchor_raw_p10_p90_spread=float(anchor_p90 - anchor_p10),
        wire_raw_range=[
            float(wire_values.min()),
            float(wire_values.max()),
        ],
        wire_raw_p10_p90=[float(wire_p10), float(wire_p90)],
        wire_p10_p90_overlap_fraction=float(overlap / wire_band),
        outside_anchor_range_fraction=float(np.mean(distance > 0)),
        max_distance_in_anchor_spreads=float(distance.max() / span),
        p90_distance_in_anchor_spreads=float(
            np.percentile(distance, 90) / span
        ),
    )
    return result


def compact_report(result):
    report = result["report"]
    return {
        "status": report["status"],
        "accepted": report["accepted"],
        "reasons": report["reasons"],
        "a": report["a"],
        "b": report["b"],
        "positive_slope": report["positive_slope"],
        "candidate_anchors": report["candidate_anchors"],
        "finite_anchor_candidates": report["finite_anchor_candidates"],
        "usable_anchors": report["usable_anchors"],
        "spatial_blocks": report["spatial_blocks"],
        "target_depth_bins": report["target_depth_bins"],
        "surface_interior_groups": report["surface_interior_groups"],
        "heldout_normalized_median": report["heldout_normalized_median"],
        "heldout_normalized_p90": report["heldout_normalized_p90"],
        "fold_scale_variation": report["fold_scale_variation"],
        "anchor_spread_fractions": report.get("anchor_spread_fractions"),
        "attempted_fold_slopes": report.get("attempted_fold_slopes"),
        "folds": [
            {
                "fold": f["fold"],
                "training_count": f["training_count"],
                "validation_count": f["validation_count"],
                "a": f.get("a"),
                "b": f.get("b"),
                "median": f.get("median"),
                "p90": f.get("p90"),
                "usable": f.get("usable"),
                "positive_slope": f.get("positive_slope"),
                "reason": f.get("reason"),
            }
            for f in report["folds"]
        ],
    }


def make_band_overlay(rgb, distance, candidate, radii, path):
    base = np.asarray(rgb, dtype=np.uint8).copy()

    # Draw candidate anchors in white first.
    out = base.copy()
    out[candidate] = (
        0.55 * out[candidate]
        + 0.45 * np.array([255, 255, 255])
    ).astype(np.uint8)

    image = Image.fromarray(out).convert("RGB")

    # Contours for radii.
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)

    for radius in radii:
        inside = distance <= radius
        boundary = inside ^ ndi.binary_erosion(inside)
        yy, xx = np.where(boundary)
        # Alternate deterministic colors.
        color = (
            int(40 + (radius * 3) % 200),
            int(80 + (radius * 5) % 170),
            int(120 + (radius * 7) % 130),
        )
        for y, x in zip(yy, xx):
            if 0 <= x < image.width and 0 <= y < image.height:
                image.putpixel((int(x), int(y)), color)

    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--old-crop", type=Path, default=DEFAULT_OLD_CROP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--radii",
        type=int,
        nargs="*",
        default=list(RADII),
    )
    args = parser.parse_args()

    context = args.context.resolve()
    old_crop = args.old_crop.resolve()
    output = args.output.resolve()

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    report = json.loads((context / "alignment.json").read_text())
    config = AlignmentConfig(**report["config"])

    raw = np.load(context / "raw_tinyvim_relative.npy")
    target = np.load(context / "sharp_visible_inverse_m.npy")
    raw_fit = np.load(context / "tinyvim_common_bandwidth.npy")
    target_fit = np.load(context / "sharp_common_bandwidth.npy")
    regions = np.load(context / "fitting_regions.npy")
    rgb = np.asarray(Image.open(context / "rgb_crop.png").convert("RGB"))

    candidate = (
        np.asarray(
            Image.open(context / "anchor_candidates.png").convert("L")
        )
        > 0
    )

    if not (
        raw.shape
        == target.shape
        == raw_fit.shape
        == target_fit.shape
        == regions.shape
        == candidate.shape
        == rgb.shape[:2]
    ):
        raise ValueError("Context artifacts do not share coordinates")

    wire, mapping = map_old_proxy(
        report,
        old_crop,
        raw.shape,
    )

    if not wire.any():
        raise RuntimeError("Mapped wire proxy is empty")

    # Distance is zero on wire pixels, Euclidean native pixels elsewhere.
    distance = ndi.distance_transform_edt(~wire)

    output.mkdir(parents=True, exist_ok=False)

    radii = sorted(set(int(r) for r in args.radii if int(r) > 0))
    results = []

    for radius in radii:
        local_candidate = candidate & (distance <= radius)

        aligned = align_affine(
            raw,
            target,
            local_candidate,
            raw_fit=raw_fit,
            target_fit=target_fit,
            regions=regions,
            foreground_mask=wire,
            config=config,
        )

        selected = aligned["selected_anchor_mask"]

        item = {
            "radius_native_px": radius,
            "candidate_pixels": int(local_candidate.sum()),
            "region_ids_with_candidates": [
                int(v)
                for v in np.unique(regions[local_candidate])
                if int(v) > 0
            ],
            "raw_support": raw_support(
                raw,
                selected,
                wire,
            ),
            "alignment": compact_report(aligned),
        }

        results.append(item)

    make_band_overlay(
        rgb,
        distance,
        candidate,
        radii,
        output / "wire_support_bands.png",
    )

    summary = {
        "purpose": (
            "wire-centric local anchor support and affine identifiability; "
            "no inference/fusion/replacement/K5"
        ),
        "context": str(context),
        "old_crop": str(old_crop),
        "mapping": mapping,
        "wire_pixels": int(wire.sum()),
        "radii_native_px": radii,
        "results": results,
        "interpretation_guardrails": [
            "A passing radius is diagnostic evidence only, not authorization for fusion.",
            "Candidate fitting regions are connected interiors, not semantic surface ownership.",
            "Wire raw-range extrapolation must be considered independently of residual gates.",
            "Do not use a fit whose wire values remain materially outside its anchor raw range.",
        ],
    }

    save_json(output / "summary.json", summary)

    print("K4 wire-centric anchor-support diagnostic complete")
    print(f"context: {context}")
    print(f"wire pixels: {int(wire.sum())}")
    print()

    for item in results:
        a = item["alignment"]
        s = item["raw_support"]

        print(f"===== radius {item['radius_native_px']} px =====")
        print(
            "regions:",
            item["region_ids_with_candidates"],
        )
        print(
            "support:",
            f"candidates={item['candidate_pixels']}",
            f"selected={a['usable_anchors']}",
            f"blocks={a['spatial_blocks']}",
            f"depth_bins={a['target_depth_bins']}",
        )
        print(
            "fit:",
            f"status={a['status']}",
            f"a={a['a']}",
            f"b={a['b']}",
            f"median={a['heldout_normalized_median']}",
            f"p90={a['heldout_normalized_p90']}",
            f"scale_var={a['fold_scale_variation']}",
        )
        print(
            "wire support:",
            f"outside={s.get('outside_anchor_range_fraction')}",
            f"p90_extrap_spreads={s.get('p90_distance_in_anchor_spreads')}",
            f"p10-p90_overlap={s.get('wire_p10_p90_overlap_fraction')}",
        )
        print(
            "reasons:",
            a["reasons"],
        )
        print()


if __name__ == "__main__":
    main()
