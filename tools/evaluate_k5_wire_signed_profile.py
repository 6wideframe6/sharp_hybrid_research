#!/usr/bin/env python3
"""K5-A.4 signed wire-edge profile evaluation.

Diagnostic only:
- no TinyViM/SHARP inference
- no confidence regeneration
- no depth modification
- no fusion/integration

This refines K5-A.3 by separating confidence INSIDE and OUTSIDE the known
wire proxy, so we can determine whether the observed 2-4 px peak is:
- inside the proxy (proxy thicker than TinyViM edge support),
- outside the proxy (registration/context displacement),
- or symmetric (expected smoothing/edge width).

Signed distance convention:
- negative = inside wire proxy
- positive = outside wire proxy
- immediately adjacent pixel centers are approximately +/-0.5 px

Reported signed bins:
[-8,-4), [-4,-2), [-2,-1), [-1,0),
[0,1), [1,2), [2,4), [4,8), [8,16)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


BINS = (
    (-8.0, -4.0, "inside_4_to_8px"),
    (-4.0, -2.0, "inside_2_to_4px"),
    (-2.0, -1.0, "inside_1_to_2px"),
    (-1.0,  0.0, "inside_0_to_1px"),
    ( 0.0,  1.0, "outside_0_to_1px"),
    ( 1.0,  2.0, "outside_1_to_2px"),
    ( 2.0,  4.0, "outside_2_to_4px"),
    ( 4.0,  8.0, "outside_4_to_8px"),
    ( 8.0, 16.0, "outside_8_to_16px"),
)

THRESHOLDS = (0.01, 0.025, 0.05, 0.10, 0.25, 0.50)


def parse_box(values):
    if len(values) != 4:
        raise ValueError("Expected [x0,y0,x1,y1]")
    return tuple(int(v) for v in values)


def map_wire_proxy(wire_crop: Path, overlap_box, overlap_shape):
    meta = json.loads((wire_crop / "metadata.json").read_text())
    wire_box = parse_box(meta["box"])
    wire = np.asarray(Image.open(wire_crop / "rgb_proxy_mask.png").convert("L")) > 0

    wx0, wy0, wx1, wy1 = wire_box
    ox0, oy0, ox1, oy1 = overlap_box

    if wire.shape != (wy1 - wy0, wx1 - wx0):
        raise ValueError("Wire proxy shape does not match metadata box")

    out = np.zeros(overlap_shape, dtype=bool)
    ix0, iy0 = max(wx0, ox0), max(wy0, oy0)
    ix1, iy1 = min(wx1, ox1), min(wy1, oy1)
    if ix0 >= ix1 or iy0 >= iy1:
        return out

    out[iy0-oy0:iy1-oy0, ix0-ox0:ix1-ox0] = wire[
        iy0-wy0:iy1-wy0, ix0-wx0:ix1-wx0
    ]
    return out


def signed_boundary_distance(mask):
    mask = np.asarray(mask, dtype=bool)
    inside = ndi.distance_transform_edt(mask)
    outside = ndi.distance_transform_edt(~mask)

    signed = np.empty(mask.shape, dtype=np.float64)
    signed[mask] = -(inside[mask] - 0.5)
    signed[~mask] = outside[~mask] - 0.5
    return signed


def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {"n": 0}

    return {
        "n": int(len(a)),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        **{
            f"gt_{t:g}": float(np.mean(a > t))
            for t in THRESHOLDS
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k5-output", type=Path, required=True)
    ap.add_argument("--wire-crop", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    k5_root = args.k5_output.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    source = json.loads((k5_root / "summary.json").read_text())
    overlap_box = parse_box(source["overlap_box_native"])
    overlap_shape = tuple(int(v) for v in source["overlap_shape"])

    wire = map_wire_proxy(
        args.wire_crop.resolve(),
        overlap_box,
        overlap_shape,
    )
    if not wire.any():
        raise RuntimeError("Wire proxy has no pixels in overlap")

    signed = signed_boundary_distance(wire)

    variants = []
    csv_rows = []

    for variant_dir in sorted(k5_root.glob("fine_*_coarse_*")):
        if not variant_dir.is_dir():
            continue

        confidence = np.load(variant_dir / "final_confidence.npy")
        detail = np.load(variant_dir / "detail_consensus.npy")
        direction = np.load(variant_dir / "direction_confidence.npy")

        if confidence.shape != overlap_shape:
            raise ValueError(f"{variant_dir.name}: shape mismatch")

        bins_out = []
        for lo, hi, name in BINS:
            mask = (signed >= lo) & (signed < hi)
            row = {
                "name": name,
                "distance_range_px": [lo, hi],
                "pixels": int(mask.sum()),
                "final_confidence": summarize(confidence[mask]),
                "detail_consensus": summarize(detail[mask]),
                "direction_confidence": summarize(direction[mask]),
            }
            bins_out.append(row)

            flat = {
                "variant": variant_dir.name,
                "band": name,
                "lo_px": lo,
                "hi_px": hi,
                "pixels": int(mask.sum()),
            }
            for k, v in row["final_confidence"].items():
                flat[f"final_{k}"] = v
            csv_rows.append(flat)

        variants.append({
            "variant": variant_dir.name,
            "bins": bins_out,
        })

    if not variants:
        raise RuntimeError("No variants found")

    output.mkdir(parents=True)

    summary = {
        "purpose": "K5-A.4 signed wire-edge localization evaluation",
        "source_k5_output": str(k5_root),
        "wire_crop": str(args.wire_crop.resolve()),
        "overlap_box_native": list(overlap_box),
        "wire_pixels": int(wire.sum()),
        "variants": variants,
        "guardrails": [
            "Wire proxy is evaluation evidence only.",
            "No confidence generation changes are made.",
            "No inference or depth modification is performed.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fields = [
        "variant", "band", "lo_px", "hi_px", "pixels",
        "final_n", "final_mean", "final_median",
        "final_p75", "final_p90", "final_p95", "final_p99",
        "final_gt_0.01", "final_gt_0.025", "final_gt_0.05",
        "final_gt_0.1", "final_gt_0.25", "final_gt_0.5",
    ]
    with (output / "signed_distance_profile.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("K5-A.4 signed wire-edge profile complete")
    print()

    for variant in variants:
        print(f"===== {variant['variant']} =====")
        for row in variant["bins"]:
            s = row["final_confidence"]
            print(
                f"{row['name']}: "
                f"n={row['pixels']} "
                f"mean={s.get('mean')} "
                f"median={s.get('median')} "
                f"p90={s.get('p90')} "
                f"p99={s.get('p99')} "
                f"gt0.025={s.get('gt_0.025')} "
                f"gt0.05={s.get('gt_0.05')}"
            )
        print()


if __name__ == "__main__":
    main()
