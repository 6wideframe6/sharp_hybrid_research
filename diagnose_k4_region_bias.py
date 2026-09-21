#!/usr/bin/env python3
"""
K4 region-bias diagnosis ONLY.

Uses existing held-out points from diagnose_k4_model_family.py and existing
far_1480_512 context artifacts. No inference, no refit, no fusion, no K5.

Outputs:
- region_summary.csv
- leave_one_region_out.csv
- grouped_region_sets.json
- region_map_overlay.png
- m2_signed_residual_overlay.png
- m2_signed_residual.npy
- summary.json

Purpose:
Determine whether M2's remaining held-out error is concentrated in a few
disconnected fitting interiors, and visualize where those interiors are.

Important:
fitting_regions are connected fitting interiors, NOT semantic surface labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTEXT = ROOT / "results/sharp_hybrid/k4_context/far_1480_512"
DEFAULT_ABLATION = ROOT / "results/sharp_hybrid/k4_model_ablation/far_1480_512"
DEFAULT_OUTPUT = ROOT / "results/sharp_hybrid/k4_region_bias/far_1480_512"


def summarize_signed(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {
            "n": 0,
            "signed_mean": None,
            "signed_median": None,
            "signed_p10": None,
            "signed_p90": None,
            "abs_mean": None,
            "abs_median": None,
            "abs_p90": None,
            "gt_0.10": None,
            "gt_0.25": None,
        }

    aa = np.abs(a)
    return {
        "n": int(len(a)),
        "signed_mean": float(np.mean(a)),
        "signed_median": float(np.median(a)),
        "signed_p10": float(np.percentile(a, 10)),
        "signed_p90": float(np.percentile(a, 90)),
        "abs_mean": float(np.mean(aa)),
        "abs_median": float(np.median(aa)),
        "abs_p90": float(np.percentile(aa, 90)),
        "gt_0.10": float(np.mean(aa > 0.10)),
        "gt_0.25": float(np.mean(aa > 0.25)),
    }


def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_m2_points(csv_path, yspan):
    rows = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row["model"] != "M2":
                continue

            pred = float(row["predicted_inverse_m"])
            target = float(row["sharp_fold_smoothed_inverse_m"])
            signed = (pred - target) / yspan

            rows.append({
                "fold": int(row["fold"]),
                "x": int(row["x"]),
                "y": int(row["y"]),
                "block": int(row["block"]),
                "region": int(row["region"]),
                "depth_bin": int(row["depth_bin"]),
                "signed": float(signed),
                "abs": float(abs(signed)),
            })
    return rows


def region_bbox(mask):
    yy, xx = np.where(mask)
    if not len(xx):
        return None
    return [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]


def signed_color(value, limit):
    # blue = negative; white = near zero; red = positive
    t = float(np.clip(value / max(limit, np.finfo(float).tiny), -1.0, 1.0))
    if t < 0:
        k = -t
        return (
            int(round(255 * (1 - k))),
            int(round(255 * (1 - k))),
            255,
        )
    k = t
    return (
        255,
        int(round(255 * (1 - k))),
        int(round(255 * (1 - k))),
    )


def make_region_overlay(rgb, regions, region_rows, path):
    image = Image.fromarray(rgb).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Deterministic visually separated palette.
    palette = [
        (255, 64, 64, 90),
        (64, 255, 64, 90),
        (64, 128, 255, 90),
        (255, 192, 64, 90),
        (192, 64, 255, 90),
        (64, 255, 224, 90),
        (255, 64, 192, 90),
        (192, 255, 64, 90),
    ]

    sorted_regions = sorted(
        [r for r in region_rows if r["n"] > 0],
        key=lambda r: r["abs_p90"],
        reverse=True,
    )

    arr = np.array(overlay)
    for index, row in enumerate(sorted_regions):
        rid = row["region"]
        mask = regions == rid
        if not mask.any():
            continue
        color = palette[index % len(palette)]
        arr[mask] = color

    overlay = Image.fromarray(arr, mode="RGBA")
    result = Image.alpha_composite(image.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(result)

    for row in sorted_regions:
        rid = row["region"]
        bbox = row["bbox"]
        if bbox is None:
            continue

        x0, y0, x1, y1 = bbox
        draw.rectangle(
            (x0, y0, x1 - 1, y1 - 1),
            outline=(255, 255, 255, 220),
            width=1,
        )
        label = (
            f"R{rid} n={row['n']} "
            f"med={row['signed_median']:+.3f} "
            f"p90={row['abs_p90']:.3f}"
        )
        draw.text(
            (x0 + 2, max(0, y0 - 12)),
            label,
            fill=(255, 255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )

    result.convert("RGB").save(path)


def make_signed_overlay(rgb, shape, rows, path_png, path_npy):
    accum = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.int32)

    for row in rows:
        y, x = row["y"], row["x"]
        accum[y, x] += row["signed"]
        counts[y, x] += 1

    signed = np.full(shape, np.nan, dtype=np.float64)
    valid = counts > 0
    signed[valid] = accum[valid] / counts[valid]
    np.save(path_npy, signed)

    vals = np.abs(signed[np.isfinite(signed)])
    if not vals.size:
        Image.fromarray(rgb).save(path_png)
        return

    limit = float(np.percentile(vals, 95))
    base = Image.fromarray(rgb).convert("RGBA")
    overlay = np.zeros((*shape, 4), dtype=np.uint8)

    yy, xx = np.where(np.isfinite(signed))
    for y, x in zip(yy, xx):
        r, g, b = signed_color(float(signed[y, x]), limit)
        overlay[y, x] = (r, g, b, 180)

    out = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))
    out.convert("RGB").save(path_png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--ablation", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    context = args.context.resolve()
    ablation = args.ablation.resolve()
    output = args.output.resolve()

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    alignment = json.loads((context / "alignment.json").read_text())
    yspan = float(alignment["target_anchor_range"]["p90_minus_p10"])

    rgb = np.asarray(Image.open(context / "rgb_crop.png").convert("RGB"))
    regions = np.load(context / "fitting_regions.npy")
    points = load_m2_points(ablation / "heldout_points.csv", yspan)

    if regions.shape != rgb.shape[:2]:
        raise ValueError("RGB/region shape mismatch")

    by_region = defaultdict(list)
    by_region_bin = defaultdict(list)
    coords = defaultdict(list)

    for row in points:
        rid = row["region"]
        by_region[rid].append(row["signed"])
        by_region_bin[(rid, row["depth_bin"])].append(row["signed"])
        coords[rid].append((row["x"], row["y"]))

    region_rows = []
    for rid, vals in by_region.items():
        item = {
            "region": int(rid),
            **summarize_signed(vals),
        }

        xy = np.asarray(coords[rid], dtype=np.float64)
        item["heldout_x_min"] = int(xy[:, 0].min())
        item["heldout_x_max"] = int(xy[:, 0].max())
        item["heldout_y_min"] = int(xy[:, 1].min())
        item["heldout_y_max"] = int(xy[:, 1].max())
        item["heldout_x_mean"] = float(xy[:, 0].mean())
        item["heldout_y_mean"] = float(xy[:, 1].mean())
        item["bbox"] = region_bbox(regions == rid)

        bins_present = sorted(
            int(bin_id)
            for (region_id, bin_id) in by_region_bin
            if region_id == rid
        )
        item["depth_bins"] = bins_present
        region_rows.append(item)

    region_rows.sort(key=lambda x: x["abs_p90"], reverse=True)

    global_values = [row["signed"] for row in points]
    global_summary = summarize_signed(global_values)

    leave_one = []
    all_regions = sorted(by_region)
    for rid in all_regions:
        vals = [
            row["signed"]
            for row in points
            if row["region"] != rid
        ]
        item = {
            "excluded_region": int(rid),
            **summarize_signed(vals),
        }
        item["delta_abs_median_vs_all"] = (
            item["abs_median"] - global_summary["abs_median"]
        )
        item["delta_abs_p90_vs_all"] = (
            item["abs_p90"] - global_summary["abs_p90"]
        )
        leave_one.append(item)

    leave_one.sort(key=lambda x: x["delta_abs_p90_vs_all"])

    region_bin_rows = []
    for (rid, bin_id), vals in by_region_bin.items():
        if len(vals) < 8:
            continue
        region_bin_rows.append({
            "region": int(rid),
            "depth_bin": int(bin_id),
            **summarize_signed(vals),
        })
    region_bin_rows.sort(key=lambda x: (x["region"], x["depth_bin"]))

    passing_regions = [
        r["region"]
        for r in region_rows
        if r["n"] >= 20 and r["abs_p90"] < 0.25
    ]
    failing_regions = [
        r["region"]
        for r in region_rows
        if r["n"] >= 20 and r["abs_p90"] >= 0.25
    ]

    passing_vals = [
        row["signed"]
        for row in points
        if row["region"] in passing_regions
    ]
    failing_vals = [
        row["signed"]
        for row in points
        if row["region"] in failing_regions
    ]

    output.mkdir(parents=True, exist_ok=False)

    write_csv(
        output / "region_summary.csv",
        region_rows,
        [
            "region",
            "n",
            "signed_mean",
            "signed_median",
            "signed_p10",
            "signed_p90",
            "abs_mean",
            "abs_median",
            "abs_p90",
            "gt_0.10",
            "gt_0.25",
            "heldout_x_min",
            "heldout_x_max",
            "heldout_y_min",
            "heldout_y_max",
            "heldout_x_mean",
            "heldout_y_mean",
            "bbox",
            "depth_bins",
        ],
    )

    write_csv(
        output / "leave_one_region_out.csv",
        leave_one,
        [
            "excluded_region",
            "n",
            "signed_mean",
            "signed_median",
            "signed_p10",
            "signed_p90",
            "abs_mean",
            "abs_median",
            "abs_p90",
            "gt_0.10",
            "gt_0.25",
            "delta_abs_median_vs_all",
            "delta_abs_p90_vs_all",
        ],
    )

    write_csv(
        output / "region_depth_bin_summary.csv",
        region_bin_rows,
        [
            "region",
            "depth_bin",
            "n",
            "signed_mean",
            "signed_median",
            "signed_p10",
            "signed_p90",
            "abs_mean",
            "abs_median",
            "abs_p90",
            "gt_0.10",
            "gt_0.25",
        ],
    )

    make_region_overlay(
        rgb,
        regions,
        region_rows,
        output / "region_map_overlay.png",
    )
    make_signed_overlay(
        rgb,
        regions.shape,
        points,
        output / "m2_signed_residual_overlay.png",
        output / "m2_signed_residual.npy",
    )

    summary = {
        "context": str(context),
        "ablation": str(ablation),
        "purpose": "M2 region-bias diagnosis only; no inference/refit/fusion/K5",
        "global_M2": global_summary,
        "regions": region_rows,
        "leave_one_region_out": leave_one,
        "region_depth_bins": region_bin_rows,
        "diagnostic_grouping": {
            "regions_with_n_ge_20_and_abs_p90_lt_0_25": passing_regions,
            "regions_with_n_ge_20_and_abs_p90_ge_0_25": failing_regions,
            "passing_regions_pooled": summarize_signed(passing_vals),
            "failing_regions_pooled": summarize_signed(failing_vals),
            "warning": (
                "This threshold grouping is diagnostic only. "
                "Do not reinterpret fitting-region IDs as semantic surface ownership "
                "and do not use this grouping as a replacement/fusion mask."
            ),
        },
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("K4 M2 region-bias diagnostic complete")
    print(f"context:  {context}")
    print(f"ablation: {ablation}")
    print(f"output:   {output}")
    print()
    print("GLOBAL M2")
    print(json.dumps(global_summary, indent=2))
    print()
    print("REGIONS, worst abs p90 first")
    for row in region_rows:
        print(
            f"R{row['region']:>3} "
            f"n={row['n']:>5} "
            f"signed_med={row['signed_median']:+.6f} "
            f"abs_med={row['abs_median']:.6f} "
            f"abs_p90={row['abs_p90']:.6f} "
            f"bbox={row['bbox']} "
            f"bins={row['depth_bins']}"
        )
    print()
    print("BEST LEAVE-ONE-REGION-OUT CHANGES")
    for row in leave_one[:10]:
        print(
            f"exclude R{row['excluded_region']:>3}: "
            f"median={row['abs_median']:.6f} "
            f"p90={row['abs_p90']:.6f} "
            f"dP90={row['delta_abs_p90_vs_all']:+.6f}"
        )
    print()
    print("DIAGNOSTIC GROUPING")
    print(json.dumps(summary["diagnostic_grouping"], indent=2))


if __name__ == "__main__":
    main()
