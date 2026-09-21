#!/usr/bin/env python3
"""
K4 wire raw-support-density diagnosis ONLY.

No inference.
No modification of alignment.py.
No fusion/replacement/K5.

Purpose:
The radius-160 all-region fit brackets the wire raw range globally, but that
does not prove there are actual SHARP anchors near each wire raw value.
This diagnostic measures raw-space support density and explicitly detects
internal holes/gaps.

It reproduces the radius-160 selected anchors with the existing K4 align_affine()
implementation, then analyses:
- native TinyViM raw values of selected anchors
- smoothed TinyViM predictor values used by the fit
- wire raw distribution
- nearest selected-anchor raw distance for every wire pixel
- occupancy of wire-range raw bins
- contiguous unsupported raw intervals
- per-region raw support
- scatter plot of anchor predictor vs SHARP target, colored by fitting region,
  with the wire raw distribution shown separately.

Diagnostic only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.sharp_hybrid_research.alignment import (
    AlignmentConfig,
    align_affine,
)

DEFAULT_CONTEXT = ROOT / "results/sharp_hybrid/k4_context/far_1480_512"
DEFAULT_OLD_CROP = ROOT / "experiments/model_compare/outputs/depthart_tiny_512/far_1480_192"
DEFAULT_OUTPUT = ROOT / "results/sharp_hybrid/k4_wire_raw_support/far_1480_512"
RADIUS = 160


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
        raise RuntimeError("Old crop is not contained in context")

    proxy = np.asarray(
        Image.open(old_crop / "rgb_proxy_mask.png").convert("L")
    ) > 0

    mapped = np.zeros(shape, dtype=bool)
    lx0, ly0 = ox0 - cx0, oy0 - cy0
    lx1, ly1 = ox1 - cx0, oy1 - cy0
    mapped[ly0:ly1, lx0:lx1] = proxy
    return mapped


def quantiles(a):
    a = np.asarray(a, dtype=np.float64)
    return {
        "n": int(len(a)),
        "min": float(np.min(a)),
        "p01": float(np.percentile(a, 1)),
        "p05": float(np.percentile(a, 5)),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(np.max(a)),
    }


def nearest_distances(values, support):
    values = np.asarray(values, dtype=np.float64)
    support = np.sort(np.asarray(support, dtype=np.float64))
    pos = np.searchsorted(support, values)

    left_i = np.clip(pos - 1, 0, len(support) - 1)
    right_i = np.clip(pos, 0, len(support) - 1)

    left = np.abs(values - support[left_i])
    right = np.abs(values - support[right_i])
    return np.minimum(left, right)


def contiguous_false_intervals(edges, occupied):
    intervals = []
    start = None

    for i, flag in enumerate(occupied):
        if not flag and start is None:
            start = i

        is_last = i == len(occupied) - 1
        if start is not None and (flag or is_last):
            end_bin = i if flag else i + 1
            intervals.append(
                {
                    "start": float(edges[start]),
                    "end": float(edges[end_bin]),
                    "width": float(edges[end_bin] - edges[start]),
                    "bin_count": int(end_bin - start),
                }
            )
            start = None

    return intervals


def make_support_plot(
    raw,
    raw_fit,
    target_fit,
    regions,
    selected,
    wire,
    output_path,
):
    width = 1200
    height = 800
    margin = 70

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    sel = selected
    x = raw_fit[sel].astype(np.float64)
    y = target_fit[sel].astype(np.float64)
    rid = regions[sel].astype(np.int32)

    xmin, xmax = np.percentile(x, [0.5, 99.5])
    ymin, ymax = np.percentile(y, [0.5, 99.5])

    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    palette = {
        1: (80, 140, 255),
        2: (255, 80, 80),
        24: (170, 80, 210),
        68: (60, 170, 80),
        79: (255, 160, 40),
        80: (40, 180, 180),
    }

    def px(v):
        return int(margin + (v - xmin) / (xmax - xmin) * (width - 2 * margin))

    def py(v):
        return int(height - margin - (v - ymin) / (ymax - ymin) * (height - 2 * margin))

    draw.rectangle(
        (margin, margin, width - margin, height - margin),
        outline=(0, 0, 0),
        width=2,
    )

    # Downsample only for drawing if needed; diagnostics use full arrays.
    indices = np.arange(len(x))
    if len(indices) > 12000:
        rng = np.random.default_rng(0)
        indices = np.sort(rng.choice(indices, 12000, replace=False))

    for i in indices:
        xx = px(x[i])
        yy = py(y[i])
        color = palette.get(int(rid[i]), (120, 120, 120))
        draw.ellipse((xx - 1, yy - 1, xx + 1, yy + 1), fill=color)

    draw.text(
        (margin, 18),
        "selected anchors: smoothed TinyViM predictor vs smoothed SHARP inverse depth",
        fill=(0, 0, 0),
    )
    draw.text(
        (margin, height - 45),
        f"TinyViM smoothed predictor  [{xmin:.4g}, {xmax:.4g}]",
        fill=(0, 0, 0),
    )
    draw.text(
        (5, margin),
        f"SHARP 1/m [{ymin:.4g}, {ymax:.4g}]",
        fill=(0, 0, 0),
    )

    legend_x = width - 260
    legend_y = 20
    for j, region_id in enumerate(sorted(np.unique(rid))):
        color = palette.get(int(region_id), (120, 120, 120))
        yy = legend_y + 20 * j
        draw.rectangle((legend_x, yy, legend_x + 12, yy + 12), fill=color)
        draw.text(
            (legend_x + 18, yy - 1),
            f"R{int(region_id)}",
            fill=(0, 0, 0),
        )

    # Native raw histograms as bars at the bottom inside plot.
    native_anchor = raw[selected].astype(np.float64)
    wire_raw = raw[wire & np.isfinite(raw)].astype(np.float64)

    hist_lo = min(np.percentile(native_anchor, 0.5), np.percentile(wire_raw, 0.5))
    hist_hi = max(np.percentile(native_anchor, 99.5), np.percentile(wire_raw, 99.5))
    bins = np.linspace(hist_lo, hist_hi, 120)

    ha, _ = np.histogram(native_anchor, bins=bins)
    hw, _ = np.histogram(wire_raw, bins=bins)

    bar_base = height - margin - 5
    max_h = 80
    ha = ha / max(ha.max(), 1)
    hw = hw / max(hw.max(), 1)

    def hx(v):
        return int(margin + (v - hist_lo) / (hist_hi - hist_lo) * (width - 2 * margin))

    for i in range(len(bins) - 1):
        x0 = hx(bins[i])
        x1 = hx(bins[i + 1])
        if ha[i] > 0:
            draw.rectangle(
                (x0, bar_base - int(max_h * ha[i]), x1, bar_base),
                fill=(80, 80, 80),
            )
        if hw[i] > 0:
            ytop = bar_base - int(max_h * hw[i])
            draw.line(
                (x0, ytop, x1, ytop),
                fill=(255, 0, 255),
                width=2,
            )

    draw.text(
        (margin + 5, bar_base - max_h - 20),
        "bottom histogram: gray=selected-anchor native raw, magenta=wire native raw",
        fill=(0, 0, 0),
    )

    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--old-crop", type=Path, default=DEFAULT_OLD_CROP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bins", type=int, default=80)
    args = parser.parse_args()

    context = args.context.resolve()
    old_crop = args.old_crop.resolve()
    output = args.output.resolve()

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    source = json.loads((context / "alignment.json").read_text())
    config = AlignmentConfig(**source["config"])

    raw = np.load(context / "raw_tinyvim_relative.npy")
    target = np.load(context / "sharp_visible_inverse_m.npy")
    raw_fit = np.load(context / "tinyvim_common_bandwidth.npy")
    target_fit = np.load(context / "sharp_common_bandwidth.npy")
    regions = np.load(context / "fitting_regions.npy")
    candidate = np.asarray(
        Image.open(context / "anchor_candidates.png").convert("L")
    ) > 0

    if not (
        raw.shape
        == target.shape
        == raw_fit.shape
        == target_fit.shape
        == regions.shape
        == candidate.shape
    ):
        raise ValueError("Context artifact shape mismatch")

    wire = map_old_proxy(source, old_crop, raw.shape)
    distance = ndi.distance_transform_edt(~wire)
    local_candidate = candidate & (distance <= RADIUS)

    fit = align_affine(
        raw,
        target,
        local_candidate,
        raw_fit=raw_fit,
        target_fit=target_fit,
        regions=regions,
        foreground_mask=wire,
        config=config,
    )

    selected = fit["selected_anchor_mask"]
    if not selected.any():
        raise RuntimeError("No selected anchors")

    anchor_native = raw[selected].astype(np.float64)
    anchor_smoothed = raw_fit[selected].astype(np.float64)
    target_smoothed = target_fit[selected].astype(np.float64)
    wire_raw = raw[wire & np.isfinite(raw)].astype(np.float64)

    wire_q = quantiles(wire_raw)
    anchor_q = quantiles(anchor_native)

    # Diagnose support over the central 98% wire range, not just global min/max.
    wire_lo = wire_q["p01"]
    wire_hi = wire_q["p99"]
    edges = np.linspace(wire_lo, wire_hi, args.bins + 1)

    anchor_counts, _ = np.histogram(anchor_native, bins=edges)
    wire_counts, _ = np.histogram(wire_raw, bins=edges)

    occupied = anchor_counts > 0
    unsupported = contiguous_false_intervals(edges, occupied)

    wire_in_unsupported = 0
    for interval in unsupported:
        lo, hi = interval["start"], interval["end"]
        if hi == edges[-1]:
            m = (wire_raw >= lo) & (wire_raw <= hi)
        else:
            m = (wire_raw >= lo) & (wire_raw < hi)
        interval["wire_pixels"] = int(m.sum())
        interval["wire_fraction"] = float(m.mean())
        wire_in_unsupported += int(m.sum())

    nearest = nearest_distances(wire_raw, anchor_native)

    # Scale raw distances by robust selected-anchor range.
    anchor_span = max(
        float(np.percentile(anchor_native, 90) - np.percentile(anchor_native, 10)),
        np.finfo(float).tiny,
    )

    # Also quantify local anchor gaps directly from sorted native raw support.
    unique_anchor = np.unique(anchor_native)
    gaps = np.diff(unique_anchor)
    gap_rows = []
    if len(gaps):
        order = np.argsort(gaps)[::-1]
        for index in order[:30]:
            lo = float(unique_anchor[index])
            hi = float(unique_anchor[index + 1])
            in_gap = (wire_raw > lo) & (wire_raw < hi)
            gap_rows.append(
                {
                    "start": lo,
                    "end": hi,
                    "width": float(hi - lo),
                    "wire_pixels_strictly_inside": int(in_gap.sum()),
                    "wire_fraction_strictly_inside": float(in_gap.mean()),
                }
            )

    per_region = []
    for rid in sorted(int(v) for v in np.unique(regions[selected]) if int(v) > 0):
        m = selected & (regions == rid)
        vals = raw[m].astype(np.float64)
        fit_vals = raw_fit[m].astype(np.float64)
        targ = target_fit[m].astype(np.float64)
        per_region.append(
            {
                "region": rid,
                "selected_anchors": int(m.sum()),
                "native_raw": quantiles(vals),
                "smoothed_raw": quantiles(fit_vals),
                "smoothed_target": quantiles(targ),
            }
        )

    output.mkdir(parents=True, exist_ok=False)

    make_support_plot(
        raw,
        raw_fit,
        target_fit,
        regions,
        selected,
        wire,
        output / "anchor_scatter_and_wire_histogram.png",
    )

    summary = {
        "purpose": (
            "radius-160 wire raw-support density diagnosis; "
            "no inference/fusion/replacement/K5"
        ),
        "radius_native_px": RADIUS,
        "fit_reference": {
            "status": fit["report"]["status"],
            "a": fit["report"]["a"],
            "b": fit["report"]["b"],
            "heldout_median": fit["report"]["heldout_normalized_median"],
            "heldout_p90": fit["report"]["heldout_normalized_p90"],
            "fold_scale_variation": fit["report"]["fold_scale_variation"],
            "usable_anchors": fit["report"]["usable_anchors"],
        },
        "wire_native_raw": wire_q,
        "selected_anchor_native_raw": anchor_q,
        "selected_anchor_smoothed_raw": quantiles(anchor_smoothed),
        "selected_anchor_smoothed_target": quantiles(target_smoothed),
        "nearest_native_raw_anchor_distance_for_wire": {
            **quantiles(nearest),
            "median_in_anchor_p10_p90_spreads": float(
                np.median(nearest) / anchor_span
            ),
            "p90_in_anchor_p10_p90_spreads": float(
                np.percentile(nearest, 90) / anchor_span
            ),
            "p99_in_anchor_p10_p90_spreads": float(
                np.percentile(nearest, 99) / anchor_span
            ),
        },
        "wire_p01_p99_histogram_support": {
            "range": [float(wire_lo), float(wire_hi)],
            "bins": int(args.bins),
            "occupied_anchor_bins": int(occupied.sum()),
            "total_bins": int(len(occupied)),
            "occupied_fraction": float(occupied.mean()),
            "wire_pixels_in_unsupported_bins": int(wire_in_unsupported),
            "wire_fraction_in_unsupported_bins": float(
                wire_in_unsupported / max(len(wire_raw), 1)
            ),
            "unsupported_intervals": unsupported,
        },
        "largest_exact_native_raw_anchor_gaps": gap_rows,
        "per_region_selected_support": per_region,
        "guardrails": [
            "Global min/max bracketing is not evidence of dense local support.",
            "Histogram support is a diagnostic of shared predictor coverage only.",
            "Selected anchors are still fitting interiors, not semantic ownership.",
            "Do not interpolate across a large unsupported raw gap merely because a global affine fit is stable.",
        ],
    }

    save_json(output / "summary.json", summary)

    print("K4 wire raw-support-density diagnostic complete")
    print()
    print("FIT REFERENCE")
    print(json.dumps(summary["fit_reference"], indent=2))
    print()
    print("WIRE RAW")
    print(json.dumps(summary["wire_native_raw"], indent=2))
    print()
    print("ANCHOR RAW")
    print(json.dumps(summary["selected_anchor_native_raw"], indent=2))
    print()
    print("NEAREST RAW SUPPORT FOR WIRE")
    print(
        json.dumps(
            summary["nearest_native_raw_anchor_distance_for_wire"],
            indent=2,
        )
    )
    print()
    print("WIRE P01-P99 HISTOGRAM SUPPORT")
    print(
        json.dumps(
            summary["wire_p01_p99_histogram_support"],
            indent=2,
        )
    )
    print()
    print("LARGEST EXACT ANCHOR GAPS")
    for row in gap_rows[:15]:
        print(json.dumps(row))
    print()
    print("PER-REGION SELECTED SUPPORT")
    for row in per_region:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
