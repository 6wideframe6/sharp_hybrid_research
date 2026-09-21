#!/usr/bin/env python3
"""K5-B.4 stationarity diagnostic for the SHARP-derived amplitude scale.

No correction is integrated.

B.2/B.3 established:
- K5 direction is useful;
- K5 detail/confidence strength is not a reliable local metric amplitude;
- SHARP energy-excess is the most stable metric target tested so far.

B.4 asks whether that metric target can be represented by a bounded scalar, or
whether a spatially local SHARP scale is required.

It reports target amplitude versus:
1. K5 confidence bins;
2. K5 detail-consensus bins;
3. spatial 64 px blocks;
4. distance from the known wire proxy.

The diagnostic uses the same trusted support as B.3:
- finite overlap;
- wire exclusion;
- border exclusion;
- K5 confidence >= threshold;
- K5 direction aligned with SHARP fine gradient;
- strong SHARP fine-gradient support.

Only positive SHARP energy-excess samples are used for amplitude-location
statistics. Zero target values are reported separately.
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

from k5.overlap import crop_to_native_box
from k5.types import NativeBox


CONFIDENCE_EDGES = (0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0000001)
DETAIL_EDGES = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0000001)
WIRE_DISTANCE_BANDS = (
    (8.0, 16.0),
    (16.0, 32.0),
    (32.0, 64.0),
    (64.0, 128.0),
    (128.0, float("inf")),
)


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


def positive_stats(a):
    a = np.asarray(a, dtype=np.float64)
    finite = np.isfinite(a)
    positive = finite & (a > 0)
    out = stats(a[positive])
    out["finite_n"] = int(finite.sum())
    out["positive_fraction"] = (
        float(positive.sum() / finite.sum()) if finite.any() else None
    )
    return out


def rank_corr(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
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


def binned_report(values, target, selected, edges):
    result = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = selected & (values >= lo) & (values < hi)
        result.append({
            "lo": float(lo),
            "hi": float(hi),
            "pixels": int(mask.sum()),
            "target_positive": positive_stats(target[mask]),
        })
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
    ap.add_argument("--min-block-positive-samples", type=int, default=20)
    args = ap.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    ca, cb = args.context_a.resolve(), args.context_b.resolve()
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
    wire_distance = ndi.distance_transform_edt(~wire)

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

    if int(positive.sum()) < 100:
        raise RuntimeError(f"Only {int(positive.sum())} positive-target pixels")

    # Dependence on K5-derived quantities.
    confidence_bins = binned_report(
        confidence, target, selected, CONFIDENCE_EDGES
    )
    detail_bins = binned_report(
        detail, target, selected, DETAIL_EDGES
    )

    # Spatial block stationarity.
    yy, xx = np.mgrid[:common.height, :common.width]
    ncols = int(np.ceil(common.width / args.block_size))
    block_map = (yy // args.block_size) * ncols + xx // args.block_size

    block_reports = []
    block_medians = []
    for block in np.unique(block_map[selected]):
        m = positive & (block_map == block)
        n = int(m.sum())
        if n < args.min_block_positive_samples:
            continue

        values = target[m]
        med = float(np.median(values))
        block_medians.append(med)

        by = int(block // ncols)
        bx = int(block % ncols)
        block_reports.append({
            "block": int(block),
            "block_xy": [bx, by],
            "positive_samples": n,
            "target_positive": stats(values),
            "median": med,
        })

    block_medians = np.asarray(block_medians, dtype=np.float64)
    if len(block_medians):
        block_median_stationarity = {
            "blocks": int(len(block_medians)),
            "median_of_block_medians": float(np.median(block_medians)),
            "p10": float(np.percentile(block_medians, 10)),
            "p90": float(np.percentile(block_medians, 90)),
            "relative_p90_p10_span": float(
                (np.percentile(block_medians, 90)
                 - np.percentile(block_medians, 10))
                / max(np.median(block_medians), np.finfo(float).tiny)
            ),
            "min": float(np.min(block_medians)),
            "max": float(np.max(block_medians)),
        }
    else:
        block_median_stationarity = {"blocks": 0}

    # Distance from known wires: can nearby trusted SHARP geometry provide a
    # representative metric scale?
    wire_distance_reports = []
    for lo, hi in WIRE_DISTANCE_BANDS:
        m = selected & (wire_distance >= lo)
        if np.isfinite(hi):
            m &= wire_distance < hi

        wire_distance_reports.append({
            "lo_px": float(lo),
            "hi_px": None if not np.isfinite(hi) else float(hi),
            "pixels": int(m.sum()),
            "target_positive": positive_stats(target[m]),
        })

    # High-confidence core subsets: useful for a saturated-gate design.
    core_reports = []
    for threshold in (0.05, 0.10, 0.25, 0.50, 0.75):
        m = selected & (confidence >= threshold)
        core_reports.append({
            "min_confidence": threshold,
            "pixels": int(m.sum()),
            "target_positive": positive_stats(target[m]),
        })

    summary = {
        "purpose": "K5-B.4 SHARP metric-amplitude stationarity diagnostic",
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
            "min_block_positive_samples": args.min_block_positive_samples,
        },
        "support": {
            "base_pixels": int(base.sum()),
            "selected_pixels": int(selected.sum()),
            "positive_target_pixels": int(positive.sum()),
            "zero_target_fraction_selected": float(
                np.mean(target[selected] <= 0)
            ),
            "wire_pixels": int(wire.sum()),
            "target_selected": positive_stats(target[selected]),
        },
        "dependence": {
            "spearman_confidence_vs_positive_target": rank_corr(
                confidence[positive], target[positive]
            ),
            "spearman_detail_vs_positive_target": rank_corr(
                detail[positive], target[positive]
            ),
            "confidence_bins": confidence_bins,
            "detail_bins": detail_bins,
        },
        "spatial_blocks": {
            "stationarity": block_median_stationarity,
            "blocks": block_reports,
        },
        "wire_distance": wire_distance_reports,
        "high_confidence_cores": core_reports,
        "guardrails": [
            "Only positive SHARP energy-excess is used for amplitude-location statistics.",
            "K5 confidence/detail are tested for dependence; they are not assumed metric.",
            "Wire pixels remain excluded from calibration support.",
            "No correction is integrated or composed.",
        ],
    }

    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "selected_support_mask.npy", selected)
    np.save(output / "positive_target_mask.npy", positive)
    np.save(output / "energy_excess_target.npy", target)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("K5-B.4 amplitude stationarity diagnostic complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
