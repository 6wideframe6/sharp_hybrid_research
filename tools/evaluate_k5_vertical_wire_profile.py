#!/usr/bin/env python3
"""K5-A.5 transverse confidence profile on vertical hanger-wire segments.

Diagnostic only:
- no inference
- no confidence regeneration
- no SHARP modification
- no fusion/integration

Why:
Euclidean distance-to-boundary bands mix neighboring parallel hanger wires.
This evaluator instead detects locally vertical, thin wire-proxy runs and
samples K5 confidence horizontally across each run center.

A clean localization should peak near offset 0 / +/-1 px.
A consistent peak at +2..+4 or -2..-4 px would indicate a systematic spatial
offset between the RGB-derived proxy and the TinyViM detail response.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


OFFSETS = tuple(range(-6, 7))
THRESHOLDS = (0.01, 0.025, 0.05, 0.10, 0.25)


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

    ix0 = max(wx0, ox0)
    iy0 = max(wy0, oy0)
    ix1 = min(wx1, ox1)
    iy1 = min(wy1, oy1)
    if ix0 >= ix1 or iy0 >= iy1:
        return out

    out[iy0-oy0:iy1-oy0, ix0-ox0:ix1-ox0] = wire[
        iy0-wy0:iy1-wy0,
        ix0-wx0:ix1-wx0,
    ]
    return out


def row_runs(row):
    """Yield [x0, x1) True runs for one boolean row."""
    padded = np.pad(np.asarray(row, dtype=np.int8), (1, 1))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts, ends))


def detect_vertical_samples(
    wire: np.ndarray,
    *,
    max_width: int = 3,
    vertical_half_window: int = 4,
    min_vertical_fraction: float = 0.75,
    min_margin: int = 7,
):
    """Return (y, center_x, width) samples from thin locally vertical runs.

    A candidate row-run must:
    - be <= max_width pixels wide;
    - fit sampling margin;
    - have wire support at its center x through most rows in a vertical window.

    This intentionally rejects broad/diagonal cable regions and keeps hanger
    segments that are approximately vertical in native coordinates.
    """
    h, w = wire.shape
    samples = []
    radius = int(vertical_half_window)

    for y in range(radius, h - radius):
        for x0, x1 in row_runs(wire[y]):
            width = x1 - x0
            if width <= 0 or width > max_width:
                continue

            center = (x0 + x1 - 1) // 2
            if center < min_margin or center >= w - min_margin:
                continue

            vertical = wire[y-radius:y+radius+1, center]
            fraction = float(np.mean(vertical))
            if fraction < min_vertical_fraction:
                continue

            # Reject locally broad structures. In a 5-px horizontal window,
            # a thin vertical line should occupy only a minority of pixels.
            local = wire[y, max(0, center-2):min(w, center+3)]
            if int(local.sum()) > max_width:
                continue

            samples.append((y, center, width))

    if not samples:
        return np.empty((0, 3), dtype=np.int32)

    samples = np.asarray(samples, dtype=np.int32)

    # Avoid overweighting consecutive rows from the same line too aggressively:
    # retain every second y for each center-x bucket.
    keep = []
    last_y_for_x = {}
    for i, (y, x, width) in enumerate(samples):
        last = last_y_for_x.get(int(x))
        if last is None or int(y) - last >= 2:
            keep.append(i)
            last_y_for_x[int(x)] = int(y)

    return samples[np.asarray(keep, dtype=np.int64)]


def stats(a):
    a = np.asarray(a, dtype=np.float64)
    if not len(a):
        return {"n": 0}
    out = {
        "n": int(len(a)),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }
    for t in THRESHOLDS:
        out[f"gt_{t:g}"] = float(np.mean(a > t))
    return out


def sample_offsets(array, samples):
    values = {}
    y = samples[:, 0]
    x = samples[:, 1]
    for offset in OFFSETS:
        values[offset] = array[y, x + offset]
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k5-output", type=Path, required=True)
    ap.add_argument("--wire-crop", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-wire-width", type=int, default=3)
    ap.add_argument("--vertical-half-window", type=int, default=4)
    ap.add_argument("--min-vertical-fraction", type=float, default=0.75)
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

    samples = detect_vertical_samples(
        wire,
        max_width=args.max_wire_width,
        vertical_half_window=args.vertical_half_window,
        min_vertical_fraction=args.min_vertical_fraction,
        min_margin=max(abs(v) for v in OFFSETS) + 1,
    )

    if len(samples) < 50:
        raise RuntimeError(
            f"Only {len(samples)} vertical-wire samples found; "
            "detection is too sparse for this proxy"
        )

    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "vertical_wire_samples_y_x_width.npy", samples)

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

        conf_values = sample_offsets(confidence, samples)
        detail_values = sample_offsets(detail, samples)
        direction_values = sample_offsets(direction, samples)

        offsets = []
        for offset in OFFSETS:
            row = {
                "offset_px": int(offset),
                "final_confidence": stats(conf_values[offset]),
                "detail_consensus": stats(detail_values[offset]),
                "direction_confidence": stats(direction_values[offset]),
            }
            offsets.append(row)

            flat = {
                "variant": variant_dir.name,
                "offset_px": int(offset),
            }
            for key, value in row["final_confidence"].items():
                flat[f"final_{key}"] = value
            csv_rows.append(flat)

        # Determine maxima using several robust criteria.
        mean_peak = max(
            offsets,
            key=lambda row: row["final_confidence"]["mean"],
        )["offset_px"]
        p90_peak = max(
            offsets,
            key=lambda row: row["final_confidence"]["p90"],
        )["offset_px"]
        gt005_peak = max(
            offsets,
            key=lambda row: row["final_confidence"]["gt_0.05"],
        )["offset_px"]

        center = next(
            row for row in offsets if row["offset_px"] == 0
        )["final_confidence"]

        far_values = np.concatenate(
            [conf_values[-6], conf_values[-5], conf_values[5], conf_values[6]]
        )
        far = stats(far_values)

        variants.append({
            "variant": variant_dir.name,
            "offsets": offsets,
            "peak_offsets": {
                "mean": int(mean_peak),
                "p90": int(p90_peak),
                "gt_0.05": int(gt005_peak),
            },
            "center_vs_far": {
                "center_mean": center["mean"],
                "far_mean": far["mean"],
                "mean_ratio": (
                    None if far["mean"] <= 0
                    else float(center["mean"] / far["mean"])
                ),
                "center_p90": center["p90"],
                "far_p90": far["p90"],
                "p90_ratio": (
                    None if far["p90"] <= 0
                    else float(center["p90"] / far["p90"])
                ),
            },
        })

    if not variants:
        raise RuntimeError("No K5 variants found")

    fields = [
        "variant", "offset_px", "final_n", "final_mean",
        "final_median", "final_p75", "final_p90",
        "final_p95", "final_p99", "final_gt_0.01",
        "final_gt_0.025", "final_gt_0.05",
        "final_gt_0.1", "final_gt_0.25",
    ]
    with (output / "transverse_profile.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    summary = {
        "purpose": "K5-A.5 transverse profile on vertical hanger-wire proxy segments",
        "source_k5_output": str(k5_root),
        "wire_crop": str(args.wire_crop.resolve()),
        "overlap_box_native": list(overlap_box),
        "wire_pixels": int(wire.sum()),
        "vertical_wire_samples": int(len(samples)),
        "offsets_px": list(OFFSETS),
        "detector": {
            "max_wire_width": int(args.max_wire_width),
            "vertical_half_window": int(args.vertical_half_window),
            "min_vertical_fraction": float(args.min_vertical_fraction),
        },
        "variants": variants,
        "guardrails": [
            "Wire proxy is evaluation evidence only.",
            "Only thin locally vertical proxy runs are sampled.",
            "No K5 confidence generation is modified.",
            "No inference, metric amplitude, integration or fusion is performed.",
        ],
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("K5-A.5 vertical-wire transverse profile complete")
    print(f"vertical wire samples: {len(samples)}")
    print()

    for variant in variants:
        print(f"===== {variant['variant']} =====")
        print("peak offsets:", variant["peak_offsets"])
        print("center vs far:", variant["center_vs_far"])
        print("offset profile:")
        for row in variant["offsets"]:
            fc = row["final_confidence"]
            print(
                f"  {row['offset_px']:+d}: "
                f"mean={fc['mean']:.6f} "
                f"p90={fc['p90']:.6f} "
                f"gt0.025={fc['gt_0.025']:.6f} "
                f"gt0.05={fc['gt_0.05']:.6f}"
            )
        print()


if __name__ == "__main__":
    main()
