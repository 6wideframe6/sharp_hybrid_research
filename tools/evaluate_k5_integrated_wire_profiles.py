#!/usr/bin/env python3
"""K5-C.4 integrated wire-ridge transverse-profile diagnostic.

Evaluation only.  No correction is changed and no wire mask is used to build a
correction.

The script loads already-computed C.3 ``delta_q`` outputs, isolates thin
approximately vertical wire/hanger samples from the known evaluation proxy, and
measures the scalar correction transversely across each wire.

For every selected wire sample:
- take offsets -8..+8 px in x;
- estimate a local side baseline from the outer bands [-8,-6] and [+6,+8];
- subtract that baseline;
- align the profile sign by the center contrast, so foreground/background sign
  does not cancel the localization measurement.

This answers the next question after C.3:
Does screened integration turn the K5 edge field into a centered, thin scalar
ridge/valley, or into a displaced/broad halo?

The wire proxy is evaluation-only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


OFFSETS = np.arange(-8, 9, dtype=int)
OUTER = np.array([-8, -7, -6, 6, 7, 8], dtype=int)


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
        "p99": float(np.percentile(a, 99)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def map_wire_proxy(wire_crop: Path, target_box):
    meta = json.loads((wire_crop / "metadata.json").read_text())
    source_box = tuple(int(v) for v in meta["box"])
    tx0, ty0, tx1, ty1 = target_box
    sx0, sy0, sx1, sy1 = source_box

    mask = np.asarray(
        Image.open(wire_crop / "rgb_proxy_mask.png").convert("L")
    ) > 0
    if mask.shape != (sy1 - sy0, sx1 - sx0):
        raise ValueError("wire proxy shape/metadata mismatch")

    out = np.zeros((ty1 - ty0, tx1 - tx0), dtype=bool)

    x0 = max(tx0, sx0)
    y0 = max(ty0, sy0)
    x1 = min(tx1, sx1)
    y1 = min(ty1, sy1)
    if x0 >= x1 or y0 >= y1:
        return out

    out[
        y0 - ty0 : y1 - ty0,
        x0 - tx0 : x1 - tx0,
    ] = mask[
        y0 - sy0 : y1 - sy0,
        x0 - sx0 : x1 - sx0,
    ]
    return out


def row_runs(row):
    x = np.flatnonzero(row)
    if not len(x):
        return []
    split = np.where(np.diff(x) > 1)[0] + 1
    return [chunk for chunk in np.split(x, split) if len(chunk)]


def select_vertical_wire_samples(
    wire: np.ndarray,
    *,
    max_width: int = 3,
    vertical_half_window: int = 4,
    min_vertical_occupancy: float = 0.75,
    x_tolerance: int = 1,
    row_stride: int = 2,
    edge_margin: int = 9,
):
    """Select thin approximately vertical row-run centers.

    Vertical support tolerates +/- ``x_tolerance`` drift between neighbouring
    rows, so slightly sloped hangers are retained.
    """
    h, w = wire.shape

    # Each pixel says whether this row has wire within +/- x_tolerance.
    tolerant = ndi.maximum_filter1d(
        wire.astype(np.uint8),
        size=2 * x_tolerance + 1,
        axis=1,
        mode="constant",
        cval=0,
    ).astype(bool)

    candidates = []

    for y in range(edge_margin, h - edge_margin):
        if row_stride > 1 and (y % row_stride) != 0:
            continue

        for run in row_runs(wire[y]):
            width = int(run[-1] - run[0] + 1)
            if width > max_width:
                continue

            x = int(round((int(run[0]) + int(run[-1])) * 0.5))
            if x < edge_margin or x >= w - edge_margin:
                continue

            y0 = y - vertical_half_window
            y1 = y + vertical_half_window + 1
            occupancy = float(np.mean(tolerant[y0:y1, x]))
            if occupancy < min_vertical_occupancy:
                continue

            candidates.append((y, x, width, occupancy))

    return candidates


def load_delta(c3_root: Path, screening: float):
    label = f"{screening:g}".replace(".", "p")
    path = c3_root / f"delta_q_screening_{label}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path).astype(np.float64)


def aggregate_profile(delta: np.ndarray, samples):
    raw_profiles = []
    aligned_profiles = []
    contrasts = []
    signed_contrasts = []

    outer_indices = [int(np.where(OFFSETS == o)[0][0]) for o in OUTER]
    center_index = int(np.where(OFFSETS == 0)[0][0])

    for y, x, width, occupancy in samples:
        xs = x + OFFSETS
        if xs.min() < 0 or xs.max() >= delta.shape[1]:
            continue

        raw = delta[y, xs]
        baseline = float(np.median(raw[outer_indices]))
        local = raw - baseline
        center = float(local[center_index])

        # Ignore exactly-zero degenerate profiles.
        if abs(center) <= np.finfo(float).tiny:
            continue

        sign = 1.0 if center > 0 else -1.0
        aligned = sign * local

        raw_profiles.append(raw)
        aligned_profiles.append(aligned)
        signed_contrasts.append(center)
        contrasts.append(abs(center))

    if not aligned_profiles:
        raise RuntimeError("No usable non-degenerate wire profiles")

    raw_profiles = np.asarray(raw_profiles, dtype=np.float64)
    aligned_profiles = np.asarray(aligned_profiles, dtype=np.float64)
    signed_contrasts = np.asarray(signed_contrasts, dtype=np.float64)
    contrasts = np.asarray(contrasts, dtype=np.float64)

    med = np.median(aligned_profiles, axis=0)
    p25 = np.percentile(aligned_profiles, 25, axis=0)
    p75 = np.percentile(aligned_profiles, 75, axis=0)
    p90 = np.percentile(aligned_profiles, 90, axis=0)

    peak_index = int(np.argmax(med))
    peak_offset = int(OFFSETS[peak_index])
    peak_value = float(med[peak_index])

    half = 0.5 * peak_value
    above = np.flatnonzero(med >= half)
    fwhm_px = (
        float(OFFSETS[above[-1]] - OFFSETS[above[0]] + 1)
        if len(above)
        else None
    )

    outer_mask = np.isin(OFFSETS, OUTER)
    outer_abs = np.abs(aligned_profiles[:, outer_mask])

    return {
        "usable_samples": int(len(aligned_profiles)),
        "center_abs_contrast": stats(contrasts),
        "center_signed_contrast": stats(signed_contrasts),
        "positive_center_fraction": float(np.mean(signed_contrasts > 0)),
        "median_profile_peak_offset_px": peak_offset,
        "median_profile_peak_value": peak_value,
        "median_profile_fwhm_px": fwhm_px,
        "outer_sideband_abs": stats(outer_abs.ravel()),
        "median_profile": med,
        "p25_profile": p25,
        "p75_profile": p75,
        "p90_profile": p90,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3-output", type=Path, required=True)
    ap.add_argument("--wire-crop", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--screenings",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.10],
    )
    ap.add_argument("--max-wire-width", type=int, default=3)
    ap.add_argument("--vertical-half-window", type=int, default=4)
    ap.add_argument("--min-vertical-occupancy", type=float, default=0.75)
    ap.add_argument("--x-tolerance", type=int, default=1)
    ap.add_argument("--row-stride", type=int, default=2)
    args = ap.parse_args()

    c3_root = args.c3_output.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    summary_c3 = json.loads((c3_root / "summary.json").read_text())
    target_box = tuple(
        int(v) for v in summary_c3["overlap_box_native"]
    )
    wire = map_wire_proxy(args.wire_crop.resolve(), target_box)

    samples = select_vertical_wire_samples(
        wire,
        max_width=args.max_wire_width,
        vertical_half_window=args.vertical_half_window,
        min_vertical_occupancy=args.min_vertical_occupancy,
        x_tolerance=args.x_tolerance,
        row_stride=args.row_stride,
    )
    if len(samples) < 20:
        raise RuntimeError(
            f"Only {len(samples)} vertical wire samples; selector too strict"
        )

    reports = {}
    rows = []

    for screening in args.screenings:
        delta = load_delta(c3_root, screening)
        if delta.shape != wire.shape:
            raise ValueError(
                f"delta shape {delta.shape} != wire shape {wire.shape}"
            )

        report = aggregate_profile(delta, samples)
        reports[f"screening_{screening:g}"] = {
            k: v
            for k, v in report.items()
            if not isinstance(v, np.ndarray)
        }

        for i, offset in enumerate(OFFSETS):
            rows.append({
                "screening": float(screening),
                "offset_px": int(offset),
                "median_aligned_delta_q": float(
                    report["median_profile"][i]
                ),
                "p25_aligned_delta_q": float(
                    report["p25_profile"][i]
                ),
                "p75_aligned_delta_q": float(
                    report["p75_profile"][i]
                ),
                "p90_aligned_delta_q": float(
                    report["p90_profile"][i]
                ),
            })

    output.mkdir(parents=True, exist_ok=False)

    np.save(
        output / "vertical_wire_samples_y_x_width_occupancy.npy",
        np.asarray(samples, dtype=np.float64),
    )

    with (output / "transverse_profiles.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "purpose": "K5-C.4 integrated vertical-wire ridge localization diagnostic",
        "c3_output": str(c3_root),
        "overlap_box_native": list(target_box),
        "selector": {
            "candidate_samples": int(len(samples)),
            "max_wire_width": args.max_wire_width,
            "vertical_half_window": args.vertical_half_window,
            "min_vertical_occupancy": args.min_vertical_occupancy,
            "x_tolerance": args.x_tolerance,
            "row_stride": args.row_stride,
            "profile_offsets_px": OFFSETS.tolist(),
            "side_baseline_offsets_px": OUTER.tolist(),
        },
        "screenings": [float(v) for v in args.screenings],
        "results": reports,
        "guardrails": [
            "Wire proxy is evaluation-only.",
            "No correction is modified or recomputed.",
            "Profile sign is aligned only for localization/width statistics.",
            "Original signed center contrast fraction is reported separately.",
        ],
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("K5-C.4 integrated wire-ridge profile diagnostic complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
