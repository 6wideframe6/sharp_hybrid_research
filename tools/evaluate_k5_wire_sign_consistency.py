#!/usr/bin/env python3
"""K5-C.5 signed ridge consistency along individual vertical wire tracks.

Evaluation only. No correction is recomputed or modified.

C.4 established whether the integrated scalar correction is centered and thin
after sign-aligned aggregation. C.5 keeps the ORIGINAL sign and checks whether
that sign is stable along each selected hanger-wire track.

For every vertical-wire sample:
- sample delta_q at x offsets -8..+8;
- estimate a local side baseline from [-8,-6] and [+6,+8];
- compute signed center contrast = delta_q(center) - side_baseline.

Selected sample centers are then grouped into approximately continuous vertical
tracks using native-image proximity. The report gives per-track:
- sample count and y span;
- positive / negative center-contrast fractions;
- median signed and absolute contrast;
- sign consistency = max(positive_fraction, negative_fraction).

This does NOT decide which sign is physically correct. It answers whether the
integrated correction oscillates sign along a single wire, which would be a
failure mode before downstream composition.
"""

from __future__ import annotations

import argparse
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
    sx0, sy0, sx1, sy1 = [int(v) for v in meta["box"]]
    tx0, ty0, tx1, ty1 = [int(v) for v in target_box]

    mask = np.asarray(
        Image.open(wire_crop / "rgb_proxy_mask.png").convert("L")
    ) > 0
    if mask.shape != (sy1 - sy0, sx1 - sx0):
        raise ValueError("wire proxy shape/metadata mismatch")

    out = np.zeros((ty1 - ty0, tx1 - tx0), dtype=bool)
    x0 = max(sx0, tx0)
    y0 = max(sy0, ty0)
    x1 = min(sx1, tx1)
    y1 = min(sy1, ty1)
    if x0 >= x1 or y0 >= y1:
        return out

    out[y0-ty0:y1-ty0, x0-tx0:x1-tx0] = (
        mask[y0-sy0:y1-sy0, x0-sx0:x1-sx0]
    )
    return out


def row_runs(row):
    x = np.flatnonzero(row)
    if not len(x):
        return []
    split = np.where(np.diff(x) > 1)[0] + 1
    return [chunk for chunk in np.split(x, split) if len(chunk)]


def select_vertical_samples(
    wire,
    *,
    max_width=3,
    vertical_half_window=4,
    min_vertical_occupancy=0.75,
    x_tolerance=1,
    row_stride=2,
    edge_margin=9,
):
    h, w = wire.shape
    tolerant = ndi.maximum_filter1d(
        wire.astype(np.uint8),
        size=2*x_tolerance+1,
        axis=1,
        mode="constant",
        cval=0,
    ).astype(bool)

    samples = []
    for y in range(edge_margin, h-edge_margin):
        if row_stride > 1 and y % row_stride:
            continue
        for run in row_runs(wire[y]):
            width = int(run[-1] - run[0] + 1)
            if width > max_width:
                continue
            x = int(round((int(run[0]) + int(run[-1])) * 0.5))
            if x < edge_margin or x >= w-edge_margin:
                continue
            y0 = y - vertical_half_window
            y1 = y + vertical_half_window + 1
            occupancy = float(np.mean(tolerant[y0:y1, x]))
            if occupancy >= min_vertical_occupancy:
                samples.append((y, x, width, occupancy))
    return samples


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def group_tracks(samples, *, max_dy=4, max_dx=2):
    pts = np.asarray([(s[0], s[1]) for s in samples], dtype=int)
    uf = UnionFind(len(samples))

    order = np.argsort(pts[:, 0])
    for oi, i in enumerate(order):
        yi, xi = pts[i]
        for j in order[oi+1:]:
            yj, xj = pts[j]
            dy = int(yj - yi)
            if dy > max_dy:
                break
            if dy > 0 and abs(int(xj - xi)) <= max_dx:
                uf.union(int(i), int(j))

    groups = {}
    for i in range(len(samples)):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)
    return list(groups.values())


def signed_center_contrast(delta, y, x):
    xs = x + OFFSETS
    raw = delta[y, xs]
    outer_idx = [int(np.where(OFFSETS == o)[0][0]) for o in OUTER]
    baseline = float(np.median(raw[outer_idx]))
    return float(raw[OFFSETS.tolist().index(0)] - baseline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3-output", type=Path, required=True)
    ap.add_argument("--wire-crop", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--screening", type=float, default=0.05)

    ap.add_argument("--min-track-samples", type=int, default=8)
    ap.add_argument("--min-track-y-span", type=int, default=12)
    ap.add_argument("--track-max-dy", type=int, default=4)
    ap.add_argument("--track-max-dx", type=int, default=2)
    args = ap.parse_args()

    c3_root = args.c3_output.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    c3_summary = json.loads((c3_root / "summary.json").read_text())
    target_box = c3_summary["overlap_box_native"]
    wire = map_wire_proxy(args.wire_crop.resolve(), target_box)

    samples = select_vertical_samples(wire)
    if len(samples) < 20:
        raise RuntimeError(f"Only {len(samples)} selected wire samples")

    label = f"{args.screening:g}".replace(".", "p")
    delta_path = c3_root / f"delta_q_screening_{label}.npy"
    delta = np.load(delta_path).astype(np.float64)
    if delta.shape != wire.shape:
        raise ValueError("delta/wire shape mismatch")

    contrasts = np.asarray(
        [signed_center_contrast(delta, y, x) for y, x, _, _ in samples],
        dtype=np.float64,
    )

    groups = group_tracks(
        samples,
        max_dy=args.track_max_dy,
        max_dx=args.track_max_dx,
    )

    track_reports = []
    kept_indices = []

    for group in groups:
        ys = np.asarray([samples[i][0] for i in group], dtype=int)
        xs = np.asarray([samples[i][1] for i in group], dtype=int)
        c = contrasts[group]
        y_span = int(ys.max() - ys.min() + 1)

        if len(group) < args.min_track_samples or y_span < args.min_track_y_span:
            continue

        pos = float(np.mean(c > 0))
        neg = float(np.mean(c < 0))
        consistency = max(pos, neg)

        track_reports.append({
            "track_id": len(track_reports),
            "samples": int(len(group)),
            "y_min": int(ys.min()),
            "y_max": int(ys.max()),
            "y_span_px": y_span,
            "x_median": float(np.median(xs)),
            "x_min": int(xs.min()),
            "x_max": int(xs.max()),
            "positive_fraction": pos,
            "negative_fraction": neg,
            "sign_consistency": float(consistency),
            "signed_center_contrast": stats(c),
            "abs_center_contrast": stats(np.abs(c)),
        })
        kept_indices.extend(group)

    if not track_reports:
        raise RuntimeError("No tracks passed track-length filters")

    kept_indices = np.asarray(sorted(set(kept_indices)), dtype=int)
    kept_contrasts = contrasts[kept_indices]

    weights = np.asarray([r["samples"] for r in track_reports], dtype=float)
    consistencies = np.asarray(
        [r["sign_consistency"] for r in track_reports],
        dtype=float,
    )

    weighted_consistency = float(
        np.sum(weights * consistencies) / np.sum(weights)
    )

    summary = {
        "purpose": "K5-C.5 signed correction consistency along wire tracks",
        "screening": float(args.screening),
        "c3_output": str(c3_root),
        "overlap_box_native": target_box,
        "selection": {
            "all_vertical_samples": int(len(samples)),
            "kept_track_samples": int(len(kept_indices)),
            "tracks": int(len(track_reports)),
            "min_track_samples": args.min_track_samples,
            "min_track_y_span": args.min_track_y_span,
            "track_max_dy": args.track_max_dy,
            "track_max_dx": args.track_max_dx,
        },
        "all_kept_samples": {
            "signed_center_contrast": stats(kept_contrasts),
            "abs_center_contrast": stats(np.abs(kept_contrasts)),
            "positive_fraction": float(np.mean(kept_contrasts > 0)),
            "negative_fraction": float(np.mean(kept_contrasts < 0)),
        },
        "track_sign_consistency": {
            "weighted_mean": weighted_consistency,
            "tracks_ge_0.75": int(np.sum(consistencies >= 0.75)),
            "tracks_ge_0.90": int(np.sum(consistencies >= 0.90)),
            "tracks_ge_0.95": int(np.sum(consistencies >= 0.95)),
            "sample_fraction_in_tracks_ge_0.75": float(
                np.sum(weights[consistencies >= 0.75]) / np.sum(weights)
            ),
            "sample_fraction_in_tracks_ge_0.90": float(
                np.sum(weights[consistencies >= 0.90]) / np.sum(weights)
            ),
            "sample_fraction_in_tracks_ge_0.95": float(
                np.sum(weights[consistencies >= 0.95]) / np.sum(weights)
            ),
        },
        "tracks": track_reports,
        "guardrails": [
            "Wire proxy is evaluation-only.",
            "No correction is recomputed or modified.",
            "This diagnostic tests sign consistency, not absolute physical sign correctness.",
        ],
    }

    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("K5-C.5 signed wire-track consistency diagnostic complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
