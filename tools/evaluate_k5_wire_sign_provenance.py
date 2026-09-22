#!/usr/bin/env python3
"""K5-C.6 trace signed wire contrast from TinyViM contexts to integrated delta.

Evaluation only. No correction is recomputed or modified.

For the same thin approximately vertical hanger samples used in C.5:

1. Gaussian-filter each FULL TinyViM raw context at fine sigma.
2. Crop the filtered scalar fields to the common native overlap.
3. Measure local signed center-minus-side contrast on each context.
4. Measure the same signed contrast on the already integrated C.3 delta_q.
5. Ask:
   - do TinyViM contexts A/B agree on ridge-vs-valley sign?
   - where they agree, does integrated delta preserve that sign?
   - does agreement improve when restricting to stronger TinyViM contrasts?

TinyViM contrast magnitude remains non-metric. Only its sign and within-context
rank are used here. Positive affine remaps of either raw context preserve this
diagnostic.
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


def load_context(folder: Path):
    report = json.loads((folder / "alignment.json").read_text())
    box = tuple(int(v) for v in report["provenance"]["box_native_half_open"])
    raw = np.load(folder / "raw_tinyvim_relative.npy").astype(np.float64)
    x0, y0, x1, y1 = box
    if raw.shape != (y1-y0, x1-x0):
        raise ValueError(f"{folder}: raw shape / native box mismatch")
    if not np.isfinite(raw).all():
        raise ValueError(f"{folder}: TinyViM raw must be finite")
    return box, raw


def intersect_box(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    out = (
        max(ax0, bx0),
        max(ay0, by0),
        min(ax1, bx1),
        min(ay1, by1),
    )
    if out[0] >= out[2] or out[1] >= out[3]:
        raise ValueError("contexts do not overlap")
    return out


def crop_native(array, source_box, target_box):
    sx0, sy0, sx1, sy1 = source_box
    tx0, ty0, tx1, ty1 = target_box
    if not (
        sx0 <= tx0 <= tx1 <= sx1
        and sy0 <= ty0 <= ty1 <= sy1
    ):
        raise ValueError("target not contained in source")
    return array[
        ty0-sy0:ty1-sy0,
        tx0-sx0:tx1-sx0,
    ]


def map_wire_proxy(folder: Path, target_box):
    meta = json.loads((folder / "metadata.json").read_text())
    source_box = tuple(int(v) for v in meta["box"])
    sx0, sy0, sx1, sy1 = source_box
    tx0, ty0, tx1, ty1 = target_box

    mask = np.asarray(
        Image.open(folder / "rgb_proxy_mask.png").convert("L")
    ) > 0
    if mask.shape != (sy1-sy0, sx1-sx0):
        raise ValueError("wire proxy shape/metadata mismatch")

    out = np.zeros((ty1-ty0, tx1-tx0), dtype=bool)
    x0, y0 = max(sx0, tx0), max(sy0, ty0)
    x1, y1 = min(sx1, tx1), min(sy1, ty1)
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
            if dy > 0 and abs(int(xj-xi)) <= max_dx:
                uf.union(int(i), int(j))

    groups = {}
    for i in range(len(samples)):
        groups.setdefault(uf.find(i), []).append(i)
    return list(groups.values())


def center_contrast(field, y, x):
    profile = field[y, x+OFFSETS]
    outer_idx = [int(np.where(OFFSETS == o)[0][0]) for o in OUTER]
    baseline = float(np.median(profile[outer_idx]))
    return float(profile[OFFSETS.tolist().index(0)] - baseline)


def sign_array(values, eps=0.0):
    a = np.asarray(values)
    out = np.zeros(a.shape, dtype=np.int8)
    out[a > eps] = 1
    out[a < -eps] = -1
    return out


def agreement_report(ca, cb, cd, mask):
    mask = np.asarray(mask, dtype=bool)
    sa, sb, sd = sign_array(ca), sign_array(cb), sign_array(cd)

    nonzero_ab = mask & (sa != 0) & (sb != 0)
    agree_ab = nonzero_ab & (sa == sb)
    disagree_ab = nonzero_ab & (sa != sb)

    if agree_ab.any():
        delta_match = float(np.mean(sd[agree_ab] == sa[agree_ab]))
        delta_nonzero = float(np.mean(sd[agree_ab] != 0))
    else:
        delta_match = None
        delta_nonzero = None

    return {
        "samples": int(mask.sum()),
        "nonzero_both_contexts": int(nonzero_ab.sum()),
        "tinyvim_context_sign_agreement_fraction": (
            float(agree_ab.sum() / nonzero_ab.sum())
            if nonzero_ab.any() else None
        ),
        "tinyvim_context_sign_disagreement_fraction": (
            float(disagree_ab.sum() / nonzero_ab.sum())
            if nonzero_ab.any() else None
        ),
        "delta_matches_tinyvim_consensus_fraction": delta_match,
        "delta_nonzero_on_tinyvim_agreement_fraction": delta_nonzero,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-a", type=Path, required=True)
    ap.add_argument("--context-b", type=Path, required=True)
    ap.add_argument("--c3-output", type=Path, required=True)
    ap.add_argument("--wire-crop", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fine-sigma", type=float, default=1.0)
    ap.add_argument("--screening", type=float, default=0.05)
    ap.add_argument("--min-track-samples", type=int, default=8)
    ap.add_argument("--min-track-y-span", type=int, default=12)
    args = ap.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    box_a, raw_a = load_context(args.context_a.resolve())
    box_b, raw_b = load_context(args.context_b.resolve())
    common = intersect_box(box_a, box_b)

    c3_root = args.c3_output.resolve()
    c3_summary = json.loads((c3_root / "summary.json").read_text())
    c3_box = tuple(int(v) for v in c3_summary["overlap_box_native"])
    if c3_box != common:
        raise ValueError("C3 overlap does not match TinyViM context overlap")

    # Full-context filtering first: same boundary-safe rule as K5-A.1b.
    fa_full = ndi.gaussian_filter(
        raw_a, sigma=args.fine_sigma, mode="nearest"
    )
    fb_full = ndi.gaussian_filter(
        raw_b, sigma=args.fine_sigma, mode="nearest"
    )
    fa = crop_native(fa_full, box_a, common)
    fb = crop_native(fb_full, box_b, common)

    label = f"{args.screening:g}".replace(".", "p")
    delta = np.load(
        c3_root / f"delta_q_screening_{label}.npy"
    ).astype(np.float64)

    wire = map_wire_proxy(args.wire_crop.resolve(), common)
    if delta.shape != wire.shape or fa.shape != wire.shape or fb.shape != wire.shape:
        raise ValueError("overlap array shape mismatch")

    samples = select_vertical_samples(wire)
    if len(samples) < 20:
        raise RuntimeError(f"Only {len(samples)} wire samples")

    ca = np.asarray(
        [center_contrast(fa, y, x) for y, x, _, _ in samples],
        dtype=np.float64,
    )
    cb = np.asarray(
        [center_contrast(fb, y, x) for y, x, _, _ in samples],
        dtype=np.float64,
    )
    cd = np.asarray(
        [center_contrast(delta, y, x) for y, x, _, _ in samples],
        dtype=np.float64,
    )

    # Context-local strength ranks; no cross-context metric magnitude assumption.
    abs_a = np.abs(ca)
    abs_b = np.abs(cb)

    strength_reports = {}
    for percentile in (0, 25, 50, 75):
        ta = float(np.percentile(abs_a, percentile))
        tb = float(np.percentile(abs_b, percentile))
        mask = (abs_a >= ta) & (abs_b >= tb)
        strength_reports[f"both_contexts_ge_p{percentile}"] = {
            "context_a_abs_threshold": ta,
            "context_b_abs_threshold": tb,
            **agreement_report(ca, cb, cd, mask),
        }

    groups = group_tracks(samples)
    tracks = []

    for group in groups:
        ys = np.asarray([samples[i][0] for i in group], dtype=int)
        xs = np.asarray([samples[i][1] for i in group], dtype=int)
        span = int(ys.max() - ys.min() + 1)
        if len(group) < args.min_track_samples or span < args.min_track_y_span:
            continue

        idx = np.asarray(group, dtype=int)
        base = agreement_report(
            ca[idx], cb[idx], cd[idx],
            np.ones(len(idx), dtype=bool),
        )

        tracks.append({
            "track_id": len(tracks),
            "samples": int(len(idx)),
            "y_min": int(ys.min()),
            "y_max": int(ys.max()),
            "y_span_px": span,
            "x_median": float(np.median(xs)),
            "tinyvim_a_signed_contrast": stats(ca[idx]),
            "tinyvim_b_signed_contrast": stats(cb[idx]),
            "delta_signed_contrast": stats(cd[idx]),
            **base,
        })

    summary = {
        "purpose": "K5-C.6 signed wire contrast provenance diagnostic",
        "fine_sigma": float(args.fine_sigma),
        "screening": float(args.screening),
        "overlap_box_native": list(common),
        "selection": {
            "vertical_samples": int(len(samples)),
            "tracks": int(len(tracks)),
        },
        "all_samples": {
            "tinyvim_a_signed_contrast": stats(ca),
            "tinyvim_b_signed_contrast": stats(cb),
            "delta_signed_contrast": stats(cd),
            **agreement_report(
                ca, cb, cd,
                np.ones(len(samples), dtype=bool),
            ),
        },
        "strength_conditioned": strength_reports,
        "tracks": tracks,
        "guardrails": [
            "Wire proxy is evaluation-only.",
            "TinyViM raw contrasts are non-metric.",
            "Only sign and within-context strength ranks are compared.",
            "TinyViM full contexts are filtered before overlap crop.",
            "No correction is recomputed or modified.",
        ],
    }

    output.mkdir(parents=True, exist_ok=False)
    np.save(
        output / "sample_contrasts_a_b_delta.npy",
        np.column_stack((ca, cb, cd)),
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("K5-C.6 signed wire contrast provenance diagnostic complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
