#!/usr/bin/env python3
"""K5-C.7 production-usable TinyViM polarity-reliability diagnostic.

Evaluation only. No correction is recomputed.

C.6 used a wire-specific transverse center-minus-side contrast and showed that
signed reliability rises sharply with signal strength. That measurement cannot
be used in production because it depends on the evaluation wire proxy.

C.7 tests a generic proxy available everywhere:

    polarity_i = sign(G_fine(r_i) - G_coarse(r_i))

for each TinyViM context i.

The signed DoG is invariant to additive offsets and preserves sign under
positive affine scaling of the TinyViM raw prediction. Its absolute magnitude
is normalized independently per context over the same halo-safe overlap and is
used only as a non-metric reliability measure.

On evaluation wire samples, the script reports:
- A/B DoG sign agreement;
- coverage after minimum normalized DoG-strength thresholds;
- whether integrated delta_q center contrast matches the agreed DoG polarity;
- whether DoG polarity agrees with the wire-specific transverse contrast used
  in C.6.

The wire proxy is evaluation-only and is never used to construct the DoG maps,
their normalization, or their thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


OFFSETS = np.arange(-8, 9, dtype=int)
OUTER = np.array([-8, -7, -6, 6, 7, 8], dtype=int)


def load_context(folder: Path):
    report = json.loads((folder / "alignment.json").read_text())
    box = tuple(int(v) for v in report["provenance"]["box_native_half_open"])
    raw = np.load(folder / "raw_tinyvim_relative.npy").astype(np.float64)
    x0, y0, x1, y1 = box
    if raw.shape != (y1-y0, x1-x0):
        raise ValueError("raw/box shape mismatch")
    if not np.isfinite(raw).all():
        raise ValueError("TinyViM raw must be finite")
    return box, raw


def intersect_box(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x0 >= x1 or y0 >= y1:
        raise ValueError("contexts do not overlap")
    return (x0, y0, x1, y1)


def crop_native(array, source_box, target_box):
    sx0, sy0, sx1, sy1 = source_box
    tx0, ty0, tx1, ty1 = target_box
    return array[ty0-sy0:ty1-sy0, tx0-sx0:tx1-sx0]


def halo_mask(source_box, target_box, halo):
    sx0, sy0, sx1, sy1 = source_box
    tx0, ty0, tx1, ty1 = target_box
    yy, xx = np.mgrid[ty0:ty1, tx0:tx1]
    return (
        (xx >= sx0 + halo)
        & (xx < sx1 - halo)
        & (yy >= sy0 + halo)
        & (yy < sy1 - halo)
    )


def robust_normalize_abs(value, valid, low=50.0, high=99.0):
    a = np.abs(np.asarray(value, dtype=np.float64))
    samples = a[valid]
    lo, hi = np.percentile(samples, [low, high])
    if hi <= lo:
        return np.zeros_like(a), float(lo), float(hi)
    out = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    out[~valid] = 0.0
    return out, float(lo), float(hi)


def map_wire_proxy(folder: Path, target_box):
    meta = json.loads((folder / "metadata.json").read_text())
    source_box = tuple(int(v) for v in meta["box"])
    sx0, sy0, sx1, sy1 = source_box
    tx0, ty0, tx1, ty1 = target_box
    mask = np.asarray(
        Image.open(folder / "rgb_proxy_mask.png").convert("L")
    ) > 0
    out = np.zeros((ty1-ty0, tx1-tx0), dtype=bool)

    x0=max(sx0,tx0); y0=max(sy0,ty0)
    x1=min(sx1,tx1); y1=min(sy1,ty1)
    if x0 < x1 and y0 < y1:
        out[y0-ty0:y1-ty0, x0-tx0:x1-tx0] = (
            mask[y0-sy0:y1-sy0, x0-sx0:x1-sx0]
        )
    return out


def row_runs(row):
    x = np.flatnonzero(row)
    if not len(x):
        return []
    split = np.where(np.diff(x) > 1)[0] + 1
    return [c for c in np.split(x, split) if len(c)]


def select_vertical_samples(wire):
    h, w = wire.shape
    tolerant = ndi.maximum_filter1d(
        wire.astype(np.uint8), size=3, axis=1,
        mode="constant", cval=0
    ).astype(bool)

    samples = []
    for y in range(9, h-9):
        if y % 2:
            continue
        for run in row_runs(wire[y]):
            width = int(run[-1]-run[0]+1)
            if width > 3:
                continue
            x = int(round((int(run[0])+int(run[-1]))*0.5))
            if x < 9 or x >= w-9:
                continue
            occupancy = float(np.mean(tolerant[y-4:y+5, x]))
            if occupancy >= 0.75:
                samples.append((y,x))
    return samples


def center_contrast(field, y, x):
    p = field[y, x+OFFSETS]
    outer_idx = [int(np.where(OFFSETS == o)[0][0]) for o in OUTER]
    baseline = float(np.median(p[outer_idx]))
    return float(p[OFFSETS.tolist().index(0)] - baseline)


def sign(v):
    a = np.asarray(v)
    return np.where(a > 0, 1, np.where(a < 0, -1, 0))


def report_mask(mask, dog_a, dog_b, delta_c, trans_a, trans_b):
    sa = sign(dog_a)
    sb = sign(dog_b)
    sd = sign(delta_c)
    sta = sign(trans_a)
    stb = sign(trans_b)

    m = np.asarray(mask, dtype=bool)
    agree = m & (sa != 0) & (sb != 0) & (sa == sb)

    out = {
        "samples": int(m.sum()),
        "dog_context_sign_agreement_fraction": (
            float(np.mean(sa[m] == sb[m])) if m.any() else None
        ),
        "agreed_dog_samples": int(agree.sum()),
        "delta_matches_agreed_dog_fraction": (
            float(np.mean(sd[agree] == sa[agree])) if agree.any() else None
        ),
        "dog_a_matches_transverse_a_fraction": (
            float(np.mean(sa[m] == sta[m])) if m.any() else None
        ),
        "dog_b_matches_transverse_b_fraction": (
            float(np.mean(sb[m] == stb[m])) if m.any() else None
        ),
        "transverse_context_sign_agreement_fraction": (
            float(np.mean(sta[m] == stb[m])) if m.any() else None
        ),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-a", type=Path, required=True)
    ap.add_argument("--context-b", type=Path, required=True)
    ap.add_argument("--c3-output", type=Path, required=True)
    ap.add_argument("--wire-crop", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fine-sigma", type=float, default=1.0)
    ap.add_argument("--coarse-sigma", type=float, default=2.0)
    ap.add_argument("--screening", type=float, default=0.05)
    ap.add_argument(
        "--strength-thresholds", type=float, nargs="+",
        default=[0.0, 0.05, 0.10, 0.25, 0.50]
    )
    args = ap.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if args.coarse_sigma <= args.fine_sigma:
        raise ValueError("coarse sigma must exceed fine sigma")

    box_a, raw_a = load_context(args.context_a.resolve())
    box_b, raw_b = load_context(args.context_b.resolve())
    common = intersect_box(box_a, box_b)

    fine_a = ndi.gaussian_filter(raw_a, sigma=args.fine_sigma, mode="nearest")
    coarse_a = ndi.gaussian_filter(raw_a, sigma=args.coarse_sigma, mode="nearest")
    fine_b = ndi.gaussian_filter(raw_b, sigma=args.fine_sigma, mode="nearest")
    coarse_b = ndi.gaussian_filter(raw_b, sigma=args.coarse_sigma, mode="nearest")

    dog_a = crop_native(fine_a - coarse_a, box_a, common)
    dog_b = crop_native(fine_b - coarse_b, box_b, common)
    fine_a_c = crop_native(fine_a, box_a, common)
    fine_b_c = crop_native(fine_b, box_b, common)

    halo = int(math.ceil(4.0 * args.coarse_sigma))
    valid = halo_mask(box_a, common, halo) & halo_mask(box_b, common, halo)

    norm_a, lo_a, hi_a = robust_normalize_abs(dog_a, valid)
    norm_b, lo_b, hi_b = robust_normalize_abs(dog_b, valid)
    strength = np.minimum(norm_a, norm_b)

    c3_root = args.c3_output.resolve()
    c3_summary = json.loads((c3_root / "summary.json").read_text())
    if tuple(c3_summary["overlap_box_native"]) != common:
        raise ValueError("C3 overlap mismatch")

    label = f"{args.screening:g}".replace(".", "p")
    delta = np.load(c3_root / f"delta_q_screening_{label}.npy").astype(np.float64)

    wire = map_wire_proxy(args.wire_crop.resolve(), common)
    samples = select_vertical_samples(wire)
    if len(samples) < 20:
        raise RuntimeError("too few vertical wire samples")

    ys = np.asarray([p[0] for p in samples], dtype=int)
    xs = np.asarray([p[1] for p in samples], dtype=int)

    dog_a_s = dog_a[ys, xs]
    dog_b_s = dog_b[ys, xs]
    strength_s = strength[ys, xs]
    valid_s = valid[ys, xs]

    delta_c = np.asarray(
        [center_contrast(delta, y, x) for y,x in samples],
        dtype=np.float64
    )
    trans_a = np.asarray(
        [center_contrast(fine_a_c, y, x) for y,x in samples],
        dtype=np.float64
    )
    trans_b = np.asarray(
        [center_contrast(fine_b_c, y, x) for y,x in samples],
        dtype=np.float64
    )

    reports = {}
    for threshold in args.strength_thresholds:
        mask = valid_s & (strength_s >= threshold)
        reports[f"strength_ge_{threshold:g}"] = {
            "threshold": float(threshold),
            "coverage_fraction_of_wire_samples": float(np.mean(mask)),
            **report_mask(
                mask, dog_a_s, dog_b_s, delta_c, trans_a, trans_b
            ),
        }

    summary = {
        "purpose": "K5-C.7 generic DoG polarity-reliability diagnostic",
        "overlap_box_native": list(common),
        "fine_sigma": float(args.fine_sigma),
        "coarse_sigma": float(args.coarse_sigma),
        "screening": float(args.screening),
        "normalization": {
            "domain": "shared halo-safe overlap, independently per context",
            "low_percentile": 50.0,
            "high_percentile": 99.0,
            "context_a_abs_dog_low": lo_a,
            "context_a_abs_dog_high": hi_a,
            "context_b_abs_dog_low": lo_b,
            "context_b_abs_dog_high": hi_b,
        },
        "wire_samples": int(len(samples)),
        "strength_thresholds": [float(v) for v in args.strength_thresholds],
        "results": reports,
        "guardrails": [
            "Wire proxy is evaluation-only.",
            "DoG maps/normalization/thresholds do not use the wire proxy.",
            "DoG magnitude is non-metric and context-local.",
            "No correction is recomputed or modified.",
        ],
    }

    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "dog_a.npy", dog_a)
    np.save(output / "dog_b.npy", dog_b)
    np.save(output / "dog_strength_consensus.npy", strength)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("K5-C.7 generic DoG polarity diagnostic complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
