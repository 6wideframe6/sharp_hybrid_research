#!/usr/bin/env python3
"""
K4 overlap/context-consistency diagnosis for TinyViM raw depth.

No inference.
No SHARP modification.
No fusion/replacement/K5.

Question:
The same far-wire pixels receive very different absolute TinyViM raw values
under far_1480_512 vs far_1495_512 context. Is that context dependence mostly
an affine value remapping, while local depth gradients/detail remain stable?

This script compares the two already-cached 512 TinyViM predictions in their
native-image overlap.

Diagnostics:
1. Robust raw-value affine transfer:
       r_1495 ~= a * r_1480 + b
   fit on NON-WIRE overlap pixels, evaluated on:
   - non-wire overlap
   - exact far_1480_192 wire proxy
   - a small dilated wire neighborhood

2. Gradient consistency at Gaussian scales sigma = 0,1,2,4 px:
   - best positive scalar between gradient fields, fitted on non-wire overlap
   - gradient-vector cosine similarity
   - normalized vector residual
   - gx/gy/magnitude Spearman correlations
   evaluated on non-wire overlap and wire neighborhood.

Interpretation:
- Good affine raw transfer would mean context changes mostly scale/offset.
- Poor raw transfer but good gradient transfer would support using TinyViM as
  a differential/detail source rather than absolute metric calibration.
- Poor gradient transfer too would mean the wire detail itself is context
  unstable and should not be trusted without a stronger consistency mechanism.

All results are diagnostic only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.sharp_hybrid_research.alignment import (
    AlignmentConfig,
    robust_affine,
    _spread,
)

DEFAULT_A = ROOT / "results/sharp_hybrid/k4_context/far_1480_512"
DEFAULT_B = ROOT / "results/sharp_hybrid/k4_context/far_1495_512"
DEFAULT_WIRE = ROOT / "experiments/model_compare/outputs/depthart_tiny_512/far_1480_192"
DEFAULT_OUTPUT = ROOT / "results/sharp_hybrid/k4_context_overlap/far_1480_vs_1495"

SIGMAS = (0.0, 1.0, 2.0, 4.0)

def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\\n")


def load_report(folder: Path):
    return json.loads((folder / "alignment.json").read_text())


def overlap(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x0 >= x1 or y0 >= y1:
        raise RuntimeError(f"No overlap between {a} and {b}")
    return [x0, y0, x1, y1]


def roi(array, source_box, box):
    return array[
        box[1] - source_box[1] : box[3] - source_box[1],
        box[0] - source_box[0] : box[2] - source_box[0],
    ]


def map_wire_to_overlap(old_crop: Path, overlap_box, shape):
    meta = json.loads((old_crop / "metadata.json").read_text())
    box = list(map(int, meta["box"]))
    mask = np.asarray(
        Image.open(old_crop / "rgb_proxy_mask.png").convert("L")
    ) > 0

    ox0, oy0, ox1, oy1 = overlap_box
    wx0, wy0, wx1, wy1 = box

    ix0, iy0 = max(ox0, wx0), max(oy0, wy0)
    ix1, iy1 = min(ox1, wx1), min(oy1, wy1)

    out = np.zeros(shape, dtype=bool)
    if ix0 >= ix1 or iy0 >= iy1:
        return out, {
            "wire_native_box": box,
            "overlap_intersection": None,
            "wire_pixels_in_overlap": 0,
        }

    out[
        iy0 - oy0 : iy1 - oy0,
        ix0 - ox0 : ix1 - ox0,
    ] = mask[
        iy0 - wy0 : iy1 - wy0,
        ix0 - wx0 : ix1 - wx0,
    ]

    return out, {
        "wire_native_box": box,
        "overlap_intersection": [ix0, iy0, ix1, iy1],
        "wire_pixels_in_overlap": int(out.sum()),
    }


def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {
            "n": 0,
            "median": None,
            "p90": None,
            "mean": None,
            "max": None,
        }
    return {
        "n": int(len(a)),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "mean": float(np.mean(a)),
        "max": float(np.max(a)),
    }


def signed_summary(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0}
    return {
        "n": int(len(a)),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
    }


def safe_spearman(a, b, mask):
    av = np.asarray(a, np.float64)[mask]
    bv = np.asarray(b, np.float64)[mask]
    good = np.isfinite(av) & np.isfinite(bv)
    av, bv = av[good], bv[good]
    if len(av) < 3:
        return None
    r = spearmanr(av, bv)
    if not np.isfinite(r.statistic):
        return None
    return {
        "rho": float(r.statistic),
        "p": float(r.pvalue),
        "n": int(len(av)),
    }


def affine_transfer(a, b, fit_mask, eval_masks):
    x = a[fit_mask].astype(np.float64)
    y = b[fit_mask].astype(np.float64)

    weights = np.ones(len(x), dtype=np.float64)
    cfg = AlignmentConfig(
        sigma=0,
        border=0,
        shoulder=0,
        min_region_pixels=1,
        block_size=32,
        depth_bins=4,
        bucket_cap=64,
        min_anchors=16,
        min_blocks=3,
        folds=3,
    )
    scale, offset = robust_affine(x, y, weights, cfg)

    denom = max(_spread(y), np.finfo(np.float64).tiny)
    out = {
        "a": float(scale),
        "b": float(offset),
        "positive_slope": bool(scale > 0),
        "fit_target_p10_p90_spread": float(denom),
        "evaluations": {},
    }

    pred = scale * a + offset

    for name, mask in eval_masks.items():
        m = mask & np.isfinite(a) & np.isfinite(b)
        if not m.any():
            out["evaluations"][name] = {"n": 0}
            continue
        signed = (pred[m] - b[m]) / denom
        out["evaluations"][name] = {
            "signed": signed_summary(signed),
            "absolute": summarize(np.abs(signed)),
            "raw_a": summarize(a[m]),
            "raw_b": summarize(b[m]),
        }

    return out


def gradient_field(raw, sigma):
    v = raw.astype(np.float64)
    if sigma > 0:
        v = ndi.gaussian_filter(v, sigma=sigma, mode="nearest")
    gy, gx = np.gradient(v)
    mag = np.hypot(gx, gy)
    return gx, gy, mag


def fit_gradient_scale(gax, gay, gbx, gby, mask):
    ax = gax[mask]
    ay = gay[mask]
    bx = gbx[mask]
    by = gby[mask]

    finite = (
        np.isfinite(ax)
        & np.isfinite(ay)
        & np.isfinite(bx)
        & np.isfinite(by)
    )
    ax, ay, bx, by = ax[finite], ay[finite], bx[finite], by[finite]

    if not len(ax):
        return None

    denom = np.sum(ax * ax + ay * ay)
    if denom <= np.finfo(float).tiny:
        return None

    scale = float(np.sum(ax * bx + ay * by) / denom)
    return scale


def gradient_eval(gax, gay, gbx, gby, scale, mask, ref_scale):
    ax = gax[mask]
    ay = gay[mask]
    bx = gbx[mask]
    by = gby[mask]

    finite = (
        np.isfinite(ax)
        & np.isfinite(ay)
        & np.isfinite(bx)
        & np.isfinite(by)
    )
    ax, ay, bx, by = ax[finite], ay[finite], bx[finite], by[finite]

    if not len(ax):
        return {"n": 0}

    amag = np.hypot(ax, ay)
    bmag = np.hypot(bx, by)

    dot = ax * bx + ay * by
    denom_cos = amag * bmag
    strong = denom_cos > np.finfo(float).eps

    cosine = np.full(len(ax), np.nan, dtype=np.float64)
    cosine[strong] = dot[strong] / denom_cos[strong]

    residual = np.hypot(scale * ax - bx, scale * ay - by)
    normalized = residual / max(ref_scale, np.finfo(float).tiny)

    # Direction agreement is only meaningful where both gradients are nontrivial.
    mag_cut_a = np.percentile(amag, 50)
    mag_cut_b = np.percentile(bmag, 50)
    direction_mask = strong & (amag >= mag_cut_a) & (bmag >= mag_cut_b)

    return {
        "n": int(len(ax)),
        "gradient_scale_a_to_b": float(scale),
        "vector_residual_normalized": summarize(normalized),
        "cosine_all_nonzero": summarize(cosine[strong]),
        "cosine_strong_half": summarize(cosine[direction_mask]),
        "fraction_cosine_positive_strong_half": (
            float(np.mean(cosine[direction_mask] > 0))
            if direction_mask.any()
            else None
        ),
        "magnitude_a": summarize(amag),
        "magnitude_b": summarize(bmag),
    }


def make_visual(a, b, wire, output_path):
    # Show robust-normalized raw A, raw B, and signed difference after median/IQR normalization.
    def norm(v):
        lo, hi = np.percentile(v[np.isfinite(v)], [5, 95])
        if hi <= lo:
            hi = lo + 1.0
        return np.clip((v - lo) / (hi - lo), 0, 1)

    na = norm(a)
    nb = norm(b)
    diff = np.clip((na - nb) * 0.5 + 0.5, 0, 1)

    h, w = a.shape
    canvas = Image.new("RGB", (w * 3, h), "black")
    for i, arr in enumerate((na, nb, diff)):
        gray = np.uint8(arr * 255)
        rgb = np.repeat(gray[..., None], 3, axis=2)
        if i == 2:
            rgb[..., 0] = np.uint8(np.clip((na - nb), 0, 1) * 255)
            rgb[..., 2] = np.uint8(np.clip((nb - na), 0, 1) * 255)
            rgb[..., 1] = np.uint8((1 - np.clip(np.abs(na - nb), 0, 1)) * 80)
        rgb[wire] = (255, 0, 255)
        canvas.paste(Image.fromarray(rgb), (i * w, 0))

    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), "far_1480 context", fill="white")
    draw.text((w + 5, 5), "far_1495 context", fill="white")
    draw.text((2 * w + 5, 5), "normalized context difference", fill="white")
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-a", type=Path, default=DEFAULT_A)
    parser.add_argument("--context-b", type=Path, default=DEFAULT_B)
    parser.add_argument("--wire-crop", type=Path, default=DEFAULT_WIRE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    a_dir = args.context_a.resolve()
    b_dir = args.context_b.resolve()
    wire_dir = args.wire_crop.resolve()
    output = args.output.resolve()

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    report_a = load_report(a_dir)
    report_b = load_report(b_dir)

    box_a = list(map(int, report_a["provenance"]["box_native_half_open"]))
    box_b = list(map(int, report_b["provenance"]["box_native_half_open"]))
    common = overlap(box_a, box_b)

    raw_a_full = np.load(a_dir / "raw_tinyvim_relative.npy")
    raw_b_full = np.load(b_dir / "raw_tinyvim_relative.npy")

    raw_a = roi(raw_a_full, box_a, common)
    raw_b = roi(raw_b_full, box_b, common)

    if raw_a.shape != raw_b.shape:
        raise RuntimeError("Overlap extraction shape mismatch")

    wire, wire_meta = map_wire_to_overlap(
        wire_dir,
        common,
        raw_a.shape,
    )

    finite = np.isfinite(raw_a) & np.isfinite(raw_b)

    # Ignore one-pixel derivative border.
    interior = finite.copy()
    interior[:2] = False
    interior[-2:] = False
    interior[:, :2] = False
    interior[:, -2:] = False

    wire_dilated = ndi.binary_dilation(wire, iterations=3)
    nonwire = interior & ~wire_dilated
    wire_exact = interior & wire
    wire_neighborhood = interior & wire_dilated

    if nonwire.sum() < 100:
        raise RuntimeError("Insufficient non-wire overlap support")

    eval_masks = {
        "nonwire_overlap": nonwire,
        "wire_exact": wire_exact,
        "wire_neighborhood_3px": wire_neighborhood,
    }

    raw_transfer = affine_transfer(
        raw_a,
        raw_b,
        nonwire,
        eval_masks,
    )

    gradient_results = []

    for sigma in SIGMAS:
        gax, gay, amag = gradient_field(raw_a, sigma)
        gbx, gby, bmag = gradient_field(raw_b, sigma)

        scale = fit_gradient_scale(
            gax, gay, gbx, gby, nonwire
        )

        if scale is None:
            gradient_results.append(
                {"sigma": sigma, "status": "degenerate"}
            )
            continue

        # Normalization scale from target-context non-wire gradient magnitude.
        ref = np.percentile(
            bmag[nonwire & np.isfinite(bmag)],
            90,
        )

        item = {
            "sigma": sigma,
            "gradient_scale_a_to_b_fitted_nonwire": float(scale),
            "positive_scale": bool(scale > 0),
            "reference_b_nonwire_gradient_p90": float(ref),
            "evaluations": {},
            "spearman": {},
        }

        for name, mask in eval_masks.items():
            item["evaluations"][name] = gradient_eval(
                gax, gay, gbx, gby, scale, mask, ref
            )
            item["spearman"][name] = {
                "gx": safe_spearman(gax, gbx, mask),
                "gy": safe_spearman(gay, gby, mask),
                "magnitude": safe_spearman(amag, bmag, mask),
            }

        gradient_results.append(item)

    output.mkdir(parents=True, exist_ok=False)

    make_visual(
        raw_a,
        raw_b,
        wire,
        output / "context_overlap_raw_comparison.png",
    )

    summary = {
        "purpose": (
            "TinyViM context-consistency diagnostic: absolute raw vs local gradients; "
            "no inference/fusion/replacement/K5"
        ),
        "context_a": str(a_dir),
        "context_b": str(b_dir),
        "box_a": box_a,
        "box_b": box_b,
        "overlap_box_native": common,
        "overlap_shape": list(raw_a.shape),
        "wire": wire_meta,
        "mask_counts": {
            "nonwire_overlap": int(nonwire.sum()),
            "wire_exact": int(wire_exact.sum()),
            "wire_neighborhood_3px": int(wire_neighborhood.sum()),
        },
        "raw_affine_transfer_a_to_b": raw_transfer,
        "gradient_consistency": gradient_results,
        "guardrails": [
            "Raw transfer is fitted only on non-wire overlap and evaluated on wire separately.",
            "Gradient scale is fitted only on non-wire overlap and evaluated on wire separately.",
            "Good gradient consistency would support differential use only; it would not establish metric depth.",
            "Poor gradient consistency means the TinyViM wire detail itself is context-sensitive.",
        ],
    }

    save_json(output / "summary.json", summary)

    print("K4 TinyViM context-overlap diagnostic complete")
    print(f"A: {a_dir}")
    print(f"B: {b_dir}")
    print(f"overlap: {common} shape={raw_a.shape}")
    print(f"wire pixels in overlap: {wire_meta['wire_pixels_in_overlap']}")
    print()

    print("RAW AFFINE TRANSFER A->B")
    print(json.dumps(raw_transfer, indent=2))
    print()

    print("GRADIENT CONSISTENCY")
    for item in gradient_results:
        print(f"===== sigma {item['sigma']} =====")
        if item.get("status") == "degenerate":
            print("degenerate")
            continue
        print(
            "scale:",
            item["gradient_scale_a_to_b_fitted_nonwire"],
            "positive:",
            item["positive_scale"],
        )
        for name in ("nonwire_overlap", "wire_exact", "wire_neighborhood_3px"):
            ev = item["evaluations"][name]
            sp = item["spearman"][name]
            print(
                name,
                "res_med=",
                ev["vector_residual_normalized"]["median"],
                "res_p90=",
                ev["vector_residual_normalized"]["p90"],
                "cos_strong_med=",
                ev["cosine_strong_half"]["median"],
                "cos_pos_frac=",
                ev["fraction_cosine_positive_strong_half"],
                "rho_mag=",
                None if sp["magnitude"] is None else sp["magnitude"]["rho"],
            )
        print()


if __name__ == "__main__":
    main()
