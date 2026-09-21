#!/usr/bin/env python3
"""
K4 radius-160 region-subset diagnosis ONLY.

No inference.
No modification of alignment.py.
No fusion/replacement/K5.

At the already-identified 160 px wire neighborhood, evaluate which disconnected
fitting regions are responsible for the remaining held-out p90 tail.

For every tested region subset:
- restrict candidate anchors to distance <= 160 px from the real far_1480_192
  wire proxy
- retain only selected fitting-region IDs
- call the existing K4 align_affine() implementation
- report held-out median/p90, fold scale variation, slope stability
- independently report wire raw-range support/extrapolation

This is a diagnostic ablation. Region IDs are connected fitting interiors,
not semantic surface ownership.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.sharp_hybrid_research.alignment import (
    AlignmentConfig,
    _spread,
    align_affine,
)

DEFAULT_CONTEXT = ROOT / "results/sharp_hybrid/k4_context/far_1480_512"
DEFAULT_OLD_CROP = ROOT / "experiments/model_compare/outputs/depthart_tiny_512/far_1480_192"
DEFAULT_OUTPUT = ROOT / "results/sharp_hybrid/k4_radius160_regions/far_1480_512"

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


def raw_support(raw, selected_mask, wire):
    anchors = raw[selected_mask & np.isfinite(raw)].astype(np.float64)
    wire_values = raw[wire & np.isfinite(raw)].astype(np.float64)

    if not len(anchors) or not len(wire_values):
        return {"available": False}

    span = max(_spread(anchors), np.finfo(float).tiny)
    lo = float(anchors.min())
    hi = float(anchors.max())

    distance = np.maximum.reduce(
        (lo - wire_values, wire_values - hi, np.zeros_like(wire_values))
    )

    wp10, wp90 = np.percentile(wire_values, [10, 90])
    ap10, ap90 = np.percentile(anchors, [10, 90])

    overlap = max(0.0, min(wp90, ap90) - max(wp10, ap10))
    wire_band = max(float(wp90 - wp10), np.finfo(float).tiny)

    return {
        "available": True,
        "anchor_raw_range": [lo, hi],
        "anchor_raw_p10_p90": [float(ap10), float(ap90)],
        "wire_raw_range": [float(wire_values.min()), float(wire_values.max())],
        "wire_raw_p10_p90": [float(wp10), float(wp90)],
        "wire_p10_p90_overlap_fraction": float(overlap / wire_band),
        "outside_anchor_range_fraction": float(np.mean(distance > 0)),
        "p90_distance_in_anchor_spreads": float(np.percentile(distance, 90) / span),
        "max_distance_in_anchor_spreads": float(distance.max() / span),
    }


def compact(result):
    r = result["report"]
    return {
        "status": r["status"],
        "accepted": r["accepted"],
        "a": r["a"],
        "b": r["b"],
        "positive_slope": r["positive_slope"],
        "usable_anchors": r["usable_anchors"],
        "spatial_blocks": r["spatial_blocks"],
        "target_depth_bins": r["target_depth_bins"],
        "surface_interior_groups": r["surface_interior_groups"],
        "heldout_median": r["heldout_normalized_median"],
        "heldout_p90": r["heldout_normalized_p90"],
        "scale_variation": r["fold_scale_variation"],
        "attempted_fold_slopes": r.get("attempted_fold_slopes"),
        "reasons": r["reasons"],
        "folds": [
            {
                "fold": f["fold"],
                "a": f.get("a"),
                "b": f.get("b"),
                "median": f.get("median"),
                "p90": f.get("p90"),
                "usable": f.get("usable"),
                "positive_slope": f.get("positive_slope"),
                "reason": f.get("reason"),
            }
            for f in r["folds"]
        ],
    }


def region_raw_stats(raw, target, candidate, regions, ids):
    rows = []
    for rid in ids:
        mask = candidate & (regions == rid) & np.isfinite(raw) & np.isfinite(target) & (target > 0)
        rv = raw[mask].astype(np.float64)
        tv = target[mask].astype(np.float64)
        if not len(rv):
            continue
        rows.append({
            "region": int(rid),
            "candidate_pixels": int(len(rv)),
            "raw_min": float(rv.min()),
            "raw_max": float(rv.max()),
            "raw_p10": float(np.percentile(rv, 10)),
            "raw_p90": float(np.percentile(rv, 90)),
            "target_min": float(tv.min()),
            "target_max": float(tv.max()),
            "target_p10": float(np.percentile(tv, 10)),
            "target_p90": float(np.percentile(tv, 90)),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--old-crop", type=Path, default=DEFAULT_OLD_CROP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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

    wire = map_old_proxy(source, old_crop, raw.shape)
    distance = ndi.distance_transform_edt(~wire)
    local = candidate & (distance <= RADIUS)

    region_ids = sorted(
        int(v) for v in np.unique(regions[local]) if int(v) > 0
    )

    if not region_ids:
        raise RuntimeError("No fitting regions inside radius")

    # We know from prior diagnosis that these are the currently interesting small
    # regions, but build subsets from actual local region IDs rather than hard-code
    # assumptions about availability.
    suspected = [rid for rid in (2, 68, 79) if rid in region_ids]
    stable_small = [rid for rid in (24, 80) if rid in region_ids]
    primary = [rid for rid in (1,) if rid in region_ids]

    subsets = []

    def add(name, ids):
        ids = sorted(set(ids))
        if ids and ids not in [x["regions"] for x in subsets]:
            subsets.append({"name": name, "regions": ids})

    add("all_local_regions", region_ids)
    add("primary_only", primary)
    add("primary_plus_stable_small", primary + stable_small)
    add("drop_all_suspected", [r for r in region_ids if r not in suspected])

    for rid in suspected:
        add(f"drop_R{rid}", [r for r in region_ids if r != rid])

    # All combinations of keeping/dropping suspected regions while always keeping
    # the non-suspected local regions. This gives the full 2^N diagnostic lattice.
    base = [r for r in region_ids if r not in suspected]
    for keep_count in range(len(suspected) + 1):
        for keep in itertools.combinations(suspected, keep_count):
            add(
                "base_plus_" + ("none" if not keep else "_".join(f"R{x}" for x in keep)),
                base + list(keep),
            )

    results = []

    for subset in subsets:
        ids = subset["regions"]
        mask = local & np.isin(regions, ids)

        fit = align_affine(
            raw,
            target,
            mask,
            raw_fit=raw_fit,
            target_fit=target_fit,
            regions=regions,
            foreground_mask=wire,
            config=config,
        )

        item = {
            "name": subset["name"],
            "regions": ids,
            "candidate_pixels": int(mask.sum()),
            "alignment": compact(fit),
            "wire_raw_support": raw_support(
                raw,
                fit["selected_anchor_mask"],
                wire,
            ),
        }
        results.append(item)

    stats = region_raw_stats(raw, target, local, regions, region_ids)

    # Sort useful summary: passing p90 first, then median, then support overlap.
    def score(item):
        a = item["alignment"]
        s = item["wire_raw_support"]
        p90 = a["heldout_p90"]
        med = a["heldout_median"]
        overlap = s.get("wire_p10_p90_overlap_fraction", -1.0)
        return (
            999 if p90 is None else p90,
            999 if med is None else med,
            -overlap,
        )

    ranked = sorted(results, key=score)

    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "purpose": (
            "radius-160 region subset ablation; no inference/fusion/replacement/K5"
        ),
        "radius_native_px": RADIUS,
        "local_region_ids": region_ids,
        "per_region_raw_target_support": stats,
        "results": results,
        "ranked_subset_names": [r["name"] for r in ranked],
        "guardrails": [
            "Region IDs are connected fitting interiors, not semantic surface ownership.",
            "A numerically passing subset is diagnostic evidence only.",
            "Wire raw support must remain adequate after region exclusion.",
            "Do not use excluded-region sets as replacement masks.",
        ],
    }
    save_json(output / "summary.json", summary)

    print("K4 radius-160 region-subset diagnostic complete")
    print(f"local regions: {region_ids}")
    print()

    print("PER-REGION RAW/TARGET SUPPORT")
    for row in stats:
        print(json.dumps(row, sort_keys=True))
    print()

    print("SUBSETS (ranked by held-out p90)")
    for item in ranked:
        a = item["alignment"]
        s = item["wire_raw_support"]
        print(
            f"{item['name']}: regions={item['regions']} "
            f"anchors={a['usable_anchors']} blocks={a['spatial_blocks']} "
            f"a={a['a']} median={a['heldout_median']} p90={a['heldout_p90']} "
            f"scale_var={a['scale_variation']} "
            f"outside={s.get('outside_anchor_range_fraction')} "
            f"overlap={s.get('wire_p10_p90_overlap_fraction')} "
            f"p90_extrap={s.get('p90_distance_in_anchor_spreads')} "
            f"status={a['status']} reasons={a['reasons']}"
        )


if __name__ == "__main__":
    main()
