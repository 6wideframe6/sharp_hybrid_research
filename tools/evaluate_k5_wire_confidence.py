#!/usr/bin/env python3
"""Evaluate saved K5-A.1 confidence against the known far-wire proxy.

Diagnostic only: no inference, no SHARP modification, no fusion/integration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


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

    mapped = np.zeros(overlap_shape, dtype=bool)
    ix0, iy0 = max(wx0, ox0), max(wy0, oy0)
    ix1, iy1 = min(wx1, ox1), min(wy1, oy1)
    if ix0 >= ix1 or iy0 >= iy1:
        return mapped

    mapped[iy0-oy0:iy1-oy0, ix0-ox0:ix1-ox0] = wire[
        iy0-wy0:iy1-wy0, ix0-wx0:ix1-wx0
    ]
    return mapped


def stats(values):
    a = np.asarray(values, dtype=np.float64)
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


def threshold_stats(values, thresholds):
    a = np.asarray(values, dtype=np.float64)
    return {f"gt_{t:g}": float(np.mean(a > t)) for t in thresholds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k5-output", type=Path, required=True)
    ap.add_argument("--wire-crop", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.10, 0.25, 0.50])
    args = ap.parse_args()

    k5_root = args.k5_output.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    source = json.loads((k5_root / "summary.json").read_text())
    overlap_box = parse_box(source["overlap_box_native"])
    overlap_shape = tuple(int(v) for v in source["overlap_shape"])
    wire = map_wire_proxy(args.wire_crop.resolve(), overlap_box, overlap_shape)
    if not wire.any():
        raise RuntimeError("Known wire proxy has no pixels inside overlap")

    wire3 = ndi.binary_dilation(wire, iterations=3)
    wire8 = ndi.binary_dilation(wire, iterations=8)
    groups = {
        "wire_exact": wire,
        "wire_dilated_3px": wire3,
        "local_background_ring_3_to_8px": wire8 & ~wire3,
        "global_background": ~wire3,
    }

    variants = []
    for variant_dir in sorted(k5_root.glob("fine_*_coarse_*")):
        if not variant_dir.is_dir():
            continue
        confidence = np.load(variant_dir / "final_confidence.npy")
        detail = np.load(variant_dir / "detail_consensus.npy")
        direction = np.load(variant_dir / "direction_confidence.npy")
        if confidence.shape != overlap_shape:
            raise ValueError(f"{variant_dir.name}: shape mismatch")

        result = {
            "variant": variant_dir.name,
            "pixels": {name: int(mask.sum()) for name, mask in groups.items()},
            "final_confidence": {},
            "detail_consensus": {},
            "direction_confidence": {},
            "thresholds": {},
            "enrichment": {},
        }

        for name, mask in groups.items():
            result["final_confidence"][name] = stats(confidence[mask])
            result["detail_consensus"][name] = stats(detail[mask])
            result["direction_confidence"][name] = stats(direction[mask])
            result["thresholds"][name] = threshold_stats(confidence[mask], args.thresholds)

        for t in args.thresholds:
            key = f"gt_{t:g}"
            wr = result["thresholds"]["wire_exact"][key]
            lr = result["thresholds"]["local_background_ring_3_to_8px"][key]
            gr = result["thresholds"]["global_background"][key]
            result["enrichment"][key] = {
                "wire_rate": wr,
                "local_background_rate": lr,
                "global_background_rate": gr,
                "wire_vs_local_background": None if lr == 0 else float(wr / lr),
                "wire_vs_global_background": None if gr == 0 else float(wr / gr),
            }

        variants.append(result)

    if not variants:
        raise RuntimeError("No fine_*_coarse_* variants found")

    output.mkdir(parents=True)
    summary = {
        "purpose": "K5-A.2 known-wire confidence evaluation only",
        "source_k5_output": str(k5_root),
        "wire_crop": str(args.wire_crop.resolve()),
        "overlap_box_native": list(overlap_box),
        "overlap_shape": list(overlap_shape),
        "wire_pixels": int(wire.sum()),
        "thresholds": [float(t) for t in args.thresholds],
        "variants": variants,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("K5-A.2 wire confidence evaluation complete")
    print(f"wire pixels: {int(wire.sum())}")
    print()
    for result in variants:
        print(f"===== {result['variant']} =====")
        for group in ("wire_exact", "local_background_ring_3_to_8px", "global_background"):
            s = result["final_confidence"][group]
            print(group, f"median={s['median']:.6f}", f"p90={s['p90']:.6f}", result["thresholds"][group])
        print("enrichment:")
        for key, value in result["enrichment"].items():
            print(" ", key, value)
        print()


if __name__ == "__main__":
    main()
