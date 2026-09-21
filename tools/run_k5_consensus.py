#!/usr/bin/env python3
"""K5-A real-data diagnostic: TinyViM multi-context gradient consensus only.

Reads existing cached TinyViM K4 context predictions and writes K5-A consensus
artifacts. It does NOT modify SHARP depth, assign metric amplitude, integrate a
correction, or perform fusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from k5.consensus import consensus_gradient_direction
from k5.detail_signal import extract_gradient_field
from k5.overlap import extract_native_overlap
from k5.types import NativeBox


def load_context(folder: Path):
    report = json.loads((folder / "alignment.json").read_text())
    box = NativeBox.from_sequence(report["provenance"]["box_native_half_open"])
    raw = np.load(folder / "raw_tinyvim_relative.npy")
    if raw.shape != box.shape:
        raise ValueError(
            f"{folder}: raw shape {raw.shape} does not match native box {box.shape}"
        )
    return box, raw


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "mean": float(np.mean(values)),
    }


def save_direction_png(path: Path, dx, dy, confidence, valid):
    rgb = np.zeros((*dx.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.uint8(np.clip((dx + 1.0) * 0.5, 0, 1) * 255)
    rgb[..., 1] = np.uint8(np.clip((dy + 1.0) * 0.5, 0, 1) * 255)
    rgb[..., 2] = np.uint8(np.clip(confidence, 0, 1) * 255)
    rgb[~valid] = 0
    Image.fromarray(rgb).save(path)


def save_scalar_png(path: Path, values, valid, *, signed=False):
    image = np.zeros(values.shape, dtype=np.uint8)
    mapped = (
        np.clip(values * 0.5 + 0.5, 0, 1)
        if signed
        else np.clip(values, 0, 1)
    )
    image[valid] = np.uint8(mapped[valid] * 255)
    Image.fromarray(image).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-a", type=Path, required=True)
    parser.add_argument("--context-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--min-agreement", type=float, default=0.0)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    box_a, raw_a_full = load_context(args.context_a.resolve())
    box_b, raw_b_full = load_context(args.context_b.resolve())
    common, raw_a, raw_b = extract_native_overlap(
        raw_a_full, box_a, raw_b_full, box_b
    )

    output.mkdir(parents=True, exist_ok=False)

    summary = {
        "purpose": "K5-A TinyViM gradient-orientation consensus only",
        "context_a": str(args.context_a.resolve()),
        "context_b": str(args.context_b.resolve()),
        "overlap_box_native": [common.x0, common.y0, common.x1, common.y1],
        "overlap_shape": list(common.shape),
        "min_agreement": float(args.min_agreement),
        "scales": [],
        "guardrails": [
            "No SHARP depth is modified.",
            "TinyViM gradient magnitude is not treated as metric amplitude.",
            "Consensus confidence is orientation-only and is not a replacement mask.",
            "No integration, Poisson solve, depth replacement or Gaussian update is performed.",
        ],
    }

    for sigma in args.sigmas:
        field_a = extract_gradient_field(raw_a, sigma_px=sigma)
        field_b = extract_gradient_field(raw_b, sigma_px=sigma)
        consensus = consensus_gradient_direction(
            [field_a, field_b],
            min_contexts=2,
            min_agreement=args.min_agreement,
        )

        scale_dir = output / f"sigma_{sigma:g}"
        scale_dir.mkdir()

        np.save(scale_dir / "direction_x.npy", consensus.direction_x)
        np.save(scale_dir / "direction_y.npy", consensus.direction_y)
        np.save(scale_dir / "agreement.npy", consensus.agreement)
        np.save(scale_dir / "confidence.npy", consensus.confidence)
        np.save(scale_dir / "valid.npy", consensus.valid)

        save_direction_png(
            scale_dir / "consensus_direction.png",
            consensus.direction_x,
            consensus.direction_y,
            consensus.confidence,
            consensus.valid,
        )
        save_scalar_png(
            scale_dir / "agreement.png",
            consensus.agreement,
            np.isfinite(consensus.agreement),
            signed=True,
        )
        save_scalar_png(
            scale_dir / "confidence.png",
            consensus.confidence,
            consensus.valid,
        )

        both_valid = field_a.valid & field_b.valid
        agreement_values = consensus.agreement[both_valid]
        summary["scales"].append(
            {
                "sigma_px": float(sigma),
                "field_a_valid_fraction": float(field_a.valid.mean()),
                "field_b_valid_fraction": float(field_b.valid.mean()),
                "both_valid_fraction": float(both_valid.mean()),
                "consensus_valid_fraction": float(consensus.valid.mean()),
                "agreement": stats(agreement_values),
                "confidence": stats(consensus.confidence[consensus.valid]),
                "positive_agreement_fraction_among_both_valid": (
                    float(np.mean(agreement_values > 0))
                    if agreement_values.size
                    else None
                ),
            }
        )

    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("K5-A consensus diagnostic complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
