#!/usr/bin/env python3
"""K5-A.1 real-data diagnostic: context-consistent TinyViM detail confidence.

Pipeline:
1. exact native overlap of cached TinyViM contexts;
2. fine-scale gradient direction extraction;
3. multi-context cosine direction consensus;
4. per-context multi-scale high-frequency detail detector;
5. context-local robust detector normalization;
6. conservative min(detail_A, detail_B);
7. final confidence = direction confidence * detail consensus.

No SHARP depth is modified.
No TinyViM magnitude is interpreted as metric amplitude.
No correction integration or fusion is performed.
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

from k5.confidence import (
    combine_direction_and_detail,
    detail_consensus_min,
    multiscale_detail_strength,
    robust_normalize_detail,
)
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
        "p99": float(np.percentile(values, 99)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def save_direction_png(path: Path, dx, dy, confidence, valid):
    rgb = np.zeros((*dx.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.uint8(np.clip((dx + 1.0) * 0.5, 0, 1) * 255)
    rgb[..., 1] = np.uint8(np.clip((dy + 1.0) * 0.5, 0, 1) * 255)
    rgb[..., 2] = np.uint8(np.clip(confidence, 0, 1) * 255)
    rgb[~valid] = 0
    Image.fromarray(rgb).save(path)


def save_unit_png(path: Path, values):
    image = np.uint8(np.clip(values, 0.0, 1.0) * 255)
    Image.fromarray(image).save(path)


def save_signed_unit_png(path: Path, values):
    finite = np.isfinite(values)
    image = np.zeros(values.shape, dtype=np.uint8)
    mapped = np.clip(values * 0.5 + 0.5, 0.0, 1.0)
    image[finite] = np.uint8(mapped[finite] * 255)
    Image.fromarray(image).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-a", type=Path, required=True)
    parser.add_argument("--context-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument(
        "--fine-sigmas",
        type=float,
        nargs="+",
        default=[1.0],
        help="Fine native-pixel scales used for direction and detail detection.",
    )
    parser.add_argument(
        "--coarse-sigmas",
        type=float,
        nargs="+",
        default=[2.0, 4.0, 8.0],
        help="Coarse scales for high-frequency detail gating.",
    )
    parser.add_argument("--min-agreement", type=float, default=0.0)
    parser.add_argument("--detail-low-percentile", type=float, default=50.0)
    parser.add_argument("--detail-high-percentile", type=float, default=99.0)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    box_a, raw_a_full = load_context(args.context_a.resolve())
    box_b, raw_b_full = load_context(args.context_b.resolve())
    common, raw_a, raw_b = extract_native_overlap(
        raw_a_full, box_a, raw_b_full, box_b
    )

    fine_sigmas = sorted(set(float(v) for v in args.fine_sigmas))
    coarse_sigmas = sorted(set(float(v) for v in args.coarse_sigmas))

    pairs = [
        (fine, coarse)
        for fine in fine_sigmas
        for coarse in coarse_sigmas
        if coarse > fine
    ]
    if not pairs:
        raise ValueError("No valid fine/coarse sigma pairs: require coarse > fine")

    output.mkdir(parents=True, exist_ok=False)

    summary = {
        "purpose": "K5-A.1 TinyViM orientation + high-frequency detail confidence",
        "context_a": str(args.context_a.resolve()),
        "context_b": str(args.context_b.resolve()),
        "overlap_box_native": [common.x0, common.y0, common.x1, common.y1],
        "overlap_shape": list(common.shape),
        "min_agreement": float(args.min_agreement),
        "detail_normalization_percentiles": [
            float(args.detail_low_percentile),
            float(args.detail_high_percentile),
        ],
        "variants": [],
        "guardrails": [
            "No SHARP depth is modified.",
            "TinyViM gradient magnitude is used only as a context-local detail detector.",
            "Each context's detail detector is normalized independently.",
            "Final confidence is dimensionless and is not metric correction amplitude.",
            "No integration, Poisson solve, depth replacement or Gaussian update is performed.",
        ],
    }

    field_cache_a = {}
    field_cache_b = {}

    for fine, coarse in pairs:
        if fine not in field_cache_a:
            field_cache_a[fine] = extract_gradient_field(raw_a, sigma_px=fine)
            field_cache_b[fine] = extract_gradient_field(raw_b, sigma_px=fine)

        field_a = field_cache_a[fine]
        field_b = field_cache_b[fine]

        consensus = consensus_gradient_direction(
            [field_a, field_b],
            min_contexts=2,
            min_agreement=args.min_agreement,
        )

        detail_a_raw = multiscale_detail_strength(
            raw_a,
            fine_sigma_px=fine,
            coarse_sigma_px=coarse,
        )
        detail_b_raw = multiscale_detail_strength(
            raw_b,
            fine_sigma_px=fine,
            coarse_sigma_px=coarse,
        )

        detail_a = robust_normalize_detail(
            detail_a_raw,
            low_percentile=args.detail_low_percentile,
            high_percentile=args.detail_high_percentile,
        )
        detail_b = robust_normalize_detail(
            detail_b_raw,
            low_percentile=args.detail_low_percentile,
            high_percentile=args.detail_high_percentile,
        )
        detail_common = detail_consensus_min([detail_a, detail_b])

        final_confidence = combine_direction_and_detail(
            consensus.confidence,
            detail_common,
            valid=consensus.valid,
        )

        variant_dir = output / f"fine_{fine:g}_coarse_{coarse:g}"
        variant_dir.mkdir()

        np.save(variant_dir / "direction_x.npy", consensus.direction_x)
        np.save(variant_dir / "direction_y.npy", consensus.direction_y)
        np.save(variant_dir / "agreement.npy", consensus.agreement)
        np.save(variant_dir / "direction_confidence.npy", consensus.confidence)
        np.save(variant_dir / "detail_strength_a_raw.npy", detail_a_raw)
        np.save(variant_dir / "detail_strength_b_raw.npy", detail_b_raw)
        np.save(variant_dir / "detail_strength_a_norm.npy", detail_a)
        np.save(variant_dir / "detail_strength_b_norm.npy", detail_b)
        np.save(variant_dir / "detail_consensus.npy", detail_common)
        np.save(variant_dir / "final_confidence.npy", final_confidence)
        np.save(variant_dir / "valid.npy", consensus.valid)

        save_direction_png(
            variant_dir / "consensus_direction.png",
            consensus.direction_x,
            consensus.direction_y,
            final_confidence,
            consensus.valid,
        )
        save_signed_unit_png(
            variant_dir / "agreement.png",
            consensus.agreement,
        )
        save_unit_png(
            variant_dir / "direction_confidence.png",
            consensus.confidence,
        )
        save_unit_png(
            variant_dir / "detail_strength_a.png",
            detail_a,
        )
        save_unit_png(
            variant_dir / "detail_strength_b.png",
            detail_b,
        )
        save_unit_png(
            variant_dir / "detail_consensus.png",
            detail_common,
        )
        save_unit_png(
            variant_dir / "final_confidence.png",
            final_confidence,
        )

        both_valid = field_a.valid & field_b.valid
        agreement_values = consensus.agreement[both_valid]
        variant_summary = {
            "fine_sigma_px": fine,
            "coarse_sigma_px": coarse,
            "both_gradient_valid_fraction": float(both_valid.mean()),
            "direction_consensus_valid_fraction": float(consensus.valid.mean()),
            "positive_agreement_fraction": (
                float(np.mean(agreement_values > 0))
                if agreement_values.size
                else None
            ),
            "agreement": stats(agreement_values),
            "direction_confidence": stats(consensus.confidence[consensus.valid]),
            "detail_strength_a_norm": stats(detail_a),
            "detail_strength_b_norm": stats(detail_b),
            "detail_consensus": stats(detail_common),
            "final_confidence": stats(final_confidence),
            "fractions": {
                "detail_consensus_gt_0.10": float(np.mean(detail_common > 0.10)),
                "detail_consensus_gt_0.25": float(np.mean(detail_common > 0.25)),
                "detail_consensus_gt_0.50": float(np.mean(detail_common > 0.50)),
                "final_confidence_gt_0.10": float(np.mean(final_confidence > 0.10)),
                "final_confidence_gt_0.25": float(np.mean(final_confidence > 0.25)),
                "final_confidence_gt_0.50": float(np.mean(final_confidence > 0.50)),
            },
        }
        summary["variants"].append(variant_summary)

    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("K5-A.1 detail confidence diagnostic complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
