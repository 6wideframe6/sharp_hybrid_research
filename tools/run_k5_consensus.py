#!/usr/bin/env python3
"""K5-A.1b real-data diagnostic with boundary-safe full-context filtering.

Pipeline:
1. load each full TinyViM context;
2. Gaussian filtering / gradients on each FULL context;
3. crop computed fields to exact native overlap;
4. exclude pixels whose Gaussian+gradient footprint touches either source
   context boundary;
5. independently normalize each context's detail detector over that same
   halo-safe physical overlap;
6. multi-context direction/detail consensus;
7. final confidence = direction confidence * detail consensus.

No SHARP depth is modified.
No TinyViM magnitude is interpreted as metric amplitude.
No correction integration or fusion is performed.
"""

from __future__ import annotations

import argparse
import json
import math
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
from k5.overlap import (
    crop_to_native_box,
    native_halo_support_mask,
    native_overlap,
)
from k5.types import GradientField, NativeBox


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


def gaussian_gradient_halo_px(sigma_px: float) -> int:
    """Strict raw-data support radius for Gaussian filter + np.gradient.

    scipy.ndimage.gaussian_filter uses truncate=4 by default. The following
    np.gradient central difference needs one additional neighbouring filtered
    sample.
    """
    if sigma_px < 0:
        raise ValueError("sigma_px must be non-negative")
    gaussian_radius = int(math.ceil(4.0 * float(sigma_px)))
    return gaussian_radius + 1


def crop_gradient_field(
    field: GradientField,
    source_box: NativeBox,
    target_box: NativeBox,
    *,
    valid_mask: np.ndarray,
) -> GradientField:
    """Crop a full-context field and apply common halo-safe validity."""
    def c(a):
        return np.array(
            crop_to_native_box(a, source_box, target_box),
            copy=True,
        )

    valid = c(field.valid).astype(bool) & valid_mask
    direction_x = c(field.direction_x)
    direction_y = c(field.direction_y)
    direction_x[~valid] = 0.0
    direction_y[~valid] = 0.0

    return GradientField(
        gx=c(field.gx),
        gy=c(field.gy),
        magnitude=c(field.magnitude),
        direction_x=direction_x,
        direction_y=direction_y,
        valid=valid,
        sigma_px=field.sigma_px,
    )


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
    )
    parser.add_argument(
        "--coarse-sigmas",
        type=float,
        nargs="+",
        default=[2.0, 4.0, 8.0],
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
    common = native_overlap(box_a, box_b)

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
        "purpose": "K5-A.1b boundary-safe TinyViM orientation + detail confidence",
        "context_a": str(args.context_a.resolve()),
        "context_b": str(args.context_b.resolve()),
        "overlap_box_native": [common.x0, common.y0, common.x1, common.y1],
        "overlap_shape": list(common.shape),
        "filtering_order": "full context -> filter/gradient -> overlap crop",
        "min_agreement": float(args.min_agreement),
        "detail_normalization_percentiles": [
            float(args.detail_low_percentile),
            float(args.detail_high_percentile),
        ],
        "variants": [],
        "guardrails": [
            "Gaussian filtering and gradients are computed before overlap cropping.",
            "Pixels without full source-context filter support are invalidated.",
            "Detail normalization uses only the same halo-safe physical overlap in both contexts.",
            "No SHARP depth is modified.",
            "TinyViM gradient magnitude is used only as a context-local detail detector.",
            "Final confidence is dimensionless and is not metric correction amplitude.",
            "No integration, Poisson solve, depth replacement or Gaussian update is performed.",
        ],
    }

    full_field_cache_a = {}
    full_field_cache_b = {}
    full_detail_cache_a = {}
    full_detail_cache_b = {}

    for fine, coarse in pairs:
        halo_px = max(
            gaussian_gradient_halo_px(fine),
            gaussian_gradient_halo_px(coarse),
        )
        safe_a = native_halo_support_mask(
            box_a, common, halo_px=halo_px
        )
        safe_b = native_halo_support_mask(
            box_b, common, halo_px=halo_px
        )
        safe_both = safe_a & safe_b

        if not safe_both.any():
            raise RuntimeError(
                f"No halo-safe overlap for fine={fine:g}, coarse={coarse:g}, "
                f"halo={halo_px}px"
            )

        if fine not in full_field_cache_a:
            full_field_cache_a[fine] = extract_gradient_field(
                raw_a_full, sigma_px=fine
            )
            full_field_cache_b[fine] = extract_gradient_field(
                raw_b_full, sigma_px=fine
            )

        field_a = crop_gradient_field(
            full_field_cache_a[fine],
            box_a,
            common,
            valid_mask=safe_both,
        )
        field_b = crop_gradient_field(
            full_field_cache_b[fine],
            box_b,
            common,
            valid_mask=safe_both,
        )

        consensus = consensus_gradient_direction(
            [field_a, field_b],
            min_contexts=2,
            min_agreement=args.min_agreement,
        )

        key = (fine, coarse)
        if key not in full_detail_cache_a:
            full_detail_cache_a[key] = multiscale_detail_strength(
                raw_a_full,
                fine_sigma_px=fine,
                coarse_sigma_px=coarse,
            )
            full_detail_cache_b[key] = multiscale_detail_strength(
                raw_b_full,
                fine_sigma_px=fine,
                coarse_sigma_px=coarse,
            )

        detail_a_raw = np.array(
            crop_to_native_box(
                full_detail_cache_a[key], box_a, common
            ),
            copy=True,
        )
        detail_b_raw = np.array(
            crop_to_native_box(
                full_detail_cache_b[key], box_b, common
            ),
            copy=True,
        )

        detail_a = robust_normalize_detail(
            detail_a_raw,
            low_percentile=args.detail_low_percentile,
            high_percentile=args.detail_high_percentile,
            valid_mask=safe_both,
        )
        detail_b = robust_normalize_detail(
            detail_b_raw,
            low_percentile=args.detail_low_percentile,
            high_percentile=args.detail_high_percentile,
            valid_mask=safe_both,
        )
        detail_common = detail_consensus_min([detail_a, detail_b])

        final_valid = consensus.valid & safe_both
        final_confidence = combine_direction_and_detail(
            consensus.confidence,
            detail_common,
            valid=final_valid,
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
        np.save(variant_dir / "valid.npy", final_valid)
        np.save(variant_dir / "halo_safe.npy", safe_both)

        save_direction_png(
            variant_dir / "consensus_direction.png",
            consensus.direction_x,
            consensus.direction_y,
            final_confidence,
            final_valid,
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
        save_unit_png(
            variant_dir / "halo_safe.png",
            safe_both.astype(np.float64),
        )

        both_valid = field_a.valid & field_b.valid
        agreement_values = consensus.agreement[both_valid]

        variant_summary = {
            "fine_sigma_px": fine,
            "coarse_sigma_px": coarse,
            "strict_halo_px": int(halo_px),
            "halo_safe_fraction": float(safe_both.mean()),
            "halo_safe_pixels": int(safe_both.sum()),
            "both_gradient_valid_fraction": float(both_valid.mean()),
            "direction_consensus_valid_fraction": float(final_valid.mean()),
            "positive_agreement_fraction": (
                float(np.mean(agreement_values > 0))
                if agreement_values.size
                else None
            ),
            "agreement": stats(agreement_values),
            "direction_confidence": stats(
                consensus.confidence[final_valid]
            ),
            "detail_strength_a_norm": stats(detail_a[safe_both]),
            "detail_strength_b_norm": stats(detail_b[safe_both]),
            "detail_consensus": stats(detail_common[safe_both]),
            "final_confidence": stats(final_confidence[safe_both]),
            "fractions": {
                "detail_consensus_gt_0.10": float(
                    np.mean(detail_common[safe_both] > 0.10)
                ),
                "detail_consensus_gt_0.25": float(
                    np.mean(detail_common[safe_both] > 0.25)
                ),
                "detail_consensus_gt_0.50": float(
                    np.mean(detail_common[safe_both] > 0.50)
                ),
                "final_confidence_gt_0.10": float(
                    np.mean(final_confidence[safe_both] > 0.10)
                ),
                "final_confidence_gt_0.25": float(
                    np.mean(final_confidence[safe_both] > 0.25)
                ),
                "final_confidence_gt_0.50": float(
                    np.mean(final_confidence[safe_both] > 0.50)
                ),
            },
        }
        summary["variants"].append(variant_summary)

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("K5-A.1b boundary-safe detail confidence diagnostic complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
