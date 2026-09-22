#!/usr/bin/env python3
"""K5-D.1 transfer final native correction onto SHARP's 1536 grid and round-trip it.

This is a resolution/registration gate before any Gaussian replay.

Input:
- frozen K3 baseline capture;
- C.8 composition output;
- optional evaluation-only wire proxy.

The selected C.8 correction is sampled from native-image coordinates onto the
authoritative SHARP metric inverse-depth grid.  That candidate grid is converted
back to metric depth and sampled again into the same native overlap using the
exact K4 mapping:

    x_grid = x_native * (W_grid - 1) / (W_native - 1)
    y_grid = y_native * (H_grid - 1) / (H_native - 1)

The round-trip answers whether the thin scalar correction actually survives the
1536x1536 SHARP internal depth grid before we spend time on K3 replay.

No SHARP model inference, initializer edit, Gaussian decoder, or PLY export is
performed here.  The wire proxy is evaluation-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party/ml-sharp/src"))
sys.path.insert(0, str(ROOT))

from experiments.sharp_hybrid_research.alignment import sample_sharp_inverse_native
from experiments.sharp_hybrid_research.sharp_adapter import load_capture
from experiments.sharp_hybrid_research.k5.types import NativeBox


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


def abs_stats(values):
    return stats(np.abs(np.asarray(values, dtype=np.float64)))


def correlation(a, b, mask):
    m = np.asarray(mask, dtype=bool)
    x = np.asarray(a, dtype=np.float64)[m]
    y = np.asarray(b, dtype=np.float64)[m]
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 2 or np.std(x) <= 0 or np.std(y) <= 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def sign_agreement(a, b, mask):
    m = np.asarray(mask, dtype=bool)
    sa = np.sign(np.asarray(a)[m])
    sb = np.sign(np.asarray(b)[m])
    valid = (sa != 0) & (sb != 0)
    if not valid.any():
        return None
    return float(np.mean(sa[valid] == sb[valid]))


def map_wire_proxy(folder: Path, target_box: NativeBox):
    meta = json.loads((folder / "metadata.json").read_text())
    source_box = NativeBox.from_sequence(meta["box"])
    mask = np.asarray(Image.open(folder / "rgb_proxy_mask.png").convert("L")) > 0
    if mask.shape != source_box.shape:
        raise ValueError("wire proxy shape mismatch")

    out = np.zeros(target_box.shape, dtype=bool)
    try:
        common = source_box.intersect(target_box)
    except ValueError:
        return out

    sy, sx = source_box.local_slices(common)
    ty, tx = target_box.local_slices(common)
    out[ty, tx] = mask[sy, sx]
    return out


def native_overlap_to_sharp_grid(
    values: np.ndarray,
    box: NativeBox,
    *,
    native_hw: tuple[int, int],
    grid_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a native-overlap scalar map at SHARP grid pixel centres."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != box.shape:
        raise ValueError("native overlap values/box mismatch")

    native_h, native_w = native_hw
    grid_h, grid_w = grid_hw

    # Grid pixel centres correspond exactly to these native coordinates under
    # align_corners=True square stretch.
    gy, gx = np.mgrid[:grid_h, :grid_w].astype(np.float64)
    native_y = gy * (native_h - 1) / (grid_h - 1)
    native_x = gx * (native_w - 1) / (grid_w - 1)

    inside = (
        (native_x >= box.x0)
        & (native_x <= box.x1 - 1)
        & (native_y >= box.y0)
        & (native_y <= box.y1 - 1)
    )

    out = np.zeros((grid_h, grid_w), dtype=np.float64)
    if inside.any():
        local_y = native_y[inside] - box.y0
        local_x = native_x[inside] - box.x0
        out[inside] = ndi.map_coordinates(
            values,
            [local_y, local_x],
            order=1,
            mode="nearest",
            prefilter=False,
        )

    return out, inside


def save_signed(array, path, scale):
    a = np.asarray(array, dtype=np.float64)
    if scale <= np.finfo(float).tiny:
        image = np.full(a.shape, 128, dtype=np.uint8)
    else:
        mapped = 0.5 + 0.5 * np.clip(a / scale, -1.0, 1.0)
        image = np.round(mapped * 255.0).astype(np.uint8)
    Image.fromarray(image).save(path)


def row_runs(row):
    x = np.flatnonzero(row)
    if not len(x):
        return []
    split = np.where(np.diff(x) > 1)[0] + 1
    return [chunk for chunk in np.split(x, split) if len(chunk)]


def select_vertical_wire_samples(wire):
    h, w = wire.shape
    tolerant = ndi.maximum_filter1d(
        wire.astype(np.uint8), size=3, axis=1, mode="constant", cval=0
    ).astype(bool)

    samples = []
    for y in range(9, h - 9):
        if y % 2:
            continue
        for run in row_runs(wire[y]):
            width = int(run[-1] - run[0] + 1)
            if width > 3:
                continue
            x = int(round((int(run[0]) + int(run[-1])) * 0.5))
            if x < 9 or x >= w - 9:
                continue
            occupancy = float(np.mean(tolerant[y - 4 : y + 5, x]))
            if occupancy >= 0.75:
                samples.append((y, x))
    return samples


def aligned_wire_profile(field, samples):
    outer_idx = [int(np.where(OFFSETS == o)[0][0]) for o in OUTER]
    center_idx = int(np.where(OFFSETS == 0)[0][0])
    profiles = []
    contrasts = []

    for y, x in samples:
        p = np.asarray(field[y, x + OFFSETS], dtype=np.float64)
        baseline = float(np.median(p[outer_idx]))
        local = p - baseline
        center = float(local[center_idx])
        if abs(center) <= np.finfo(float).tiny:
            continue
        sign = 1.0 if center > 0 else -1.0
        profiles.append(sign * local)
        contrasts.append(abs(center))

    if not profiles:
        return {"samples": 0}

    p = np.asarray(profiles)
    med = np.median(p, axis=0)
    peak_idx = int(np.argmax(med))
    peak = float(med[peak_idx])
    above = np.flatnonzero(med >= 0.5 * peak)

    return {
        "samples": int(len(p)),
        "center_abs_contrast": stats(contrasts),
        "peak_offset_px": int(OFFSETS[peak_idx]),
        "peak_value": peak,
        "fwhm_px": (
            float(OFFSETS[above[-1]] - OFFSETS[above[0]] + 1)
            if len(above) else None
        ),
        "median_profile": [float(v) for v in med],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--composition", type=Path, required=True)
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--wire-crop", type=Path)
    args = ap.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    composition = args.composition.resolve()
    summary = json.loads((composition / "summary.json").read_text())
    box = NativeBox.from_sequence(summary["overlap_box_native"])

    q_native_reference = np.load(composition / "q_sharp.npy").astype(np.float64)
    delta_native = np.load(
        composition / f"delta_q_{args.variant}.npy"
    ).astype(np.float64)
    q_final_native = np.load(
        composition / f"q_final_{args.variant}.npy"
    ).astype(np.float64)
    gate_native = np.load(
        composition / f"application_gate_{args.variant}.npy"
    ).astype(np.float64)

    for name, a in (
        ("q_sharp", q_native_reference),
        ("delta_q", delta_native),
        ("q_final", q_final_native),
        ("application_gate", gate_native),
    ):
        if a.shape != box.shape:
            raise ValueError(f"{name} shape {a.shape} != overlap {box.shape}")

    capture = load_capture(args.capture.resolve(), device="cpu")
    if capture.camera is None:
        raise ValueError("capture must contain camera/native resize metadata")
    if capture.camera["resize"] != {
        "mode": "bilinear",
        "align_corners": True,
        "aspect": "square_stretch",
    }:
        raise ValueError("unexpected capture resize contract")

    native_hw = tuple(int(v) for v in capture.camera["native_hw"])
    depth_grid = capture.metric_depth[0, 0].detach().cpu().numpy().astype(np.float64)
    if depth_grid.ndim != 2:
        raise ValueError("expected visible SHARP metric depth grid [H,W]")
    q_grid = 1.0 / depth_grid

    # Verify the K4/C8 native SHARP reference is exactly the current capture mapping.
    native_from_baseline_grid = sample_sharp_inverse_native(
        depth_grid, native_hw, [box.x0, box.y0, box.x1, box.y1]
    )
    baseline_mapping_error = native_from_baseline_grid - q_native_reference

    delta_grid, grid_overlap = native_overlap_to_sharp_grid(
        delta_native,
        box,
        native_hw=native_hw,
        grid_hw=depth_grid.shape,
    )
    gate_grid, _ = native_overlap_to_sharp_grid(
        gate_native,
        box,
        native_hw=native_hw,
        grid_hw=depth_grid.shape,
    )

    q_candidate_grid = q_grid + delta_grid
    if not np.isfinite(q_candidate_grid).all() or np.any(q_candidate_grid <= 0):
        raise RuntimeError("candidate SHARP inverse-depth grid is invalid")
    z_candidate_grid = 1.0 / q_candidate_grid

    q_roundtrip_native = sample_sharp_inverse_native(
        z_candidate_grid, native_hw, [box.x0, box.y0, box.x1, box.y1]
    )
    delta_roundtrip = q_roundtrip_native - q_native_reference
    transfer_error = delta_roundtrip - delta_native

    active = gate_native > 0
    strong = np.abs(delta_native) >= np.percentile(
        np.abs(delta_native[np.abs(delta_native) > 0]), 50
    )

    report = {
        "purpose": "K5-D.1 native correction -> SHARP 1536 grid -> native round-trip",
        "variant": args.variant,
        "overlap_box_native": [box.x0, box.y0, box.x1, box.y1],
        "native_hw": list(native_hw),
        "sharp_grid_hw": list(depth_grid.shape),
        "grid_overlap_pixels": int(grid_overlap.sum()),
        "grid_overlap_fraction": float(grid_overlap.mean()),
        "baseline_mapping_error_1_per_m": abs_stats(baseline_mapping_error),
        "original_delta_abs_1_per_m": abs_stats(delta_native),
        "roundtrip_delta_abs_1_per_m": abs_stats(delta_roundtrip),
        "transfer_error_abs_1_per_m": abs_stats(transfer_error),
        "global": {
            "pearson": correlation(
                delta_native, delta_roundtrip, np.ones(box.shape, dtype=bool)
            ),
            "sign_agreement": sign_agreement(
                delta_native, delta_roundtrip, np.abs(delta_native) > 0
            ),
            "l1_mass_retention": float(
                np.sum(np.abs(delta_roundtrip))
                / np.sum(np.abs(delta_native))
            ),
        },
        "active_support": {
            "pixels": int(active.sum()),
            "pearson": correlation(delta_native, delta_roundtrip, active),
            "sign_agreement": sign_agreement(
                delta_native, delta_roundtrip, active
            ),
            "original_abs": abs_stats(delta_native[active]),
            "roundtrip_abs": abs_stats(delta_roundtrip[active]),
        },
        "strong_original_delta": {
            "pixels": int(strong.sum()),
            "pearson": correlation(delta_native, delta_roundtrip, strong),
            "sign_agreement": sign_agreement(
                delta_native, delta_roundtrip, strong
            ),
            "original_abs": abs_stats(delta_native[strong]),
            "roundtrip_abs": abs_stats(delta_roundtrip[strong]),
        },
        "candidate_grid": {
            "q_min": float(np.min(q_candidate_grid)),
            "q_max": float(np.max(q_candidate_grid)),
            "nonpositive": int(np.count_nonzero(q_candidate_grid <= 0)),
            "nonfinite": int(np.count_nonzero(~np.isfinite(q_candidate_grid))),
            "gate_grid_nonzero_pixels": int(np.count_nonzero(gate_grid > 0)),
        },
        "guardrails": [
            "No SHARP predictor/model inference is executed.",
            "No K3 initializer or Gaussian replay is executed.",
            "Transfer uses the exact K4 native<->SHARP align_corners=True geometry.",
            "Wire proxy is evaluation-only.",
        ],
    }

    wire = None
    if args.wire_crop is not None:
        wire = map_wire_proxy(args.wire_crop.resolve(), box)
        report["evaluation_only_wire"] = {
            "pixels": int(wire.sum()),
            "pearson": correlation(delta_native, delta_roundtrip, wire),
            "sign_agreement": sign_agreement(
                delta_native, delta_roundtrip, wire
            ),
            "original_abs": abs_stats(delta_native[wire]),
            "roundtrip_abs": abs_stats(delta_roundtrip[wire]),
            "p90_retention": float(
                np.percentile(np.abs(delta_roundtrip[wire]), 90)
                / np.percentile(np.abs(delta_native[wire]), 90)
            ),
        }

        samples = select_vertical_wire_samples(wire)
        report["evaluation_only_wire"]["vertical_profile_samples"] = int(len(samples))
        report["evaluation_only_wire"]["original_profile"] = aligned_wire_profile(
            delta_native, samples
        )
        report["evaluation_only_wire"]["roundtrip_profile"] = aligned_wire_profile(
            delta_roundtrip, samples
        )

    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "delta_sharp_grid.npy", delta_grid)
    np.save(output / "q_candidate_sharp_grid.npy", q_candidate_grid)
    np.save(output / "gate_sharp_grid.npy", gate_grid)
    np.save(output / "q_roundtrip_native.npy", q_roundtrip_native)
    np.save(output / "delta_roundtrip_native.npy", delta_roundtrip)
    np.save(output / "transfer_error_native.npy", transfer_error)

    scale = max(
        float(np.percentile(np.abs(delta_native), 99)),
        float(np.percentile(np.abs(delta_roundtrip), 99)),
    )
    save_signed(
        delta_native,
        output / "delta_original_native_signed_common.png",
        scale,
    )
    save_signed(
        delta_roundtrip,
        output / "delta_roundtrip_native_signed_common.png",
        scale,
    )
    save_signed(
        transfer_error,
        output / "transfer_error_native_signed_common.png",
        scale,
    )

    (output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print("K5-D.1 SHARP-grid round-trip diagnostic complete")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
