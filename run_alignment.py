"""K4 cached two-crop registration/alignment diagnostics; no fusion or inference."""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party/ml-sharp/src"))
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from experiments.sharp_hybrid_research.alignment import (
    AlignmentConfig, align_affine, make_fitting_maps, sample_sharp_inverse_native,
)
from experiments.sharp_hybrid_research.sharp_adapter import load_capture
from experiments.sharp_hybrid_research.capture_baseline import sha256, write_json

BASELINE = ROOT / "results/sharp_hybrid/baseline_cache"
COMPARE = ROOT / "experiments/model_compare"
CROPS = ("far_1495_192", "far_1480_192")


def display_gray(values, bounds):
    lo, hi = bounds
    valid = np.isfinite(values)
    fraction = np.where(valid, (values - lo) / max(hi - lo, np.finfo(float).tiny), 0.)
    gray = np.uint8(np.clip(fraction, 0, 1) * 255)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    rgb[~valid] = (180, 0, 180)
    return Image.fromarray(rgb)


def finite_bounds(values):
    valid = values[np.isfinite(values)]
    if not valid.size:
        return [0., 1.]
    lo, hi = np.percentile(valid, [1, 99])
    return [float(lo), float(hi if hi > lo else lo + 1.)]


def difference_image(values, limit):
    f = np.clip(np.nan_to_num(values / max(limit, np.finfo(float).tiny)), -1, 1)
    image = np.full((*values.shape, 3), 255., dtype=float)
    image[..., 0] -= 255 * np.maximum(-f, 0)
    image[..., 1] -= 255 * np.abs(f)
    image[..., 2] -= 255 * np.maximum(f, 0)
    image[~np.isfinite(values)] = [180, 0, 180]
    return Image.fromarray(image.astype(np.uint8))


def panel(folder, rgb, raw, sharp, result, rows):
    aligned = result["aligned"]
    joint = np.r_[sharp.ravel(), aligned[np.isfinite(aligned)]]
    bounds = finite_bounds(joint)
    difference = aligned - sharp
    finite_difference = np.abs(difference[np.isfinite(difference)])
    limit = float(np.percentile(finite_difference, 99)) if finite_difference.size else 1.
    images = [Image.fromarray(rgb), display_gray(raw, finite_bounds(raw)), display_gray(sharp, bounds),
              Image.fromarray(np.uint8(result["selected_anchor_mask"]) * 255).convert("RGB"),
              display_gray(aligned, bounds), difference_image(difference, limit)]
    labels = ["Native RGB", "TinyViM RAW: arbitrary relative units", "SHARP: 1/metre",
              "Selected fitting anchors (NOT replacement)", "Aligned TinyViM: 1/metre (ATTEMPT)",
              "Aligned - SHARP: red positive / blue negative"]
    width, tile_height = 1200, 440
    canvas = Image.new("RGB", (width, 2 * tile_height + 360), "#181818")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        x, y = (index % 3) * 400, (index // 3) * tile_height
        draw.text((x + 8, y + 5), label, fill="white")
        if index in (2, 4):
            draw.text((x + 8, y + 24), f"shared [{bounds[0]:.6g}, {bounds[1]:.6g}] 1/m", fill="white")
        if index == 5:
            draw.text((x + 8, y + 24), f"range +/-{limit:.6g} 1/m; magenta=undefined", fill="white")
        canvas.paste(image.resize((384, 384), Image.Resampling.NEAREST), (x + 8, y + 48))
    top = 2 * tile_height
    row = rows[2]
    draw.text((15, top + 6), f"Native row y={row}; SHARP black, aligned TinyViM green; units 1/metre", fill="white")
    draw.text((15, top + 26), f"Status: {result['report']['status']}; " + "; ".join(result['report']['reasons'])[:155], fill="white")
    draw.rectangle((40, top + 60, 1160, top + 310), fill="white")
    lo, hi = bounds
    for signal, color in ((sharp[row], "black"), (aligned[row], "green")):
        points = [(40 + 1120 * x / (len(signal) - 1), top + 310 - 250 * float(np.clip((v - lo) / (hi - lo), 0, 1)))
                  for x, v in enumerate(signal) if np.isfinite(v)]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
    draw.text((5, top + 60), f"{hi:.4g}", fill="white")
    draw.text((5, top + 310), f"{lo:.4g}", fill="white")
    canvas.save(folder / "panel.png")
    return {"sharp_and_aligned_shared_inverse_m_bounds": bounds, "difference_symmetric_limit_inverse_m": limit,
            "raw_scale": "independent arbitrary relative units, explicitly labeled",
            "profile_plot_clips_to_display_bounds": True, "full_unclipped_profiles": "wire_profiles.npz"}


def save_crop_diagnostics(folder, rgb, raw, target, wire, config, provenance):
    folder.mkdir(parents=True, exist_ok=False)
    maps = make_fitting_maps(raw, target, rgb, wire, unpadded=np.ones(raw.shape, bool), config=config)
    result = align_affine(raw, target, maps["anchor_mask"], raw_fit=maps["raw_fit"],
                          target_fit=maps["target_fit"], regions=maps["regions"],
                          foreground_mask=wire, config=config)
    report = result["report"]
    report["anchor_construction"] = maps["stats"]
    report["provenance"] = provenance
    report["focal_caveat"] = "1/metre in captured SHARP coordinates; focal is a numerical placeholder, not calibrated"
    Image.fromarray(rgb).save(folder / "rgb_crop.png")
    mask = result["selected_anchor_mask"]
    Image.fromarray(np.uint8(mask) * 255).save(folder / "anchor_mask.png")
    Image.fromarray(np.uint8(maps["anchor_mask"]) * 255).save(folder / "anchor_candidates.png")
    Image.fromarray(np.uint8(wire) * 255).save(folder / "wire_exclusion_mask.png")
    arrays = {"raw_tinyvim_relative": raw, "sharp_visible_depth_m": 1 / target,
              "sharp_visible_inverse_m": target, "anchor_mask": mask,
              "sharp_common_bandwidth": maps["target_fit"], "tinyvim_common_bandwidth": maps["raw_fit"],
              "aligned_tinyvim_inverse_m": result["aligned"],
              "aligned_minus_sharp_inverse_m": result["aligned"] - target,
              "valid_transformed": result["valid_transformed"], "fitting_regions": maps["regions"],
              "kernel_mass": maps["kernel_mass"]}
    for name, values in arrays.items():
        np.save(folder / (name + ".npy"), values)
    rows = [round(raw.shape[0] * f) for f in (.25, .4, .55, .7)]
    np.savez(folder / "wire_profiles.npz", rows=np.array(rows), x=np.arange(raw.shape[1]),
             rgb=rgb[rows], raw_tinyvim_relative=raw[rows], sharp_inverse_m=target[rows],
             aligned_tinyvim_inverse_m=result["aligned"][rows], wire_proxy=wire[rows])
    report["display"] = panel(folder, rgb, raw, target, result, rows)
    report["outputs_are_attempts_not_fusion"] = True
    write_json(folder / "alignment.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("buckets", "folds", "provenance", "display")}, indent=2), flush=True)
    return report


def load_baseline():
    verification = json.loads((BASELINE / "verification.json").read_text())
    assert verification["status"] == "PASS" and verification["ply_byte_identical"]
    assert sha256(BASELINE / "capture.pt") == verification["cache_sha256"]
    capture = load_capture(BASELINE / "capture.pt")
    assert capture.camera["resize"] == {"mode": "bilinear", "align_corners": True, "aspect": "square_stretch"}
    assert tuple(capture.metric_depth.shape) == (1, 2, 1536, 1536)
    depth = capture.metric_depth[0, 0].numpy().copy()
    native_hw = tuple(capture.camera["native_hw"])
    del capture
    gc.collect()
    return depth, native_hw, verification


def load_cached_crop(name, rgb):
    source = COMPARE / "outputs/depthart_tiny_512" / name
    meta = json.loads((source / "metadata.json").read_text())
    coordinates = json.loads((COMPARE / "regions.json").read_text())[name]["box"]
    assert coordinates == meta["box"]
    x0, y0, x1, y1 = coordinates
    crop = rgb[y0:y1, x0:x1]
    assert hashlib.sha256(crop.tobytes()).hexdigest() == meta["source_crop_sha256"]
    assert np.array_equal(crop, np.array(Image.open(source / "rgb.png").convert("RGB")))
    assert meta["native_hw"] == list(crop.shape[:2]) and meta["semantics"] == "relative_inverse_depth"
    assert meta["input_hw"] == meta["content_hw"] == [512, 512]  # No padded pixels in these two crops.
    raw = np.load(source / "depth_raw.npy")
    grid = np.load(source / "depth_model_grid.npy")
    reconstructed = F.interpolate(torch.from_numpy(grid)[None, None], crop.shape[:2],
                                  mode="bilinear", align_corners=False)[0, 0].numpy()
    mapping_error = float(np.max(np.abs(raw - reconstructed)))
    # Original output was resampled on CUDA; tiny arithmetic differences on CPU are permitted.
    assert mapping_error <= 1e-5 * max(float(np.ptp(raw)), 1e-8)
    wire = np.array(Image.open(source / "rgb_proxy_mask.png")) > 0
    provenance = {"crop_name": name, "box_native_half_open": coordinates, "tinyvim_source_array": str(source / "depth_raw.npy"),
                  "tinyvim_source_array_sha256": sha256(source / "depth_raw.npy"), "tinyvim_metadata": meta,
                  "preprocessing_source": str(COMPARE / "backends.py"),
                  "preprocessing_source_sha256": sha256(COMPARE / "backends.py"),
                  "metadata_sampling_note": "metadata says linear but depthart source explicitly uses cubic RGB; raw mapping is bilinear align_corners=False",
                  "native_grid_reconstruction_max_abs_error": mapping_error, "padded_pixels": 0,
                  "wire_mask_source": str(source / "rgb_proxy_mask.png"),
                  "sharp_cache": str(BASELINE / "capture.pt"),
                  "sharp_mapping": "inverse metric depth grid bilinear sampled at x*(1535)/(W-1), y*(1535)/(H-1)",
                  "depth_crop_definition": "reciprocal of native-resampled inverse depth; unresampled grid remains authoritative"}
    return crop, raw, wire, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/sharp_hybrid/k4_alignment")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    depth, native_hw, verification = load_baseline()
    rgb = np.array(Image.open(ROOT / "bridge.png").convert("RGB"))
    assert rgb.shape[:2] == native_hw and sha256(ROOT / "bridge.png") == verification["input_sha256"]
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"baseline_cache_sha256": verification["cache_sha256"], "crops": {}}
    for name in CROPS:
        crop, raw, wire, provenance = load_cached_crop(name, rgb)
        target = sample_sharp_inverse_native(depth, native_hw, provenance["box_native_half_open"])
        main_report = save_crop_diagnostics(args.output / name, crop, raw, target, wire, AlignmentConfig(), provenance)
        native_report = save_crop_diagnostics(args.output / name / "sigma0_control", crop, raw, target, wire,
                                              replace(AlignmentConfig(), sigma=0), provenance)
        summary["crops"][name] = {"sigma24_status": main_report["status"], "sigma0_status": native_report["status"],
                                  "sigma24_reasons": main_report["reasons"], "sigma0_reasons": native_report["reasons"]}
    write_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
