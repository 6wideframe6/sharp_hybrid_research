"""Bounded K4 identifiability follow-up: two verified contexts, then overlap audit.

Exactly two TinyViM-S CPU passes, only after the cached far-crop fits are shown
underdetermined. Does not change thresholds or perform fusion/ownership.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party/ml-sharp/src"))
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from experiments.model_compare.backends import Backend
from experiments.sharp_hybrid_research.alignment import AlignmentConfig, sample_sharp_inverse_native
from experiments.sharp_hybrid_research.run_alignment import (
    CROPS, COMPARE, load_baseline, load_cached_crop, save_crop_diagnostics,
)
from experiments.sharp_hybrid_research.capture_baseline import sha256, write_json

OUT = ROOT / "results/sharp_hybrid/k4_context"


def overlap(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x0 >= x1 or y0 >= y1:
        return None
    return [x0, y0, x1, y1]


def roi(array, source_box, box):
    return array[box[1] - source_box[1]:box[3] - source_box[1],
                 box[0] - source_box[0]:box[2] - source_box[0]]


def main():
    initial = ROOT / "results/sharp_hybrid/k4_alignment_crossfit"
    reports = {name: json.loads((initial / name / "alignment.json").read_text()) for name in CROPS}
    assert all(r["status"] == "underdetermined" for r in reports.values())
    if OUT.exists():
        raise FileExistsError(OUT)
    depth, native_hw, verification = load_baseline()
    rgb = np.array(Image.open(ROOT / "bridge.png").convert("RGB"))
    known_candidates = np.array(Image.open(ROOT / "results/final/details/candidate_mask.png")) > 0
    original_tiles = []
    for name in CROPS:
        crop, raw, wire, provenance = load_cached_crop(name, rgb)
        box = provenance["box_native_half_open"]
        x0, y0, x1, y1 = box
        known_candidates[y0:y1, x0:x1] |= wire
        original_tiles.append((name, box, initial / name, reports[name]))
    torch.set_num_threads(2)
    backend = Backend("depthart_tiny", device="cpu", resolution=512)
    checkpoint = ROOT / backend.meta["checkpoint"]
    OUT.mkdir(parents=True, exist_ok=False)
    tiles = list(original_tiles)
    stages = []
    for stage, name in enumerate(("far_1495_512", "far_1480_512"), 1):
        metadata_source = ROOT / "results/08_remaining" / name / "metadata.json"
        historic = json.loads(metadata_source.read_text())
        box = historic["box"]
        x0, y0, x1, y1 = box
        assert historic["native_hw"] == [y1 - y0, x1 - x0] == [512, 512]
        assert 0 <= x0 < x1 <= native_hw[1] and 0 <= y0 < y1 <= native_hw[0]
        assert all(overlap(box, old_box) == old_box for _, old_box, _, _ in original_tiles)
        crop = rgb[y0:y1, x0:x1]
        wire = known_candidates[y0:y1, x0:x1]
        start = time.monotonic()
        raw, grid, input_rgb, metadata = backend.infer(crop)
        seconds = time.monotonic() - start
        assert raw.shape == (512, 512) and np.isfinite(raw).all()
        provenance = {"stage": stage, "role": "expanded context" if stage == 1 else "shifted overlapping context",
                      "box_native_half_open": box, "coordinate_source": str(metadata_source),
                      "coordinate_source_sha256": sha256(metadata_source),
                      "input_sha256": verification["input_sha256"],
                      "crop_rgb_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
                      "tinyvim_checkpoint": str(checkpoint), "tinyvim_checkpoint_sha256": sha256(checkpoint),
                      "tinyvim_metadata": metadata, "new_inference_seconds": seconds,
                      "known_wire_exclusions": ["results/final/details/candidate_mask.png",
                                                 "model_compare cached far192 rgb_proxy_mask.png union"],
                      "sharp_source_cache": str(ROOT / "results/sharp_hybrid/baseline_cache/capture.pt"),
                      "fitting_config_unchanged": True}
        target = sample_sharp_inverse_native(depth, native_hw, box)
        folder = OUT / name
        report = save_crop_diagnostics(folder, crop, raw, target, wire, AlignmentConfig(), provenance)
        np.save(folder / "depth_model_grid.npy", grid)
        Image.fromarray(input_rgb).save(folder / "model_input_rgb.png")
        native_report = save_crop_diagnostics(folder / "sigma0_control", crop, raw, target, wire,
                                              replace(AlignmentConfig(), sigma=0), provenance)
        tiles.append((name, box, folder, report))
        stages.append({"name": name, "role": provenance["role"], "box": box,
                       "sigma24_status": report["status"], "sigma24_reasons": report["reasons"],
                       "sigma0_status": native_report["status"], "new_inference_seconds": seconds})
    edges = []
    for i, (name, box, folder, report) in enumerate(tiles):
        for other_name, other_box, other_folder, other_report in tiles[i + 1:]:
            common = overlap(box, other_box)
            if common is None:
                continue
            left_mask = roi(np.load(folder / "anchor_mask.npy"), box, common)
            right_mask = roi(np.load(other_folder / "anchor_mask.npy"), other_box, common)
            shared = left_mask & right_mask
            edges.append({"tiles": [name, other_name], "overlap_box_native": common,
                          "overlap_pixels": int(shared.size), "shared_selected_anchor_pixels": int(shared.sum()),
                          "endpoint_has_accepted_sharp_fit": bool(report["accepted"] or other_report["accepted"])})
    any_anchor = any(report["accepted"] for _, _, _, report in tiles)
    # A pairwise-consistent but wholly unanchored graph cannot set metric scale.
    summary = {"stages": stages, "overlap_edges": edges, "accepted_sharp_anchor_exists": any_anchor,
               "joint_overlap_refinement_performed": False,
               "joint_refinement_reason": ("Review usable overlaps before any optional joint fit" if any_anchor else
                    "No accepted SHARP anchor in connected overlap component; affine gauge remains unresolved"),
               "new_tinyvim_passes": 2, "fusion_performed": False,
               "outcome": "wire geometry detected; absolute SHARP-coordinate depth unresolved" if not any_anchor else "context review required"}
    write_json(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
