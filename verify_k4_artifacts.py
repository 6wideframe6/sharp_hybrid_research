"""Read-only checks of authoritative K4 artifacts; no inference or fitting."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party/ml-sharp/src"))
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
from experiments.sharp_hybrid_research.capture_baseline import sha256, write_json
from experiments.sharp_hybrid_research.run_alignment import load_baseline
from experiments.sharp_hybrid_research.alignment import sample_sharp_inverse_native

HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "results/sharp_hybrid"


def main():
    output = HERE / "k4_artifact_verification.json"
    if output.exists():
        raise FileExistsError(output)
    grid, native_hw, baseline = load_baseline()
    for filename, expected in baseline["ply_sha256"].items():
        assert sha256(RESULTS / "baseline_cache" / filename) == expected
    reports = sorted((RESULTS / "k4_alignment_crossfit").rglob("alignment.json"))
    reports += sorted((RESULTS / "k4_context").rglob("alignment.json"))
    assert len(reports) == 8
    checks = []
    for path in reports:
        folder = path.parent
        report = json.loads(path.read_text())
        box = report["provenance"]["box_native_half_open"]
        raw = np.load(folder / "raw_tinyvim_relative.npy")
        sharp_q = np.load(folder / "sharp_visible_inverse_m.npy")
        sharp_z = np.load(folder / "sharp_visible_depth_m.npy")
        aligned = np.load(folder / "aligned_tinyvim_inverse_m.npy")
        difference = np.load(folder / "aligned_minus_sharp_inverse_m.npy")
        anchors = np.load(folder / "anchor_mask.npy")
        valid = np.load(folder / "valid_transformed.npy")
        shape = (box[3] - box[1], box[2] - box[0])
        assert all(a.shape == shape for a in (raw, sharp_q, sharp_z, aligned, anchors, valid))
        assert np.isfinite(raw).all() and np.isfinite(sharp_q).all() and (sharp_q > 0).all()
        np.testing.assert_array_equal(sharp_q, sample_sharp_inverse_native(grid, native_hw, box))
        np.testing.assert_array_equal(sharp_z, 1 / sharp_q)
        assert anchors.dtype == bool and valid.dtype == bool
        assert int(anchors.sum()) == report["usable_anchors"]
        assert report["target_units"] == report["output_units"] == "1/metre"
        if report["positive_slope"]:
            np.testing.assert_array_equal(aligned, report["a"] * raw.astype(np.float64) + report["b"])
        else:
            assert np.isnan(aligned).all() and not valid.any()
        np.testing.assert_array_equal(difference, aligned - sharp_q)
        if not report["accepted"]:
            assert not valid.any(), "Rejected fit must not expose a usable correction mask"
        fit_x = np.load(folder / "tinyvim_common_bandwidth.npy")
        fit_y = np.load(folder / "sharp_common_bandwidth.npy")
        assert np.isfinite(fit_x[anchors]).all() and np.isfinite(fit_y[anchors]).all()
        for fold in report["folds"]:
            assert not (set(fold["training_blocks"]) & set(fold["validation_blocks"]))
        if report["config"]["sigma"]:
            assert "training-only" in report["holdout_smoothing"]
        profiles = np.load(folder / "wire_profiles.npz")
        np.testing.assert_array_equal(profiles["raw_tinyvim_relative"], raw[profiles["rows"]])
        np.testing.assert_array_equal(profiles["sharp_inverse_m"], sharp_q[profiles["rows"]])
        np.testing.assert_array_equal(profiles["aligned_tinyvim_inverse_m"], aligned[profiles["rows"]])
        for filename in ("rgb_crop.png", "anchor_mask.png", "panel.png"):
            with Image.open(folder / filename) as image:
                image.verify()
        checks.append({"path": str(folder.relative_to(ROOT)), "status": report["status"],
                       "anchors": int(anchors.sum()), "passed": True})
    inventory = []
    for path in sorted(RESULTS.rglob("*")):
        if path.is_file():
            inventory.append({"path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size})
    record = {"passed": True, "verified_diagnostic_runs": len(checks), "runs": checks,
              "real_sharp_registration_verified": True, "raw_affine_application_verified": True,
              "baseline_cache_sha256": baseline["cache_sha256"],
              "baseline_ply_hashes_reverified": True,
              "historical_alignment_sha256": sha256(ROOT / "src/alignment.py"),
              "artifact_inventory": inventory,
              "historical_first_pass": "k4_alignment retained; superseded by leakage-free k4_alignment_crossfit"}
    write_json(output, record)
    print(json.dumps({k: v for k, v in record.items() if k != "artifact_inventory"}, indent=2))


if __name__ == "__main__":
    main()
