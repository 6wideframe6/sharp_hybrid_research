"""One non-overwriting real bridge capture using the existing K3 cache schema."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party/ml-sharp/src"))
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from sharp.models import PredictorParams, create_predictor
from experiments.sharp_hybrid_research.sharp_adapter import (
    SharpAdapter, save_capture, load_capture, export_ply,
)

CHECKPOINT = Path(__file__).resolve().parent / "checkpoints/sharp_2572gikvuh.pt"
OUTPUT = ROOT / "results/sharp_hybrid/baseline_cache"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def write_json(path, data):
    with Path(path).open("x") as stream:
        json.dump(json_value(data), stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    gate = json.loads((Path(__file__).resolve().parent / "released_checkpoint_gate.json").read_text())
    assert gate["status"] == "PASS" and gate["full_monodepth_executed"]
    checkpoint_hash = sha256(CHECKPOINT)
    assert checkpoint_hash == gate["checkpoint_sha256"]
    if OUTPUT.exists():
        raise FileExistsError(f"Baseline already exists; refusing another capture: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)  # Same CPU setting as the successful existing K3 test.
    start = time.monotonic()
    rgb = np.array(Image.open(ROOT / "bridge.png").convert("RGB"))
    height, width = rgb.shape[:2]
    provenance = {"checkpoint_path": str(CHECKPOINT), "checkpoint_sha256": checkpoint_hash,
                  "checkpoint_size_bytes": CHECKPOINT.stat().st_size,
                  "input_path": str(ROOT / "bridge.png"), "input_sha256": sha256(ROOT / "bridge.png"),
                  "focal_px": float(width), "focal_source": "numerical placeholder: native image_width",
                  "native_hw": [height, width], "device": "cpu", "full_capture_count": 1}
    write_json(OUTPUT / "provenance.json", provenance)
    model = create_predictor(PredictorParams())
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    adapter = SharpAdapter(model)
    print("Running exactly one full bridge capture", flush=True)
    capture_start = time.monotonic()
    capture = adapter.capture_rgb(rgb, float(width), device="cpu", provenance=provenance)
    capture_seconds = time.monotonic() - capture_start
    assert capture.image.shape == (1, 3, 1536, 1536)
    original = capture.gaussians_ndc
    camera = capture.camera
    manifest = {**provenance, "sharp_revision": capture.metadata["source"]["commit"],
                "camera": camera, "disparity_factor": capture.disparity_factor,
                "depth_factor": capture.depth_factor, "global_scale": capture.initializer.global_scale,
                "canonical_disparity_shape": list(capture.monodepth_output.disparity.shape),
                "metric_depth_shape": list(capture.metric_depth.shape),
                "encoder_feature_shapes": [list(x.shape) for x in capture.monodepth_output.encoder_features],
                "delta_shape": list(capture.delta.shape), "gaussian_count": original.mean_vectors.shape[1],
                "capture_seconds": capture_seconds, "cache_schema": 1}
    save_capture(capture, OUTPUT / "capture.pt")
    del capture
    gc.collect()
    print("Reloading K3 cache and replaying original tail", flush=True)
    restored = load_capture(OUTPUT / "capture.pt")
    replay_start = time.monotonic()
    replayed = adapter.replay(restored)
    manifest["replay_seconds"] = time.monotonic() - replay_start
    comparison = {}
    for name in original._fields:
        a, b, c = getattr(original, name), getattr(restored.gaussians_ndc, name), getattr(replayed, name)
        comparison[name] = {"reload_bit_exact": bool(torch.equal(a, b)),
                            "replay_bit_exact": bool(torch.equal(a, c)),
                            "max_abs_error": float((a - c).abs().max()), "shape": list(a.shape)}
    assert all(x["reload_bit_exact"] and x["replay_bit_exact"] for x in comparison.values())
    print("Exporting original and replayed PLY via upstream SHARP", flush=True)
    export_ply(original, camera, OUTPUT / "original.ply")
    export_ply(replayed, restored.camera, OUTPUT / "replayed.ply")
    ply_hashes = {name: sha256(OUTPUT / name) for name in ("original.ply", "replayed.ply")}
    assert ply_hashes["original.ply"] == ply_hashes["replayed.ply"]
    verification = {**manifest, "status": "PASS", "full_1536_inference_executed": True,
                    "gaussian_comparison": comparison, "comparison_atol": 0., "comparison_rtol": 0.,
                    "ply_sha256": ply_hashes, "ply_byte_identical": True,
                    "cache_path": str(OUTPUT / "capture.pt"), "cache_sha256": sha256(OUTPUT / "capture.pt"),
                    "elapsed_seconds": time.monotonic() - start,
                    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "hybrid_correction_performed": False}
    write_json(OUTPUT / "verification.json", verification)
    print(json.dumps(json_value(verification), indent=2), flush=True)


if __name__ == "__main__":
    main()
