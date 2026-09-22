#!/usr/bin/env python3
"""K5-D.2b live SHARP Gaussian-decoder replay for the order-safe K5 correction.

This is the first real hybrid-scene replay.

Inputs are frozen from the validated upstream stages:
- K3 baseline capture;
- released SHARP checkpoint matching that capture;
- D.2a pre-K3 accept mask and component ownership IDs;
- D.2a.2 order-safe inverse-depth grid for one chosen gap fraction.

The script runs the SAME accepted metric-depth edit twice:
1. frozen_delta=True  -- preservation/control replay;
2. frozen_delta=False -- live feature_model + prediction_head on owned visible cells.

The live path is exactly SharpAdapter.replay_corrected().  The adapter preserves
secondary depth/deltas and unowned visible deltas by construction.  This runner
asserts those contracts again on the real checkpoint, verifies that the K3
ownership mask is exactly the one produced by D.2a.2, and exports baseline and
hybrid world-space PLY files for direct scene inspection.

No TinyViM magnitude is used here. No wire mask is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party/ml-sharp/src"))
sys.path.insert(0, str(ROOT))

from sharp.models import PredictorParams, create_predictor
from sharp.utils.gaussians import save_ply

from experiments.sharp_hybrid_research.sharp_adapter import (
    SharpAdapter,
    flattened_layer_indices,
    load_capture,
    world_gaussians,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values):
    if isinstance(values, torch.Tensor):
        a = values.detach().cpu().numpy()
    else:
        a = np.asarray(values)
    a = np.asarray(a, dtype=np.float64)
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
    if isinstance(values, torch.Tensor):
        return stats(values.abs())
    return stats(np.abs(np.asarray(values)))


def tensor_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return (
            left.keys() == right.keys()
            and all(tensor_tree_equal(left[k], right[k]) for k in left)
        )
    if isinstance(left, (tuple, list)):
        return (
            len(left) == len(right)
            and all(tensor_tree_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def gaussian_change_report(capture, live):
    """Verify secondary/unowned preservation and summarize owned visible change."""
    if capture.image.shape[0] != 1:
        raise ValueError("D.2b diagnostic currently requires batch size 1")

    h, w = capture.delta.shape[-2:]
    visible_idx = flattened_layer_indices(h, w, 0)
    secondary_idx = flattened_layer_indices(h, w, 1)
    owned_flat = live.edit.owned_cells[0, 0].flatten()

    report = {}

    for name in capture.gaussians_ndc._fields:
        old = getattr(capture.gaussians_ndc, name)
        new = getattr(live.gaussians_ndc, name)

        old_visible = old[:, visible_idx]
        new_visible = new[:, visible_idx]
        old_secondary = old[:, secondary_idx]
        new_secondary = new[:, secondary_idx]

        if not torch.equal(old_secondary, new_secondary):
            raise RuntimeError(f"{name}: secondary Gaussian layer changed")

        if not torch.equal(
            old_visible[:, ~owned_flat],
            new_visible[:, ~owned_flat],
        ):
            raise RuntimeError(f"{name}: unowned visible Gaussians changed")

        owned_old = old_visible[:, owned_flat]
        owned_new = new_visible[:, owned_flat]
        diff = owned_new - owned_old

        report[name] = {
            "secondary_bit_exact": True,
            "unowned_visible_bit_exact": True,
            "owned_values": int(owned_old.numel()),
            "owned_changed_values": int(torch.count_nonzero(diff).item()),
            "owned_abs_change": abs_stats(diff),
        }

    return report


def save_owned_mask(mask: torch.Tensor, path: Path):
    image = (
        mask[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
        * 255
    )
    Image.fromarray(image).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--d2a-output", type=Path, required=True)
    ap.add_argument("--d2a2-output", type=Path, required=True)
    ap.add_argument("--gap-fraction", type=float, default=0.25)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if (
        not np.isfinite(args.gap_fraction)
        or args.gap_fraction < 0
        or args.gap_fraction > 1
    ):
        raise ValueError("gap fraction must be finite in [0,1]")

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    capture_path = args.capture.resolve()
    checkpoint_path = args.checkpoint.resolve()
    d2a = args.d2a_output.resolve()
    d2a2 = args.d2a2_output.resolve()

    capture = load_capture(capture_path, device="cpu")
    if capture.camera is None:
        raise ValueError("capture needs camera metadata")
    if capture.image.shape[0] != 1:
        raise ValueError("D.2b runner expects the real batch-1 bridge capture")

    grid_hw = tuple(int(v) for v in capture.metric_depth.shape[-2:])

    pre_accept = np.load(
        d2a / "pre_k3_accept_mask.npy"
    ).astype(bool)
    surface_ids_np = np.load(
        d2a / "surface_ids_grid.npy"
    ).astype(np.int64)

    key = f"gap_{args.gap_fraction:g}"
    q_safe_path = d2a2 / f"q_safe_grid_{key}.npy"
    expected_applied_path = d2a2 / f"applied_pixels_{key}.npy"

    q_safe = np.load(q_safe_path).astype(np.float32)
    expected_applied = np.load(expected_applied_path).astype(bool)

    if any(
        a.shape != grid_hw
        for a in (
            pre_accept,
            surface_ids_np,
            q_safe,
            expected_applied,
        )
    ):
        raise ValueError("grid shape mismatch")

    if not np.array_equal(surface_ids_np >= 0, pre_accept):
        raise ValueError("D.2a accept/surface-id contract mismatch")

    d2a2_summary = json.loads(
        (d2a2 / "summary.json").read_text()
    )
    expected = d2a2_summary["results"][key]
    if expected["layer_order_reversal_pixels"] != 0:
        raise ValueError("chosen D.2a.2 candidate is not order-safe")

    baseline_q = capture.metric_depth[:, :1].reciprocal()
    inverse_depth = baseline_q.clone()
    q_t = torch.from_numpy(q_safe)[None, None]
    accept_t = torch.from_numpy(pre_accept)[None, None]
    surface_ids = torch.from_numpy(surface_ids_np)[None, None]
    inverse_depth[accept_t] = q_t[accept_t]

    torch.set_num_threads(2)

    model = create_predictor(PredictorParams())
    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    del state
    model.eval()

    adapter = SharpAdapter(model)

    # Same exact correction, first with the learned Gaussian deltas frozen,
    # then with the real feature decoder/head live on owned visible cells.
    frozen = adapter.replay_corrected(
        capture,
        inverse_depth,
        accept_t,
        surface_ids,
        cell_allow=None,
        frozen_delta=True,
    )

    live = adapter.replay_corrected(
        capture,
        inverse_depth,
        accept_t,
        surface_ids,
        cell_allow=None,
        frozen_delta=False,
    )

    # Ownership/depth must be identical between frozen and live replays and must
    # exactly reproduce the already validated D.2a.2 run.
    for name, left, right in (
        (
            "accepted_pixels frozen/live",
            frozen.edit.accepted_pixels,
            live.edit.accepted_pixels,
        ),
        (
            "owned_cells frozen/live",
            frozen.edit.owned_cells,
            live.edit.owned_cells,
        ),
        (
            "metric_depth frozen/live",
            frozen.edit.metric_depth,
            live.edit.metric_depth,
        ),
        (
            "normalized_depth frozen/live",
            frozen.edit.normalized_depth,
            live.edit.normalized_depth,
        ),
    ):
        if not torch.equal(left, right):
            raise RuntimeError(f"{name} mismatch")

    applied_np = (
        live.edit.accepted_pixels[0, 0]
        .detach()
        .cpu()
        .numpy()
    )
    if not np.array_equal(applied_np, expected_applied):
        raise RuntimeError(
            "live replay does not reproduce D.2a.2 applied-pixel mask"
        )

    if live.edit.diagnostics["layer_order_reversal"].any():
        raise RuntimeError("live replay contains layer-order reversal")

    # K3 contract: secondary depth and secondary learned deltas remain exact.
    secondary_depth_exact = torch.equal(
        live.edit.metric_depth[:, 1:],
        capture.metric_depth[:, 1:],
    )
    secondary_delta_exact = torch.equal(
        live.delta[:, :, 1],
        capture.delta[:, :, 1],
    )
    if not secondary_depth_exact:
        raise RuntimeError("secondary metric depth changed")
    if not secondary_delta_exact:
        raise RuntimeError("secondary learned Gaussian deltas changed")

    owned = live.edit.owned_cells.expand(
        -1,
        live.delta.shape[1],
        -1,
        -1,
    )

    if not torch.equal(
        live.delta[:, :, 0][~owned],
        capture.delta[:, :, 0][~owned],
    ):
        raise RuntimeError("unowned visible learned deltas changed")

    delta_diff_owned = (
        live.delta[:, :, 0][owned]
        - capture.delta[:, :, 0][owned]
    )
    frozen_delta_exact = torch.equal(
        frozen.delta,
        capture.delta,
    )
    if not frozen_delta_exact:
        raise RuntimeError("frozen replay did not preserve learned deltas")

    gaussian_report = gaussian_change_report(
        capture,
        live,
    )

    # Export both scenes with the exact camera metadata captured from SHARP.
    baseline_world = world_gaussians(
        capture.gaussians_ndc,
        capture.camera,
    )
    hybrid_world = world_gaussians(
        live.gaussians_ndc,
        capture.camera,
    )

    output.mkdir(parents=True, exist_ok=False)

    baseline_ply = output / "baseline_world.ply"
    hybrid_ply = output / f"hybrid_{key}_world.ply"

    f_px = float(capture.camera["f_px"])
    native_hw = tuple(int(v) for v in capture.camera["native_hw"])

    save_ply(
        baseline_world,
        f_px,
        native_hw,
        baseline_ply,
    )
    save_ply(
        hybrid_world,
        f_px,
        native_hw,
        hybrid_ply,
    )

    save_owned_mask(
        live.edit.owned_cells,
        output / "owned_cells.png",
    )
    np.save(
        output / "owned_cells.npy",
        live.edit.owned_cells[0, 0]
        .detach()
        .cpu()
        .numpy(),
    )
    np.save(
        output / "accepted_pixels.npy",
        applied_np,
    )
    np.save(
        output / "live_visible_delta.npy",
        live.delta[:, :, 0]
        .detach()
        .cpu()
        .numpy(),
    )

    report = {
        "purpose": (
            "K5-D.2b live SHARP Gaussian-decoder replay "
            "for order-safe K5 correction"
        ),
        "gap_fraction": float(args.gap_fraction),
        "capture_sha256": sha256(capture_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "ownership": {
            "pre_k3_accept_pixels": int(pre_accept.sum()),
            "applied_pixels": int(
                live.edit.accepted_pixels.sum().item()
            ),
            "owned_cells": int(
                live.edit.owned_cells.sum().item()
            ),
            "collision_cells": int(
                live.edit.diagnostics[
                    "collision_cells"
                ].sum().item()
            ),
            "pool_unchanged_cells": int(
                live.edit.diagnostics[
                    "pool_unchanged_cells"
                ].sum().item()
            ),
            "layer_order_reversal_pixels": int(
                live.edit.diagnostics[
                    "layer_order_reversal"
                ].sum().item()
            ),
            "exactly_matches_d2a2_applied_mask": True,
        },
        "preservation": {
            "frozen_replay_delta_bit_exact": True,
            "secondary_metric_depth_bit_exact": (
                secondary_depth_exact
            ),
            "secondary_learned_delta_bit_exact": (
                secondary_delta_exact
            ),
            "unowned_visible_learned_delta_bit_exact": True,
        },
        "live_decoder": {
            "owned_visible_delta_values": int(
                delta_diff_owned.numel()
            ),
            "owned_visible_delta_changed_values": int(
                torch.count_nonzero(
                    delta_diff_owned
                ).item()
            ),
            "owned_visible_delta_abs_change": abs_stats(
                delta_diff_owned
            ),
        },
        "gaussians": gaussian_report,
        "exports": {
            "baseline_world_ply": str(baseline_ply),
            "hybrid_world_ply": str(hybrid_ply),
            "native_hw": list(native_hw),
            "f_px": f_px,
        },
        "guardrails": [
            "Exact D.2a.2 order-safe metric-depth candidate is reused.",
            "Exact D.2a component ownership IDs are reused.",
            "No cell-level reversal veto is added.",
            "Live decoder updates only owned visible-layer deltas.",
            "Secondary depth and secondary learned deltas must remain bit-exact.",
            "Unowned visible learned deltas must remain bit-exact.",
            "Secondary and unowned visible final Gaussians must remain bit-exact.",
        ],
    }

    (output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print("K5-D.2b live Gaussian-decoder replay complete")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
