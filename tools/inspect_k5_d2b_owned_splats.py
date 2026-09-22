#!/usr/bin/env python3
"""Inspect exactly which world-space Gaussians changed in K5-D.2b.

This is diagnostic only. It does not alter the hybrid pipeline.

Outputs:
- baseline_owned_only.ply
- hybrid_owned_only.ply
- hybrid_owned_highlight_full.ply
- hybrid_owned_displacement_x5.ply
- summary.json

The highlight PLY keeps geometry unchanged but makes owned visible Gaussians
bright red/high-opacity and dims everything else. The x5 PLY contains only the
owned Gaussians and exaggerates baseline->hybrid world-position displacement
5x for visual inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party/ml-sharp/src"))
sys.path.insert(0, str(ROOT))

from sharp.utils.gaussians import Gaussians3D, load_ply, save_ply
from experiments.sharp_hybrid_research.sharp_adapter import flattened_layer_indices


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


def subset_gaussians(g: Gaussians3D, idx: torch.Tensor) -> Gaussians3D:
    return Gaussians3D(
        mean_vectors=g.mean_vectors[:, idx],
        singular_values=g.singular_values[:, idx],
        quaternions=g.quaternions[:, idx],
        colors=g.colors[:, idx],
        opacities=g.opacities[:, idx],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-ply", type=Path, required=True)
    ap.add_argument("--hybrid-ply", type=Path, required=True)
    ap.add_argument("--owned-cells", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--displacement-scale", type=float, default=5.0)
    args = ap.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    baseline, meta_b = load_ply(args.baseline_ply.resolve())
    hybrid, meta_h = load_ply(args.hybrid_ply.resolve())

    if meta_b != meta_h:
        raise ValueError(f"PLY metadata mismatch: {meta_b} != {meta_h}")

    for name in baseline._fields:
        a = getattr(baseline, name)
        b = getattr(hybrid, name)
        if a.shape != b.shape:
            raise ValueError(f"{name} shape mismatch: {a.shape} != {b.shape}")

    owned = np.load(args.owned_cells.resolve()).astype(bool)
    if owned.ndim != 2:
        raise ValueError("owned_cells must be [Hc,Wc]")

    cell_h, cell_w = owned.shape
    expected_per_layer = cell_h * cell_w
    total_gaussians = baseline.mean_vectors.shape[1]

    if total_gaussians != 2 * expected_per_layer:
        raise ValueError(
            f"Expected two layer-major Gaussian grids: "
            f"{total_gaussians} != 2*{expected_per_layer}"
        )

    visible_idx = flattened_layer_indices(cell_h, cell_w, 0).cpu()
    owned_flat = torch.from_numpy(owned.reshape(-1))
    owned_idx = visible_idx[owned_flat]

    if len(owned_idx) != int(owned.sum()):
        raise RuntimeError("owned index construction mismatch")

    baseline_owned = subset_gaussians(baseline, owned_idx)
    hybrid_owned = subset_gaussians(hybrid, owned_idx)

    displacement = hybrid_owned.mean_vectors - baseline_owned.mean_vectors
    displacement_norm = torch.linalg.vector_norm(displacement, dim=-1)

    color_delta = torch.linalg.vector_norm(
        hybrid_owned.colors - baseline_owned.colors,
        dim=-1,
    )
    scale_delta = torch.linalg.vector_norm(
        hybrid_owned.singular_values - baseline_owned.singular_values,
        dim=-1,
    )
    opacity_delta = (
        hybrid_owned.opacities - baseline_owned.opacities
    ).abs()

    # Full-scene diagnostic highlight:
    # owned visible = bright red, high opacity
    # everything else = dim low-opacity gray-ish original color
    highlight_colors = hybrid.colors.clone()
    highlight_opacities = hybrid.opacities.clone()

    highlight_colors *= 0.25
    highlight_opacities[:] = torch.clamp(
        highlight_opacities, max=0.03
    )

    highlight_colors[:, owned_idx] = torch.tensor(
        [1.0, 0.01, 0.01],
        dtype=highlight_colors.dtype,
        device=highlight_colors.device,
    )
    highlight_opacities[:, owned_idx] = 0.95

    highlight = Gaussians3D(
        mean_vectors=hybrid.mean_vectors,
        singular_values=hybrid.singular_values,
        quaternions=hybrid.quaternions,
        colors=highlight_colors,
        opacities=highlight_opacities,
    )

    # Owned-only exaggerated displacement diagnostic.
    exaggerated_means = (
        baseline_owned.mean_vectors
        + float(args.displacement_scale) * displacement
    )
    exaggerated = Gaussians3D(
        mean_vectors=exaggerated_means,
        singular_values=hybrid_owned.singular_values,
        quaternions=hybrid_owned.quaternions,
        colors=torch.tensor(
            [1.0, 0.01, 0.01],
            dtype=hybrid_owned.colors.dtype,
        ).view(1, 1, 3).expand_as(hybrid_owned.colors).clone(),
        opacities=torch.full_like(hybrid_owned.opacities, 0.95),
    )

    width, height = meta_h.resolution_px
    save_shape = (height, width)

    output.mkdir(parents=True, exist_ok=False)

    save_ply(
        baseline_owned,
        meta_h.focal_length_px,
        save_shape,
        output / "baseline_owned_only.ply",
    )
    save_ply(
        hybrid_owned,
        meta_h.focal_length_px,
        save_shape,
        output / "hybrid_owned_only.ply",
    )
    save_ply(
        highlight,
        meta_h.focal_length_px,
        save_shape,
        output / "hybrid_owned_highlight_full.ply",
    )
    save_ply(
        exaggerated,
        meta_h.focal_length_px,
        save_shape,
        output / f"hybrid_owned_displacement_x{args.displacement_scale:g}.ply",
    )

    summary = {
        "purpose": "K5-D.2b owned-Gaussian visual diagnostic",
        "total_gaussians": int(total_gaussians),
        "gaussians_per_layer": int(expected_per_layer),
        "owned_visible_gaussians": int(len(owned_idx)),
        "owned_fraction_of_visible_layer": float(
            len(owned_idx) / expected_per_layer
        ),
        "owned_fraction_of_total_scene": float(
            len(owned_idx) / total_gaussians
        ),
        "world_position_displacement_m": stats(displacement_norm),
        "linear_rgb_change_norm": stats(color_delta),
        "singular_value_change_norm": stats(scale_delta),
        "opacity_abs_change": stats(opacity_delta),
        "displacement_scale_visualization": float(
            args.displacement_scale
        ),
        "outputs": {
            "baseline_owned_only": "baseline_owned_only.ply",
            "hybrid_owned_only": "hybrid_owned_only.ply",
            "highlight_full": "hybrid_owned_highlight_full.ply",
            "exaggerated_owned": (
                f"hybrid_owned_displacement_x{args.displacement_scale:g}.ply"
            ),
        },
        "notes": [
            "Highlight/exaggerated PLYs are diagnostic and not production outputs.",
            "The owned-only PLYs contain exactly the K3 owned visible Gaussian cells.",
            "No pipeline state or correction is recomputed.",
        ],
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
