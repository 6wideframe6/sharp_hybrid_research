"""K4 only: robust affine relative-disparity -> SHARP metric inverse-depth fitting.

No replacement mask, fusion, surface ownership or Gaussian operations. Regional
labels below identify fitting interiors only. Rejected attempts are diagnostics,
not calibrated geometry. All regression arithmetic is float64.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class AlignmentConfig:
    sigma: float = 24.0
    truncate: float = 3.0
    border: int = 16
    shoulder: int = 3
    rgb_edge_threshold: float = 0.035  # gradient of RGB in [0,1], not depth units
    depth_edge_threshold: float = 0.035  # per-pixel gradient / robust depth spread
    min_region_pixels: int = 64
    min_kernel_mass: float = 0.05
    block_size: int = 32
    depth_bins: int = 4
    bucket_cap: int = 64
    min_anchors: int = 128
    min_blocks: int = 4
    folds: int = 3
    iterations: int = 15
    trim_fraction: float = 0.05
    min_anchor_spread_fraction: float = 0.02
    heldout_median_limit: float = 0.1
    heldout_p90_limit: float = 0.25
    scale_variation_limit: float = 0.2
    seed: int = 0

    def __post_init__(self):
        if (self.sigma < 0 or self.truncate <= 0 or self.border < 0 or self.shoulder < 0
                or self.block_size < 1 or self.depth_bins < 1 or self.bucket_cap < 1
                or self.folds < 2 or self.min_blocks < self.folds or self.min_anchors < 4
                or not 0 <= self.trim_fraction < .5 or self.iterations < 1):
            raise ValueError("Invalid alignment configuration")


def _spread(values):
    return float(np.percentile(values, 90) - np.percentile(values, 10))


def _resolvable(values):
    # Relative to numerical representation, not an absolute depth/slope threshold.
    scale = max(float(np.max(np.abs(values))), np.finfo(np.float64).tiny)
    return _spread(values) > 64 * np.finfo(np.float64).eps * scale


def sample_sharp_inverse_native(depth_grid, native_hw, box):
    """Invert the authoritative metric grid BEFORE align_corners=True sampling.

    Output depth is reciprocal of the sampled inverse depth, not an independent
    interpolation of metric depth. Crops are half-open native [x0,y0,x1,y1].
    """
    z = np.asarray(depth_grid)
    h, w = native_hw
    x0, y0, x1, y1 = box
    if (z.ndim != 2 or min(z.shape) < 2 or min(h, w) < 2
            or not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h)):
        raise ValueError("Invalid grid/native dimensions/crop")
    if not (np.isfinite(z).all() and (z > 0).all()):
        raise ValueError("Captured SHARP depth must be finite positive metric z")
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    yy *= (z.shape[0] - 1) / (h - 1)
    xx *= (z.shape[1] - 1) / (w - 1)
    q_grid = 1.0 / z.astype(np.float64)
    return ndi.map_coordinates(q_grid, [yy, xx], order=1, mode="nearest", prefilter=False)


def restricted_smoothing(values, valid, regions, config):
    """Normalized Gaussian filtering separately within each fitting-interior ID.

    Invalid and boundary pixels never enter numerators. Different IDs never mix.
    A 3-sigma shoulder erosion is not needed across excluded boundaries because
    both numerators and denominators are restricted to the same interior; actual
    kernel support/mass is returned and low-support anchors are excluded.
    """
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, bool) & np.isfinite(values) & (regions > 0)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    mass = np.zeros_like(values)
    if config.sigma == 0:
        output[valid], mass[valid] = values[valid], 1.
        return output, mass
    radius = math.ceil(config.sigma * config.truncate)
    for region_id in np.unique(regions[valid]):
        region_mask = valid & (regions == region_id)
        ys, xs = np.where(region_mask)
        sl = (slice(max(0, ys.min() - radius), min(values.shape[0], ys.max() + radius + 1)),
              slice(max(0, xs.min() - radius), min(values.shape[1], xs.max() + radius + 1)))
        mask = region_mask[sl]
        weight = ndi.gaussian_filter(mask.astype(float), config.sigma, mode="constant",
                                     truncate=config.truncate)
        numerator = ndi.gaussian_filter(np.where(mask, values[sl], 0.), config.sigma,
                                        mode="constant", truncate=config.truncate)
        output[sl][mask] = numerator[mask] / weight[mask]
        mass[sl][mask] = weight[mask]
    return output, mass


def make_fitting_maps(raw, sharp_inverse, rgb, wire_mask, *, unpadded=None,
                      region_labels=None, config=AlignmentConfig()):
    """Fitting-anchor exclusions only; does not decide where to replace SHARP."""
    raw, target = np.asarray(raw, float), np.asarray(sharp_inverse, float)
    wire_mask = np.asarray(wire_mask, bool)
    if raw.shape != target.shape or raw.ndim != 2 or rgb.shape != (*raw.shape, 3) or wire_mask.shape != raw.shape:
        raise ValueError("Native RGB/depth/mask coordinates must match")
    valid = np.isfinite(raw) & np.isfinite(target) & (target > 0)
    if unpadded is not None:
        if unpadded.shape != raw.shape:
            raise ValueError("Unpadded mask shape mismatch")
        valid &= unpadded
    rgb_float = np.asarray(rgb, float) / 255.
    rgb_gradient = np.sqrt(sum(np.sum(g * g, axis=2) for g in np.gradient(rgb_float, axis=(0, 1))))
    safe_target = np.where(valid, target, np.median(target[valid]) if valid.any() else 0.)
    spread = _spread(target[valid]) if valid.any() else 0.
    target_grad = np.hypot(*np.gradient(safe_target))
    strong_depth = (target_grad > config.depth_edge_threshold * spread) if spread > 0 else target_grad > 0
    strong_rgb = rgb_gradient > config.rgb_edge_threshold
    barriers = wire_mask | strong_rgb | strong_depth | ~valid
    if config.shoulder:
        barriers = ndi.binary_dilation(barriers, iterations=config.shoulder)
    interior = valid & ~barriers
    if config.border:
        n = config.border
        interior[:n] = False
        interior[-n:] = False
        interior[:, :n] = False
        interior[:, -n:] = False
    if region_labels is None:
        regions, _ = ndi.label(interior)
    else:
        if region_labels.shape != raw.shape:
            raise ValueError("Fitting region labels have wrong coordinates")
        regions = np.where(interior, region_labels, 0).astype(np.int32)
    counts = {int(i): int(np.sum(regions == i)) for i in np.unique(regions) if i > 0}
    for label, count in counts.items():
        if count < config.min_region_pixels:
            regions[regions == label] = 0
    interior &= regions > 0
    r_fit, r_mass = restricted_smoothing(raw, interior, regions, config)
    q_fit, q_mass = restricted_smoothing(target, interior, regions, config)
    anchor = interior & (r_mass >= config.min_kernel_mass) & (q_mass >= config.min_kernel_mass)
    return {"anchor_mask": anchor, "regions": regions, "raw_fit": r_fit, "target_fit": q_fit,
            "kernel_mass": np.minimum(r_mass, q_mass), "excluded_barriers": barriers,
            "stats": {"candidate_anchors": int(valid.sum()), "interior_anchors": int(interior.sum()),
                      "accepted_anchor_candidates": int(anchor.sum()),
                      "interior_groups": len(np.unique(regions[anchor])),
                      "known_wire_pixels": int(wire_mask.sum()), "rgb_boundary_pixels": int(strong_rgb.sum()),
                      "sharp_boundary_pixels": int(strong_depth.sum()),
                      "smoothing": "normalized Gaussian within disconnected fitting interiors, not surface ownership",
                      "kernel_support_radius_native_px": math.ceil(config.sigma * config.truncate)}}


def stratify_anchors(target, mask, regions, config):
    """Cap each (interior, target-quantile, spatial-block) bucket.

    Each (interior, quantile) pair has equal total mass, then capped sample-count
    block mass within the pair. Sparse buckets are not amplified to the mass of
    full buckets: doing so can turn a minority of spatially scattered outliers
    into a weighted majority. Large buckets still cannot exceed bucket_cap.
    """
    indices = np.flatnonzero(mask)
    if not len(indices):
        return {"indices": indices, "weights": np.array([]), "blocks": indices,
                "bins": indices, "regions": indices, "buckets": []}
    width = target.shape[1]
    ncols = math.ceil(width / config.block_size)
    blocks = (indices // width // config.block_size) * ncols + indices % width // config.block_size
    edges = np.unique(np.quantile(target.flat[indices], np.linspace(0, 1, config.depth_bins + 1)))
    bins = np.searchsorted(edges[1:-1], target.flat[indices], side="right")
    labels = regions.flat[indices]
    keys = np.column_stack((labels, bins, blocks))
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    rng = np.random.default_rng(config.seed)
    chosen, chosen_blocks, chosen_bins, chosen_regions, weights, buckets = [], [], [], [], [], []
    pairs = np.unique(unique_keys[:, :2], axis=0)
    for bucket_id, (region, bin_id, block) in enumerate(unique_keys):
        group = indices[inverse == bucket_id]
        subset = np.sort(rng.choice(group, min(len(group), config.bucket_cap), replace=False))
        pair_bucket_ids = np.flatnonzero(np.all(unique_keys[:, :2] == [region, bin_id], axis=1))
        pair_count = sum(min(int(np.sum(inverse == k)), config.bucket_cap) for k in pair_bucket_ids)
        mass = len(subset) / (len(pairs) * pair_count)
        chosen.extend(subset)
        chosen_blocks.extend([block] * len(subset))
        chosen_bins.extend([bin_id] * len(subset))
        chosen_regions.extend([region] * len(subset))
        weights.extend([mass / len(subset)] * len(subset))
        buckets.append({"region": int(region), "depth_bin": int(bin_id), "block": int(block),
                        "available": len(group), "selected": len(subset), "weight_mass": mass})
    return {"indices": np.asarray(chosen, dtype=np.int64), "weights": np.asarray(weights),
            "blocks": np.asarray(chosen_blocks), "bins": np.asarray(chosen_bins),
            "regions": np.asarray(chosen_regions), "buckets": buckets}


class DegenerateFit(ValueError):
    pass


def robust_affine(x, y, weights, config):
    """Weighted LS -> normalized float64 Huber IRLS, optional residual trimming.

    Returns attempted coefficients; callers enforce a>0 as an acceptance gate.
    A nonpositive attempt never produces a usable alignment mask.
    """
    x, y, weights = (np.asarray(v, np.float64) for v in (x, y, weights))
    if len(x) < 4 or not (_resolvable(x) and _resolvable(y)):
        raise DegenerateFit("unresolvable predictor or target variation")
    mx, my, sx, sy = np.median(x), np.median(y), _spread(x), _spread(y)
    xn, yn = (x - mx) / sx, (y - my) / sy
    design = np.column_stack((xn, np.ones_like(xn)))
    work = weights / weights.sum()
    coefficient = np.zeros(2)
    for iteration in range(config.iterations + 1):
        previous = coefficient
        matrix = design * np.sqrt(work[:, None])
        if np.linalg.matrix_rank(matrix) != 2:
            raise DegenerateFit("rank-deficient weighted affine fit")
        coefficient = np.linalg.lstsq(matrix, yn * np.sqrt(work), rcond=None)[0]
        residual = design @ coefficient - yn
        sigma = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        delta = max(1.345 * sigma, 64 * np.finfo(float).eps)
        robust_weight = np.minimum(1., delta / np.maximum(np.abs(residual), np.finfo(float).tiny))
        if config.trim_fraction:
            robust_weight[np.abs(residual) > np.quantile(np.abs(residual), 1 - config.trim_fraction)] = 0
        work = weights * robust_weight
        if work.sum() <= 0:
            raise DegenerateFit("no IRLS weight remains")
        work /= work.sum()
        if iteration and np.linalg.norm(coefficient - previous) < 1e-11:
            break
    a = float(coefficient[0] * sy / sx)
    b = float(my + coefficient[1] * sy - a * mx)
    return a, b


def align_affine(raw, sharp_inverse, anchor_mask, *, raw_fit=None, target_fit=None,
                 regions=None, foreground_mask=None, config=AlignmentConfig()):
    """Fit/validate a*r+b in 1/metre; return diagnostics even for rejected fits."""
    raw, target = np.asarray(raw, float), np.asarray(sharp_inverse, float)
    if raw.ndim != 2 or target.shape != raw.shape or anchor_mask.shape != raw.shape:
        raise ValueError("Arrays must share a native 2D crop coordinate system")
    xmap = raw if raw_fit is None else np.asarray(raw_fit, float)
    ymap = target if target_fit is None else np.asarray(target_fit, float)
    regions = np.ones(raw.shape, np.int32) if regions is None else np.asarray(regions)
    if any(a.shape != raw.shape for a in (xmap, ymap, regions)):
        raise ValueError("Fitting maps/regions shape mismatch")
    if foreground_mask is not None and foreground_mask.shape != raw.shape:
        raise ValueError("Foreground diagnostic mask shape mismatch")
    valid = (np.asarray(anchor_mask, bool) & np.isfinite(raw) & np.isfinite(target) & (target > 0)
             & np.isfinite(xmap) & np.isfinite(ymap) & (ymap > 0) & (regions > 0))
    sample = stratify_anchors(ymap, valid, regions, config)
    indices, weights, blocks = sample["indices"], sample["weights"], sample["blocks"]
    report = {"input_units": "arbitrary relative disparity", "target_units": "1/metre",
              "output_units": "1/metre", "equation": "q=a*r+b; raw r is never inverted",
              "config": asdict(config), "candidate_anchors": int(np.asarray(anchor_mask, bool).sum()),
              "finite_anchor_candidates": int(valid.sum()), "usable_anchors": len(indices),
              "spatial_blocks": len(np.unique(blocks)), "target_depth_bins": len(np.unique(sample["bins"])),
              "surface_interior_groups": len(np.unique(sample["regions"])), "buckets": sample["buckets"],
              "a": None, "b": None, "positive_slope": None, "target_anchor_range": None,
              "heldout_normalized_median": None, "heldout_normalized_p90": None,
              "fold_scale_variation": None, "foreground_extrapolation": None,
              "invalid_transformed_pixels": None, "folds": [], "status": "underdetermined",
              "accepted": False, "reasons": [], "attempt_only": True}
    aligned = np.full(raw.shape, np.nan, float)
    selected = np.zeros_like(valid)
    selected.flat[indices] = True
    result = {"report": report, "aligned": aligned, "valid_transformed": np.zeros_like(valid),
              "selected_anchor_mask": selected, "sample": sample}
    if len(indices) < config.min_anchors or len(np.unique(blocks)) < config.min_blocks:
        report["reasons"].append("insufficient anchors/spatial blocks for affine identification and holdout")
        return result
    x, y = xmap.flat[indices], ymap.flat[indices]
    report["target_anchor_range"] = {"min": float(y.min()), "max": float(y.max()),
                                     "p10": float(np.percentile(y, 10)), "p90": float(np.percentile(y, 90)),
                                     "p90_minus_p10": _spread(y)}
    full_raw = raw[np.isfinite(raw)]
    full_target = target[np.isfinite(target) & (target > 0)]
    if not (_resolvable(x) and _resolvable(y)):
        report["reasons"].append("nearly constant TinyViM predictor or SHARP target anchors")
        return result
    xspan, yspan = _spread(x), _spread(y)
    report["anchor_spread_fractions"] = {
        "raw": xspan / max(_spread(full_raw), np.finfo(float).tiny),
        "target": yspan / max(_spread(full_target), np.finfo(float).tiny)}
    weak_range = min(report["anchor_spread_fractions"].values()) < config.min_anchor_spread_fraction
    try:
        a, b = robust_affine(x, y, weights, config)
    except DegenerateFit as exc:
        report["reasons"].append(str(exc))
        return result
    report.update(a=a, b=b, positive_slope=bool(a > 0))
    # Do not transform a nonpositive-slope attempt into usable physical geometry.
    with np.errstate(invalid="ignore", over="ignore"):
        algebraic_attempt = a * raw + b
    algebraic_valid = np.isfinite(algebraic_attempt) & (algebraic_attempt > 0)
    report["invalid_transformed_pixels"] = int((~algebraic_valid).sum())
    report["invalid_count_definition"] = "nonfinite or nonpositive a*r+b, even if slope is rejected"
    if a > 0 and np.isfinite([a, b]).all():
        aligned = algebraic_attempt
        result["aligned"] = aligned
        transformed_valid = algebraic_valid
    else:
        transformed_valid = np.zeros_like(valid)
        report["reasons"].append("nonpositive slope: rejected, not clipped or sign-flipped")
    report["withheld_due_to_nonpositive_slope"] = int(raw.size if a <= 0 else 0)
    fg = np.isfinite(raw) if foreground_mask is None else np.asarray(foreground_mask, bool) & np.isfinite(raw)
    if fg.any():
        values = raw[fg]
        distance = np.maximum.reduce((x.min() - values, values - x.max(), np.zeros_like(values)))
        report["foreground_extrapolation"] = {
            "raw_foreground_range": [float(values.min()), float(values.max())],
            "raw_anchor_range": [float(x.min()), float(x.max())],
            "outside_anchor_range_fraction": float(np.mean(distance > 0)),
            "max_distance_in_anchor_spreads": float(distance.max() / xspan),
            "p90_distance_in_anchor_spreads": float(np.percentile(distance, 90) / xspan),
            "label": "known RGB wire proxy; extrapolation is not distance accuracy"}
    unique_blocks = np.unique(blocks)
    np.random.default_rng(config.seed).shuffle(unique_blocks)
    scales, attempted_scales, all_errors, bad_folds = [], [], [], 0
    crossfit = config.sigma > 0 and raw_fit is not None and target_fit is not None
    report["holdout_smoothing"] = ("recomputed separately using training-only and heldout-only raw samples"
                                    if crossfit else "no smoothing shared across folds")
    for number, validation_blocks in enumerate(np.array_split(unique_blocks, config.folds)):
        validation = np.isin(blocks, validation_blocks)
        training = ~validation
        fold = {"fold": number, "training_blocks": np.unique(blocks[training]).tolist(),
                "validation_blocks": validation_blocks.tolist(), "training_count": int(training.sum()),
                "validation_count": int(validation.sum())}
        try:
            if (len(np.unique(blocks[training])) < 2 or training.sum() < config.min_anchors // 2
                    or not validation.any()):
                raise DegenerateFit("insufficient fold support")
            xf, yf = x, y
            if crossfit:
                yy, xx = np.mgrid[:raw.shape[0], :raw.shape[1]]
                block_map = (yy // config.block_size) * math.ceil(raw.shape[1] / config.block_size) + xx // config.block_size
                validation_domain = np.isin(block_map, validation_blocks)
                xf, yf = np.full_like(x, np.nan), np.full_like(y, np.nan)
                for subset, domain in ((training, ~validation_domain), (validation, validation_domain)):
                    rf, rm = restricted_smoothing(raw, valid & domain, regions, config)
                    tf, tm = restricted_smoothing(target, valid & domain, regions, config)
                    supported = (rm.flat[indices] >= config.min_kernel_mass) & (tm.flat[indices] >= config.min_kernel_mass)
                    use = subset & supported
                    xf[use], yf[use] = rf.flat[indices[use]], tf.flat[indices[use]]
                training &= np.isfinite(xf) & np.isfinite(yf)
                validation &= np.isfinite(xf) & np.isfinite(yf)
                if training.sum() < config.min_anchors // 2 or not validation.any():
                    raise DegenerateFit("insufficient disjoint-kernel fold support")
            fold.update(training_count=int(training.sum()), validation_count=int(validation.sum()))
            fa, fb = robust_affine(xf[training], yf[training], weights[training], config)
            error = np.abs(fa * xf[validation] + fb - yf[validation]) / yspan
            attempted_scales.append(fa)
            all_errors.extend(error)
            fold.update(a=fa, b=fb, median=float(np.median(error)), p90=float(np.percentile(error, 90)),
                        usable=bool(fa > 0), positive_slope=bool(fa > 0))
            if fa > 0:
                scales.append(fa)
            else:
                bad_folds += 1
                fold["reason"] = "nonpositive fold slope; residual reported for diagnosis only"
        except DegenerateFit as exc:
            bad_folds += 1
            fold.update(usable=False, reason=str(exc))
        report["folds"].append(fold)
    if all_errors:
        report["heldout_normalized_median"] = float(np.median(all_errors))
        report["heldout_normalized_p90"] = float(np.percentile(all_errors, 90))
    if len(scales) >= 2:
        report["fold_scale_variation"] = float(np.ptp(scales) / np.median(scales))
    report["attempted_fold_slopes"] = attempted_scales
    if weak_range:
        report["reasons"].append("anchor variation covers too little of the available predictor/target range")
    if bad_folds:
        report["reasons"].append("one or more spatial holdout folds cannot identify a positive stable scale")
    if all_errors and report["heldout_normalized_median"] >= config.heldout_median_limit:
        report["reasons"].append("held-out normalized median exceeds development gate")
    if all_errors and report["heldout_normalized_p90"] >= config.heldout_p90_limit:
        report["reasons"].append("held-out normalized p90 exceeds development gate")
    if report["fold_scale_variation"] is not None and report["fold_scale_variation"] >= config.scale_variation_limit:
        report["reasons"].append("fold scale variation exceeds development gate")
    if weak_range or bad_folds:
        report["status"] = "underdetermined"
    elif report["reasons"]:
        report["status"] = "rejected"
    else:
        report.update(status="accepted", accepted=True, attempt_only=False)
        result["valid_transformed"] = transformed_valid
    return result
