# ML-SHARP + local TinyViM: architecture and implementation investigation

**Latest execution result:** [K4.md](K4.md) supersedes this original report's
checkpoint/K4 status. Full released-checkpoint capture/replay and real baseline
export passed exactly. K4 is implemented and tested; the two cached far-wire fits
are underdetermined, and verified expanded/shifted contexts remain rejected by
fixed held-out residual gates. K5/K6 remain future work.

Date: 2026-09-18. Workspace: `/mnt/c/depth_issue`.

**Implementation follow-up:** K3's capture/replay and preserved-secondary adapter
are now implemented in `sharp_adapter.py`, with22 passing source-contract tests
and1 skipped optional released-checkpoint test. See [K3.md](K3.md) for the actual
API and verification. The investigation below is retained as the original design
record; references to K3 as a proposed module are superseded by that implementation.
Alignment, fusion, full-checkpoint inference and rendered quality validation remain
future stages.

## Decision summary

**Proceed with global SHARP, selective TinyViM-S crops, robust affine inverse-depth alignment, and surface-constrained gradient reconstruction.** Use hard surface ownership at occlusion boundaries, rather than feathering foreground depth into background. Preserve SHARP's original normalization, encoder features, and secondary Gaussian parameters explicitly.

Three gates precede a quality claim:

1. The local model must actually resolve the structure before fusion. Existing TinyViM-S results justify this direction, but only on one bridge image.
2. Corrected geometry must survive SHARP's 1536×1536 input-depth grid, 768×768 Gaussian grid, and learned Gaussian offsets/scales/opacities. A beautiful native-resolution depth map is insufficient.
3. Source and nearby-view renders must show thinner, correctly placed geometry without a halo. Poisson reconstruction is not inherently halo-free.

**Status:** workspace/source/paper investigation completed; a small executable source-contract/synthetic probe is included and passed. The full hybrid estimator and SHARP checkpoint inference have **not** been implemented or run. No actual SHARP depth, hybrid PLY, or rendered quality result is claimed. Future files/commands are identified as such below.

## A. Existing Workspace Audit

### A1. Current setup, not historical setup

The top-level workspace contains `src/`, `scripts/`, `results/`, `experiments/model_compare/`, `third_party/ZipDepth/`, `venv/`, `.cache/pip/`, and the original `bridge.png`. It is a research folder, not one top-level Git repository. No `AGENTS.md` was found beneath it. The only discovered virtual-environment configuration is `venv/pyvenv.cfg`. The only discovered notebook is upstream DA3's `notebooks/da3.ipynb`; no separate fusion notebook was found.

Verified current environment:

| Property | Observed |
|---|---|
| Interpreter | `/mnt/c/depth_issue/venv/bin/python`, Python 3.10.12 |
| Torch | package version 2.4.1; historical build recorded as cu121 |
| Torchvision | 0.19.1+cpu |
| timm / NumPy / OpenCV | 0.9.16 / 2.1.3 / 4.11.0.86 |
| CUDA | `torch.cuda.is_available() == False`; `nvidia-smi` unavailable |
| Dependencies | `python -m pip check` passed; SciPy absent |
| CPU record | existing `results/tinyvim_local/environment.json`: i7-13700, WSL |
| Environment activation | shell `python` initially unavailable; use explicit `venv/bin/python` |

The older README and reports reference `.venv`, `/mnt/e/weird_stuff/depth_issue`, and RTX 3060 Ti 8 GB inference. Those are historical provenance, **not the current execution environment**. Conda and Python 3.13 were not found on PATH during this audit.

There was no ML-SHARP checkout or matching `sharp*.pt` checkpoint under this workspace. For this investigation, the official repository was cloned into:

```text
/mnt/c/depth_issue/third_party/ml-sharp
commit aed6527499ef91cba3b54c18d49a870f25947190
```

Its working tree was clean after cloning. No SHARP weights or dependencies were installed. Upstream recommends Python 3.13; its lock file uses torch 2.8.0, torchvision 0.23.0, timm 1.0.20, NumPy 2.3.3, SciPy 1.16.2, gsplat 1.5.3. Installing that lock file into the existing Python 3.10 research environment is not a drop-in continuation. Keep the validated TinyViM environment; provision a separate SHARP environment when running the learned model and exchange arrays/cache manifests between them. Python 3.13 is a recommendation/tooling target here, not a `requires-python` declaration in the inspected package metadata.

### A2. Reusable code and artifacts

| Existing location | Actual purpose / reuse |
|---|---|
| `src/zipdepth_backend.py`, `src/backend.py` | Validated raw inverse-depth baseline and backend interface |
| `src/depthart_backend.py` | TinyViM-S adapter delegating to the previously validated model-comparison wrapper |
| `experiments/model_compare/backends.py` | Strict checkpoint loading; model-specific normalization; isotropic crops; pad/unpad and pixel-center native mapping |
| `src/prediction_cache.py` | Provenance-checked input/checkpoint/box/resolution cache; reuse and extend with source/preprocessing hashes |
| `src/detail_detection.py`, `src/roi.py` | RGB ridge/line proposals and bridge diagnostic coordinates; useful proposal baseline, not a general thin-object segmenter |
| `src/alignment.py` | Huber/trimmed affine regression, value-balanced anchors, spatial holdout, common-bandwidth fitting |
| `src/fusion.py` | Existing residual/high-pass/feather variants; retain as historical ablations, not the new main fusion |
| `src/diagnostics.py`, `src/visualization.py` | Profiles, gradients, FWHM proxies and identical-coordinate panels; extend with true labels and halo measurements |
| `scripts/run_experiment.py` | Current runner supports CPU, ZipDepth global, TinyViM local and `--reuse-from`; it does not support SHARP global |
| `scripts/validate_tinyvim_cpu.py` | Pending raw CPU parity and ROI 0/5 alignment probe; **not completed** according to available artifacts |
| `results/final/` | Frozen preferred ZipDepth result; authoritative comparison baseline |
| `results/iteration1` through `iteration4_budget10` | Alignment/mask/coverage failures and improvements |
| `results/05_alignment_diagnostic/`, `06_fusion_diagnostic/`, `07_quality_variants/`, `08_remaining/` | Cached alignment, masking, context, guidance and far-wire evidence |
| `experiments/model_compare/outputs/`, `profiles/`, `regions/` | Raw alternative-model evidence, including TinyViM recovery; reuse without re-inference |
| `results/tinyvim_local/` | Only environment, frozen hashes and progress note present; no completed CPU validation/full TinyViM fusion result |

Existing repositories: ZipDepth (`91f3fd21e131641f51e8d35736d1958350180e3a`, recorded provenance), DepthART (`b307c123f19051e2d969960bba93ee3684874b90`, checked clean), DA3 (`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`, recorded), Apple Depth Pro (`9e65e4dbe9568d23c546fcec53302b10445e109e`, recorded). The last three reside under `experiments/model_compare/`, not top-level `third_party/`.

Weights exist under `experiments/model_compare/weights/{depthart,depthart_tiny,da3,depthpro}` and ZipDepth's checkpoint directory. The promising local checkpoint is:

```text
experiments/model_compare/weights/depthart_tiny/relative/depthart_relative_s_448.pth
SHA256 e17adf70a87b4d2b7665bf0546aad68f8c5b8b63866cbca604602a7859fadebe
```

No saved SHARP features, depth tensors, camera metadata or splats were found in the audited research outputs. `ml-depth-pro` is a separate metric-depth model, not the SHARP setup.

### A3. What the existing experiments establish

Read in full: top-level `PROGRESS.md`, `README.md`, `results/final/REPORT.md`, `experiments/model_compare/REPORT.md`, and `results/tinyvim_local/PROGRESS.md`. Also inspected the actual alignment/fusion/backend/runner code and opened `results/final/crops/right_suspenders_strip.png` and `experiments/model_compare/regions/far_1480_192.png` during this audit.

* Native sharp-local versus blurred-global fitting rejected cable-only ROIs. Common-bandwidth inverse-depth fitting at native sigma 24 accepted all six preferred ROIs; the transform was applied to the original sharp local map.
* Feathered high-pass residual fusion improved cable continuity, but retained thickening, halos, incorrect lattice-hole depth and coverage transitions. Final median depth FWHM versus RGB ridge width was 9/4, 7/3, 5/2 px for left/upper/right. These are proxies, not depth truth.
* Hard/dilated RGB-edge masks around a **signed high-pass residual** cut grooves and fragmented existing diagonals. Positive-only novel-foreground masks lost junctions. Guided filtering transferred clamp/texture details and failed to separate pairs.
* Ten ROIs increased coverage from about 31.6% to 56.1% without resolving the finest ZipDepth far wires. Native192→input512 magnification also failed for ZipDepth.
* The subsequent **raw-model** comparison changes the model conclusion: TinyViM-S recovered separate far-wire ridges. On the two far crops, reported foreground-center contrast proxies were 84.0%/95.8%, versus ZipDepth 12.0%/12.5%. TinyViM far median profile width was 3 px versus RGB 3 px, but near wires were still over-wide and tower-adjacent background was wrong.
* TinyViM-B/L and an explicitly identified ViT-L local model were **not** evaluated in those artifacts. DA3-Small and Depth Pro are not evidence about every ViT-L depth checkpoint.

The current task begins where these observations end: local neural evidence exists, but width, surface ownership, robust cross-model units, SHARP sampling and Gaussian integration remain unresolved.

## B. ML-SHARP Internals

Source references below are relative to `third_party/ml-sharp/src/sharp/` at the pinned commit. Shapes are derived from source/default CLI configuration, not observed from a checkpoint run.

### B1. Exact inference data flow

```text
cli/predict.py::predict_image
  original full RGB [H,W,3], focal f_px
  image [B,3,1536,1536], bilinear align_corners=True (square resize)
  disparity_factor k = f_px / original_width
      ↓
models/predictor.py::RGBGaussianPredictor.forward
  monodepth_output = self.monodepth_model(image)
      ↓
models/monodepth.py::MonodepthWithEncodingAdaptor.forward
  RGB [0,1] → [-1,1]
  Depth-Pro-style SlidingPyramidNetwork encoder
  depth decoder + replicated two-channel ReLU disparity head
  MonodepthOutput(disparity, encoder_features, decoder_features, output_features, ...)
      ↓
  monodepth = k / disparity.clamp(1e-4, 1e4)
  self.depth_alignment(monodepth, depth=None, decoder_features) → identity
      ↓  ← semantic depth-override location
  init_output = self.init_model(image, monodepth)
      ↓
  self.feature_model(init_output.feature_input, encodings=output_features)
  self.prediction_head(image_features) → delta_values
  self.gaussian_composer(delta_values, base_values, global_scale)
      ↓
cli/predict.py → unproject_gaussians → save_ply
```

This is one global SHARP inference. Its **internal pretrained encoder** already uses 25+9+1 overlapping 384-pixel patches and a low-resolution image encoder (`encoders/spn_encoder.py:205–310`). Leave that architecture intact. It is different from running independent SHARP scenes on external crops.

There is no ordinary per-scene iterative Gaussian optimization/refinement after this CLI prediction. The learned decoder predicts parameter deltas in a feed-forward pass; composition, camera unprojection and export follow.

### B2. Tensor contract

| Tensor | Default shape / semantics |
|---|---|
| `image` | `[B,3,1536,1536]`, RGB float `[0,1]` before monodepth normalization |
| `monodepth_output.disparity` | `[B,2,1536,1536]`, nonnegative canonical disparity; not meters and not simply `1/z` |
| `monodepth` | `[B,2,1536,1536]`, camera-scaled metric depth `k / clamp(disparity)` |
| `encoder_features` / `output_features` | Default **five** assembled levels: `[B,256,768,768]`, `[B,256,384,384]`, `[B,512,192,192]`, `[B,1024,96,96]`, `[B,1024,48,48]` |
| `decoder_features` | `[B,256,768,768]`; not included in `output_features` by default |
| `init_output.feature_input` | `[B,5,1536,1536]`; concatenated RGB + both normalized inverse-depth channels, mapped by `2*x-1` |
| `global_scale` | `[B]`, approximately the minimum metric depth across **both** layers |
| `GaussianBaseValues.mean_inverse_z_ndc` | `[B,1,2,768,768]`, inverse normalized depth after 2×2 max-disparity pooling |
| `mean_x_ndc`, `mean_y_ndc` | `[B,1,2,768,768]`, regular NDC grid |
| base `scales` / `colors` | `[B,1,2,768,768]` / `[B,3,2,768,768]`; rotation and opacity use broadcastable constants |
| `ImageFeatures` | `texture_features`, `geometry_features`, each `[B,32,768,768]` |
| `delta_values` | `[B,14,2,768,768]`; xyz, scales, quaternion, RGB, opacity |
| final `Gaussians3D` | `N=2*768*768=1,179,648`; means/scales/colors `[B,N,3]`, quaternions `[B,N,4]`, opacities `[B,N]` |

The paper describes four encoder feature maps conceptually; the released SPN returns five assembled levels. Match the code when caching/replaying.

The disparity clamp implies `z ∈ [k/1e4, k/1e-4]` before initializer normalization. These are computational bounds, not an empirically verified scene-depth range. Save actual per-layer quantiles and invalid/clamped counts during the first checkpoint run.

### B3. Two layers do exist, but their meaning is qualified

`params.py:199–201` sets two layers and **sorting off**. `monodepth.py:244–251` duplicates the final convolution during construction, then the checkpoint supplies learned weights. Channel 0 has primary visible-depth supervision in the paper. Channel 1 may represent occluded content and view-dependent effects, with a smoothness regularizer. It is not guaranteed to be a strictly ordered back surface or complete hidden-scene reconstruction.

Distinguish:

1. The first depth channel: a view-synthesis-adapted visible-surface estimate.
2. The second depth channel: learned auxiliary geometry, possibly occluded/view-dependent.
3. Encoder/decoder features: multidimensional latent tensors, not depth layers.
4. Gaussian parameters: learned position, scale, orientation, color, opacity; both layers can move from their base surfaces.
5. Inpainting: the paper uses novel-view perceptual supervision to encourage plausible disocclusion. There is no separate arbitrary 360° geometry-inpainting module in this inference graph. Its stated target is nearby-view/headbox motion.

### B4. Three hidden dependencies matter

**Normalization:** `_rescale_depth` (`initializer.py:281–297`) computes one depth factor from the minimum over both channels, then clamps normalized depth to 100. A new nearer sample can change all normalized depths, far-depth clipping, the decoder input and eventual Gaussians.

**Decoder coupling:** `prepare_feature_input` concatenates both depths; the Gaussian decoder mixes that with cached encoder features. Re-running it after changing only channel 0 may change deltas for both layers and pixels outside the edited area. Group normalization also means influence is not bounded by a small local convolution radius.

**Final depth differs from injected depth:** `composer.py:198–208` uses

```text
q_final_norm = softplus(inverse_softplus(q_base_norm) + 0.001 * delta_z)
z_final_norm = 1 / (q_final_norm + 0.001)
```

and changes scales relative to the predicted means. Thus even zero delta does not give an exactly identical reciprocal due to the `0.001` stabilizer. Do not bypass this learned contract or claim exact final visible depth after an override.

## C. Integration Point

### C1. Least-invasive semantic insertion

In `models/predictor.py::RGBGaussianPredictor.forward`, insert a validated correction **after** `self.depth_alignment(...)` and **before** `self.init_model(image, monodepth)`. At inference pass `depth=None`. The existing `depth=` argument is a training-style alignment target, not a direct replacement API: it invokes a learned scale map affecting both channels.

For an initial hook-based capture, a forward hook on `monodepth_model` exposes `MonodepthOutput`; an `init_model` pre-hook sees the post-alignment metric depth. A permanent experimental wrapper should explicitly split encode/initialize/decode/compose to make cache ownership and teardown clearer. Keep upstream files clean until the no-op wrapper is validated.

**No-op gate:** replay the cached `MonodepthOutput` through the unmodified tail with the original image, camera and depth. Compare every Gaussian field against the standard forward, before unprojection and after export. Same-device deterministic tensor equality is the target; otherwise report tolerances and covariance equality rather than quaternion sign equality.

### C2. Recommended preservation adapter at `InitializerOutput`

The semantic hook above is useful for tracing, but naïvely running the initializer again fails the strongest preservation requirement. The practical first hybrid should build a corrected `InitializerOutput` **from the original one**:

1. Run global SHARP once and cache `MonodepthOutput`, original metric depths `Zs`, original `init0`, baseline `delta0`, camera/resize metadata, and baseline Gaussians.
2. Let `S = init0.global_scale`. The initializer's normalized depth is `Zn = min(Zs/S,100)`. For the corrected visible surface use the **same S**, never recompute its minimum.
3. Replace only channel 0 in `init0.feature_input`, and only where a correction is accepted. Its value is `2 * initializer.disparity_factor / Zn0 - 1` (factor defaults to 1).
4. Recompute the corrected visible base inverse depth using the original **2×2 max-pooling** of `1/Zn0`. In `GaussianBaseValues`, replace only `mean_inverse_z_ndc[:,:,0]` and visible base `scales[:,:,0]` at affected cells. Use the original formula `scales = (2*scale_factor*stride/image_width)/q_base_norm`. Keep x/y, colors, quaternions, opacities and every secondary base value unchanged.
5. Run the **original** `feature_model` and `prediction_head` with this corrected feature input and original `output_features`. No encoder or depth-decoder rerun is required.
6. Select the new 14 deltas only for accepted visible Gaussian cells. Everywhere else, including **all secondary cells**, retain `delta0`. Use a Boolean cell ownership mask, not soft cross-surface mixing.
7. Run the **original** `gaussian_composer` with selected deltas/base values and original `S`; then the original `unproject_gaussians` and `save_ply`.

This is a small orchestration/initializer-output adapter, not a new Gaussian decoder. It copies two initializer formulas; unit-check these against the pinned initializer source. If upstream later changes normalization or scale initialization, fail the contract check until the adapter is reviewed.

It preserves all secondary Gaussian inputs and therefore their composed parameters, rather than merely preserving secondary *depth*. It also preserves unedited base/delta slots exactly. Rendering can still change there due to projected splat support and occlusion; parameter preservation is not pixel preservation.

Require candidate visible normalized depth in the original computational support `[1,100]` for this first contract-preserving trial, and reject rather than silently clamp out-of-support corrections. The small epsilon in the original normalization makes its minimum marginally below one; preserve original fallback values exactly. This acceptance restriction may reject a genuinely nearer new object; count such rejected structures explicitly. Relaxing it is a later learned-decoder distribution-shift experiment.

The baseline and corrected Gaussian decoder passes are relatively small compared with the shared encoder. Once cached, many fusion trials reuse the expensive global depth/features. A **frozen-delta** ablation can reuse `delta0` without any decoder rerun, but its old color/opacity/scale predictions may suppress the new wire. It is an ablation, not evidence that base-depth replacement alone suffices.

### C3. Sampling and camera registration are part of the integration

SHARP's outer CLI stretches the full image to 1536² with `align_corners=True`; TinyViM's existing adapter uses isotropic resize/padding and native output `align_corners=False`. Preserve each model's native preprocessing and explicitly map coordinates between them. For SHARP's RGB sample centers:

```text
x_native = x_sharp * (W_native - 1) / 1535
y_native = y_sharp * (H_native - 1) / 1535
```

Retain the original CLI intrinsics construction/rescaling and `k=f_px/W`. Do not estimate a separate camera for each relative TinyViM crop. If testing a metric local model later, use crop-adjusted principal point and resize/pad intrinsics; it is not needed for the relative branch proposed here.

On `bridge.png`, one Gaussian-grid column spans about `5388/768=7.02` native pixels; one row spans `3368/768=4.39`. Gaussian offsets and anisotropic footprints can represent sub-cell content, but two separate wires within the same cell are not guaranteed to survive a single visible-depth sample.

First compare ordinary depth sampling against **surface-aware categorical rasterization**: transform accepted native foreground supports into the 1536 grid, choose a single surface value per occupied sample, then let the unchanged 2×2 min-depth initializer operate. Never average foreground and background metric depth. Log coverage and foreground fattening introduced by rasterization. Pure max-nearness footprint selection can retain a wire while making it too wide; it is not automatically the winner.

If independent 1–5 px structures collapse into the same Gaussian cells or retain background color/opacity, stop claiming the depth-only integration meets the goal. A later selective Gaussian densification/color-support mechanism may be necessary. External tiled SHARP is not the first remedy and is not part of this implementation plan.

## D. ViT-L/TinyViM Tiled Pipeline

### D1. Model identity and selection

TinyViM S/B/L are **DepthART** encoders with convolutional local blocks and selective-scan/state-space blocks, feeding a DPT head. They are not simply smaller versions of a ViT-L transformer. Inspectable paths:

```text
experiments/model_compare/DepthART/relative/models.py
experiments/model_compare/DepthART/relative/tinyvim/model/tinyvim.py
experiments/model_compare/DepthART/relative/tinyvim/model/dpt.py
```

The four encoder channel schedules are S `[48,64,168,224]`, B `[48,96,192,384]`, L `[64,128,384,512]`; DPT fusion widths are 48/48/64. Output is one full-input-resolution ReLU scalar channel, used/evaluated as affine relative disparity. The released inference head provides **no calibrated per-pixel uncertainty**. Internal high/low-frequency feature outputs are not a substitute for uncertainty.

Choose **TinyViM-S448 at input short side 512** for continuity with the positive far-crop experiment, with trained-resolution 448 as the first scale ablation. Reuse the 6.03M-parameter checkpoint already present. Test B448 then L448 only on failures plus held-out controls; neither increased capacity nor upstream average depth accuracy demonstrates narrower ropes.

“ViT-L” alone does not identify a depth estimator. No corresponding local checkpoint was found. For an explicitly reproducible comparison, use **Depth Anything V2-L relative**, `depth_anything_v2_vitl.pth`, 335.3M parameters, DINOv2 intermediate features plus DPT, official input size 518 and multiple-of-14 preprocessing. This is a proposed identified comparator, not an assertion that it is the model used in the user's earlier tests. Pin its source/checkpoint before adding it. Do not conflate it with DA3-Small, Depth Pro, or SHARP's own ViT-L encoders.

### D2. Concrete first settings

Separate native crop extent from network input size:

| Stage | Initial configuration | Why |
|---|---|---|
| SHARP | one original full-image inference, standard 1536² preprocessing | Global geometry/camera and features stay consistent |
| Optional TinyViM full-frame | short side 448, about 448×736 padded on this bridge | Context/alignment diagnostic only; never replaces SHARP low frequencies |
| Controlled continuation | existing six native 768×1536 portrait boxes → 512×1024 inputs | Isolates model/fusion changes from prior ROI selection |
| Far-wire probes | the two existing native192 squares → 512² | Reuses proven raw TinyViM evidence; test a shifted/context-expanded crop before trusting scale |
| General tile baseline | native768 square, stride384 (50% overlap), input512² | Explicit overlap for independent boundary consensus |
| Trusted tile core | central512 native pixels, 128-pixel context on each side | Discard/deprioritize border artifacts, rather than feathering depth surfaces |
| Adaptive finer tier | native384 square → 512², only uncertain thin candidates | Greater sampling density at limited extra cost |
| Very fine tier | native192 → 448/512, only where raw inference supports a real object | Proven far-wire case; high texture/context risk |
| Long structures | portrait768×1536; compare one shifted portrait or overlapping squares | Preserve along-wire context without reducing horizontal density |
| Padding | existing replicate padding to multiple32; keep pad mask | Matches validated wrapper; reflected-edge context is a later ablation |
| Batch | CPU batch1; GPU start1, test2 then4 for equal padded shapes | Existing adapter is single-image; add batching below it, not around squeeze assumptions |

These are **starting settings, not an established optimum**. The six historical portrait ROIs do not provide uniform 50% overlap and do not cover the far tower; record their actual overlap graph.

Use full-frame TinyViM only if it improves transform stability on weak-anchor tiles. SHARP already supplies global context, and a bad local-model full frame can become another source of bias. First correct the model-unit alignment guards; do not add a full-frame pass to conceal that bug.

### D3. Seam handling

Align all accepted tiles to the same SHARP inverse-depth coordinate system. Solve overlapping connected regions jointly. Use center/Hann weights for **confidence and within-surface gradient evidence**, not averaging conflicting absolute depths at boundaries. At a depth edge, select one coherent tile/scale owner along the connected contour, or reject a conflicting component. Pixelwise switching between owners can itself create seams.

Overlap disagreement is measured after alignment, using both surface values and boundary position. If two crops put a wire 3 pixels apart, averaging them can create a 6-pixel stripe even when their scales match. Prefer the crop with independently stronger location/width evidence; require an uncertainty margin to switch owners.

### D4. Performance: passes, memory and practical variants

For a regular tile extent `T` and stride `s`, a boundary-covering axis uses `1+ceil(max(0,L-T)/s)` tiles. Native768/stride384 on this bridge gives **14×8=112 tiles**, not six. Uniform multiscale processing would be expensive and would refine much irrelevant texture.

| Variant | Additional local passes per image | Concrete trade-off |
|---|---:|---|
| Cached quality investigation | 0 for cached raw boxes | Fast fusion comparisons; no new overlap/multiscale evidence |
| Minimal controlled continuation | 6 portrait + 2 far crops = 8 | Reuses strong evidence; coverage is incomplete and consistency sparse |
| Recommended selective validation | 8 above + 4 targeted shifted/context checks + optional1 full-frame = 12–13 | Tests uncertainty where it matters; no blanket coverage guarantee |
| Dense square control | 112 + optional1 full-frame | Exhaustive coverage, many texture false-positive opportunities, substantial CPU cost |
| Dense two-scale/offset ensemble | roughly 2×112 or more | Better disagreement estimates; not justified before selective variants pass |
| S/B/L voting | 3× selected tile count | Correlated teacher/training errors remain; three models are not independent ground truth |
| ViT-L comparator | same selected tiles, much larger model | Useful failure-focused quality comparison; much higher weight/activation cost |

Historical wrapper `sanity_seconds` at TinyViM512 on the two far crops was 0.840 and 0.086 seconds on CUDA. These are single-call mixed-overhead observations, **not benchmark latency or current CPU timings**. DepthART upstream A6000 PyTorch FP32/TF32 tables report S/B/L448 model-only 2.937/3.356/5.015 ms and E2E 5.860/6.310/7.876 ms. They use an optimized scan path and a different device/protocol. The reference scan wrapper here can be much slower; do not extrapolate those FPS to this CPU or historical 3060 Ti.

Useful planning arithmetic, not measurements: if a warm S512 square pass takes 0.09 s and portrait cost is roughly twice square, six portraits + two squares cost about `14*0.09=1.26 s` model/wrapper time; 112 squares cost about 10.1 s. Replace 0.09 with a measured CPU value before scheduling. If CPU square inference were 1 s, the corresponding budgets would be approximately 14 and 112 s. These scenarios exclude SHARP, alignment, solver, I/O and rendering.

S weights alone occupy about 23 MiB FP32; V2-L weights about 1.25 GiB. SHARP's paper reports 702M total parameters, about **2.62 GiB FP32 weights alone**. The five cached default encoder feature maps total about **0.82 GiB** FP32; caching the unused monodepth decoder feature adds about 0.56 GiB. Cache only what replay needs. SHARP transient attention/convolution activations and the 35-patch encoder batch are additional, so an 8 GB card is not certified by these lower bounds. Measure peak allocated/reserved memory and host RSS on an actual run. Sequential local and SHARP inference avoids unnecessary simultaneous weight residency; retaining SHARP features on CPU trades transfer latency for VRAM.

DepthART upstream S/B/L448 reported PyTorch peaks 59.3/79.8/162.7 MiB are useful references only. Portrait/512/batched/reference-scan execution differs. Allocate batch1 first and record `max_memory_allocated`, `max_memory_reserved`, RSS, synchronized wall time, preprocessing, alignment, solving, decoder and export separately. Discarding unneeded cached features, selective second-scale checks, sparse-component solves and avoiding native18MP diagnostic duplication are the first practical savings.

## E. Depth Alignment

### E1. Work in inverse metric depth

Define `q_s=1/Zs_visible=clamp(d_s,1e-4,1e4)/k`. For TinyViM raw output `r_t`, fit

```text
q_t(p) = a_t * r_t(p) + b_t,    a_t > 0.
```

The target has units 1/m. If instead fitting to canonical SHARP disparity, convert with the **same** `k`; never mix canonical disparity, metric depth and arbitrary relative inverse depth in one regression. Do not take `1/r_t` before removing its arbitrary affine offset: zero/negative denominators were already an observed failure.

Metric-depth affine fitting is a secondary comparison for a genuinely metric-output model. It is not the default for TinyViM relative weights. Median-only scale ratios assume zero shift and are insufficient; quantile/median/MAD statistics are useful for normalization and initialization, not a proof of affine calibration.

### E2. Concrete robust fit

1. Map SHARP `q_s` to the native tile coordinate frame with recorded transforms. Keep the unresampled SHARP arrays as authoritative.
2. Form a reliable anchor mask excluding invalid/padded pixels, thin candidates, strong depth/RGB discontinuities and their uncertain shoulders, and unreliable tile borders. Include spatially separated stable foreground **and** background anchors. Blue RGB alone is not a sky classifier; remove the old absolute `glob<0.003` heuristic from the generalized algorithm.
3. Match common spatial bandwidth on fitting maps only. Start with native sigma24 to compare with prior work, but use **surface-restricted normalized smoothing** and expand excluded boundary bands to cover filter support. If confident surface labels are unavailable, fit only interiors whose kernel support cannot cross a known boundary. Ordinary Gaussian smoothing across a foreground/background edge can corrupt otherwise low-gradient anchors.
4. Stratify anchors by spatial block, surface and target-depth quantile; cap each bucket's influence. A huge sky region must not dominate, but excluding every sky anchor can lose the offset constraint. Require at least128 anchors spread over at least4 blocks; these are initial engineering gates to calibrate.
5. Normalize regressor and target with robust location/spread (`median`, `P90-P10` or MAD); reject nearly constant distributions. This makes numerical conditioning and slope checks unit-independent.
6. Solve weighted affine least squares in float64, then 10–15 Huber IRLS iterations with `delta=1.345*MAD` and optional top5% residual trimming, reusing `robust_fit`'s structure. Enforce positive scale; reject an optimum at the positivity boundary. RANSAC is a diagnostic for multimodal mismatched anchors, followed by IRLS, not a default random fit on sky.
7. Hold out whole spatial blocks and perform leave-one-surface/block sensitivity checks. Initial thresholds: held-out median residual below0.1 of target anchor `P90-P10`, p90 below0.25, stable positive slope, and scale variation below20% across usable folds. Tune on development scenes and freeze before held-out evaluation. Report acceptance/coverage together with accuracy.
8. Apply `a,b` to the **original sharp** prediction. Reject nonfinite/nonpositive transformed inverse depth locally; fall back to SHARP rather than clamp it to arbitrary tiny depth/disparity. Retain uncertainty on extrapolated foreground values outside the anchor range.

The old `0.025<a<8`, `spread>=0.002`, and raw upper-range thresholds in `src/alignment.py` are ZipDepth-unit assumptions. Do not reuse them unchanged for TinyViM→SHARP or conclude that a small cross-model slope means a bad fit. Create a new generalized module; keep the historical code/results intact.

### E3. Joint overlap refinement

Refine the per-tile transforms with robust shared-surface overlap constraints:

```text
min_{a_t>0,b_t}
  Σ_t Σ_{p in A_t} w_tp ρ(a_t r̄_t(p)+b_t-q̄_s(p))
  + η Σ_(t,u) Σ_{p in O_tu} w_tup ρ(a_t r̄_t(p)+b_t-a_u r̄_u(p)-b_u)
```

Bars denote the common fitting bandwidth, not the final data. Begin `η=0.25` after normalizing term counts/units and ablate it. Each overlap component needs a trustworthy SHARP anchor; pairwise consistency alone has an affine gauge ambiguity and can be consistently wrong. The existing DepthART `relative/utils/tiled_kitti.py::joint_affine_align` provides a useful structural reference, but pins a central tile, lacks these surface/positivity/conditioning checks, and Hann-blends depth. It is not a ready-made halo-safe merger.

An isolated sky-plus-missing-wire crop may not have two distinct shared depth populations. Its scale is **unidentifiable** from sky alone. Expand context, connect to an anchored overlapping crop, or obtain an optional full-frame relation. If none works, preserve SHARP and record a scale-uncertain missed structure; a guessed wire distance is not a valid alignment.

## F. Fusion

### F1. Why simple high-frequency transfer and plain Poisson are insufficient

The previous method was `q_s + M*(R-GσR)`, with `R=q_t-q_s`. A narrow positive correction has a broad negative low-pass component. Subtracting it creates shoulders/undershoot; multiplying by a spatial mask reintroduces low-frequency terms. In derivatives, `∇(MR)=M∇R+R∇M`: the mask itself can create gradients. The existing grooves and halos are consistent with these mechanisms.

Likewise, minimizing `||∇q-g||² + λ||q-q_s||²` over an unrestricted image can distribute an inconsistent gradient field into ramps and halos. Pinning a wire to an incorrect blurred base suppresses its amplitude; pinning its foreground/background transition to the wrong surface can pull background forward. A “strongest gradient wins” rule can select texture, inconsistent edge signs, or two displaced contours.

The new method differs fundamentally: **surface labels constrain where jumps occur, both flanks of a thin object are represented, contaminated background shoulders can be corrected, and distant stable background is fixed exactly.** A hard foreground-only mask around the old signed high-pass residual is not this method.

### F2. Construct a small surface-labeled problem

For each connected candidate and its overlapping tiles:

* Detect a raw local depth ridge with two flanks, with RGB location support and at least one repeat/context check for an uncertain component. For curves use local tangent/normal profiles, not only vertical lines.
* Label pixels as foreground surface `F`, stable background `B`, contaminated background shoulder `H`, or uncertain `U`. Build `H` only where a foreground band in SHARP is wider than independently supported local geometry; do not erase a legitimate sloped surface just because it is smooth.
* Include foreground interiors and **both** jump flanks in the correction support `Ω=F∪H∪accepted surface details`. Preserve existing connected cable junctions if consistent; do not carve holes via a dilated edge mask.
* Outside `Ω`, set `q=q_s` exactly. Unknown labels cause component rejection or a targeted extra crop, not a soft foreground/background interpolation.
* For `H`, estimate the local background surface from stable **same-background** samples, retaining its SHARP level and slope. A robust plane suffices only on locally planar inverse depth; use a supported curved model or reject otherwise. Never fill a lattice hole from the foreground median.

Foreground shape/width can be better identified than absolute distance. Require an aligned foreground value or reliable jump anchor as well as good location. RGB may constrain the edge position within a model-supported band; it must not create an object missing from all raw depth predictions.

### F3. Objective: constrained, screened gradient reconstruction

Use oriented 4-neighbor pixel edges `e=(p,r)` and incidence `B_e q=q_r-q_p`. Work in normalized inverse-depth units within a component, and convert back using the fixed SHARP reference scale afterward.

Define disjoint edge sets:

* `E_s`: within one surface, target gradient `g_e` from SHARP unless reliable local differential detail is selected.
* `E_j`: confirmed foreground/background interface, target signed jump `J_e` from the aligned, surface-owned local hypothesis or foreground/background plateaus. The correct opposing signs on the two flanks of a wire are essential.
* Uncertain crossing edges: remove from smoothness coupling; do not impose a zero jump. Their connected components still require anchors.

Solve:

```text
min_q  Σ_(e∈E_s) w_e ρ(B_e q - g_e)
     + Σ_(e∈E_j) v_e ρ(B_e q - J_e)
     + λ_A Σ_(p∈A) u_p (q_p-q_s,p)^2
     + λ_H Σ_(p∈H) h_p (q_p-q_bg,p)^2
     + λ_F Σ_(p∈F_seed) c_p (q_p-q_fg,p)^2
     + λ_L || C_L(q-q_s) ||²

subject to:
  q_p=q_s,p                        outside Ω and on stable fixed anchors
  0<l_label(p)≤q_p≤u_label(p)       within each accepted surface
```

`C_L` contains low-frequency/surface-mean constraints for **unchanged supported surfaces**. Do not force a coarse foreground/background average to equal an old value that omitted the wire; that would compensate the new foreground by moving background and reintroduce a halo. For the first implementation set `λ_L=0`: hard background anchors plus a small support already preserve the global reference outside it. Add this term only as a labeled-surface ablation.

Use confidence-dependent **weights on constraints**, not soft blending of foreground and background absolute depth. Surface intervals come from robust aligned plateau estimates and their fit/overlap uncertainty. If foreground/background intervals overlap so much that ownership is unreliable, reject/flag the component rather than inventing an intermediate sheet. Background interval bounds explicitly limit its pull toward foreground. They are priors with uncertainty, not an unconditional guarantee of physical accuracy.

Initial normalized weights: same-surface gradient1, confirmed jump4, accepted foreground-seed1, contaminated-background tether4; other stable anchors are hard constraints. Normalize aggregate terms by sample counts and record all weights. These are development starting points. Set Huber transition using fit/overlap noise (e.g. `max(1e-4,1.345*MAD)` in component-normalized units), not a raw ZipDepth constant.

### F4. Solving and validity

For quadratic/IRLS weights, the unconstrained reduced system has the form:

```text
(B_Uᵀ W B_U + Λ + λ_L C_Uᵀ C_U) q_U
 = B_Uᵀ W (g - B_K q_K) + Λ d + low_frequency_rhs
```

`K` are fixed nodes, `U` variable nodes. Use sparse/matrix-free conjugate gradients with Jacobi preconditioning inside an active-set or projected solver for the interval constraints. Repeatedly solving then clipping once is not a correct constrained optimum. A simpler projected-gradient/FISTA QP implementation is a valid first reference. Every variable connected component needs a fixed neighbor or positive data anchor to remove the nullspace. Do not install a full-image dense solver for18MP.

Solve the union of overlapping supports once per connected component; do not average independently solved tile depths. Initial stopping rule: relative residual/KKT tolerance `1e-5` and maximum200 inner iterations, recording convergence; reject a failed component to original SHARP. Check positive/finite values, boundary residual, interval violations, cycle inconsistency `curl(g)`, and halo-width change. Uncertainty or nonintegrability is not a reason to smear the jump.

The no-halo property is conditional: background fixed outside support cannot change there, and surface intervals bound contamination inside it. Wrong ownership, wrong plateau estimates or a raw-model halo can still fail. This architecture localizes those errors and makes them measurable; it does not prove monocular geometry is correct.

### F5. Laplacian/multiband alternative

Retain the existing Gaussian high-pass method as the historical control. A new Laplacian alternative should use **surface-restricted** pyramids/masks: filter within a surface, retain SHARP coarse surface values, transfer local bands only where the same surface is established, and reconstruct each label without smoothing across jumps. Local Laplacian filtering research explains how amplitude-aware remapping can preserve edges, but does not by itself resolve cross-model surface ownership or scale ambiguity.

Do not adopt an ordinary Gaussian pyramid plus soft cross-edge mask as the “halo fix.” Narrow edges occupy multiple bands, so retaining only “high frequency” can weaken a wire's foreground plateau and produce negative lobes.

## G. Confidence

Maintain separate interpretable components rather than one opaque confidence image:

| Component | Concrete evidence |
|---|---|
| Validity | Finite raw/aligned values; positive accepted inverse depth; no pad; valid source mapping |
| Alignment | Held-out normalized residual, anchor spread/coverage, fold stability, transform extrapolation |
| Raw geometric support | Foreground ridge/edge exists before fusion; both flanks and correct foreground contrast sign |
| Localization/width | Distance between local depth boundaries and RGB-supported object band; paired-wire separation and background trough |
| Overlap consistency | Median/MAD of aligned values within same label; boundary Hausdorff/distance profile across shifted crops |
| Scale/context consistency | Structure location/jump sign retained at448/512 or alternate context; amplitude discrepancy is separately recorded |
| Tile reliability | Interior distance, valid unpadded context and agreement with full-frame reference if available |
| Texture rejection | Depth-vs-RGB support, component continuity and control-surface false positives; variance alone is not confidence |
| Representation survival | Native→1536→768 cell occupancy, collision/width inflation and composed Gaussian depth/opacity |

A useful first score is `C=min(C_align,C_overlap,C_geometry,C_location)` after validity gates; keep the full vector and a reason code. A product is also possible but can over-penalize long thin objects and hide the reason for rejection. A single supported tile with no independent check receives “unverified,” not zero-confidence truth or automatic acceptance.

Initial development gates: local edge agreement within1 native pixel where the source supports that precision; component jump sign agreement in all usable overlaps; plateau MAD below10% of foreground/background separation; no increase in the labeled halo band; no gross collision at the output grid. These thresholds must scale with source resolution/blur and be validated on held-out scenes.

The replacement mask is **binary surface/component ownership**, derived from this evidence. A continuous confidence affects transform/gradient fitting strength inside an accepted surface. It is not multiplied into a depth jump to create a partial surface. If confidence fails, use SHARP exactly and count the missed thin structure.

SHARP/local disagreement is necessary for many useful corrections, but disagreement alone cannot identify the better model. Agreement among S/B/L is also correlated evidence, since the models share training supervision. Do not run three models everywhere initially.

## H. Hybrid ML-SHARP Representation

### H1. Metric-depth input contract

Conceptual pseudocode at the semantic insertion point (not a shipped API):

```python
# Shapes: Zs [B,2,1536,1536], q_fused and accept [B,1,1536,1536].
# q_fused is metric inverse depth, already spatially registered.
ok = accept & torch.isfinite(q_fused) & (q_fused > 0)
q_safe = torch.where(ok, q_fused, Zs[:, :1].reciprocal())
candidate_z = q_safe.reciprocal()
ok &= torch.isfinite(candidate_z) & (candidate_z > 0)
Zhybrid = Zs.clone()
Zhybrid[:, :1] = torch.where(ok, candidate_z, Zs[:, :1])
assert torch.equal(Zhybrid[:, 1:], Zs[:, 1:])
assert torch.equal(Zhybrid[:, :1][~ok], Zs[:, :1][~ok])
```

Sanitize before reciprocal/division. Do not allow a NaN to enter a Gaussian blur, a normal equation or `0*NaN` arithmetic. No valid update means an exact no-op. Never use `max(fused,0)` as validity repair: zero inverse depth is not a finite valid distance and zero metric depth collapses Gaussians to the camera.

Then apply the fixed-normalization initializer-output adapter in C2. Preserve all original encoder features, original secondary depth/base/delta tensors, original camera factors, and colorspace conventions. Preserve original decoder features if cached; the corrected Gaussian decoder features are newly computed and are **not** claimed to remain unchanged.

### H2. Exact layer selection downstream

At 768², derive a Boolean ownership mask from the footprint of accepted1536 corrections. A cell is eligible only when its chosen visible surface is unambiguous and the pooling winner is tracked. A wire crossing two cells or two wires sharing a cell must be recorded; blind max-pooling can choose the nearest surface and erase the other.

```text
base_h.layer1 = base0.layer1
delta_h[:,:,1,:,:] = delta0[:,:,1,:,:]
base_h.layer0[outside_owned_cells] = base0.layer0[outside_owned_cells]
delta_h[:,:,0,outside_owned_cells] = delta0[:,:,0,outside_owned_cells]
global_scale_h = global_scale0
```

For final flattened arrays, layer0 occupies indices `[0,768*768)` and layer1 `[768*768,2*768*768)` according to `composer.py:135–142`. Assert that mapping rather than assuming interleaving. Original unprojection applies a common camera transform and covariance transformation, with CPU float64 SVD; compare covariance matrices if quaternion signs differ numerically.

Do not enforce `Z0<Z1` everywhere: sorting is disabled and the second layer is not guaranteed to be behind the first. Record order reversals caused by accepted edits; inspect occlusion behavior rather than sorting away the learned representation.

### H3. New source/synthetic probes actually run

`probe_contracts.py` extracts and executes the **unmodified initializer definitions** from the pinned source without importing the full SHARP package. It uses the existing CPU PyTorch, NumPy, OpenCV and Pillow. Results: `probes/contracts.json`, `probes/synthetic_profiles.npz`, `probes/synthetic_profiles.png`.

Observed:

* Original primary10m / secondary300m; one primary sample changed to1m. Shared normalization scale changed about10→1, and secondary initialized metric base depth changed300→100.0001m despite identical secondary input. This is an actual initializer counterexample, not a learned-model inference result.
* Invalid NaN/zero/negative/infinite candidate inverse depths fell back exactly to the original visible layer, with secondary layer unchanged; guards passed.
* Synthetic three-pixel foreground plateau: high-pass residual produced maximum nearby background inverse-depth error0.01287; ordinary screened Poisson0.07583. A restricted gradient solve with **oracle support and consistent jumps** produced approximately1e-15 error. The corresponding plot was opened. This verifies the mathematics on an ideal case; automatic labels and real-data interval constraints remain unvalidated.
* Sampling20 horizontal positions in a native5388-wide image: ordinary bilinear5388→1536 plus2× max-disparity pooling entirely missed8/20 one-pixel wires and3/20 two-pixel wires. Three-pixel wires retained peaks between0.515 and1.0 for true foreground1.0/background0.1. This tests a resampling path, not the learned model's ability to infer wires from RGB.

## I. Failure Cases

1. **Unidentifiable scale:** sky plus a missing wire provides no common foreground scale anchor. More confident width does not solve depth ambiguity.
2. **Raw local halo:** TinyViM itself predicts a curtain or merged pair. Gradient transfer preserves that error unless ownership/width rejects it.
3. **Wrong hard labels:** background texture accepted as a wire creates a crisp false object; hard selection is not automatically correct.
4. **Texture/shadow ambiguity:** RGB edges alone can be paint, shadows, specular stripes or compression. Hair can mix several depths and partial coverage inside a pixel.
5. **Intersecting wires/lattice holes:** a single label or plane can bridge a true hole or erase one strand at a crossing.
6. **Far background:** inverse depths approach zero, affine offsets become influential and reciprocal depth magnifies small errors. Reject invalid/uncertain values, preserve finite SHARP sky representation.
7. **Conflicting tiles:** crop context changes depth order, edge position or width; ordinary overlap averaging creates duplicate/fat contours.
8. **Subpixel sampling:** native thin structures disappear during mapping; footprint selection preserves presence but inflates width.
9. **Gaussian bottleneck:** min-depth pooling, background-contaminated base RGB, learned low opacity, old feature evidence and large splat scales can defeat a good depth map.
10. **Distribution shift:** abrupt edited depth may be outside the decoder's training distribution; corrected primary deltas can amplify or suppress the change.
11. **Disocclusion interaction:** retaining original secondary Gaussians avoids accidental edits but may leave a hole or duplicate surface inconsistent with a newly added wire. Nearby-view rendering is essential.
12. **Over-constrained preservation:** rejecting new global minima or scale-uncertain tiny objects loses recall. Report this rather than choosing flattering accepted-only metrics.

If depth is good before composition but fog appears afterward, diagnose Gaussian scale/opacity and secondary-layer contributions. Repeatedly sharpening the depth would then address the wrong stage.

## J. Experiments

### J1. Dataset and protocol

Keep the bridge as a regression case, not the validation set. Acquire at least two independently selected scenes per class: ropes/straps, wires, branches, hair, grids, thin metal, and small foreground objects. Include distant backgrounds and native1/2/3/4/5-pixel width strata. Count per-object performance, not only per-pixel averages dominated by background. Separate development from held-out scenes before tuning confidence thresholds.

Use annotated foreground support, boundary polylines, individual centerlines and known background/foreground depth where possible. High-resolution synthetic rendered scenes with exact geometry provide controlled width/halo truth; registered multi-view captures provide a real-rendering check. Commodity RGB-D edges can themselves be mixed/invalid, so use valid masks and do not treat missing sensor depth as background truth. Hair/transparency should additionally evaluate alpha/coverage, not force every mixed pixel into an opaque surface.

### J2. Minimal sequential ablations

Use cached predictions and the same proposal coordinates for each fusion comparison:

| Experiment | Isolated question / pass evidence |
|---|---|
| 0. Unmodified SHARP + no-op replay | Is the integration numerically faithful? Check both layers and exported renders |
| 1. Raw TinyViM-S448/512 on fixed boxes | Are both wire flanks and background trough already present? No fusion |
| 2. Unit-aware native vs common-bandwidth alignment | Does robust registration work without suppressing a missing foreground? Holdout/fold/anchor diagnostics |
| 3. Existing high-pass vs ordinary screened Poisson vs hard surface substitution vs constrained-gradient method | Does gradient reconstruction add value beyond a correct ownership mask? Measure halo/width/recall together |
| 4. Constrained method minus background-shoulder correction, then minus paired-flank/interval constraints | Which explicit anti-halo constraints matter? Use identical local predictions |
| 5. Independent tile fits vs joint overlap fits; single vs shifted/context check | Are seam and false-positive improvements worth extra passes? |
| 6. Ordinary native→SHARP resampling vs categorical surface rasterization | Does recall survive without unacceptable width inflation? |
| 7. Naïve pre-init override vs fixed-normalization preserved-secondary adapter; frozen vs updated visible deltas | What changes downstream and does it improve render quality? |
| 8. S vs B vs L vs identified ViT-L on remaining failures + held-out controls | Does model cost buy actual width/halo improvements? No blanket ensemble initially |

This is a sequential study, not an exhaustive combinatorial sweep. Stop a branch when its raw evidence or render quality fails. Full-frame local inference and the `C_L` constraint are optional additions only if an earlier ablation exposes a problem they could solve.

### J3. Metrics with operational definitions

* **Edge localization:** symmetric distance from predicted depth discontinuity to labeled boundary, median/p95 in native pixels; boundary precision/recall at1px and2px. Sweep a fixed normalized jump threshold rather than choosing per-image display contrast.
* **Thin-structure recall:** fraction of each annotated centerline with a correctly signed, localized foreground-depth response and correct neighboring background; report continuity gaps and pair separation separately.
* **Depth-boundary accuracy:** boundary F-score plus foreground/background jump sign and magnitude error where depth truth exists.
* **Halo width:** along normal profiles beyond the labeled foreground support, measure contiguous length with depth pulled toward the foreground by more than10% of the known/annotated foreground-background inverse-depth separation. Report median/p95/max and the complete profile. For blurred mixed RGB pixels also report an uncertainty band.
* **Background leakage:** mean/p95 normalized foreground-directed inverse-depth error in fixed1–3/4–8/9–16px exterior rings and false-foreground pixel fraction.
* **Foreground fattening:** predicted foreground support area outside truth divided by true foreground area; signed width excess and pair-merging rate. FWHM alone is inadequate for adjacent peaks.
* **Tile seams:** correction-gradient discontinuity at artificial boundaries excluding real annotated edges, versus matched non-seam controls; also depth/edge variance when shifting the tile grid. A true wire crossing a seam must not be excluded from the continuity metric.
* **Local gradient error:** signed same-surface and jump-gradient errors separately; a larger gradient magnitude is not inherently better.
* **Global consistency:** exact max change outside support; robust low-pass/surface-plane drift and scale ratio on stable controls; metric depth errors where calibrated truth exists.
* **Integration preservation:** layer1 base/delta equality, final Gaussian covariance/means/color/opacity differences; unchanged normalization and camera; valid finite positive depths.
* **Render quality:** source view and small calibrated translations in both directions, multi-axis parallax, alpha holes, projected halo width, pair recall, disoccluded-region consistency; LPIPS/DISTS supplement these local measurements.

Set acceptance criteria before held-out runs: recall improvement with no meaningful halo/width regression, no outside-support parameter changes, and no new nearby-view fog. Do not report high contrast-support percentages as complete geometry success.

### J4. Required diagnostic outputs

For every real hybrid experiment save float32 arrays with units/transforms and shared-scale visualizations:

```text
sharp/depth_layer0_m.npy              sharp/depth_layer1_m.npy
sharp/canonical_disparity.npy         sharp/camera_and_resize.json
local/tile_*/raw_relative.npy         local/tile_*/aligned_inverse_m.npy
fusion/inverse_m.npy                 fusion/depth_m.npy
fusion/difference_inverse_m.npy      fusion/difference_m.npy
diagnostics/gradient_sharp.png        diagnostics/gradient_fused.png
diagnostics/confidence.png           diagnostics/confidence_components.npz
diagnostics/replacement_mask.png     diagnostics/surface_labels.png
diagnostics/tile_boundaries.png      diagnostics/overlap_disagreement.png
diagnostics/alignment_holdout.json   diagnostics/jump_residual_and_curl.npz
diagnostics/native_1536_768_profiles.png
diagnostics/before_after_composer.png
diagnostics/layer1_preservation.json
renders/source/                     renders/nearby_views/
```

Include `D_sharp`, raw `D_vitl`/TinyViM, aligned local, fused, differences, gradients, confidence, replacement and tile boundaries in a side-by-side panel. Raw relative predictions require their own clearly labeled scale; aligned/fused maps share a SHARP reference display range. Use signed diverging scales for differences. Never infer metrics from percentile-normalized PNG16.

Only the synthetic/source-probe diagnostics exist in this new folder today; these real-SHARP diagnostics await checkpoint inference. The existing bridge panels remain accessible in their original folders.

## K. Implementation Roadmap

### K0. Reproduce this investigation's lightweight checks — available now

From `/mnt/c/depth_issue`:

```bash
/mnt/c/depth_issue/venv/bin/python -m pip check
/mnt/c/depth_issue/venv/bin/python experiments/sharp_hybrid_research/probe_contracts.py
/mnt/c/depth_issue/venv/bin/python experiments/sharp_hybrid_research/verify_preservation.py
git -C /mnt/c/depth_issue/third_party/ml-sharp rev-parse HEAD
git -C /mnt/c/depth_issue/third_party/ml-sharp status --short
```

All commands in this block were run successfully during this investigation. `probe_contracts.py` writes only its `probes/` subdirectory. It requires no added dependencies/checkpoints. `verify_preservation.py` checked all **325 files (1,743,721,386 bytes)** against the existing frozen manifest, with no mismatches; its record is `preservation.json`. The original algorithms, completed experiments and outputs were not edited. The new SHARP checkout remains clean.

### K1. Resolve the pending local CPU integration gate

Existing command, not run in this investigation:

```bash
/mnt/c/depth_issue/venv/bin/python scripts/validate_tinyvim_cpu.py
```

This runs the existing TinyViM raw parity probe and ROI0/5 tests into `results/tinyvim_local/`. Inspect its raw-parity images and alignment rejection reasons. The ZipDepth-scale guards are deliberately unchanged there; rejection can be a unit-contract problem, not evidence that TinyViM lost the wire. Do not rerun the already completed global ZipDepth/model-comparison experiments merely to regain context.

The existing runner can later reproduce a controlled **ZipDepth-global/TinyViM-local historical fusion** experiment with its actual supported flags:

```bash
/mnt/c/depth_issue/venv/bin/python scripts/run_experiment.py \
  --input ./bridge.png --output results/tinyvim_ab_control \
  --global-backend zipdepth --local-backend depthart_tiny --device cpu \
  --reuse-from results/final --global-res 512 --roi-res 512 \
  --max-rois 6 --roi-height-scale 2 --fusion-mode feather_hp --skip-joint
```

This is a historical-control command, **not the proposed SHARP hybrid**. It may be deferred because it retains the known high-pass halo mechanism. `results/final` is protected by the current runner; the old README command targeting it is historical.

### K2. Provision SHARP and record one global baseline

Next environment destination: `/mnt/c/depth_issue/venv_sharp` (proposed, not present). Upstream source is already at the actual path above. Provision a Python3.13 interpreter using the machine's chosen package/environment mechanism first; it was not available during this audit. After that prerequisite, create the environment and install from the SHARP repository working directory:

```bash
python3.13 -m venv /mnt/c/depth_issue/venv_sharp
/mnt/c/depth_issue/venv_sharp/bin/python -m pip install \
  -r /mnt/c/depth_issue/third_party/ml-sharp/requirements.txt
```

The requirements contain `-e .`, so the installation command's working directory must be `/mnt/c/depth_issue/third_party/ml-sharp`. These setup commands have **not** been executed or compatibility-tested here. Select a device-appropriate torch build if not reproducing the upstream GPU lock; record any deviation.

Then, from `/mnt/c/depth_issue`, the existing upstream CLI supports:

```bash
TORCH_HOME=/mnt/c/depth_issue/.cache/torch \
  /mnt/c/depth_issue/venv_sharp/bin/sharp predict \
  -i /mnt/c/depth_issue/bridge.png \
  -o /mnt/c/depth_issue/results/sharp_baseline --device cpu
```

It downloads the official `sharp_2572gikvuh.pt` on first use. CPU prediction is supported upstream; CUDA is needed for its rendering CLI. Record the checkpoint hash, focal-length source and actual camera. This run is a potentially expensive next step, not a command reported as completed. For quality evaluation use a CUDA-capable renderer once available, preserving the same PLY/camera.

### K3. Add capture/replay before any fusion

Proposed new file: `experiments/sharp_hybrid_research/sharp_adapter.py`. Implement explicit calls matching `RGBGaussianPredictor.forward`; save post-alignment depths, required encoder features, original initializer output, baseline deltas, camera and hashes to a new `results/sharp_hybrid/baseline_cache/` directory. Keep a small schema containing shape, dtype, units, layer ordering, original H/W, resize convention and source/checkpoint/input hashes. Do not confuse normalized initializer depth with metric depth.

Validate no-op output against the upstream baseline, then a simple known visible-depth patch with invalid holes. Visual inspection: both raw layers, normalized channels, pooled base depth, composer output and layer1 difference. Only after this pass implement C2's fixed-normalization adapter. Source tests must cover new-nearest-point normalization, far clamp, secondary-layer equality, all-invalid/empty correction, cell ownership and flattened indexing.

### K4. Generalized alignment on cached raw maps

Proposed `alignment.py` in this experiment folder, importing/reusing `src.alignment.robust_fit` where appropriate. Add unit-invariant conditioning, surface-safe anchor selection, heldout/fold reports and positive-valid output masks. Use cached TinyViM far predictions and the newly captured SHARP depth. Show before/after and spatial holdout errors. Never accept a transform only because a depth image looks contrast-normalized.

### K5. Surface/gradient fusion reference

Proposed `surface_fusion.py`: implement labeled support, paired jumps, background anchors, matrix-free/sparse constrained solver and exact fallback. First test synthetic1–5px foregrounds with sloped backgrounds, paired wires, outlier jumps, overlap scale drift and invalid pixels. Then use manually checked surface labels on bridge patches as an **oracle diagnostic** to separate label failures from solver failures. Add automatic labels only after the oracle solver passes.

Visual checkpoint: shared-range before/after profiles show a foreground plateau and clean background trough, not merely higher contrast. Save the complete J4 depth/confidence/mask/tile diagnostics. Compare hard substitution as well as the two halo-prone baselines; if the solver does not improve on substitution, choose the simpler method rather than keeping Poisson for its name.

### K6. Grid survival and Gaussian integration

Proposed `rasterize_surfaces.py`: retain categorical boundary ownership across native/1536/768 transforms, and explicitly report collisions. Feed the corrected representation through the C2 adapter and unchanged composer/unprojection. Save native, pooled and final-Gaussian profiles and source/nearby renders. Recovered depth that becomes a broad Gaussian curtain fails this stage.

### K7. Independent scenes, then cost reduction

Proposed `evaluate.py` and `benchmark.py` in the new experiment folder. Implement J's fixed metrics and ablation manifests. Measure wall time and memory stage by stage with batch1, then batch2/4 for compatible tiles. Stop generating full18MP copies of every internal intermediate once correctness is established; retain necessary native crops and authoritative arrays. Remove full-frame local inference, blanket multiscale/model voting or expensive solving only when its ablation shows no meaningful thin-structure/halo regression.

These proposed modules do not yet exist, so no fictitious runnable CLI flags are provided for them. Implement them in this order and add one tested command per completed stage to this folder's progress log.

## Primary references and claim boundaries

1. **SHARP paper**, especially §§3.1–3.4 and motion/failure discussion: https://arxiv.org/html/2512.10685v1 . Supports the two-layer representation, training-only depth adjustment, nearby-view purpose and learned Gaussian refinement. Source determines exact released tensor flow.
2. **Pinned SHARP source:** https://github.com/apple-aiml-research/ml-sharp/tree/aed6527499ef91cba3b54c18d49a870f25947190 . Key files: `models/predictor.py`, `monodepth.py`, `initializer.py`, `gaussian_decoder.py`, `heads.py`, `composer.py`, `params.py`, `encoders/spn_encoder.py`, `cli/predict.py`, `utils/gaussians.py`.
3. **DepthART official source/release:** https://github.com/xuefeng-cvr/DepthART/tree/b307c123f19051e2d969960bba93ee3684874b90 ; paper https://arxiv.org/abs/2607.17099 . Local source verifies S/B/L structure and tiled affine stitching; upstream runtime figures are not current-workspace measurements.
4. **Depth Anything V2:** https://github.com/DepthAnything/Depth-Anything-V2 and https://arxiv.org/abs/2406.09414 . Identifies a reproducible proposed ViT-L comparator; it has not been benchmarked in this workspace.
5. **Boosting Monocular Depth**, CVPR2021 official implementation/method explanation: https://github.com/compphoto/BoostingMonocularDepth . Supports the resolution/context trade-off and content-adaptive local inference; its learned merger is not a fixed-background, halo-proof fusion operator.
6. **PatchFusion**, CVPR2024 official implementation: https://github.com/zhyever/PatchFusion and https://arxiv.org/abs/2312.02284 . Demonstrates global/local feature fusion, consistency and batched tiles. It is an end-to-end trained architecture, not a training-free drop-in SHARP depth hook.
7. **Poisson Image Editing**, Di Martino, Facciolo, Meinhardt-Llopis, IPOL2016: https://www.ipol.im/pub/art/2016/163/ . Provides the gradient integration/boundary-value numerical framework. The original Pérez et al.2003 PDF endpoint returned binary content in this harness; no extracted text from that download is used as evidence here.
8. **Local Laplacian Filters**, Paris, Hasinoff, Kautz, SIGGRAPH2011, author page: https://people.csail.mit.edu/sparis/publi/2011/siggraph/ . Supports edge-aware nonlinear pyramid processing; image-editing halo results do not establish depth-geometry correctness.
9. **MiDaS scale/shift-invariant depth lineage**, Ranftl et al.: https://arxiv.org/abs/1907.01341 . Establishes the ambiguity/invariant-learning context. The actual affine equations, robustness choices and guards in this plan are specified explicitly rather than inferred from a missing upstream loss file.
10. **Workspace experimental evidence:** `results/final/REPORT.md`, `experiments/model_compare/REPORT.md`, `PROGRESS.md`, cached arrays/profiles and the new `probes/contracts.json`. Prior successes/failures refer to those exact data, not generalized model rankings.

**Next decisive experiment:** cached raw TinyViM + captured global SHARP on the two far-wire crops, first with verified surface labels, through fixed-normalization/preserved-secondary composition and native-grid survival diagnostics. If it passes depth profiles but fails Gaussian renders, the limitation is downstream representation/appearance rather than a reason to repeat the old high-pass blend.
