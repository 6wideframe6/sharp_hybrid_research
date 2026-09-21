# SHARP + TinyViM Hybrid Depth — Architecture

## 1. Goal

Build a hybrid depth pipeline where:

- **SHARP** remains the authoritative source of low-frequency / metric geometry.
- **TinyViM-S** contributes thin-detail differential geometry that SHARP misses.
- TinyViM is **not** treated as an absolute metric depth source.
- K3/K4 behavior remains preserved and testable.
- K5 introduces detail correction without violating SHARP's global geometry.

The motivating failure mode is far/thin structures such as bridge wires, straps,
hair, branches and similar geometry that can disappear in SHARP but remain visible
in tiled TinyViM predictions.

---

## 2. Research conclusions carried forward

K4 established the following constraints:

1. A single global affine mapping

   `q_sharp = a * r_tinyvim + b`

   is not reliable for the target far-wire geometry.

2. TinyViM raw depth is strongly context dependent across overlapping crops.

3. Absolute TinyViM raw values and gradient magnitudes are not stable enough to be
   treated as metric quantities.

4. Local gradient **orientation** is substantially more stable across overlapping
   TinyViM contexts.

Therefore K5 should use TinyViM primarily as a **differential detail signal**, not
as an absolute replacement depth map.

---

## 3. Non-negotiable invariants

### K3

K3 remains frozen.

It owns:

- upstream SHARP capture/replay;
- released-checkpoint compatibility;
- SHARP normalization;
- multilayer initialization;
- Gaussian generation;
- camera unprojection;
- PLY export;
- cache provenance;
- no-op bit exactness.

K5 must not replace these components.

### K4

K4 remains frozen as the diagnostic alignment layer.

It owns:

- SHARP inverse-depth sampling into native coordinates;
- anchor construction;
- leakage-safe cross-fit;
- robust affine diagnostics;
- context-overlap diagnostics;
- identifiability conclusions.

K5 must not silently reinterpret a rejected K4 transform as accepted geometry.

---

## 4. K5 design principle

K5 should decompose depth into:

`q_final = q_sharp + delta_q_detail`

where:

- `q_sharp` carries low-frequency metric geometry;
- `delta_q_detail` carries only detail-scale correction;
- `delta_q_detail` has constrained low-frequency/DC content;
- TinyViM determines **where** and **in which direction** detail changes occur;
- TinyViM raw magnitude is not directly interpreted as metric inverse-depth change.

The correction must be zero outside its validated support and must not globally
rescale or translate SHARP depth.

---

## 5. Proposed source layout

Do not move or rename existing K3/K4 files yet.

Add K5 as a separate package:

```text
sharp_hybrid_research/
├── README.md
├── REPORT.md
├── PROGRESS.md
├── K3.md
├── K4.md
├── ARCHITECTURE.md
│
├── sharp_adapter.py               # frozen K3
├── alignment.py                   # frozen K4
├── capture_baseline.py
├── run_alignment.py
├── investigate_context.py
├── verify_preservation.py
├── verify_k4_artifacts.py
│
├── k5/
│   ├── __init__.py
│   ├── types.py
│   ├── overlap.py
│   ├── detail_signal.py
│   ├── consensus.py
│   ├── confidence.py
│   ├── amplitude.py
│   ├── integration.py
│   ├── compose.py
│   ├── validate.py
│   └── pipeline.py
│
├── tools/
│   ├── run_k5_detail_diagnostic.py
│   ├── run_k5_consensus.py
│   ├── run_k5_integration.py
│   └── verify_k5.py
│
├── configs/
│   └── k5_default.json
│
└── tests/
    ├── test_sharp_adapter.py
    ├── test_alignment.py
    ├── test_k5_detail_signal.py
    ├── test_k5_consensus.py
    ├── test_k5_confidence.py
    ├── test_k5_integration.py
    └── test_k5_preservation.py
```

Existing historical diagnostic scripts may remain where they are for now.
Reorganization should happen only after K5 behavior is validated.

---

## 6. K5 modules

### 6.1 `k5/types.py`

Small dataclasses only.

Suggested data types:

- `NativeBox`
- `ContextPrediction`
- `GradientField`
- `ConsensusField`
- `DetailConfidence`
- `DetailCorrection`
- `K5Diagnostics`

All arrays must explicitly document:

- coordinate system;
- units;
- channel semantics;
- dtype.

No implicit resizing or coordinate conversion.

---

### 6.2 `k5/overlap.py`

Responsibility:

- native-coordinate crop intersection;
- exact mapping between context tiles;
- extraction of common regions;
- mapping of wire/detail masks;
- no depth inference.

This must preserve the existing native-coordinate conventions.

---

### 6.3 `k5/detail_signal.py`

Responsibility:

Convert each TinyViM raw context prediction into a differential detail signal.

Initial version:

1. optionally smooth TinyViM at a small scale;
2. compute native-coordinate gradients;
3. separate direction from magnitude;
4. expose:

   - `gx`
   - `gy`
   - `magnitude`
   - normalized direction `(dx, dy)`

Initial diagnostic scales:

- sigma = 1 px
- sigma = 2 px

No metric conversion here.

---

### 6.4 `k5/consensus.py`

Responsibility:

Combine two or more overlapping TinyViM contexts.

For context gradients `g_i`:

1. normalize direction;
2. calculate pairwise cosine agreement;
3. reject sign disagreement;
4. produce a consensus direction.

For two contexts:

`confidence_dir = max(0, dot(n1, n2))`

Possible consensus direction:

`n = normalize(w1*n1 + w2*n2)`

where the initial weights should be simple and diagnostic, not learned.

Outputs:

- consensus direction;
- direction agreement;
- context count;
- invalid/disagreement mask.

This module must not assign metric amplitude.

---

### 6.5 `k5/confidence.py`

Responsibility:

Determine where TinyViM detail should be trusted.

Initial confidence should combine only observable diagnostics:

- multi-context gradient direction agreement;
- TinyViM detail strength above noise floor;
- valid crop support;
- optional distance from crop boundary;
- optional RGB thin-structure evidence.

It must not use future corrected depth as an input.

Important distinction:

`confidence != replacement mask`

It is only a weight for detail correction.

---

### 6.6 `k5/amplitude.py`

This is the main unresolved research component.

Do **not** use:

`A = |grad(TinyViM)|`

as metric amplitude.

Initial experiments should compare bounded amplitude sources.

Candidate A — SHARP-local scale:

Estimate a metric gradient scale from trusted SHARP neighborhoods and apply it
only to the consensus TinyViM direction.

Candidate B — band-energy matching:

Within trusted shared-geometry regions, match TinyViM high-frequency energy to
the SHARP inverse-depth residual/gradient energy at a nearby valid scale.

Candidate C — bounded global scalar:

A single small diagnostic scalar for the whole target region, selected only from
held-out validation.

Every amplitude estimator must have:

- a maximum correction bound;
- no extrapolation claim outside validated support;
- separate training/validation diagnostics.

---

### 6.7 `k5/integration.py`

Responsibility:

Turn a desired detail gradient field into a scalar inverse-depth correction
`delta_q`.

Do not begin with unconstrained global Poisson reconstruction.

Development order:

#### Stage 1 — diagnostic local integration

Integrate only in a bounded ROI and inspect whether the signal has the expected
sign and spatial structure.

#### Stage 2 — screened / constrained integration

Solve something of the form:

`argmin(delta_q) ||grad(delta_q) - g_detail||^2 + lambda ||L_low(delta_q)||^2`

subject to:

- zero or near-zero DC;
- boundary anchoring;
- no uncontrolled low-frequency drift;
- finite correction magnitude.

The exact solver can be chosen after the consensus/amplitude diagnostics are
validated.

---

### 6.8 `k5/compose.py`

Responsibility:

Produce:

`q_corrected = q_sharp + w * delta_q`

where `w` is bounded confidence.

Must preserve:

- finite positive inverse depth;
- SHARP outside supported detail regions;
- SHARP low-frequency geometry;
- original K3 secondary fields unless explicitly owned by K5.

No direct TinyViM depth replacement.

---

### 6.9 `k5/validate.py`

Validation is mandatory before any corrected depth enters K3's Gaussian path.

Required gates should include:

#### No-op preservation

With zero confidence or zero amplitude:

- corrected depth is bit-identical to SHARP;
- K3 replay remains bit-identical;
- PLY remains byte-identical.

#### Low-frequency preservation

After low-pass filtering:

`LP(q_corrected) ~= LP(q_sharp)`

with a strict numeric tolerance.

#### Bounded correction

Report:

- max `|delta_q|`;
- p90 `|delta_q|`;
- percentage of modified pixels;
- invalid/nonpositive output count.

#### Context consistency

Corrections generated from overlapping TinyViM contexts should agree inside
their shared trusted region.

#### Thin-detail diagnostics

Measure the correction specifically at known wire/detail pixels separately from
background and broad surfaces.

---

## 7. K5 staged implementation

### K5-A — consensus only

No SHARP modification.

Inputs:

- cached TinyViM context A;
- cached TinyViM context B;
- known overlap;
- wire proxy.

Outputs:

- gradient direction for A;
- gradient direction for B;
- cosine agreement;
- consensus direction;
- confidence map;
- visual diagnostics.

Success criterion:

The wire/detail signal must be coherent across contexts while unrelated texture
is sufficiently suppressed.

### K5-B — amplitude diagnostic

Still no final SHARP replacement.

Evaluate candidate amplitude sources on trusted regions.

Output:

- proposed metric detail gradient;
- held-out diagnostics;
- correction bound.

### K5-C — bounded integration

Generate `delta_q` in a restricted ROI.

Validate:

- zero/near-zero low-frequency drift;
- finite positive corrected depth;
- no broad surface deformation.

### K5-D — SHARP adapter integration

Only after K5-A/B/C pass.

Inject `q_corrected` through the existing K3-owned visible-depth path.

Do not change:

- secondary SHARP fields;
- camera/unprojection;
- Gaussian construction;
- PLY export.

### K5-E — real scene gate

Run:

- released checkpoint;
- real bridge;
- baseline replay;
- corrected-depth replay;
- Gaussians;
- PLY;
- visual and numeric preservation diagnostics.

---

## 8. Repository policy

Git should contain:

- Python source;
- tests;
- Markdown reports;
- architecture docs;
- small configs;
- small manually curated evidence summaries.

Git should NOT contain:

- model weights/checkpoints;
- `capture.pt`;
- raw `.npy/.npz`;
- generated panels;
- bridge/source images;
- PLY files;
- full result directories;
- local virtual environments;
- upstream SHARP checkout.

The external SHARP dependency should remain pinned by commit hash in documentation.

Current known upstream revision:

`aed6527499ef91cba3b54c18d49a870f25947190`

The released SHARP checkpoint should be documented by filename + SHA256 rather
than committed to Git.

---

## 9. GitHub workflow

Recommended branch structure:

```text
main
├── k5-consensus
├── k5-amplitude
├── k5-integration
└── k5-adapter-integration
```

Rules:

- `main` contains frozen K3/K4 plus passing tests.
- each K5 stage gets its own branch;
- each stage must preserve K3/K4 tests;
- no branch merges until its stage-specific validation passes;
- generated data stays local.

Recommended commit granularity:

1. pure types / coordinate utilities;
2. gradient extraction;
3. context consensus;
4. confidence diagnostics;
5. amplitude experiment;
6. integration experiment;
7. K3 adapter connection;
8. preservation gate.

---

## 10. First implementation target

The first code to write should be:

```text
k5/detail_signal.py
k5/consensus.py
tools/run_k5_consensus.py
tests/test_k5_detail_signal.py
tests/test_k5_consensus.py
```

The first executable experiment should:

1. load the two cached 512 TinyViM contexts;
2. compute sigma=1 and sigma=2 gradient fields;
3. map them into native overlap;
4. compute normalized gradient directions;
5. compute cosine agreement;
6. produce consensus direction;
7. save diagnostic arrays/images;
8. make NO modification to SHARP.

That keeps the first K5 step narrow, reversible and directly supported by K4.
