# Research implementation progress

## Latest: released-checkpoint K3 and K4 investigation completed

- Read the full `sharp_hybrid_k3_checkpoint_k4_prompt.md`, K3.md and REPORT.md first.
- Verified supplied checkpoint:2,809,738,232 bytes, SHA256
  `94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d`.
- Existing opt-in CPU test:23/23 passed in121.947s, including real1536² monodepth.
- One additional full bridge baseline captured through unchanged K3 API:
  `results/sharp_hybrid/baseline_cache/capture.pt`. Focal5388px explicitly numerical
  placeholder. Reloaded replay exact for all Gaussian fields; PLY bytes identical.
- Implemented isolated K4 alignment,17 mathematical tests, cached-crop diagnostics
  and two targeted512px context probes. Historical alignment and K3 code unchanged.
- Corrected sparse-bucket outlier amplification from a failing synthetic test.
- Corrected smoothing leakage across held-out blocks. The first diagnostic run at
  `results/sharp_hybrid/k4_alignment` is retained; authoritative validation is
  `results/sharp_hybrid/k4_alignment_crossfit`.
- Both cached192px fits underdetermined. Expanded/shifted512px fits have positive
  scales but fail residual gates. Overlap component has no accepted SHARP anchor;
  no joint refinement, fusion, ownership or K5/K6 implementation.
- Opened authoritative far-crop and context panels. All required data diagnostics
  are present. Artifact verification passes for8 runs and baseline cache/PLY hashes.
- Final default suite:40 run,39 pass,1 optional checkpoint skip. The real checkpoint
  test already passed before K4; no unnecessary repeat of the full model.
- Required preservation checks pass: pip consistent,325 frozen files unchanged,
  clean pinned upstream at `aed6527499ef91cba3b54c18d49a870f25947190`.
- Completion report, commands, file inventory and per-crop numbers: [K4.md](K4.md).

The older K3 notes below describe the preceding implementation session, not the
current checkpoint availability or verification state.

## K3 completed — source-contract implementation

- Added `sharp_adapter.py`: exact upstream-forward capture, cached monodepth replay,
  fixed-normalization initializer adapter, visible-cell delta selection, original
  composer/unprojection/export, portable capture cache and provenance checks.
- Reused upstream `_rescale_depth` and `MultiLayerInitializer` directly. The adapter
  does not recreate their formulas or mutate upstream source. It preserves the
  original depth factor to avoid reciprocal-rounding changes.
- Added `tests/test_sharp_adapter.py`. The real upstream predictor, initializer,
  reduced-width Gaussian DPT decoder, nonzero prediction head, composer, camera
  unprojection and PLY writer execute in tests. Only monodepth inference is replaced
  by deterministic known disparities/features.
- Final suite result: 23 run, 22 passed, 1 optional full-checkpoint test skipped.
  No-op equality is bit-exact in these CPU tests; original/replayed PLY bytes match.
- Added only SciPy1.15.3 and plyfile1.1.2 to the existing venv for real upstream
  geometry/export imports. `pip check` passed. Torch and core research versions
  were retained.
- Upstream checkout remains clean at
  `aed6527499ef91cba3b54c18d49a870f25947190`.
- Existing frozen manifest: all325 files (1,743,721,386 bytes) reverified unchanged.
- Alignment and fusion remain outside the implementation. Surface labels and
  accepted registered inverse depths are explicit caller inputs.

Commands run from `/mnt/c/depth_issue`:

```bash
/mnt/c/depth_issue/venv/bin/python -m unittest discover -s experiments/sharp_hybrid_research/tests -v
/mnt/c/depth_issue/venv/bin/python experiments/sharp_hybrid_research/verify_preservation.py
/mnt/c/depth_issue/venv/bin/python -m pip check
git -C /mnt/c/depth_issue/third_party/ml-sharp status --short
git -C /mnt/c/depth_issue/third_party/ml-sharp rev-parse HEAD
```

## Remaining verification boundary

No released SHARP checkpoint is currently available in the workspace, so a full
1536² pretrained capture/replay run has not been performed. `K3.md` documents the
opt-in `SHARP_K3_CHECKPOINT` test, cache/API usage, and the caller's cell-ownership
obligations. No real hybrid scene or rendering-quality result is claimed.
