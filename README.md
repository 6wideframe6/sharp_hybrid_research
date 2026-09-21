# SHARP hybrid-depth investigation

**Latest continuation:** [K4.md](K4.md) records the successful full released-checkpoint
gate and real bridge capture, plus K4 alignment implementation/results. The real
baseline has bit-exact replay and byte-identical PLY export. Both cached far192
alignments remain underdetermined; expanded/shifted contexts fail held-out gates.
Authoritative diagnostics: `../../results/sharp_hybrid/k4_alignment_crossfit/`.
The milestones below retain the earlier K3 history; current status is in K4.md.

Read [REPORT.md](REPORT.md) for the full A–K workspace audit, source-level tensor
flow, integration plan, surface-constrained gradient fusion, performance estimates,
validation protocol and staged implementation roadmap.

**K3 is implemented:** see [K3.md](K3.md) for the capture/replay and
fixed-normalization preserved-secondary API, cache format, ownership policy and
tests. Implementation: [sharp_adapter.py](sharp_adapter.py).

## Completed

- Audited the existing research, including its positive TinyViM-S result and
  negative halo/width evidence.
- Added the official SHARP source at `../../third_party/ml-sharp`, pinned at
  `aed6527499ef91cba3b54c18d49a870f25947190`; working tree clean.
- Executed CPU probes using original SHARP initializer definitions and synthetic
  depth profiles. [Results](probes/contracts.json), [profile plot](probes/synthetic_profiles.png).
- Verified all325 files in the existing frozen manifest unchanged.
  [Preservation record](preservation.json).
- Implemented K3 with actual upstream modules. The test suite ran23 tests:
  **22 passed, 1 optional released-checkpoint test skipped**. Exact no-op replay,
  normalization/fallback/ownership/preservation contracts and byte-identical PLY
  exports passed on deterministic source-contract fixtures.

## Reproduce lightweight checks

Working directory: `/mnt/c/depth_issue`.

```bash
/mnt/c/depth_issue/venv/bin/python experiments/sharp_hybrid_research/probe_contracts.py
/mnt/c/depth_issue/venv/bin/python experiments/sharp_hybrid_research/verify_preservation.py
/mnt/c/depth_issue/venv/bin/python -m unittest discover -s experiments/sharp_hybrid_research/tests -v
```

The original lightweight probes need no added packages. The K3 tests use SciPy1.15.3
and plyfile1.1.2, now installed in the project venv, to execute actual upstream
unprojection and PLY I/O. No model weights are needed for the default tests.

## Next development gate

Use the implemented adapter to capture one global SHARP depth/features/Gaussian
baseline and run the optional released-checkpoint no-op gate. Later stages can
test cached TinyViM far-wire predictions with verified surface labels through
the fixed-normalization/preserved-secondary integration. Inspect native→1536→768
survival and final renders before expanding tiling or optimizing performance.

Current `venv/` is CPU-only. SHARP weights and the full locked CLI/rendering environment
have not been provisioned; K3 imports and source-contract tests run in this venv.
The original pending TinyViM CPU parity command and
actual upstream SHARP setup/CLI commands are documented in report section K.
