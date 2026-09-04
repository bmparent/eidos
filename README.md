# Eidos Brain

Eidos Brain is a proof-first research repo for testing Eidos/Sentinel ideas with reproducible runs, visible artifacts, and conservative interpretation. The project is not treated as production infrastructure or a trading system; it is a place to make claims testable, preserve raw evidence, and keep every run easy to audit.

## Read More

Start here when you need to understand what `/eidos` contains and what the current proof work is meant to show:

- `eidos/repo/README.md` - package quickstart, local demo commands, and benchmark entry points.
- `eidos/docs/proof_runs/` - plain-language run notes, Codex journals, Drive status, and test summaries by date.
- `eidos/docs/proof/merge_state_report_2026_06_07.md` - merge-state audit for proof branches and GitHub integration status.
- `eidos/AUDIT/` - static audit maps for the core loop, reservoir, predictor, normalization, surprise gate, and artifact receipts.
- `docs/controlled_memory_benchmark.md` - isolated state-rounding benchmark, frozen protocol, exact controls, and reproduction commands.
- `eidos/artifacts/` - repo-local proof receipts, including manifests, summaries, ledgers, reports, and Drive-copy records.

## Proof Posture

The standing rule is proof over novelty. Changes should make the system more reproducible, measurable, testable, and reviewable before they add new behavior. Core behavior stays untouched unless a task explicitly asks to change it.

The proof artifacts should keep raw, calibrated, merged, deduped, skipped, and failed results visible side by side. A skipped baseline, missing dependency, unavailable GPU, failed Drive copy, or weak receipt should be recorded directly instead of hidden.

## Current Workstreams

- Sentinel false-positive and calibration receipts for CICIDS/WebAttacks labeled proof runs.
- Baseline and competitor-scaffold reporting that keeps skip reasons visible.
- Artifact hygiene: manifests, journals, plain-language summaries, environment capture, git state, and Drive mirror status.
- Eidos Life Lab experiments under `eidos-life-lab/` and `experiments/eidos-life/`, kept separate from core model behavior.

## Local Orientation

Useful paths in this checkout:

```text
eidos/
  repo/                       Python package and service/demo code
  tools/                      proof, calibration, comparison, and Drive helpers
  tests/                      focused pytest coverage
  docs/proof_runs/            plain-language analyses and Codex journals
  artifacts/                  repo-local proof artifacts
  AUDIT/                      architecture and receipt audit notes
  eidos-life-lab/             local synthetic lab app
experiments/eidos-life/       browser-based Eidos Life experiment
```

## Validation

For the main Eidos proof checkout, the standard focused validation command is:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest -q
```

Run proof and benchmark commands from the Eidos checkout root and preserve their generated artifact folders. When Google Drive is mounted or configured, proof artifacts should also be mirrored under:

```text
Eidos_Brain_Proof_Phase/YYYY-MM-DD/<run_id>/
```

Drive mirroring is evidence storage, not a substitute for repo-local artifacts. If Drive is unavailable, keep the local artifacts and record the skipped copy in `drive_manifest.json`.
