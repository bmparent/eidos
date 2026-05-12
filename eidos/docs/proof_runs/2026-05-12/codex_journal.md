# Codex Journal -- 2026-05-12

## What happened today
Week 1 of the proof plan froze a reproducible baseline package with config, manifest, environment, git state, pytest XML, scenario receipts, and CSV/Markdown summaries.

## What was accomplished
- Added a repo-root proof baseline runner.
- Captured seed, frames, suite, config hash, git state, Python/runtime details, and scenario-level outputs.
- Kept Sentinel thresholds and core model behavior unchanged.

## Tests and commands run
- `python tools/run_proof_baseline.py --suite smoke --seed 42 --frames 1200 --out artifacts/proof_baseline_2026_05` -> see `benchmark_summary.md` and `pytest_results.xml`.
- Pytest status: passed (pytest completed successfully).

## Problems encountered
- External compression baselines are not implemented in Week 1.
- Smoke synthetic data does not provide ground-truth anomaly labels.
- Google Drive status: copied; reason: copy completed to local Google Drive sync folder.

## What changed
- tools/run_proof_baseline.py
- tests/test_proof_baseline_runner.py
- docs/proof_baseline_contract.md
- artifacts/proof_baseline_2026_05

## What did not change
Core model behavior, Sentinel labels, thresholds, reservoir dynamics, compression behavior, and false-positive suppression logic were not changed.

## Artifacts generated
- artifacts/proof_baseline_2026_05/benchmark_summary.csv
- artifacts/proof_baseline_2026_05/benchmark_summary.md
- artifacts/proof_baseline_2026_05/codex_journal.md
- artifacts/proof_baseline_2026_05/config.json
- artifacts/proof_baseline_2026_05/drive_manifest.json
- artifacts/proof_baseline_2026_05/environment.txt
- artifacts/proof_baseline_2026_05/git_commit.txt
- artifacts/proof_baseline_2026_05/plain_language_test_analysis.md
- artifacts/proof_baseline_2026_05/plots/README.md
- artifacts/proof_baseline_2026_05/pytest_results.xml
- artifacts/proof_baseline_2026_05/pytest_stderr.txt
- artifacts/proof_baseline_2026_05/pytest_stdout.txt
- artifacts/proof_baseline_2026_05/run_manifest.json
- artifacts/proof_baseline_2026_05/scenarios/_engine_artifacts/compression/proof_baseline_synthetic_smoke/20260512_203618_bicameral_stream_meta_proof_baseline_synthetic_smoke.json
- artifacts/proof_baseline_2026_05/scenarios/_engine_artifacts/compression/proof_baseline_synthetic_smoke/20260512_203618_bicameral_stream_proof_baseline_synthetic_smoke.bin
- artifacts/proof_baseline_2026_05/scenarios/_engine_artifacts/eidos_brain_archive/20260512_202930_proof_baseline_synthetic_smoke/anomalies.jsonl
- artifacts/proof_baseline_2026_05/scenarios/_engine_artifacts/eidos_brain_archive/20260512_202930_proof_baseline_synthetic_smoke/clusters.json
- artifacts/proof_baseline_2026_05/scenarios/_engine_artifacts/eidos_brain_archive/20260512_202930_proof_baseline_synthetic_smoke/report.txt
- artifacts/proof_baseline_2026_05/scenarios/_engine_artifacts/eidos_brain_archive/20260512_202930_proof_baseline_synthetic_smoke/session_meta.json
- artifacts/proof_baseline_2026_05/scenarios/_engine_artifacts/eidos_brain_archive/20260512_202930_proof_baseline_synthetic_smoke/state_capsule.json
- artifacts/proof_baseline_2026_05/scenarios/_engine_artifacts/eidos_brain_archive/20260512_202930_proof_baseline_synthetic_smoke/steps.csv
- artifacts/proof_baseline_2026_05/scenarios/_engine_artifacts/eidos_brain_archive/20260512_202930_proof_baseline_synthetic_smoke/summary.json
- artifacts/proof_baseline_2026_05/scenarios/_engine_artifacts/forecast.jsonl
- artifacts/proof_baseline_2026_05/scenarios/_engine_artifacts/hippocampus/proof_baseline_synthetic_smoke/20260512_203616_hippocampus_snapshot_proof_baseline_synthetic_smoke.pt
- artifacts/proof_baseline_2026_05/scenarios/_engine_artifacts/incident_cards.jsonl
- artifacts/proof_baseline_2026_05/scenarios/_engine_artifacts/manifest.jsonl
- artifacts/proof_baseline_2026_05/scenarios/_engine_artifacts/reservoir_checkpoints/proof_baseline_synthetic_smoke/20260512_203616_reservoir_checkpoint_proof_baseline_synthetic_smoke.pt
- artifacts/proof_baseline_2026_05/scenarios/_engine_artifacts/reservoir_geometry/proof_baseline_synthetic_smoke/20260512_203616_reservoir_states_proof_baseline_synthetic_smoke.npy
- artifacts/proof_baseline_2026_05/scenarios/_engine_artifacts/reservoir_geometry/proof_baseline_synthetic_smoke/20260512_203618_reservoir_geom_proof_baseline_synthetic_smoke.json
- artifacts/proof_baseline_2026_05/scenarios/_engine_artifacts/sentinel_forensics/proof_baseline_synthetic_smoke/20260512_203618_top_100_surprises_proof_baseline_synthetic_smoke.json
- artifacts/proof_baseline_2026_05/scenarios/_engine_artifacts/sentinel_forensics/proof_baseline_synthetic_smoke/20260512_203618_top_100_surprises_text_proof_baseline_synthetic_smoke.txt
- artifacts/proof_baseline_2026_05/scenarios/synthetic_smoke/engine_output.log
- artifacts/proof_baseline_2026_05/scenarios/synthetic_smoke/report.txt
- artifacts/proof_baseline_2026_05/scenarios/synthetic_smoke/scenario_manifest.json
- artifacts/proof_baseline_2026_05/scenarios/synthetic_smoke/scenario_spec.json

## Google Drive archive status
- Drive root used: C:\Users\bmpar\Google Drive
- Drive folder used: C:\Users\bmpar\Google Drive\Eidos_Brain_Proof_Phase\2026-05-12\proof_baseline_2026_05_smoke_seed42_frames1200
- Files copied: 35
- Files skipped: 0
- Reason: copy completed to local Google Drive sync folder

## Thoughts on improvement
The next proof step should add false-positive suppression work only after this baseline remains easy to regenerate.

## Where to improve next
Week 2 false-positive suppression should be a separate PR-sized change with before/after receipts.

## Anything that stands out
The wrapper can capture a complete artifact package without touching the engine internals.

## End-of-task summary
1. Files changed: tools/run_proof_baseline.py, tests/test_proof_baseline_runner.py, docs/proof_baseline_contract.md, artifacts/proof_baseline_2026_05
2. Whether core behavior changed: no.
3. Tests added or skipped: runner/report tests added; pytest XML captured by the baseline run.
4. Repo-root commands run: `python tools/run_proof_baseline.py --suite smoke --seed 42 --frames 1200 --out artifacts/proof_baseline_2026_05`.
5. Artifacts generated: 35 files under `artifacts/proof_baseline_2026_05`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: copied; copy completed to local Google Drive sync folder.
9. Known limitations: no external compression baseline, no smoke labels, no plots.
10. Follow-up tasks not implemented: Week 2 false-positive suppression.


