# Codex Journal -- 2026-05-29

## What happened today
The proof runner generated a reproducible Eidos Brain smoke-readiness package with config, manifest, environment, git state, pytest XML, event summary, incident cards, logs, and CSV/Markdown summaries.

## What was accomplished
- Verified the smoke scenario and captured the result as local and Google Drive artifacts.
- Captured normal-only false positives, candidate events, suppressed candidates, merged events, cooldown suppressions, and incident-card-compatible records.
- Kept SentinelMonitor thresholds, reservoir dynamics, compression behavior, and prediction policy unchanged.

## Tests and commands run
- `python tools/run_proof_baseline.py --suite smoke --seed 42 --frames 96 --out artifacts/readiness_smoke_2026_05_29` -> see `benchmark_summary.md` and `pytest_results.xml`.
- Pytest status: passed (pytest completed successfully).

## Problems encountered
- External compression baselines are not implemented in Week 1.
- Smoke synthetic data does not provide ground-truth anomaly labels.
- False-positive control still uses deterministic synthetic policy checks before broader labeled telemetry work.
- Google Drive status: copied; reason: copy completed.

## What changed
- tools/run_proof_baseline.py
- repo/src/eidos_brain/engine/eidos_v0_4_7_02.py
- tests/test_integration_stream.py
- tests/test_proof_baseline_runner.py
- artifacts/readiness_smoke_2026_05_29

## What did not change
Core model behavior, Sentinel labels, SentinelMonitor thresholds, reservoir dynamics, compression behavior, and prediction decision policy were not changed.

## Artifacts generated
- artifacts/readiness_smoke_2026_05_29/benchmark_summary.csv
- artifacts/readiness_smoke_2026_05_29/benchmark_summary.md
- artifacts/readiness_smoke_2026_05_29/codex_journal.md
- artifacts/readiness_smoke_2026_05_29/config.json
- artifacts/readiness_smoke_2026_05_29/drive_manifest.json
- artifacts/readiness_smoke_2026_05_29/environment.txt
- artifacts/readiness_smoke_2026_05_29/event_summary.json
- artifacts/readiness_smoke_2026_05_29/git_commit.txt
- artifacts/readiness_smoke_2026_05_29/incident_cards/eidos_life_lifecycle_balanced_01.json
- artifacts/readiness_smoke_2026_05_29/incident_cards/eidos_life_lifecycle_balanced_02.json
- artifacts/readiness_smoke_2026_05_29/incident_cards/nearby_spikes_balanced_01.json
- artifacts/readiness_smoke_2026_05_29/incident_cards/sustained_burst_balanced_01.json
- artifacts/readiness_smoke_2026_05_29/logs/false_positive_control.jsonl
- artifacts/readiness_smoke_2026_05_29/plain_language_test_analysis.md
- artifacts/readiness_smoke_2026_05_29/plots/README.md
- artifacts/readiness_smoke_2026_05_29/pytest_results.xml
- artifacts/readiness_smoke_2026_05_29/pytest_stderr.txt
- artifacts/readiness_smoke_2026_05_29/pytest_stdout.txt
- artifacts/readiness_smoke_2026_05_29/run_manifest.json
- artifacts/readiness_smoke_2026_05_29/scenarios/_engine_artifacts/compression/proof_baseline_synthetic_smoke/20260529_221349_bicameral_stream_meta_proof_baseline_synthetic_smoke.json
- artifacts/readiness_smoke_2026_05_29/scenarios/_engine_artifacts/compression/proof_baseline_synthetic_smoke/20260529_221349_bicameral_stream_proof_baseline_synthetic_smoke.bin
- artifacts/readiness_smoke_2026_05_29/scenarios/_engine_artifacts/eidos_brain_archive/20260529_221308_proof_baseline_synthetic_smoke/anomalies.jsonl
- artifacts/readiness_smoke_2026_05_29/scenarios/_engine_artifacts/eidos_brain_archive/20260529_221308_proof_baseline_synthetic_smoke/clusters.json
- artifacts/readiness_smoke_2026_05_29/scenarios/_engine_artifacts/eidos_brain_archive/20260529_221308_proof_baseline_synthetic_smoke/report.txt
- artifacts/readiness_smoke_2026_05_29/scenarios/_engine_artifacts/eidos_brain_archive/20260529_221308_proof_baseline_synthetic_smoke/session_meta.json
- artifacts/readiness_smoke_2026_05_29/scenarios/_engine_artifacts/eidos_brain_archive/20260529_221308_proof_baseline_synthetic_smoke/state_capsule.json
- artifacts/readiness_smoke_2026_05_29/scenarios/_engine_artifacts/eidos_brain_archive/20260529_221308_proof_baseline_synthetic_smoke/steps.csv
- artifacts/readiness_smoke_2026_05_29/scenarios/_engine_artifacts/eidos_brain_archive/20260529_221308_proof_baseline_synthetic_smoke/summary.json
- artifacts/readiness_smoke_2026_05_29/scenarios/_engine_artifacts/forecast.jsonl
- artifacts/readiness_smoke_2026_05_29/scenarios/_engine_artifacts/hippocampus/proof_baseline_synthetic_smoke/20260529_221349_hippocampus_snapshot_proof_baseline_synthetic_smoke.pt
- artifacts/readiness_smoke_2026_05_29/scenarios/_engine_artifacts/incident_cards.jsonl
- artifacts/readiness_smoke_2026_05_29/scenarios/_engine_artifacts/manifest.jsonl
- artifacts/readiness_smoke_2026_05_29/scenarios/_engine_artifacts/reservoir_checkpoints/proof_baseline_synthetic_smoke/20260529_221348_reservoir_checkpoint_proof_baseline_synthetic_smoke.pt
- artifacts/readiness_smoke_2026_05_29/scenarios/_engine_artifacts/reservoir_geometry/proof_baseline_synthetic_smoke/20260529_221349_reservoir_geom_proof_baseline_synthetic_smoke.json
- artifacts/readiness_smoke_2026_05_29/scenarios/_engine_artifacts/reservoir_geometry/proof_baseline_synthetic_smoke/20260529_221349_reservoir_states_proof_baseline_synthetic_smoke.npy
- artifacts/readiness_smoke_2026_05_29/scenarios/_engine_artifacts/sentinel_forensics/proof_baseline_synthetic_smoke/20260529_221349_top_100_surprises_proof_baseline_synthetic_smoke.json
- artifacts/readiness_smoke_2026_05_29/scenarios/_engine_artifacts/sentinel_forensics/proof_baseline_synthetic_smoke/20260529_221349_top_100_surprises_text_proof_baseline_synthetic_smoke.txt
- artifacts/readiness_smoke_2026_05_29/scenarios/synthetic_smoke/engine_output.log
- artifacts/readiness_smoke_2026_05_29/scenarios/synthetic_smoke/report.txt
- artifacts/readiness_smoke_2026_05_29/scenarios/synthetic_smoke/scenario_manifest.json
- artifacts/readiness_smoke_2026_05_29/scenarios/synthetic_smoke/scenario_spec.json

## Google Drive archive status
- Drive root used: G:\My Drive
- Drive folder used: G:\My Drive\Eidos_Brain_Proof_Phase\2026-05-29\proof_baseline_2026_05_smoke_seed42_frames96
- Files copied: 40
- Files skipped: 0
- Reason: copy completed

## Thoughts on improvement
Normal-only confirmed false positives were `0` per 10k synthetic frames; the next step should compare this policy against labeled real-world streams.

## Where to improve next
Run the broader experiment/test suite on this readiness branch and compare future receipts against this smoke package.

## Anything that stands out
Recall preservation for the synthetic burst: Synthetic sustained burst confirmed.

## End-of-task summary
1. Files changed: tools/run_proof_baseline.py, repo/src/eidos_brain/engine/eidos_v0_4_7_02.py, tests/test_integration_stream.py, tests/test_proof_baseline_runner.py, artifacts/readiness_smoke_2026_05_29
2. Whether core behavior changed: no.
3. Tests added or skipped: Sentinel confirmation tests added; pytest XML captured by the proof run.
4. Repo-root commands run: `python tools/run_proof_baseline.py --suite smoke --seed 42 --frames 96 --out artifacts/readiness_smoke_2026_05_29`.
5. Artifacts generated: 41 files under `artifacts/readiness_smoke_2026_05_29`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: copied; copy completed.
9. Known limitations: no external compression baseline, no smoke labels, no plots, synthetic-only false-positive fixtures.
10. Follow-up tasks not implemented: full long-run experiment campaign and labeled real-world false-positive comparison.
