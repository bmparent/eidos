# Plain-Language Test Analysis -- 2026-05-29

## What the task attempted
The task created a reproducible smoke-readiness proof package for Eidos Brain.

## Why the test matters
A readiness smoke run checks whether the current branch can execute the engine path, write the expected proof artifacts, run the selected pytest smoke check, and mirror the result to Google Drive.

## What was tested
The smoke baseline ran the configured scenario list and deterministic confirmation fixtures for normal-only, isolated spike, sustained burst, nearby spikes, mode comparison, and Eidos Life lifecycle behavior.

## What passed
- synthetic_smoke: passed

## What failed
No scenario failure is hidden; see `benchmark_summary.csv` for row-level status and notes.

## What artifacts were generated
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

## What was saved locally
Artifacts were saved under `artifacts/readiness_smoke_2026_05_29`.

## What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-05-29\proof_baseline_2026_05_smoke_seed42_frames96; reason: copy completed.

## What remains uncertain
External compression baselines, labeled anomaly metrics for smoke data, plots, full experiment campaigns, and real lifecycle export replay remain future work.

## What should happen next
Use this branch as the next test/experiment base, then add labeled real-world comparisons and longer experiment receipts.
