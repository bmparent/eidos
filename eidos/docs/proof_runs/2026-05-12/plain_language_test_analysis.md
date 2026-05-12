# Plain-Language Test Analysis -- 2026-05-12

## What the task attempted
The task created a reproducible Week 1 baseline proof package for Eidos Brain.

## Why the test matters
A frozen baseline gives future proof work a stable comparison point before threshold or false-positive changes are attempted.

## What was tested
The smoke baseline ran the configured scenario list and captured pytest output as JUnit XML.

## What passed
- synthetic_smoke: passed

## What failed
No scenario failure is hidden; see `benchmark_summary.csv` for row-level status and notes.

## What artifacts were generated
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

## What was saved locally
Artifacts were saved under `artifacts/proof_baseline_2026_05`.

## What was saved to Google Drive
Drive status: copied; folder: C:\Users\bmpar\Google Drive\Eidos_Brain_Proof_Phase\2026-05-12\proof_baseline_2026_05_smoke_seed42_frames1200; reason: copy completed to local Google Drive sync folder.

## What remains uncertain
External compression baselines, labeled anomaly metrics for smoke data, and plots remain future work.

## What should happen next
Run the same baseline command from repo root before starting Week 2 false-positive suppression work.


