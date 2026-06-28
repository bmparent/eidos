# Plain-Language Test Analysis - 2026-06-28

## What the task attempted

This task checked whether the current Eidos Brain / Sentinel labeled-domain proof harness can measure CICIDS/WebAttacks behavior clearly enough to support the next calibration step. The goal was verification and proof discipline, not core model tuning.

## Why the test matters

The important question is no longer whether Eidos can run. The question is whether Eidos can measure attack visibility, false-positive pressure, event merging, dedupe behavior, incident-card accounting, crash cleanliness, and artifact completeness without hiding raw behavior.

## What was tested

- Tiny labeled fixture smoke run.
- Balanced 250 real-data CPU run.
- Transition 1k real-data CPU run.
- Natural-order real-data CPU run selected to include the first attack rows.
- Optional GPU 10k availability check.
- Targeted proof-runner tests.
- Full pytest suite.
- Expanded crash scan for `Traceback`, `CRASH IN INCIDENT LOGIC`, `can't convert cuda`, `RuntimeError`, `ValueError`, `NaN`, and `Inf`.

## What passed

- Targeted proof-runner tests passed.
- Final full pytest passed: 175 passed, 2 skipped, 11 warnings.
- All four CPU proof runs completed.
- Every completed proof run emitted `run_manifest.json`, `environment.txt`, `precision_ledger.json`, `precision_ledger.md`, `proof_digest.json`, `crash_scan.json`, and `drive_manifest.json`.
- Crash scans were clean for completed runs after separating known nonfatal HIPP NaN telemetry warnings.
- Raw, merged, and deduped metrics remained visible side by side.
- False positives were categorized.
- Attack-window diagnostics were populated for runs with attacks.
- Incident-card accounting was present.
- Google Drive copies succeeded.

## What failed or remains uncertain

- CUDA was unavailable, so the optional GPU 10k run was skipped.
- Balanced 250 remains useful as a controlled sample, but it is not enough as a standalone recommendation because merged/deduped recall dropped from raw recall 0.259 to 0.167.
- Natural order exposed severe raw false-positive pressure. That is useful evidence, not a failure to hide.
- Natural replay requested 13,000 frames but the engine processed 11,700 frames. The run still included attack rows and produced usable receipts.
- The natural run logged 5 nonfatal `HIPP bank=INCIDENT sim=NaN` telemetry warnings. These were not runtime crashes, but they should stay visible in receipts.

## Main result in plain language

Merged and deduped views dramatically reduce alert pressure, especially on transition and natural runs, but they cannot replace raw metrics. Balanced mode is still a useful controlled calibration input, but the next step should compare multiple policy profiles against raw views and enforce recall/coverage guardrails.

## What was saved locally

- `artifacts/eidos_current_state_audit.md`
- `artifacts/eidos_sentinel_proof_verification_summary.md`
- `artifacts/next_codex_prompt_sentinel_calibration_v1.md`
- `artifacts/proof_runs/tiny_fixture_smoke/`
- `artifacts/proof_runs/cicids_webattacks_balanced_250_cpu/`
- `artifacts/proof_runs/cicids_webattacks_transition_1k_cpu/`
- `artifacts/proof_runs/cicids_webattacks_natural_cpu/`
- `artifacts/proof_runs/cicids_webattacks_gpu_10k/`

## What was saved to Google Drive

Artifacts were copied to `G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-28\...` for the tiny, balanced, transition, natural, and GPU-skip receipts.

## What remains uncertain

The proof system still needs calibration guardrails before any recommendation can be stronger. In particular, the next step should prove that FP suppression does not erase attack visibility in balanced, transition, natural, and normal-only negative-control cases.

## What should happen next

Run a proof-harness-first Sentinel calibration v1 task. It should test candidate confirmation windows, persistence requirements, merge/cooldown tuning, normal-stream suppression, and `low_noise`, `balanced`, and `high_recall` policy profiles while preserving raw metrics and requiring recall/coverage guardrails.
