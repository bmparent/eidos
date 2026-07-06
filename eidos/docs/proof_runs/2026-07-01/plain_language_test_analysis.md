# Plain-Language Test Analysis - 2026-07-01

## What the task attempted

This task attempted to move Sentinel guardrail calibration from a tiny fixture toward a larger labeled CICIDS/WebAttacks proof matrix.

## Why the test matters

A guardrail that only passes on a tiny fixture is useful for wiring, but it is not enough evidence for larger proof. The scale matrix shows what happens when the same proof runner meets a larger real labeled source.

## What was tested

Dataset discovery, label-column resolution, balanced CPU proof, transition CPU proof, natural larger replay handling, normal-only negative control handling, optional GPU skip receipts, and core-touch policy.

## What passed

Completed proof rows are listed in the scale matrix with raw, merged, deduped, confirmed, and calibrated views visible side by side.

## What failed or remained incomplete

- `natural_larger_replay_cpu`: existing partial run lacks run_manifest.json or labeled_metrics.json; rerun skipped because --rerun-existing was not set
- `gpu_10k_optional`: CUDA unavailable; torch reports CPU-only runtime

## Artifacts generated

- `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/scale_matrix.json`
- `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/scale_matrix.csv`
- `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/scale_matrix.md`
- `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/proof_logic_meaning.md`
- `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/drive_manifest.json`
- `docs/proof/dataset_registry.json`
- `docs/proof/dataset_registry.md`
- `docs/proof_runs/2026-07-01/sentinel_guardrail_scale_matrix.md`

## What was saved locally

Local artifacts were saved under `artifacts/sentinel_guardrail_scale_matrix_2026_07_01` and docs under `docs/proof_runs/2026-07-01`.

## What was saved to Google Drive

Drive status: attempted `True`, success `True`, root `G:\My Drive`.

## What remains uncertain

The full natural-order larger replay remains incomplete, CUDA behavior is untested on this CPU-only environment, and the results should not be treated as production readiness.

## What should happen next

Make the natural replay complete through bounded attack-window sampling, checkpoint/resume, or a smaller accepted natural replay leg.

## Proof Logic + Meaning

Goal reached: larger labeled Sentinel guardrail scale packaging is `CALIBRATION_ONLY_NEEDS_TUNING`. The package discovered and registered a larger CICIDS/WebAttacks CSV, reused completed CPU proof receipts when present, and kept incomplete legs explicit.

Previous state: the earlier guardrail proof was tiny-fixture-only. It could prove the runner shape, but not whether a larger real labeled CSV existed or how far the proof harness could scale on CPU.

Technical logic utilized: the builder inspects label distributions, resolves CICIDS leading-space label headers, runs or reuses the existing labeled-domain proof runner, preserves raw/merged/deduped/confirmed/calibrated event views, and evaluates crash/core-touch receipts without changing reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, or hippocampus/familiarity behavior.

Math / scoring logic:

```text
FP/10k = false_positive_events / benign_frames * 10000
precision = true_positive_events / max(true_positive_events + false_positive_events, 1)
recall = detected_attack_windows / max(total_attack_windows, 1)
F1 = 2 * precision * recall / max(precision + recall, epsilon)
attack_window_coverage = attack_windows_with_detection / total_attack_windows
```

Philosophical meaning: Sentinel calibration is restraint before alarm, and this matrix is honesty before scale. It records what ran, what timed out, and what remains unproven.

Why this is better: the proof trail now includes a dataset registry, larger-run receipts, explicit skip reasons, a bounded negative control, and Drive status instead of relying on a tiny fixture alone.

How this moves Eidos closer to the north-star goal: Eidos Brain is a self-monitoring streaming intelligence codec. This milestone strengthens the reproducibility and incident-explanation side of that claim by making anomaly preservation and uncertainty visible on larger labeled data.

Evidence:

- `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/scale_matrix.json`
- `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/scale_matrix.csv`
- `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/scale_matrix.md`
- `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/proof_logic_meaning.md`
- `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/drive_manifest.json`
- `docs/proof/dataset_registry.json`
- `docs/proof/dataset_registry.md`
- `docs/proof_runs/2026-07-01/sentinel_guardrail_scale_matrix.md`

Remaining uncertainty: the natural larger replay did not produce a completed manifest in the resumed package, optional GPU proof was skipped when CUDA was unavailable, and the result does not prove production readiness or universal CICIDS coverage.
