# Plain-Language Test Analysis - 2026-06-30

## What the task attempted
This task checked whether Sentinel calibration can reduce alert pressure without hiding the raw truth of the stream.

## Why the test matters
A quieter system is not automatically smarter. The proof must show raw behavior, filtered behavior, false positives, recall, attack-window coverage, latency, crash cleanliness, Git state, and Drive persistence together.

## What was tested
The available CICIDS/WebAttacks fixture was tested through tiny transition, natural attack-window replay, and normal-only negative-control proof legs. The larger legs were checked for feasibility first.

## What passed
- The feasible CPU legs completed.
- Crash scans were included.
- Raw visibility stayed visible beside merged, deduped, confirmed, and calibrated views.
- Normal-only status: `completed`.
- Focused tests passed: guardrail builder, proof baseline runner, labeled domain proof runner, and core-touch policy tests.
- Full pytest passed: `119 passed, 1 skipped, 11 warnings`.
- Core-touch policy passed against `origin/main`.

## What failed or remains uncertain
- `balanced 250 CPU`: requires at least 125 benign and 125 attack rows; found 8 benign and 4 attack rows
- `transition 1k CPU`: requires at least 500 benign and 500 attack rows; found 8 benign and 4 attack rows
- `optional GPU 10k`: CUDA unavailable; torch reports CPU-only runtime

## What artifacts were generated
- artifacts/sentinel_calibration_guardrails_2026_06_30/calibration_guardrail_matrix.json
- artifacts/sentinel_calibration_guardrails_2026_06_30/calibration_guardrail_matrix.md
- artifacts/sentinel_calibration_guardrails_2026_06_30/profile_comparison.csv
- artifacts/sentinel_calibration_guardrails_2026_06_30/attack_window_guardrails.json
- artifacts/sentinel_calibration_guardrails_2026_06_30/false_positive_guardrails.json
- artifacts/sentinel_calibration_guardrails_2026_06_30/normal_only_guardrails.json
- artifacts/sentinel_calibration_guardrails_2026_06_30/dataset_availability_receipt.json
- artifacts/sentinel_calibration_guardrails_2026_06_30/core_touch_policy.json
- artifacts/sentinel_calibration_guardrails_2026_06_30/proof_logic_meaning.md
- artifacts/sentinel_calibration_guardrails_2026_06_30/drive_manifest.json

## What was saved locally
Local artifact folder: `artifacts/sentinel_calibration_guardrails_2026_06_30`.

## What was saved to Google Drive
Drive copy status: `copied` at `G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-30\sentinel_calibration_guardrails_2026_06_30`.

## What should happen next
Provide a larger labeled CICIDS/WebAttacks CSV and rerun this exact guardrail matrix. Then run GPU 10k only where CUDA is available.

## Proof Logic + Meaning
Goal reached: the branch is stronger and still correctly conservative at `CALIBRATION_ONLY_NEEDS_TUNING`.

Logic/math used: precision, recall, false-positive rate per 10k benign frames, attack-window coverage, crash-scan counts, and core-touch policy.

Why this is better: the evidence now explains why the branch should stay conservative instead of merely saying that large legs were skipped.

Philosophical principle: restraint before alarm, but never restraint by blindness.

How this moves Eidos forward: it improves the proof machinery around a self-monitoring streaming intelligence codec without changing the engine itself.

Evidence supports: local receipts, per-run manifests, crash scans, core-touch receipt, dataset receipt, and Drive manifest.

What remains unproven: scale, CUDA, broad generalization, and production readiness.
