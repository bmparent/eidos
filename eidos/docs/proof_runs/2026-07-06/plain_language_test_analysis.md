# Plain-Language Test Analysis - 2026-07-06

This task attempted to turn the current Eidos Brain/Sentinel accomplishments into a final Month 1 proof package and then add two harder CPU checks.

## What passed

- The July 4 larger labeled Sentinel guardrails are now merged into `origin/main`.
- Full pytest passed locally after merge: `129 passed, 1 skipped, 11 warnings`.
- Core-touch policy passed from the Eidos project root.
- The natural attack-window replay completed with crash hits `0`.
- The 2k normal-only run completed with crash hits `0`.
- The 2k normal-only calibrated strict view reported Cal FP/10k `0`.
- Both new proof runs copied artifacts to Google Drive.
- The original noisy workspace was inventoried without deleting anything.

## What did not pass or remains incomplete

- The natural replay requested up to three windows, but the receipt still contains only a small attack sample: `508` processed frames and raw recall `0.6`.
- Raw normal-only false-positive pressure remains visible: `433.333` FP/10k before calibration/confirmation.
- Local CUDA is unavailable, so the larger GPU guardrail was recorded as skipped rather than passed.
- Cross-domain proof beyond cyber is still partial.

## What the logs mean

The important result is not that raw Eidos became silent. It did not. The important result is that the proof harness now shows the raw pressure and the calibrated/operator-facing view side by side. That is the honesty layer: raw behavior remains visible, while confirmation/calibration can be evaluated as restraint.

## Proof Logic + Meaning

Goal reached: Month 1 proof packaging is now `partial_ready`, backed by merged guardrails and two July 6 CPU receipts.

Previous state: the evidence existed across branches, artifacts, and Drive folders, but it was not consolidated into one current proof package.

Technical logic utilized: event counting, attack-window overlap, confirmation profile sweeps, calibrated FP/10k, crash scans, core-touch policy, pytest, and Drive manifests.

Math used:

```text
precision = true_positive_events / (true_positive_events + false_positive_events)
recall = detected_attack_windows / total_attack_windows
FP_per_10k = false_positive_events / benign_frames * 10000
```

Philosophical meaning: restraint before alarm. Eidos should not merely notice more; it should know when not to escalate.

Why this is better: the project has stronger merged receipts, a harder benign pressure check, and a single place to see evidence, limits, and next work.

How this moves Eidos closer to the north-star goal: it strengthens reproducibility, anomaly preservation, internal monitoring, and human-readable incident receipts.

Evidence: see `month1_final_proof_package.md`, `eidos_progress_meter.json`, `proof_logic_ledger.json`, and the two July 6 run folders.

Workspace hygiene evidence: see `workspace_hygiene_inventory.md` and `workspace_hygiene_drive_manifest.json`.

Remaining uncertainty: GPU, broader natural replay, cross-domain proof, and human review of incident cards remain pending.
