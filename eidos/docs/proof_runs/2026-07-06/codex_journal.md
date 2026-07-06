# Codex Journal - 2026-07-06

## What happened today

Codex merged the July 4 larger labeled Sentinel guardrails into `origin/main`, validated the merge, ran two additional CPU proof legs, and generated the Month 1 final proof package.

## What was accomplished

- Merged `codex/tighten-larger-labeled-guardrail-calibration-2026-07-04` into `origin/main`.
- Ran focused pytest: `57 passed`.
- Ran full pytest: `129 passed, 1 skipped, 11 warnings`.
- Ran committed core-touch policy: passed.
- Ran natural attack-window replay with seed `43`.
- Ran 2k normal-only negative control with seed `43`.
- Generated progress and proof logic artifacts.
- Mirrored package receipts to Google Drive when available.
- Inventoried the original workspace duplicate/noise state without deleting or moving files.

## Tests and commands run

- `python -m pytest -c eidos\pytest.ini eidos\tests\test_core_touch_policy.py eidos\tests\test_labeled_domain_proof_runner.py eidos\tests\test_proof_baseline_runner.py eidos\tests\test_sentinel_calibration_guardrails.py eidos\tests\test_sentinel_guardrail_scale_matrix.py -q` -> passed.
- `python tools\check_core_touch_policy.py --base origin/main --committed-only` -> passed.
- `python -m pytest -q` -> passed.
- `python tools\run_labeled_domain_proof.py ... --sample-mode natural_attack_windows --natural-window-max-windows 3` -> passed.
- `python tools\run_labeled_domain_proof.py ... --sample-mode natural --frames 2000` -> passed.
- `python tools\build_month1_final_proof_package.py --run-date 2026-07-06` -> generated this package.

## Problems encountered

- Running core-touch from the outer git root creates false-positive path prefixes; the receipt must run from the Eidos project directory.
- CUDA is unavailable locally (`torch 2.6.0+cpu`, `cuda_available=False`).
- The broader natural replay request produced only a small sampled attack window in the saved receipt.
- The worktree still has pre-existing OneDrive duplicate-file noise outside this clean package path.
- The original workspace inventory counted `247` duplicate `(1)` paths and `3` pyc/cache modifications.

## What changed

Reporting/proof package files were generated. Core behavior was not changed.

## What did not change

Reservoir dynamics, RLS updates, Sentinel thresholds, raw anomaly policy, compression behavior, hippocampus behavior, and incident-card generation were not changed.

## Proof Logic + Meaning

Goal reached: Month 1 proof packaging is now `partial_ready`.

Previous state: proof receipts existed but were distributed across branches and artifact folders.

Technical logic used: precision ledger accounting, event confirmation, calibration profile sweeps, FP/10k, recall, crash scans, Drive manifests, and core-touch policy.

Math:

```text
precision = true_positive_events / (true_positive_events + false_positive_events)
recall = detected_attack_windows / total_attack_windows
FP_per_10k = false_positive_events / benign_frames * 10000
proof_score = sum(weight_i * status_credit_i)
```

Philosophical meaning: proof before novelty; restraint before alarm.

Why this is better: the current proof state is merged, validated, packaged, mirrored, and honest about what remains.

How this moves Eidos closer to the north-star goal: it strengthens reproducible proof, anomaly preservation, self-monitoring receipts, and incident explanation.

Evidence: `month1_final_proof_package.md`, `plain_language_test_analysis.md`, `eidos_progress_meter.json`, `proof_logic_ledger.json`, and the July 6 run receipts.

Remaining uncertainty: GPU, broader multi-window replay, cross-domain proof, and incident-card quality review.

## Artifacts generated

- `artifacts/proof_runs/2026-07-06/month1_final_proof_package/`
- `artifacts/progress/`
- `docs/proof_runs/2026-07-06/`
- `docs/proof_runs/2026-07-06/workspace_hygiene_inventory.md`

## Google Drive archive status

Drive status is recorded in `artifacts/proof_runs/2026-07-06/month1_final_proof_package/drive_manifest.json`.

## Thoughts on improvement

The next best PR-sized step is a single Month 1 rerun command that regenerates the scale matrix, next-harder checks, progress meter, and final package in one clean path.

## End-of-task summary

1. Files changed: reporting/proof package builder plus generated docs.
2. Whether core behavior changed: no.
3. Tests added or skipped: package-builder test added.
4. Repo-root commands run: pytest, core-touch policy, proof runners, package builder.
5. Artifacts generated: yes.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: recorded in Drive manifest.
9. Known limitations: GPU unavailable, natural replay sample narrow, raw normal-only pressure remains visible.
10. Follow-up tasks not implemented: one-command final rerun and cross-domain proof.
11. Proof Logic + Meaning written: yes.
12. Math/logic explanation included: yes.
13. Philosophical meaning included: yes.
14. Why this is better than previous state: yes.
15. How this moves Eidos closer to the ultimate goal: yes.
16. Evidence files cited: yes.
17. Remaining uncertainty / unproven claims: yes.
