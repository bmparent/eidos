# Codex Journal — 2026-06-07

## What happened today

Completed the missing proof merge consolidation audit report. The work checked git ancestry for the two named proof commits, confirmed the current branch and HEAD state, extracted metrics from the three existing proof receipt folders, and wrote a corrected decision-oriented report.

## What was accomplished

- Confirmed `main` contains `a6d7bb5f77a39f88edbc4530eaa2ec0c0a6b5b30`.
- Confirmed `main` contains `f3c9e708a2cbb18b48f97e728c4ef316b351b15a`.
- Confirmed the prior consolidation branch tip `5365ae61288642ff0e90a1739ecab1427da30874` is not on `main`.
- Confirmed no merge or cherry-pick was performed in this follow-up.
- Extracted raw, merged, deduped, confirmed, and calibrated precision/recall/F1/FP metrics from the proof artifacts.
- Recorded crash scan, compression, device/CUDA, attack-window coverage, false-positive taxonomy, and Drive-copy status.
- Wrote the final consolidation report at `docs/proof/merge_state_report_2026_06_07.md`.

## Tests and commands run

- `git fetch --all --prune` -> passed.
- `git status --short` -> clean before report creation.
- `git rev-parse --abbrev-ref HEAD` -> first `codex/eidos-proof-merge-consolidation-2026-06-07`, then `main` after checkout.
- `git rev-parse HEAD` -> first `5365ae61288642ff0e90a1739ecab1427da30874`, then `1092805e433a803a169d0a15316b66d9ea85c4db`.
- `git checkout main` -> passed.
- `git pull --ff-only` -> `Already up to date.`
- `git merge-base --is-ancestor a6d7bb5f77a39f88edbc4530eaa2ec0c0a6b5b30 main` -> passed; `MAIN_CONTAINS_A6D7BB5=YES`.
- `git merge-base --is-ancestor f3c9e708a2cbb18b48f97e728c4ef316b351b15a main` -> passed; `MAIN_CONTAINS_F3C9E708=YES`.
- `git branch -a --contains a6d7bb5f77a39f88edbc4530eaa2ec0c0a6b5b30` -> passed.
- `git branch -a --contains f3c9e708a2cbb18b48f97e728c4ef316b351b15a` -> passed.
- `git log --oneline --decorate --graph --all --max-count=80` -> passed.
- `git merge-base --is-ancestor 5365ae61288642ff0e90a1739ecab1427da30874 main` -> failed as expected; `MAIN_CONTAINS_5365AE6=NO`.
- `powershell -NoProfile -ExecutionPolicy Bypass -File tools\configure_proof_drive_env.ps1 -CheckOnly` -> found writable `G:\My Drive`.

## Problems encountered

- A first PowerShell attempt to test `git merge-base --is-ancestor` used `if (git merge-base ...)`, which checks command output instead of the external command exit code. The audit was corrected by rerunning the checks with `$LASTEXITCODE`.
- The proof run manifests record `git_dirty: true` from tracked `__pycache__` churn at proof-run time. The current audit observed a clean worktree before creating the report.
- Local CUDA was unavailable in the proof manifests; all three proof receipts are CPU receipts.
- The first audit Drive manifest write used a relative-path helper unavailable in this PowerShell/.NET runtime, which produced null path entries. The manifest was rewritten with a simpler relative-path calculation and recopied to Drive.

## What changed

- Added `docs/proof/merge_state_report_2026_06_07.md`.
- Added `docs/proof_runs/2026-06-07/plain_language_test_analysis.md`.
- Added `docs/proof_runs/2026-06-07/codex_journal.md`.
- Added ignored local audit artifacts under `artifacts/proof_runs/2026-06-07/merge_consolidation_audit`.

## What did not change

Core model behavior was untouched.

- Reservoir dynamics: unchanged.
- RLS updates: unchanged.
- Surprise scoring: unchanged.
- Sentinel labels and thresholds: unchanged.
- Anomaly policy: unchanged.
- Compression behavior: unchanged.
- Memory/familiarity behavior: unchanged.
- Incident and forecasting logic: unchanged.
- Domain-profile behavior: unchanged.

## Artifacts generated

- `artifacts/proof_runs/2026-06-07/merge_consolidation_audit/git_audit_summary.txt`
- `artifacts/proof_runs/2026-06-07/merge_consolidation_audit/metrics_extraction_summary.json`
- `artifacts/proof_runs/2026-06-07/merge_consolidation_audit/run_manifest.json`
- `artifacts/proof_runs/2026-06-07/merge_consolidation_audit/drive_manifest.json`

## Google Drive archive status

Drive readiness check found `G:\My Drive` and verified it as writable. The audit artifact folder was copied to `G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-07\merge_consolidation_audit`.

- Files copied: 6
- Files skipped: 0
- Reason: `copy completed after manifest path rewrite`
- Local manifest: `artifacts/proof_runs/2026-06-07/merge_consolidation_audit/drive_manifest.json`

## Thoughts on improvement

The report is now clear about the difference between candidate commits being in `main` and the prior report-only consolidation branch tip not being in `main`. Future proof merge reports should include that distinction by default.

## Where to improve next

If proof quality needs to improve, use a separate calibration or comparison task. Do not tune thresholds or change core behavior inside a merge-state audit.

## Anything that stands out

The transition1k CPU receipt is strong: confirmed/calibrated precision, recall, and F1 are all `1.0`, with `0` FP/10k and a clean crash scan. Balanced250 is clean on false positives after confirmation, but recall is low. Tiny smoke is useful as a smoke receipt but should not be oversold.

## End-of-task summary

1. Files changed: `docs/proof/merge_state_report_2026_06_07.md`, `docs/proof_runs/2026-06-07/plain_language_test_analysis.md`, `docs/proof_runs/2026-06-07/codex_journal.md`; ignored audit artifacts under `artifacts/proof_runs/2026-06-07/merge_consolidation_audit`.
2. Whether core behavior changed: no.
3. Tests added or skipped: no tests added; this was an audit/reporting task.
4. Repo-root commands run: requested git audit commands, artifact extraction commands, Drive readiness check, and report validation commands.
5. Artifacts generated: audit summary, normalized metrics summary, run manifest, and Drive manifest.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: copied to `G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-07\merge_consolidation_audit`; 6 files copied, 0 skipped.
9. Known limitations: existing proofs are CPU receipts; tiny smoke missed its attack window; balanced250 confirmed recall is low; no new proof rerun was performed.
10. Follow-up tasks not implemented: no merge, no cherry-pick, no threshold tuning, no core behavior changes, no new proof feature work.

## Sentinel calibration v1 generalization validation - 2026-06-07

What happened today: ran balanced250, balanced1000, transition1000, transition4360 max-feasible, and natural2000 with calibration enabled. The natural15000 optional leg was started and interrupted during dataset loading after the completed max-transition CPU leg took about 39 minutes.

What was accomplished:
- Wrote `generalization_report.md`, `generalization_summary.csv`, `recall_protection_audit.json`, and `decision.json`.
- Audited every confirmation/calibration suppressed event from the completed receipt folders.
- Confirmed completed crash scans reported `crash_hit_count = 0`.
- Mirrored the aggregate validation artifact folder to Google Drive when `EIDOS_PROOF_DRIVE_DIR` was set.

What changed: reporting and proof artifacts only. Core Eidos behavior did not change.

What did not change: reservoir dynamics, RLS behavior, raw Sentinel thresholds, anomaly policy, compression behavior, hippocampus memory, and incident-card generation remained untouched.

End-of-task decision: `HOLD` - FP control and crash scans are clean, but generalization remains ambiguous: balanced recall is weak or the optional larger natural run was skipped as CPU-infeasible.

## Calibration recall diagnostics implementation - 2026-06-07

What happened today: investigated the Sentinel calibration v1 HOLD by adding proof-side recall diagnostics, block-preserving balanced sampling, targeted natural attack-window sampling, and confirmation profile sweep reporting.

What was accomplished:
- Added candidate funnel reporting from raw candidates through merged, deduped, confirmed, and calibrated confirmed events.
- Added dropped/suppressed event accounting with reason codes and attack-window context.
- Added `balanced_blocks` and `natural_attack_windows` sample modes for proof receipts.
- Added proof-stage confirmation profiles: `strict`, `balanced`, `recall_guarded`, and `high_recall`.
- Produced the aggregate diagnostics folder at `artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics`.

Tests and commands run:
- `python -m py_compile tools/run_labeled_domain_proof.py proof/event_confirmation.py tools/build_calibration_recall_diagnostics.py` -> passed.
- `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest -q tests/test_labeled_domain_proof_runner.py tests/test_sentinel_calibration_generalization.py tests/test_sentinel_calibration_acceptance.py` -> `31 passed`.
- `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest -q` -> `100 passed, 1 skipped`.
- Proof matrix completed for `balanced250`, `balanced1000`, `balanced_blocks250`, `balanced_blocks1000`, `transition1000`, `transition4360_max_feasible`, `natural_attack_windows`, and `natural2000`.

Problems encountered:
- OneDrive briefly locked `run_matrix_status.jsonl` during three sidecar status writes. The authoritative per-run metrics, manifests, crash scans, and Drive manifests were still written successfully.
- Running `tools/build_calibration_recall_diagnostics.py` as a direct path could not import the repo `tools` namespace. Rerunning as `python -m tools.build_calibration_recall_diagnostics` resolved it without changing `PYTHONPATH`.

What changed: proof runner diagnostics, proof-side confirmation profiles, sampling receipts, tests, aggregate reports, and docs.

What did not change: reservoir dynamics, RLS behavior, raw Sentinel thresholds, anomaly policy, compression behavior, hippocampus memory, incident-card generation, and domain adapter math.

Artifacts generated:
- `generalization_recall_report.md`
- `generalization_recall_summary.csv`
- `candidate_funnel_report.json`
- `candidate_funnel_report.md`
- `confirmation_profile_sweep.csv`
- `confirmation_profile_sweep.md`
- `balanced_blocks_manifest.json`
- `natural_attack_windows_manifest.json`
- `recall_protection_audit.json`
- `decision.json`

Google Drive archive status: copied to `G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-07\calibration_recall_diagnostics`; aggregate `drive_manifest.json` reports `drive_copy_success = true` and `823` files copied.

Decision: `APPROVE` - recall improved on `balanced_blocks` and `natural_attack_windows` while calibrated FP/10k stayed `0.0`, crash scans were clean, coverage stayed at or above the acceptance threshold, and the recall-protection audit found no suppressed true attack-context events.
