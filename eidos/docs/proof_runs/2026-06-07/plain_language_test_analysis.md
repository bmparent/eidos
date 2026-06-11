# Plain-Language Test Analysis — 2026-06-07

## What the task attempted

This follow-up checked whether the proof work that had already produced receipts was actually reflected in git history, and then extracted the real metrics from the saved proof artifacts.

The audit focused on three questions:

- Are the two named candidate commits already in `main`?
- Was anything merged during this follow-up?
- Do the proof artifacts support a `MERGE_READY`, `HOLD`, or `BLOCKED` decision?

## Why the test matters

Proof receipts are only useful if the repository history and the reported metrics agree. A report that says "ready" without showing the git ancestry, branch state, and raw precision/recall/F1 numbers is too easy to misread.

## What was tested

The audit checked git state for:

- `a6d7bb5f77a39f88edbc4530eaa2ec0c0a6b5b30`
- `f3c9e708a2cbb18b48f97e728c4ef316b351b15a`
- the prior consolidation branch tip, `5365ae61288642ff0e90a1739ecab1427da30874`

It then read the existing proof artifacts from:

- `artifacts/proof_runs/2026-06-07/merge_consolidation_tiny_smoke`
- `artifacts/proof_runs/2026-06-07/merge_consolidation_balanced250_cpu`
- `artifacts/proof_runs/2026-06-07/merge_consolidation_transition1k_cpu`

## What passed

- `main` contains `a6d7bb5f77a39f88edbc4530eaa2ec0c0a6b5b30`.
- `main` contains `f3c9e708a2cbb18b48f97e728c4ef316b351b15a`.
- `git pull --ff-only` reported `Already up to date.`
- No merge or cherry-pick was needed.
- All three proof receipt crash scans report `crash_hit_count: 0`.
- All three runs wrote precision ledgers, calibrated ledgers, labeled metrics, proof digests, run manifests, crash scans, event confirmation reports, and Drive manifests.
- All three existing proof receipt folders were copied to Google Drive.

## What failed or remains uncertain

- The prior consolidation branch tip `5365ae6` is not on `main`. It contains report/doc receipts, not the two named candidate proof commits.
- The tiny smoke run missed its one tiny attack window, so it is useful as a crash-clean smoke receipt, not as a detection-quality acceptance result.
- The balanced250 CPU run has confirmed precision of `1.0`, but confirmed recall is only `0.166667`.
- Local CUDA was unavailable in the proof manifests, so these three receipts are CPU receipts.
- This follow-up did not rerun the proof commands; it audited and reported from the existing artifacts.

## What the metrics mean

Tiny smoke:

- Raw precision: `0.0`
- Raw recall: `0.0`
- Confirmed precision: `NA`
- Confirmed recall: `0.0`
- Crash hits: `0`
- Drive copy: copied

Balanced250 CPU:

- Raw precision: `0.583333`
- Raw recall: `0.259259`
- Confirmed precision: `1.0`
- Confirmed recall: `0.166667`
- Confirmed F1: `0.285714`
- Confirmed FP/10k: `0.0`
- Crash hits: `0`
- Drive copy: copied

Transition1k CPU:

- Raw precision: `0.5`
- Raw recall: `1.0`
- Confirmed precision: `1.0`
- Confirmed recall: `1.0`
- Confirmed F1: `1.0`
- Confirmed FP/10k: `0.0`
- Crash hits: `0`
- Drive copy: copied

## What was saved locally

- Final report: `docs/proof/merge_state_report_2026_06_07.md`
- Journal: `docs/proof_runs/2026-06-07/codex_journal.md`
- This plain-language analysis: `docs/proof_runs/2026-06-07/plain_language_test_analysis.md`
- Audit artifacts: `artifacts/proof_runs/2026-06-07/merge_consolidation_audit`

## What was saved to Google Drive

Drive was available at `G:\My Drive`. The existing proof receipt folders were already copied there. This audit also records its own Drive mirror status in:

```text
artifacts/proof_runs/2026-06-07/merge_consolidation_audit/drive_manifest.json
```

The audit folder was copied to:

```text
G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-07\merge_consolidation_audit
```

Six audit files were copied and no files were skipped.

## What remains uncertain

The strongest result is the transition1k CPU run. The balanced250 run is useful but recall is low after confirmation. The tiny smoke run should stay in the evidence set as a smoke/crash receipt, not as an approval metric.

## What should happen next

Use this report as the merge-state receipt for the named candidate commits. Do not tune thresholds or change core behavior based only on this audit. If a future task needs to improve balanced recall, it should be a separate, explicitly scoped proof/calibration task with before/after receipts.

## Sentinel calibration v1 generalization - 2026-06-07

This follow-up ran a CICIDS/WebAttacks calibration generalization matrix with proof-stage calibration enabled and core Eidos behavior unchanged.

- Decision: `HOLD`.
- Completed runs: `5`.
- Skipped optional runs: `1`.
- Suppressed events audited: `8`.
- Suppressions that may hide true attack context: `0`.
- Completed crash scans: clean (`crash_hit_count = 0`).
- Drive copy: copied; folder: `G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-07\sentinel_calibration_v1_generalization`; reason: `copy completed`.

The validation supports continued false-positive-control work, but the decision is HOLD because balanced recall/generalization remains ambiguous and the larger natural run was not feasible in this CPU pass.

## Calibration recall diagnostics - 2026-06-07

This follow-up investigated why row-shuffled balanced samples had low confirmed recall even though attack-window coverage stayed high. The work stayed proof-side: diagnostics, sample construction, confirmation profile reporting, and acceptance reporting changed; core Eidos behavior did not.

- Decision: `APPROVE`.
- Best confirmation profile: `balanced`.
- Completed proof runs: `balanced250`, `balanced1000`, `balanced_blocks250`, `balanced_blocks1000`, `transition1000`, `transition4360_max_feasible`, `natural_attack_windows`, and `natural2000`.
- Skipped proof runs: none in this matrix. GPU-only work was not attempted because the receipts reported CPU execution.
- Old balanced calibrated recall stayed low: `0.166667` on both `balanced250` and `balanced1000`.
- Window-preserving `balanced_blocks` calibrated recall improved to `1.0` on both 250-row and 1000-row samples, with calibrated FP/10k still `0.0` and coverage `100.0`.
- `natural_attack_windows` provided a feasible natural-order attack-window check: calibrated recall `1.0`, F1 `1.0`, coverage `100.0`, calibrated FP/10k `0.0`.
- `natural2000` remained useful as benign/FP pressure, but had no attack recall signal.
- Crash scans were clean across completed runs: aggregate `crash_hit_count = 0`.
- Recall-protection audit reviewed `9` dropped or suppressed events; `0` overlapped attack windows and `0` were flagged as possibly hiding true attack context.
- Artifacts: `artifacts\proof_runs\2026-06-07\calibration_recall_diagnostics`.
- Drive: `G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-07\calibration_recall_diagnostics`; success: `True`; copied files: `823`.

What this means: the HOLD was mostly a sampling/metric-interpretation problem for row-shuffled balanced samples. When attack and benign row order is preserved inside blocks, recall and coverage recover while false-positive control remains clean. The looser `recall_guarded` and `high_recall` profiles did not improve attack-bearing metrics beyond `balanced`, so `balanced` remains the best proof-stage profile.
