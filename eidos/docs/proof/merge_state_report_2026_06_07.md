# Eidos Proof Merge Consolidation Report — 2026-06-07

## Git state

Commands were run from the Eidos project working directory:

```text
C:\Users\bmpar\OneDrive\Documents\eidos-brain\eidos
```

Git top-level resolves one directory higher:

```text
C:/Users/bmpar/OneDrive/Documents/eidos-brain
```

Current state after the audit:

- Current branch before checkout: `codex/eidos-proof-merge-consolidation-2026-06-07`
- Current HEAD before checkout: `5365ae61288642ff0e90a1739ecab1427da30874`
- Current branch after requested checkout: `main`
- Current HEAD after requested checkout: `1092805e433a803a169d0a15316b66d9ea85c4db`
- Main HEAD after `git pull --ff-only`: `1092805e433a803a169d0a15316b66d9ea85c4db`
- `MAIN_CONTAINS_A6D7BB5`: YES
- `MAIN_CONTAINS_F3C9E708`: YES
- `MAIN_CONTAINS_5365AE6`: NO
- Merge performed in this audit: no
- Pull result: `Already up to date.`
- Cherry-picks performed in this audit: none
- Conflicts encountered: none

Important distinction:

- The two named candidate commits, `a6d7bb5f77a39f88edbc4530eaa2ec0c0a6b5b30` and `f3c9e708a2cbb18b48f97e728c4ef316b351b15a`, are already ancestors of `main`.
- The prior consolidation branch tip, `5365ae61288642ff0e90a1739ecab1427da30874`, is not an ancestor of `main`. That branch-tip commit contains report/doc receipts, not a new merge of the two candidate commits.

Branches containing `a6d7bb5f77a39f88edbc4530eaa2ec0c0a6b5b30`:

- `main`
- `codex/eidos-hotfix-gpu-config-2026-05-31`
- `codex/eidos-cicids-webattacks-proof-2026-05-31`
- `codex/eidos-event-confirmation-layer-v1-2026-06-01`
- `codex/eidos-sentinel-calibration-v1-2026-06-03`
- `codex/eidos-calibration-v1-generalization-2026-06-04`
- `codex/eidos-proof-merge-consolidation-2026-06-07`
- `remotes/origin/HEAD -> origin/main`
- `remotes/origin/main`
- `remotes/origin/codex/eidos-hotfix-gpu-config-2026-05-31`
- `remotes/origin/codex/eidos-cicids-webattacks-proof-2026-05-31`
- `remotes/origin/codex/eidos-event-confirmation-layer-v1-2026-06-01`

Branches containing `f3c9e708a2cbb18b48f97e728c4ef316b351b15a`:

- `main`
- `codex/eidos-cicids-webattacks-proof-2026-05-31`
- `codex/eidos-event-confirmation-layer-v1-2026-06-01`
- `codex/eidos-sentinel-calibration-v1-2026-06-03`
- `codex/eidos-calibration-v1-generalization-2026-06-04`
- `codex/eidos-proof-merge-consolidation-2026-06-07`
- `remotes/origin/HEAD -> origin/main`
- `remotes/origin/main`
- `remotes/origin/codex/eidos-cicids-webattacks-proof-2026-05-31`
- `remotes/origin/codex/eidos-event-confirmation-layer-v1-2026-06-01`

Branches containing `5365ae61288642ff0e90a1739ecab1427da30874`:

- `codex/eidos-proof-merge-consolidation-2026-06-07`

`git log --oneline --decorate --graph --all --max-count=80` confirms the named candidates are below current `main`:

```text
* 5365ae6 (codex/eidos-proof-merge-consolidation-2026-06-07) proof: consolidate validated baseline and cicids precision ledger
*   1092805 (HEAD -> main, origin/main, origin/HEAD) Merge pull request #23 from bmparent/codex/eidos-calibration-v1-generalization-2026-06-04
| * f09144a (codex/eidos-sentinel-calibration-v1-2026-06-03) Add Sentinel calibration acceptance gate
| * d96061f Record Sentinel calibration v1 proof receipts
| * c17d0a4 Add Sentinel calibration v1 proof layer
| * 7ca74f2 (codex/eidos-event-confirmation-layer-v1-2026-06-01) Add labeled run comparison receipts
* |   650cfb3 Merge pull request #21 from bmparent/codex/eidos-cicids-webattacks-proof-2026-05-31
| * 9a59238 (codex/eidos-cicids-webattacks-proof-2026-05-31) Add CICIDS precision ledger proof harness
| * f3c9e70 Add labeled CICIDS WebAttacks proof harness
| * a6d7bb5 (origin/codex/eidos-hotfix-gpu-config-2026-05-31, codex/eidos-hotfix-gpu-config-2026-05-31) Document official Colab GPU 10k proof baseline
```

PowerShell note: the final `merge-base --is-ancestor` checks were validated through `$LASTEXITCODE`, which is the PowerShell equivalent of the requested shell `&& echo ... || echo ...` behavior.

## Decision

MERGE_READY

Reason:

- `main` already contains both requested candidate commits.
- `git pull --ff-only` reported `Already up to date.`
- No merge or cherry-pick was needed in this audit.
- Crash scans are clean in all three generated proof receipts.
- The generated proof artifacts include the requested precision ledgers, calibrated ledgers, event confirmation reports, labeled metrics, proof digests, run manifests, crash scans, incident cards, and Drive manifests.
- The report-only consolidation branch tip `5365ae6` was not merged during this audit; this corrected report supplies the missing audit answer on `main`.

This is not a production-readiness claim. It means the named proof consolidation is ready from a git/proof-receipt audit perspective, with the caveats below preserved.

## Proof artifact sources read

For each of the three proof folders, the audit read:

- `benchmark_summary.csv`
- `benchmark_summary.md`
- `labeled_metrics.json`
- `labeled_metrics.md`
- `precision_ledger.json`
- `precision_ledger.md`
- `calibrated_precision_ledger.json`
- `calibrated_precision_ledger.md`
- `event_confirmation_report.json`
- `event_confirmation_report.md`
- `proof_digest.json`
- `proof_digest.md`
- `crash_scan.json`
- `run_manifest.json`
- `drive_manifest.json`

Source folders:

- `artifacts/proof_runs/2026-06-07/merge_consolidation_tiny_smoke`
- `artifacts/proof_runs/2026-06-07/merge_consolidation_balanced250_cpu`
- `artifacts/proof_runs/2026-06-07/merge_consolidation_transition1k_cpu`

## Run overview

| run | suite | sample mode | frames requested | frames processed | selected device | CUDA available | torch | Drive copy |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| `merge_consolidation_tiny_smoke` | smoke | natural | 12 | 11 | cpu | false | 2.6.0+cpu | copied, 40 files |
| `merge_consolidation_balanced250_cpu` | full | balanced | 250 | 225 | cpu | false | 2.6.0+cpu | copied, 51 files |
| `merge_consolidation_transition1k_cpu` | full | transition | 1000 | 900 | cpu | false | 2.6.0+cpu | copied, 73 files |

All three run manifests report `cpu_fallback_used: true`, `cuda_available: false`, and no device error.

## Event-view metrics

`NA` means the artifact recorded `null`, usually because there were no events in that view.

| run | view | events | precision | recall | F1 | TP | FP | FN | FP/10k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny smoke | raw | 1 | 0.000000 | 0.000000 | NA | 0 | 1 | 1 | 909.090909 |
| tiny smoke | merged | 1 | 0.000000 | 0.000000 | NA | 0 | 1 | 1 | 909.090909 |
| tiny smoke | deduped | 1 | 0.000000 | 0.000000 | NA | 0 | 1 | 1 | 909.090909 |
| tiny smoke | confirmed | 0 | NA | 0.000000 | NA | 0 | 0 | 1 | 0.000000 |
| tiny smoke | calibrated | 0 | NA | 0.000000 | NA | 0 | 0 | 1 | 0.000000 |
| balanced250 CPU | raw | 12 | 0.583333 | 0.259259 | 0.358974 | 7 | 5 | 20 | 222.222222 |
| balanced250 CPU | merged | 1 | 1.000000 | 0.166667 | 0.285714 | 1 | 0 | 5 | 0.000000 |
| balanced250 CPU | deduped | 1 | 1.000000 | 0.166667 | 0.285714 | 1 | 0 | 5 | 0.000000 |
| balanced250 CPU | confirmed | 1 | 1.000000 | 0.166667 | 0.285714 | 1 | 0 | 5 | 0.000000 |
| balanced250 CPU | calibrated | 1 | 1.000000 | 0.166667 | 0.285714 | 1 | 0 | 5 | 0.000000 |
| transition1k CPU | raw | 34 | 0.500000 | 1.000000 | 0.666667 | 17 | 17 | 0 | 188.888889 |
| transition1k CPU | merged | 3 | 1.000000 | 1.000000 | 1.000000 | 3 | 0 | 0 | 0.000000 |
| transition1k CPU | deduped | 3 | 1.000000 | 1.000000 | 1.000000 | 3 | 0 | 0 | 0.000000 |
| transition1k CPU | confirmed | 3 | 1.000000 | 1.000000 | 1.000000 | 3 | 0 | 0 | 0.000000 |
| transition1k CPU | calibrated | 3 | 1.000000 | 1.000000 | 1.000000 | 3 | 0 | 0 | 0.000000 |

## Event counts and attack-window coverage

| run | raw events | merged events | deduped events | confirmed events | suppressed events | attack-window coverage after calibration | detected windows | missed windows | first detection latency | crash_hit_count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny smoke | 1 | 1 | 1 | 0 | 1 | 0.00000% | 0 | 1 | NA | 0 |
| balanced250 CPU | 12 | 1 | 1 | 1 | 0 | 91.37931% | 53 | 5 | 0 frames | 0 |
| transition1k CPU | 34 | 3 | 3 | 3 | 0 | 100.00000% | 1 | 0 | 0 frames | 0 |

## False-positive taxonomy

| run | false-positive taxonomy counts | suppressed taxonomy counts |
| --- | --- | --- |
| tiny smoke | `raw:pre_attack_near_transition=1`; `merged:pre_attack_near_transition=1`; `deduped:pre_attack_near_transition=1` | `pre_attack_near_transition=1` |
| balanced250 CPU | `raw:pre_attack_near_transition=1`; `raw:post_attack_near_transition=4` | none |
| transition1k CPU | `raw:fully_benign=16`; `raw:pre_attack_near_transition=1` | none |

## Compression receipts

| run | Eidos compression ratio | best external compression baseline | skipped external baselines |
| --- | ---: | --- | --- |
| tiny smoke | 6.115092 | `delta_zlib` at 19.027027 | `zstd`, `lz4` |
| balanced250 CPU | 5.984105 | `lzma` at 1.074948 | `zstd`, `lz4` |
| transition1k CPU | 4.587448 | `lzma` at 1.115609 | `zstd`, `lz4` |

The skipped external baselines were skipped because the optional packages were not installed. This is recorded by the artifacts and is not treated as a proof-run failure.

## Drive status

Existing proof receipt Drive copies:

- Tiny smoke: copied to `G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-07\cicids_webattacks_proof_smoke_seed42_frames11_20260607T201046Z`
- Balanced250 CPU: copied to `G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-07\cicids_webattacks_proof_full_seed42_frames225_20260607T201401Z`
- Transition1k CPU: copied to `G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-07\cicids_webattacks_proof_full_seed42_frames900_20260607T201855Z`

Drive readiness for this audit:

- `tools\configure_proof_drive_env.ps1 -CheckOnly` found `G:\My Drive`.
- The root exists, is verified as a Drive root, and is writable.
- The helper also reported that the active shell should set `EIDOS_PROOF_DRIVE_DIR='G:\My Drive'` before proof mirror commands.

This audit's local artifact folder is:

```text
artifacts/proof_runs/2026-06-07/merge_consolidation_audit
```

This audit's Drive mirror status is recorded in:

```text
artifacts/proof_runs/2026-06-07/merge_consolidation_audit/drive_manifest.json
```

Audit Drive copy result:

- Status: copied
- Drive folder: `G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-07\merge_consolidation_audit`
- Files copied: 6
- Files skipped: 0
- Reason: `copy completed after manifest path rewrite`

## Core behavior audit

No core Eidos behavior was changed by this audit.

- Reservoir dynamics changed: no
- RLS updates changed: no
- Surprise scoring changed: no
- Sentinel labels changed: no
- Sentinel thresholds changed: no
- Anomaly policy changed: no
- Compression behavior changed: no
- Memory/familiarity behavior changed: no
- Incident logic changed: no
- Forecasting logic changed: no
- Domain-profile behavior changed: no

This task was reporting and audit work only.

## Caveats

- This follow-up did not rerun the three proof receipts; it extracted metrics from the generated artifacts named in the prompt.
- The tiny smoke run is crash-clean, but it missed its one tiny attack window. It should not be treated as a detection-quality acceptance proof.
- The balanced250 CPU run removes false positives after merged/deduped/confirmed accounting, but confirmed recall is only `0.166667`.
- The transition1k CPU run is the strongest receipt in this set: confirmed/calibrated precision, recall, and F1 are all `1.000000`, with `0` FP/10k and clean crash scan.
- All three runs are CPU receipts. Local CUDA was unavailable in the recorded manifests.
- The original proof run manifests recorded `git_dirty: true` because tracked `__pycache__` paths were dirty at proof-run time. The current audit observed a clean worktree before creating this report.

## End-of-task summary

1. Files changed: `docs/proof/merge_state_report_2026_06_07.md`, `docs/proof_runs/2026-06-07/plain_language_test_analysis.md`, `docs/proof_runs/2026-06-07/codex_journal.md`; ignored local audit artifacts under `artifacts/proof_runs/2026-06-07/merge_consolidation_audit`.
2. Whether core behavior changed: no.
3. Tests added or skipped: no tests were added; this was an audit/reporting task over existing proof receipts.
4. Repo-root commands run: the requested git fetch/status/branch/HEAD/checkout/pull/ancestry/branch/log audit commands, artifact extraction commands, Drive readiness check, and report validation commands.
5. Artifacts generated: audit summary, normalized metrics summary, run manifest, Drive manifest, and mirrored report receipts under `artifacts/proof_runs/2026-06-07/merge_consolidation_audit`.
6. Local artifact folder path: `artifacts/proof_runs/2026-06-07/merge_consolidation_audit`.
7. Google Drive copy status: copied to `G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-07\merge_consolidation_audit`; 6 files copied, 0 skipped.
8. Plain-language analysis written: yes, `docs/proof_runs/2026-06-07/plain_language_test_analysis.md`.
9. Codex journal entry written: yes, `docs/proof_runs/2026-06-07/codex_journal.md`.
10. Known limitations: CPU-only receipts, tiny smoke miss, balanced250 low confirmed recall, optional `zstd`/`lz4` baselines unavailable, and no new proof rerun in this follow-up.
11. Follow-up tasks not implemented: no threshold tuning, no Sentinel behavior changes, no core model changes, no merge/cherry-pick, and no production-readiness claim.
