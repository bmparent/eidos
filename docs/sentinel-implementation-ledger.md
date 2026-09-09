# Sentinel implementation ledger

Brief: `Eidos_Sentinel_Full_Implementation_Prompt.md`, read completely on 2026-09-08.
Working branch: `codex/sentinel-guided-analysis-20260908`.
Isolated root: `C:/Users/bmpar/codex-worktrees/sentinel-guided-analysis-20260908`.
Baseline and production: `be02b6cba3579eb10412f7d148e22e62a48a87df`, Vercel deployment `dpl_5MF1AcvzUtUEzJ8fDKfbU6Vfdsc9`, READY.
Original checkout has extensive uncommitted/OneDrive residue and was not modified.
Open PR #48 supplies separate Works member services; inspect/reuse contracts without merging or changing its unrelated work. PR #36 is unrelated memory research.

## Execution sequence

1. Versioned data contract, safe ingestion, private durable records/jobs, guided UI and evidence findings.
2. Separately versioned causal Torch Eidos readout, original-unit forecasts, rolling evaluation and mechanism ablations.
3. Scoped telemetry credentials, collector/replay package, durable offsets/state and interruption recovery.
4. Semantic document/web representations, passage retrieval and grounded result questions.
5. Integration/security/regression tests, desktop/mobile browser receipts, preview, coherent commits and review PR.

## Requirement to evidence

| Requirement | Milestone | Status | Code / planned location | Acceptance evidence / dependency |
|---|---|---|---|---|
| CSV, XLSX, Parquet, JSON/JSONL, logs, text PDF | 1,4 | implementing | runner guided ingestion | Browser upload, parse, saved result and source drilldown; malformed/oversize/image PDF rejection |
| Kaggle and public URL ingestion | 1,4 | existing Kaggle; extending | existing experiments; guided fetcher | Real public input and unsafe redirect/private destination tests |
| Schema, timestamps, entities, missingness, units | 1 | implementing | versioned guided contract | Confirmation and unordered/temporal boundary tests |
| Private datasets, jobs, results, feedback | 1 | implementing | lab-owned libSQL tables | Two-user negative access across all resources |
| Durable admission, retry, cancellation, recovery | 1,3 | existing admission; extending | AdmissionStore plus guided jobs | Independent clients, lost response, cancellation and restart tests |
| Actual Torch execution | 1,2 | legacy confirmed in code | canonical RLS reservoir plus causal adapter | Source/input/config hashes and real execution receipts |
| Causal forecasts and score before update | 2 | implementing | guided causal engine | Future perturbation, issue ledger, original units and rolling coverage |
| Baselines and ablations | 2 | pending | implementation-only evaluation | Frozen development/validation/final partitions, all outcomes retained |
| Continuous telemetry | 3 | pending | guided monitors and collector | Late/duplicate/gap/backpressure/restart/replay accounting |
| Semantic embedding and retrieval | 4 | pending | runner document adapter | Full passage influence, semantic similarity, owner-scoped vectors/cache |
| Grounded questions and incidents | 1,4 | pending | deterministic findings/retrieval | Every number/reference resolves; no generated code or instructions from data |
| Accessible guided UX, history, comparisons | all | pending | guided UI, existing research route | Desktop/mobile/keyboard/reduced-motion/screenshots |
| Preview, migration, rollback, PR | all | pending | release docs and scripts | Tested commit, preview evidence, additive schema, no production promotion |

## Decisions and boundaries

- Preserve historical engine files, imports, metrics, proof archives and sealed Grand Proof gates. Product engineering tests do not advance G0-G6.
- Use existing libSQL and shared admission infrastructure; isolate guided tables and namespace. No service purchases or new recurring scheduler.
- Existing Sandbox ceiling is one job, 4 vCPU/8 GB, 45 minutes, 25,000 rows. Initial uploads will be smaller to fit request and durable storage bounds; qualify measured limits.
- Original observations, evaluator labels and reviewer feedback remain distinct. Familiarity does not erase harmful events.
- Native Browser plugin is absent; repository browser tooling / Playwright will supply repeatable rendered verification.
- `EIDOS_PROOF_DRIVE_DIR=G:/My Drive` exists. Mirror only new sanitized implementation artifacts; leave historical Drive files intact.
- Drive historical audit read: `1DbuWi_KNHUDsdlPNttscOLPfnV8comGg`, dated 2026-09-06. Its then-missing live acceptance is historical context, not current deployment evidence.

## Checkpoint

2026-09-08 evening checkpoint: all four implementation paths exist. CSV, XLSX, Parquet, JSON, JSONL and numeric logs passed real browser upload/parse/causal analysis/download/source drill-down. Desktop/mobile/keyboard/reduced-motion, reload recovery, source queries and cross-user negative access passed. Document indexing ran with the real pinned encoder; remaining document/retrieval browser matrix is running. Scoped telemetry storage and Python checkpoint replay tests pass; live connector integration remains in progress.

Current local validation: 37 runner tests plus 14 subtests passed, 39 app TypeScript tests passed, lint/type check and production build passed. New causal invariants include fixed 24-row calibration, future-value perturbation and future-append invariance. Source/worker/app changes remain separate from legacy engine files. The legacy full-engine smoke reached completion but its console output failed under Windows cp1252; rerun with PYTHONIOENCODING=utf-8, preserving the failure log.

Independent development/validation/final synthetic periods were declared in `evaluation/partition-plan.json`. Acceptance was frozen after validation and before final generation. All four Eidos variants failed the final operational qualification gates (precision/false alerts; regulation also coverage). No model tuning after final consumption. Retain results and keep every mechanism experimental. No sealed research inputs opened and zero research gates advanced.

Upstream merged member PRs #48/#49 during this work. Current `origin/main` is `86dcaef`; the approved member table is now `eidos_email_members`. The new authorization adapter has been aligned; integrate upstream before publishing the candidate. Production config was snapshotted before preview-only setup. The first preview environment request failed because the feature branch was not yet pushed; no provider settings changed in that attempt.

Host disk pressure blocked extraction of the official OpenTelemetry 0.160.0 executable (download checksum verified). Only this task's transient files may be compacted/removed; preserve source, user work and all proof receipts. Connector validation, preview Sandbox execution, release PR and final Drive archive are still required. No completion claim or production promotion has been made.
