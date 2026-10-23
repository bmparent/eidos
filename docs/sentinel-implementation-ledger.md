# Sentinel implementation ledger

Brief: `Eidos_Sentinel_Full_Implementation_Prompt.md`, read completely on 2026-09-08.
Working branch: `codex/sentinel-guided-analysis-20260908`.
Isolated root: `C:/Users/bmpar/codex-worktrees/sentinel-guided-analysis-20260908`.
Baseline and production: `be02b6cba3579eb10412f7d148e22e62a48a87df`, Vercel deployment `dpl_5MF1AcvzUtUEzJ8fDKfbU6Vfdsc9`, READY.
Original checkout has extensive uncommitted/OneDrive residue and was not modified.
The separate Works member release was integrated through upstream PR #51. Those production changes belong to that task. PR #36 remains unrelated memory research.

## Execution sequence

1. Versioned data contract, safe ingestion, private durable records/jobs, guided UI and evidence findings.
2. Separately versioned causal Torch Eidos readout, original-unit forecasts, rolling evaluation and mechanism ablations.
3. Scoped telemetry credentials, collector/replay package, durable offsets/state and interruption recovery.
4. Semantic document/web representations, passage retrieval and grounded result questions.
5. Integration/security/regression tests, desktop/mobile browser receipts, preview, coherent commits and review PR.

## Requirement to evidence

Receipts below are in `artifacts/sentinel-guided-20260908/`. “Passed” applies only to the named check.

| Requirement | Milestone | Observed status | Receipt | Code |
|---|---|---|---|---|
| Guided CSV workflow, chronology, Torch, source, mobile, reload, two-user isolation | 1,2 | passed | `preview-final/browser-receipt.json` | `apps/sentinel-lab/components/guided-lab.tsx` |
| XLSX upload and original records | 1 | listed adapter checks passed; later failure retained and resolved in later receipt | `formats/receipt.json` | `services/sentinel-runner/sentinel_runner/guided/ingestion.py` |
| Parquet, JSON, JSONL, structured log upload | 1 | listed adapter checks passed; later failure retained and resolved in later receipt | `formats-verified/receipt.json` | `services/sentinel-runner/sentinel_runner/guided/ingestion.py` |
| Text, HTML, text PDF, plain log semantic retrieval | 4 | listed adapter checks passed; later failure retained and resolved in later receipt | `documents-verified/receipt.json` | `services/sentinel-runner/sentinel_runner/guided/semantic.py` |
| Real downloadable URL, web page, patterns, unsafe redirect rejection | 1,4 | passed | `urls-cold-server/receipt.json` | `apps/sentinel-lab/lib/guided/fetch-source.ts; patterns.ts` |
| Hosted semantic retrieval and public imports | 4 | semantic upload/retrieval passed; later URL confirmation blocked by provider HTTP 402 | `preview-formats/receipt.json` | `services/sentinel-runner/sentinel_runner/guided/semantic.py` |
| Causal invariance, labels, irregular/entity boundaries, ingestion failures | 1,2 | evidence_exists | `runner-final.xml` | `services/sentinel-runner/tests/test_guided.py` |
| Frozen final baselines and memory/multiscale/regulation/TraceSeal ablations | 2 | measured; operational qualification failed | `evaluation/qualification.json` | `services/sentinel-runner/scripts/guided_evaluate.py` |
| State-equivalent real stream replay, late/gap/backpressure/reset/revocation | 3 | passed | `stream/receipt.json` | `apps/sentinel-lab/lib/guided/telemetry.ts` |
| Hosted ingress, scoped keys and browser health; processing blocked by billing | 3 | ingress/security/billing-failure checks passed; hosted model processing blocked HTTP 402 | `preview-budget/receipt.json` | `services/sentinel-runner/sentinel_runner/guided/telemetry.py` |
| Historical reviewed outcomes with owner scope and immutable metrics | 1,4 | passed | `reviewed-history/receipt.json` | `apps/sentinel-lab/lib/guided/api.ts` |
| Official Collector forwarding and configuration validation | 3 | passed | `collector/receipt.json` | `connectors/sentinel/` |
| Cancellation, same-intent retry and subsequent real job | 1,3 | passed | `cancellation/receipt.json` | `apps/sentinel-lab/lib/guided/jobs.ts` |
| Actual legacy selection and semantic suffix/cache audit | 2,4 | passed | `semantic-final/receipt.json` | `services/sentinel-runner/scripts/guided_semantic_audit.py` |
| Legacy full engine engineering and mechanisms regression | 2 | evidence_exists | `legacy-standard.log; legacy-mechanisms.log` | `services/sentinel-runner/scripts/verify_full_engine.py` |
| Account contracts, concurrent stores, private resources, pinned public IPv4 | 1,3 | evidence_exists | `app-final-test.log` | `apps/sentinel-lab/tests/guided.test.ts` |

## Decisions and boundaries

- Preserve historical engine files, imports, metrics, proof archives and sealed Grand Proof gates. Product engineering tests do not advance G0-G6.
- Use existing libSQL and shared admission infrastructure; isolate guided tables and namespace. No service purchases or new recurring scheduler.
- Existing Sandbox ceiling is one job, 4 vCPU/8 GB, 45 minutes, 25,000 rows. Initial uploads will be smaller to fit request and durable storage bounds; qualify measured limits.
- Original observations, evaluator labels and reviewer feedback remain distinct. Familiarity does not erase harmful events.
- Native Browser plugin is absent; repository browser tooling / Playwright will supply repeatable rendered verification.
- `EIDOS_PROOF_DRIVE_DIR=G:/My Drive` exists. Mirror only new sanitized implementation artifacts; leave historical Drive files intact.
- Drive historical audit read: `1DbuWi_KNHUDsdlPNttscOLPfnV8comGg`, dated 2026-09-06. Its then-missing live acceptance is historical context, not current deployment evidence.

## Final checkpoint

2026-09-08: all four milestones implemented. PR #52 is a draft release candidate on `codex/sentinel-guided-analysis-20260908`; candidate source `24e00500431a654e167618fa6ca7543405865c45`. Preview: https://eidos-sentinel-913uebwxm-1brentbm-1876s-projects.vercel.app.

Local runner 37 tests and app 16 JavaScript plus 42 TypeScript tests passed; current CI verify/runner and Vercel build passed. All supported file adapters and URL/pattern workflows have real local browser/worker/source receipts. Official OpenTelemetry 0.160.0 forwarded three synthetic measurements; cancellation, label/forecast invariants, two-user ownership, genuine semantic suffix influence and real checkpoint replay passed. Actual hosted CSV/Torch and document/retrieval passed at `da14cc7`; the final candidate adds reviewed-history evidence and proper terminal handling for definitive provider allocation rejection.

External gate: Vercel returned HTTP 402 `payment_required` on new Sandbox creation. Full final-SHA hosted compute, hosted URL confirmation/analysis completion and hosted checkpointed telemetry processing remain blocked. Actual hosted ingress, OTLP, duplicate handling, source revocation, mobile health, reviewed history and explicit billing-failure behavior passed on the final candidate. No service purchase, paid-plan change, alternative-account bypass or production promotion occurred. Restore Sandbox creation through the owning project's Vercel usage/billing settings or provider quota reset, then rerun the documented hosted acceptance scripts.

All four Eidos variants failed the immutable final operational qualification; default precision .60 and false alerts/day 16.7033 exceed the accepted burden. No final tuning or sealed-data access. Legacy core behavior and proof gates remain unchanged; this engineering work advances zero research gates. `candidate-source-comparison.json` records which runtime files match the hosted compute-tested source.

Failures are preserved beside corrected receipts. Raw evaluation was losslessly compacted (420 files, per-file hashes); the official test Collector executable was removed after validation to recover host disk space. New artifact mirror status and checksums are recorded in `drive_manifest.json`. Historical Drive archives and the original dirty checkout were preserved.

Final Git evidence check: path-scoped byte preservation corrected automatic line-ending conversion; all 165 tracked review artifacts match their indexed Git blobs exactly. See `docs/proof_runs/2026-09-08/sentinel_guided_git_evidence_integrity.md` for the reproducible check and journal addendum. Runtime behavior is unchanged.
