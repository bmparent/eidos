# Sentinel guided implementation — 2026-09-08

Four connected milestones are implemented in PR https://github.com/bmparent/eidos/pull/52. Final application candidate: `24e00500431a654e167618fa6ca7543405865c45`. Preview: https://eidos-sentinel-913uebwxm-1brentbm-1876s-projects.vercel.app. Actual hosted CSV/Torch and semantic worker source: `da14cc77e988555fd7b607d0cb9b24ba5f5f8dc9`; the final candidate adds reviewed-history context and explicit provider-rejection handling, tested separately in preview. Final local stream replay, current CI/build and hosted noncompute checks are recorded separately. Evidence packaging revision: `24e00500431a654e167618fa6ca7543405865c45`.

The original dirty checkout was preserved. Work was isolated in `codex/sentinel-guided-analysis-20260908`. Existing Works member changes were integrated from upstream through PR #51; this task did not publish those account releases. Production started at `be02b6cba3579eb10412f7d148e22e62a48a87df`; another authorized task moved it to `233a1d8404d6b9be806cdf9c45a5ac861123e871`. This task creates preview deployments only.

## What happened today

Implemented versioned ingestion and confirmation, private durable jobs/results/feedback, actual causal Torch readouts, original-unit plots and issued forecasts, retrospective patterns, scoped OpenTelemetry/JSONL monitoring, and pinned semantic retrieval. Retained `/research`, the Kaggle connector, old engine observatory and historical evidence. Shared admission and current member tables were reused; four additive guided tables require no destructive migration.

## What was accomplished

Browser acceptance covers CSV, XLSX, Parquet, JSON, JSONL, numeric logs, plain logs, TXT, HTML, text PDF, real downloadable URLs and public pages. It includes malformed/oversize errors, source drill-down, mobile, keyboard/reduced-motion, reload, two owners and real worker receipts. Stream checks cover duplicate conflict rollback, late/gap policy, replay equivalence, interruption, backpressure, reset, revocation and an actual official Collector forwarding three synthetic points. Cancellation and subsequent capacity recovery passed. The immutable synthetic final evaluation was completed once; failed qualification remains visible.

## Tests and commands run

All local commands ran from repository root without manual PYTHONPATH edits. Focused runner suite: `.venv-guided/Scripts/python.exe -m pytest services/sentinel-runner/tests --junitxml=artifacts/sentinel-guided-20260908/runner-final.xml` — 37 passed. Existing global pytest plugin incompatibility required `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; this disables third-party discovery, not repo tests. `npm test --prefix apps/sentinel-lab` — 16 JavaScript plus 42 TypeScript tests passed. `npm run lint --prefix apps/sentinel-lab` and `npm run build --prefix apps/sentinel-lab` passed. `guided_fixtures.py`, `guided_semantic_audit.py`, both `verify_full_engine.py` profiles, `guided-browser-qa.mjs`, `guided-formats-qa.mjs`, `guided-stream-qa.mjs`, `guided-cancel-qa.mjs` and `guided-collector-qa.mjs` produced the receipts indexed below. See release guide for complete invocations/environment prerequisites. Build/CI results are source-specific, not a production-promotion receipt.

## Problems encountered

Preserved failures include the local worker's missing pyarrow, a document-query locator mismatch, public dual-stack DNS rejection, stale development-server modules, Windows cp1252 legacy console failure, initial OAuth workflow-scope push rejection, protected-preview login and disk pressure. Fixes were scoped: venv dependency availability, explicit Question label, validated/pinned IPv4 selection, owned dev-server restart, UTF-8 output, leaving the CI workflow unchanged and declaring parser base dependencies, authorized temporary preview access, and lossless verified compaction of this task's evaluation. The official Collector temporary executable was removed after successful validation to recover disk. No user archive was deleted, and earlier failed receipts were retained.

## What changed / What did not change

Code is under `apps/sentinel-lab/components/guided-*`, `lib/guided`, `app/api/lab/v1`, Python `sentinel_runner/guided`, focused tests/scripts and `connectors/sentinel`. The root experience moves to guided analysis; the existing console is retained at `/research`. Legacy reservoir/engine implementation files, historical algorithms/imports, frozen proof archives and Grand Proof gates were not changed. The new explicit causal adapter uses the existing reservoir/RLS class with separately recorded policy; it does not reinterpret legacy output as issued forecasts.

## Proof Logic + Meaning

### Goal reached
All four product milestones have working implementations and local integration receipts. Full hosted acceptance is partial: actual CSV/Torch and document retrieval passed, then Vercel blocked new workers with HTTP 402 payment_required. Hosted ingress, privacy, reviewed history and explicit failure handling passed on the final candidate. The remaining hosted URL confirmation/analysis and checkpointed telemetry processing require restored Sandbox capacity. Frozen operational alert-quality qualification failed. Production release approval is pending; this is a reviewable candidate with a documented external gate, not an operational detector qualification.

### Previous state
The main product was a research console and Kaggle workflow. General private uploads, named causal forecasts, owned monitors and document retrieval were absent. Legacy selection could choose the lower current residual after seeing the observation, and character prefixes did not represent document meaning.

### Technical logic utilized
M1: Byte checksums, explicit schema confirmation, original record references, owner-scoped libSQL records and idempotent jobs link the UI to actual isolated execution. M2: The actual canonical Torch RLS reservoir is called through a new named-feature adapter; a fixed 24-observation calibration prefix, historical-loss predictor selection, immutable issued forecasts and score-before-update preserve causal interpretation. Anomalies are skipped during adaptation by default. M3: Arrival/event timestamps, ingress deduplication, contiguous offsets, checkpoint compare-and-set and actual saved reservoir/RLS state preserve replay equivalence across batches. M4: Pinned MiniLM embeddings pool every token window; cosine retrieval returns original passages and content hashes, with no model-generated code or per-event LLM calls.

### Math / scoring logic
Reservoir: r_t = (1-alpha) r_(t-1) + alpha*tanh(W_in*x_t + W_rec*r_(t-1)). RLS: k=P*z/(lambda+z^T*P*z); W_out'=W_out+error*k^T. A forecast is committed before its target: prediction_t=f(state_before_t), residual_t=observation_t-prediction_t. Surprise uses the prior residual median and MAD scale; decision=score>=prior_threshold, then the recorded learning policy determines update. This robust score is not a probability. MAE=mean(abs(observed-issued)); coverage=count(target in issued band)/resolved intervals. Precision=TP_incidents/(TP_incidents+FP_incidents); recall=detected_truth_incidents/truth_incidents; false_alerts/day=FP_incidents/operating_seconds*86400. Matching horizon and maximum late window are explicit seconds. Pearson correlations are retrospective associations, with complete-pair counts; cosine similarity is association, not factual confidence. Unknown metrics stay null/NA.

### Philosophical meaning
Reproducibility is truth that can be revisited. Source drill-down is explanation before automation. Separating anomaly from consequence is restraint before alarm. Keeping failed qualification visible is honesty before optimization. Familiarity is recognition, never automatic permission to suppress a repeated harmful event.

### Why this is better
A user can now bring a supported source through confirmation to an actual saved result, inspect a named forecast and the records behind a finding, return after interruption, and revoke a source credential. The causal audit removes the retrospective-forecast ambiguity. The result is more inspectable; the measurements do not prove universal improvement or an acceptable operational alert burden.

### How this moves Eidos closer to the north-star goal
The north-star claim is: Eidos Brain is a self-monitoring streaming intelligence codec. It learns live streams, compresses predictable behavior, preserves meaningful anomalies, monitors its own internal state, and emits human-readable incident receipts. This release strengthens live-stream ingestion/learning, retained anomalies, visible internal state, source-linked incident explanation and reproducible execution. It does not establish new compression performance, broad domain generalization or value beyond all existing detectors/compressors.

### Evidence
The requirement-to-evidence table maps each milestone to browser, runner, provider, collector, replay, cancellation and semantic receipts. The immutable evaluation plan, acceptance freeze, final metrics, qualification decision, verified raw archive and frozen source archive retain the complete numerical result. Two-user access checks and real worker hashes support privacy/execution claims; hashes alone do not establish accuracy.

### Remaining uncertainty
All four Eidos variants failed operational precision/false-alert criteria; regulation also failed coverage. Synthetic periods do not establish natural-domain generalization. Vercel HTTP 402 payment_required blocks the remaining hosted compute matrix and a full final-SHA hosted CSV repeat; earlier actual Sandbox execution receipts identify their tested SHA separately. No sealed Grand Proof gate was opened or advanced. GPU, OCR/image PDFs, IPv6-only URLs, authenticated web imports, unrestricted stream rates, broad load testing and attributed provider cost are unqualified. No collector is installed for unrelated or live customer collection. Private sources require deliberate owner setup. Research readiness is unknown; no percentage is assigned.


## Artifacts generated

Repo-local: `artifacts/sentinel-guided-20260908/`. Reports, manifests, hashes, JUnit, CLI logs, screenshots, real source/result snapshots, issued forecasts, semantic vectors, monitor replay state, collector validation and all raw synthetic evaluation outputs are retained. The 420 evaluation files are losslessly stored in `evaluation/period-raw-artifacts.zip`, with individual hashes in `period-raw-manifest.json`. Exact frozen modules/generator are in `evaluation/frozen-code.zip`. Large generated ZIPs/fixtures are repo-local and Drive-mirrored; compact review evidence is committed. Credentials, database files and worker requests under `artifacts/sentinel-guided-private/` and `artifacts/guided-jobs/` are excluded.

## Google Drive archive status

See `drive_manifest.json` for the actual copy outcome and checksums. Configured root: `G:/My Drive`. Target: `Eidos_Brain_Proof_Phase/2026-09-08/sentinel-guided-20260908/`. Only new sanitized task artifacts are mirrored. Mounted-file verification proves the local Drive mirror, not independently observed completion of the provider's background cloud sync.

## Thoughts on improvement / Where to improve next

Keep the detector experimental. A separate development-only task should reduce false alerts while preserving recall, with a new untouched final partition and acceptance freeze. The external action is to restore Vercel Sandbox creation for this project through its usage/billing settings or the provider's quota reset; no plan or payment was changed. Then rerun the remaining hosted acceptance scripts. Before production promotion, an operator chooses the release and authorized collector target. No service purchase is needed to review the source, saved evidence or local workflow.

## Anything that stands out

Default final MAE was 0.9866 versus persistence 1.0326, but incident precision was 0.60 and false alerts/day 16.7033, above the frozen maximum of 1. Findings and empty findings both require interpretation. Local Python 3.11/Torch 2.6 and hosted Python 3.14/Torch 2.14 are recorded separately. This is no claim of identical cross-version numerical output.

## End-of-task summary

1. Files changed: guided app/runner/adapters/tests/connector and release documents; see PR diff.
2. Core behavior: legacy engine unchanged; explicitly separate causal policy added.
3. Tests: runner, app, full engine, browser, collector, semantic, cancellation and streaming acceptance; receipts retain failures/retries.
4. Commands: repo-root reproduction in `docs/sentinel-guided-release.md`.
5. Artifacts: local folder above, raw archive and compact review receipts.
6. Plain-language analysis: this report and benchmark summary.
7. Journal: separate Sentinel entry linked from the existing date journal; prior Works entry preserved.
8. Drive: exact state in drive_manifest.json; no secrets or historical archive changes.
9. Limits: 2 MB source, 5,000 rows, 16 numeric features, 8 entity/session groups, 256 passages, one 900-second 4-vCPU/8-GB worker; 4 processing batches/source/day, 1,000 buffered events, 10,000 ingress/day.
10. Follow-up: restore Sandbox capacity and finish hosted compute matrix; production approval, deliberate collector installation and operational detector improvement are not performed.
11. Proof Logic + Meaning: included above.
12. Math/logic: causal issue/score/update, RLS, MAE/coverage, incident precision/recall/FA, replay offsets and semantic similarity.
13. Philosophy: inspectable evidence, restraint and honesty.
14. Improvement: general sources now reach private, inspectable, actual computation and survive interrupted clients.
15. North-star: stream learning, anomaly retention, state visibility, explanation and reproducibility strengthened.
16. Evidence: requirement table, manifests, acceptance freeze, metrics and source hashes.
17. Unproven: operational alert qualification failed; research readiness, generalization, compression superiority, GPU and broad load remain unproven.

## Final comparisons

| Method | MAE | Incident precision | Incident recall | False alerts/day | Coverage |
|---|---:|---:|---:|---:|---:|
| eidos_adapt_all | 0.9831 | 0.6000 | 0.8000 | 18.4615 | 0.8767 |
| eidos_multiscale | 0.9723 | 0.6000 | 0.8000 | 17.5824 | 0.8602 |
| eidos_none | 0.9866 | 0.6000 | 0.8000 | 16.7033 | 0.8578 |
| eidos_regulation | 1.0072 | 0.5556 | 0.8000 | 18.4615 | 0.8443 |
| isolation_forest_prefix | NA | 0.2875 | 0.5333 | 38.6813 | NA |
| persistence | 1.0326 | 0.6000 | 0.8000 | 14.5055 | 0.8541 |
| robust_prefix | NA | 0.8333 | 1.0000 | 14.0659 | NA |
| seasonal_12 | 1.8446 | 0.5509 | 1.0000 | 24.6154 | 0.8132 |


## Archive completion
The complete 38,997,233-byte evidence ZIP (267 files, including the raw 420-file evaluation archive) was accepted by the authenticated Google Drive connector in the configured task folder. Provider metadata and authenticated raw-file response both report the same size. Local ZIP SHA-256 is `cb93d0930afa46aecdee1a84faf25a01bc8726e95ba649d2cca7d8fcb50bd044`; every archived entry was compared with local bytes. The connector did not expose an independent remote byte hash. The mounted-path copy failed with a storage error; that attempt is retained in mount-copy-attempt.json. Final upload status and updated reports are stored beside the ZIP. See [Drive folder](https://drive.google.com/drive/folders/16ObF_8iSYgJsKPQPrRnBh20hK68F1fuC). No historical archive was deleted or reorganized.
