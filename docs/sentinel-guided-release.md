# Sentinel guided analysis release candidate

This is an additive product engineering release. The guided workflow is `/`; the existing Kaggle workflow, engine observatory, historical receipts and research gates remain at `/research`. Production promotion requires the final release decision. No sealed Grand Proof data is used.

## Architecture and supported envelope

`/api/lab/v1/*` authenticates each operation, stores private input bytes and hashes in the existing libSQL service, and records a stable job before allocation. Jobs share the existing Sandbox admission slot. A pinned Git revision runs the actual canonical Torch `RLS_Reservoir` through a separately versioned causal adapter. Documents use a pinned Sentence Transformer encoder; unordered tables use explicitly identified Isolation Forest and median/MAD baselines. Those methods never masquerade as temporal Eidos runs.

The initial envelope is 2,000,000 source bytes; 5,000 rows; 64 source columns; 16 selected numeric measurements; eight entity/session groups; 256 passages; 50 PDF pages; 16 MB expanded workbook/Parquet payload and result size; 30 datasets and 120 jobs per owner; one shared worker; 4 vCPU / 8 GB provider allocation; a 900-second worker deadline. Parser limits are enforced before analysis. PDF extraction additionally runs inside the memory/time-bounded process; an image-only document is rejected. Public fetches have a 20-second limit, four redirects and address pinning. HTTP compression, authenticated URLs, private addresses and IPv6-only destinations are outside this pilot importer. No per-row LLM calls, model-generated code, paid plan changes or recurring scheduler.

Supported: CSV, values-only single-sheet XLSX, Parquet, flat JSON arrays, JSONL/NDJSON, structured or plain-text logs, text PDFs, HTML, TXT/Markdown, public downloadable dataset URLs and pages. Original bytes and source references are retained. Numeric log attributes are analyzed as measurements; logs without varying numeric measurements use semantic templates with original message references. Categories are profiled and preserved but are excluded from the numerical temporal readout. Unselected fields, candidate labels, identifiers and constants are visible in confirmation. Missing measurements are excluded or imputed from the fixed calibration prefix; arbitrary file order is never chronology.

## Walkthrough

1. Sign in with an approved Lab key, or an existing authorized Eidos member credential. Upload a file, import a public URL, or open the preserved Kaggle workflow.
2. Confirm what a row represents, reference group, feature names/units, missing-value policy, timestamp/time zone and entity/session keys. Ambiguous/duplicate entity timestamps fail explicitly. Without a confirmed timestamp the analysis remains unordered.
3. Choose unusual behavior, a named measurement forecast, or patterns. Temporal forecasts require at least 48 records per entity/session and specify horizon and late-match window in seconds. Advanced explains the method and limits.
4. Inspect actual/expected values, bands, grouped findings, original records, grounded questions and provenance. Download source/results. Feedback records a reviewed outcome without changing frozen labels, scores or training.
5. Saved work restores results and durable jobs after reload. Resume retries the same intent. Cancel fences publication before stopping compute. Monitors show arrival health, buffered offsets, warmup, late observations, gaps and resets.

## Causal policy and limitations

The first 24 observations per entity/session are a fixed calibration prefix. No forecast is issued before that prefix is known. Named original features are scaled directly; there is no lossy projection to invert. At each issuance the adapter commits the named prediction, event issuance time, target time, horizon, predictor choice, historical loss, threshold, baseline parameters, normalization hash, state hash and a chained commitment. Its JSONL audit is flushed before the next observation is read. A matching observation is scored against that committed state; only then may learning and baseline updates occur. Overlapping issues are accounted as matched, superseded, expired or awaiting a future target. Replay issuance times are historical event times; this does not claim wall-clock predictions were made in the past.

`prediction = f(state_before_target)`; `residual = observed - committed_prediction`; `score = max(0, (norm(residual / feature_scale) - prior_median) / prior_MAD_scale)`; `alert = score >= 5`. Predictor selection uses previous resolved losses only. A robust score is neither a probability nor Gaussian significance. Accepted-past absolute errors supply an empirical 90% band; drift, contamination and skipped anomalies can reduce coverage. Familiarity does not suppress a harmful event. Consequence remains unknown without contextual rules or review. Empty findings do not establish safety.

Legacy `embed_line_to_vec`, retrospective `best_pred`, error-baseline update order and CICIDS 64-feature bridge remain reproducible in their existing version. This release changes interpretation through the new explicit adapter; it does not rewrite old engine files, imports, metrics or receipts. The synthetic product evaluation reports losses as well as gains. No operational alert guarantee or general superiority follows from passing workflow tests.

## Storage, authentication and recovery

Four additive tables (`sentinel_guided_objects`, `sentinel_guided_jobs`, `sentinel_guided_keys`, `sentinel_guided_events`) plus an owner index are created idempotently on first use. Scope and owner constraints apply to sources, previews, vectors, runs, feedback, monitors and downloads. Existing member/session/key tables are reused without creating accounts or sending email. Pilot sessions are eight-hour HttpOnly, SameSite=Strict cookies and revalidate grant revocation; member credentials remain backed by the existing revocable member service. Ingest keys are hashed, source-scoped, revocable and expire after 30 days. Keys never enter the worker request or source evidence.

Dataset/result access expires after 30 days. This release does not delete retained archives. Export before expiration; an operator retention/deletion service is outside this change. A stable retry key owns one immutable request hash and one job identity. A database compare-and-set lease and the existing shared admission prevent duplicate allocation. The deterministic provider name is saved before creation, allowing reconciliation after a lost response. An uncertain provider continues to occupy capacity until its stop/absence is confirmed. Completed results survive ephemeral compute. Cancellation markers stop local workers without trusting reusable operating-system PIDs. A cancelled job cannot publish success.

Workers persist heartbeat, immutable issuance ledger, result checksum, state checkpoint and terminal status. A scoped callback publishes terminal output when available; browser/API polling can recover it from the provider snapshot if the callback is blocked. Parser or analysis process failure is a failed job, never a simulator fallback. Retrying a failed analysis creates a new intent from the saved input; completed telemetry offsets resume from the committed checkpoint. Uncommitted batch computation can be rerun, but checkpoint compare-and-set and ingress deduplication prevent counting accepted events twice.

## Configuration and local reproduction

Hosted guided analysis uses existing `EIDOS_DATABASE_URL`, `EIDOS_DATABASE_AUTH_TOKEN`, approved `EIDOS_TEST_ACCESS_GRANTS` / member credentials, Vercel Sandbox identity and `VERCEL_GIT_COMMIT_SHA`. `EIDOS_GUIDED_SCOPE` isolates a branch preview's new objects. Keep production settings intact. Do not copy production data into QA. Explicit provider token/team/project variables are optional for an already configured local operator context; hosted Sandbox normally uses Vercel OIDC. The new workflow does not need Kaggle credentials for uploads/documents/telemetry; the preserved Kaggle workflow retains its existing credential requirements.

From the repository root (Python 3.14 is the hosted/CI runtime):

```text
python -m venv .venv-guided
python -m pip install uv
uv pip install --python .venv-guided/bin/python --torch-backend cpu -e './services/sentinel-runner[guided]'
npm ci --prefix apps/sentinel-lab --no-audit --no-fund
node apps/sentinel-lab/scripts/guided-local.mjs
node apps/sentinel-lab/node_modules/next/dist/bin/next dev apps/sentinel-lab --hostname 127.0.0.1 --port 3210
```

On Windows the venv interpreter is `.venv-guided/Scripts/python.exe`; substitute it for `.venv-guided/bin/python`. The local setup refuses to overwrite an existing app `.env.local`. It saves two private QA keys under ignored `artifacts/sentinel-guided-private/`; never commit or mirror that folder. `EIDOS_GUIDED_LOCAL=1` cannot activate the local subprocess backend on Vercel.

```text
.venv-guided/Scripts/python.exe services/sentinel-runner/scripts/guided_fixtures.py
.venv-guided/Scripts/python.exe -m unittest discover -s services/sentinel-runner/tests -v
.venv-guided/Scripts/python.exe services/sentinel-runner/scripts/verify_full_engine.py --profile cpu_engineering --output artifacts/sentinel-guided-20260908/legacy-standard
.venv-guided/Scripts/python.exe services/sentinel-runner/scripts/verify_full_engine.py --profile cpu_mechanisms --output artifacts/sentinel-guided-20260908/legacy-mechanisms
npm test --prefix apps/sentinel-lab
npm run lint --prefix apps/sentinel-lab
npm run build --prefix apps/sentinel-lab
node apps/sentinel-lab/scripts/guided-browser-qa.mjs
node apps/sentinel-lab/scripts/guided-formats-qa.mjs
```

Browser scripts use installed Playwright or `PLAYWRIGHT_MODULE` and optional `CHROME_BIN`; `EIDOS_QA_URL` selects an authenticated preview. Scripts exercise real API/runner execution, not mocked successful responses. The evaluation uses `guided_evaluate.py plan`, `development`, `freeze`, `final` in order; plan/freeze are write-once and final refuses repeated consumption. Preserve receipts when changing a later protocol. Initial Windows validation used Python 3.11/Torch 2.6 CPU; hosted qualification must identify its actual Python/Torch separately.

## Connector and rollback

See `connectors/sentinel/README.md` for the scoped OpenTelemetry/JSONL package. It monitors only a source deliberately configured by its owner. No collector is installed as part of this release. Public page import does not grant process visibility.

Deploy the feature branch as a preview, verify the exact Git SHA plus real source → worker → durable result path, and inspect the PR before production authorization. Compare provider production configuration fingerprints before/after preview setup. After authorization, the rollout is an ordinary forward deployment; migrations are additive. Rollback promotes the previously approved production deployment, with no DROP/TRUNCATE, archive deletion or reversal of unrelated Works account changes. Existing guided data remains private in its namespaced tables. A schema rollback is unnecessary and would discard evidence.

## Proof Logic + Meaning

Milestone 1 provides an inspectable upload-to-findings implementation; acceptance depends on actual browser/worker receipts. Previously the main experience was a research console with no general upload workflow. Byte hashes, owner constraints, immutable requests and explicit methods now connect each finding to reproducible input. This strengthens evidence preservation and repeatable execution, not research-gate readiness.

Milestone 2 issues forecasts before observations, with named units and bounded empirical uncertainty. Previously a minimum residual could be chosen after seeing the observation. Historical loss selection and score-before-update remove that interpretation ambiguity. MAE, coverage and incident metrics measure usefulness separately: `precision=TP_incidents/(TP_incidents+FP_incidents)`, `recall=detected_truth_incidents/truth_incidents`, `false_alerts/day=FP_incidents/operating_seconds*86400`. False positives remain visible. Mechanisms remain experimental if frozen acceptance fails. This represents honesty before optimization.

Milestone 3 records arrival/event times, deduplication keys, contiguous offsets and model snapshots. Previously this general scoped ingestion/monitor path was absent. Replay-equivalent state transitions and checkpoint compare-and-set make interruption auditable. This strengthens live-stream learning and self-monitoring within a declared budget. It does not qualify unlimited continuous throughput or unattended production collection.

Milestone 4 uses pinned semantic embeddings across all token windows, owner-contained content caching and exact source-linked retrieval. Previously character prefixes were not adequate semantic evidence. Similarity supports finding related passages; it does not verify factual claims or establish causation. This strengthens human-readable explanations and the principle that another person should be able to inspect and challenge a conclusion.

Evidence is indexed in `docs/sentinel-implementation-ledger.md` and `artifacts/sentinel-guided-20260908/`. The final journal and plain-language analysis record observed status, failed comparisons, runtime differences, Drive archive status, exact tested source and any external limitation. Overall research readiness remains unknown; no gate or percentage is invented.
