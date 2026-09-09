# Implementation acceptance evidence

All paths below are relative to this artifact folder. Failed attempts remain alongside their corrected acceptance receipts.

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
