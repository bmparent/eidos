# Plain-Language Test Analysis — Eidos Agent Lab v1

## What the task attempted

The task built a controlled research organization around Eidos. One Director can ask specialists for
evidence, experiment design, Sentinel analysis, mathematics, implementation, measurement and audit,
but code prevents those agents from acquiring authority they should not have.

## Why the tests matter

A multi-agent system can sound careful while still letting a builder approve itself, skipping failed
tests, hiding recall loss or spending money without a visible limit. These tests target those concrete
failure modes rather than checking prompts alone.

## What was tested and passed

- 46 no-cost agent-lab tests passed; the one paid live smoke stayed opt-in and skipped.
- Schemas reject invalid statuses and unsupported `KNOWN` claims.
- Router fixtures choose Archivist, Curie, Sentry, Gauss, Forge, Bench, Auditor or the gated Council.
- Archivist/Auditor/Bench/Council cannot write source; Forge cannot merge or deploy.
- Main writes, destructive git commands, secret-like text and out-of-scope files are blocked.
- Implementation cannot skip Bench; Forge cannot audit itself; Auditor BLOCK cannot promote.
- Human approval survives process/store reconstruction; SDK run state has a durable resume adapter.
- Specialist fan-out is bounded and scientific failure is not blindly retried.
- Mocked full workflow generated benchmark, audit, manifest, cost and decision receipts.
- Sentinel dry-run generated evidence/hypothesis/experiment receipts without modifying engine source.
- The current Agents SDK graph and all requested models were available under Python 3.12.14.
- Existing Eidos, packaged-engine and Sentinel-runner suites passed.

## What failed

Five of fourteen focused root RNG-null tests fail before agent-lab code is involved. The existing
`eidos/sentinel/event_merge.py` imports `operator_explanation` as a root module, while that file is
located inside `eidos/`. The standard Eidos suite itself passed 140/1 skipped. This unrelated defect
was not silently fixed in this infrastructure change.

## Artifacts generated

Mocked task receipts include TaskSpec, EvidencePacket, Hypothesis, ExperimentSpec, ImplementationSpec,
BenchmarkReceipt, AuditResult, CostReceipt, transitions, structured events, RunManifest and final
JSON/Markdown decisions.

## Local and Drive storage

Local receipts are under `artifacts/agent_lab/`. The build-validation bundle is mirrored to
`G:\My Drive\Eidos_Brain_Proof_Phase\2026-09-15\eidos_agent_lab_v1_20260915\` with a file/hash manifest.

## What remains uncertain and what should happen next

No paid agent inference ran, so live response quality and remote trace appearance remain untested.
Current Sentinel performance was not measured; the mission is an experiment specification, not a
result. Brent should review the code and receipts, choose a dollar budget, then explicitly authorize a
tiny Archivist smoke before any implementation mission.

## Proof Logic + Meaning

### Goal reached

The local architecture gate passed. The scientific Sentinel gate remains `missing` because no new
frozen experiment ran.

### Previous state

Roles and proof practices existed as project conventions, but no typed, persistent orchestration layer
enforced them end to end.

### Technical logic utilized

The workflow is a directed state graph. Capabilities are allowlists. File changes are checked against
allowed and forbidden patterns. Approvals and transitions are stored. Builder, measurer and auditor
are separate identities. The Director owns the conversation by calling specialists as tools.

### Math / scoring logic

Sentinel comparisons require precision, recall, F1, false positives per 10k benign frames, coverage
and latency together. Cost estimates sum token classes against a versioned price registry; unknown
prices yield unknown cost rather than a false zero.

### Philosophical meaning

The milestone represents restraint before alarm and evidence before consensus.

### Why this is better

Important constraints can now fail tests. A persuasive agent answer cannot make a prohibited state
transition or grant itself a merge, deployment or audit capability.

### How this moves Eidos closer to the north-star goal

It makes future claims about learning streams, compression, anomaly preservation, self-monitoring and
incident explanation easier to reproduce and independently review.

### Evidence

The agent-lab suite, existing regression suites, mocked receipt bundles, doctor output and dry-run
mission are the evidence.

### Remaining uncertainty

Scientific and production value are still unproven; GPU/device/provider and paid live orchestration
were not tested.

---

## Live smoke addendum

The first paid Director call cost an estimated `$0.017534` against a `$1.00` cap. It created a remote trace and
durable token/cost/task receipts without changing Sentinel source. Those infrastructure checks passed.

The intended research organization did not pass. Director skipped Archivist, Sentry and Curie, retrieved no
current project evidence, and returned an evidence-free `INCONCLUSIVE` answer. The recommendation for a temporal
holdout calibration experiment may be reasonable, but this run did not establish that it is the repository's
highest-value missing experiment.

The next change should be a deterministic gate requiring a persisted Archivist EvidencePacket before a research
task can leave evidence gathering. This embodies evidence before consensus and makes Eidos more reproducible and
self-monitoring without changing any core detection behavior. Full receipts and limitations are in
`docs/agent_lab/LIVE_SMOKE_REPORT_2026-09-15.md`.

## Routing smoke v2 addendum

The workflow repair succeeded at its core safety purpose: it stopped Director's answer from reaching human review
without Archivist evidence, Sentry analysis and Curie's structured experiment. Sixty-one Agent Lab tests passed.

The paid live organization still failed. Director attempted all required tools, but an open-ended JSON field in the
specialist output schema was incompatible with the SDK's strict structured-output rules. No specialist model ran,
no repository evidence was read, and no ExperimentSpec was produced. Five Director turns cost an estimated
`$0.0396056` against the `$0.75` cap. Source code and Sentinel behavior stayed unchanged.

This is better than the first smoke because the system failed closed and named the missing requirements. It is not
ready for the full Sentinel precision mission. The exact receipts and next repair are documented in
`docs/agent_lab/LIVE_SMOKE_REPORT_V2_2026-09-15.md`.
