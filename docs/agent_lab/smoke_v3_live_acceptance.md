# Agent Lab Smoke v3 — live acceptance

Date: 2026-09-16

## Pre-live acceptance

- Deterministic Agent Lab suite and mocked ordered-routing mission are green after the live-derived regressions.
- Required chain: Archivist -> Sentry -> Curie -> Director.
- Host-owned SpecialistAccounting, CostReceipt, RunManifest, and workflow state cannot be requested as specialist outputs.
- Routed work-order schemas pin Archivist, Sentry, and Curie to their exact SDK output contracts.
- Required specialist failures/skips cannot satisfy READY_FOR_HUMAN.
- Live failure persistence writes forensic events, specialist accounting, cost receipt, run failure, and run manifest.
- Per-session ceiling: $0.85; aggregate daily ceiling: $2.00; Council disabled.
- Daily ledger is restored/saved across live GitHub runners for the UTC budget day.

## Live-derived fixes

1. The first credentialed live attempt completed Archivist but Sentry failed structured output because Director requested host-owned artifacts outside Sentry's output contract. Routed work-order schemas now make that combined request invalid and Sentry has an explicit compact SDKSentryResult.
2. The next live attempt proved Sentry's structured result parsed successfully, but the host rejected its provenance because Director forwarded persisted evidence ids as namespaced refs such as `Archivist:E1` while the verifier requires canonical `E1`. SDKSentryWorkOrder now canonicalizes `Archivist:E1` and `Archivist:E1=path` to the persisted evidence id before Sentry sees them. Regression tests reproduce this exact condition while retaining the strict packet provenance gate.

## Final acceptance condition

Accept Smoke v3 only if the credentialed live run persists successful Archivist, Sentry, and Curie results in order; persists a valid EvidencePacket and ExperimentSpec; satisfies the deterministic research gate; reaches READY_FOR_HUMAN; retains cost and run evidence; and remains within all budget limits.
