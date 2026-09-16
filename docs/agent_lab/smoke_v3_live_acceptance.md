# Agent Lab Smoke v3 — live acceptance

Date: 2026-09-16

## Pre-live acceptance

- Deterministic Agent Lab suite: 74 passed; 1 paid live-only test intentionally deselected.
- Mocked ordered-routing mission: PASS.
- Required chain: Archivist -> Sentry -> Curie -> Director.
- Host-owned SpecialistAccounting, CostReceipt, RunManifest, and workflow state cannot be requested as specialist outputs.
- Routed work-order schemas pin Archivist, Sentry, and Curie to their exact SDK output contracts.
- Required specialist failures/skips cannot satisfy READY_FOR_HUMAN.
- Live failure persistence writes forensic events, specialist accounting, cost receipt, run failure, and run manifest.
- Per-session ceiling: $0.85; aggregate daily ceiling: $2.00; Council disabled.
- Daily ledger is restored/saved across live GitHub runners for the UTC budget day.

## Prior live finding

The first credentialed live attempt correctly completed Archivist, blocked after Sentry failed structured output twice, skipped Curie, and refused READY_FOR_HUMAN. The retained evidence identified a contract mismatch: Director requested host-owned artifacts from Sentry although Sentry's SDK result contract did not contain those artifacts. The routed contract fix now makes that request invalid at the tool-schema boundary and gives Sentry a compact, explicit SDKSentryResult.

## Final acceptance condition

Accept Smoke v3 only if the credentialed live run persists successful Archivist, Sentry, and Curie results in order; persists a valid EvidencePacket and ExperimentSpec; satisfies the deterministic research gate; reaches READY_FOR_HUMAN; retains cost and run evidence; and remains within all budget limits.
