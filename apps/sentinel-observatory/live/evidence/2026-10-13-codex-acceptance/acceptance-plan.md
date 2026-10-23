# Sentinel Observatory continuation acceptance

Task date: 2026-10-13 (America/New_York).

Scope: finish the existing owner-private Observatory and draft PR #73. Preserve real retained evidence, numerical Sentinel behavior, the recorded capture, database migrations, collector configuration, and current audience.

## Required gates

| Gate | Required evidence | Initial state |
| --- | --- | --- |
| Existing state preserved | Fetched PR head, Sites source/version/deployment, environment key names, owner-only policy | Inspected; PR head 00d49ff2e8ef74931503fb7a27046857228d0b04 |
| Credential activated | Dedicated key from supported OpenAI workflow; only server secret OPENAI_API_KEY; saved source redeployed at new environment revision | Pending secure organization/project picker selection; no key created |
| Actual Sol accepted | UI-initiated completed provider receipt, exact model, response ID, usage, model_ran and output_accepted, authoritative input IDs | Pending credential |
| Cache accepted | Repeat UI interpretation, same provider receipt, one durable reservation | Pending first actual response |
| Safe rejection and failures | Hosted negative checks where possible; isolated provider-failure, quota and concurrency regressions without credential mutation | Pending rerun |
| Rendered desktop/mobile accepted | Authenticated owner browser; screenshots and observed interaction states for connection, history, graph, filters, interpretation, freshness, citations, export and keyboard | In-app owner sign-in encountered upstream content-type error and security verification; recovery pending |
| Documentation and receipts | Sanitized evidence, updated audit/PR, exact source/deployment/environment, journal, plain-language analysis, Drive status | In progress |

## Execution order

1. Inspect current source, runtime configuration and access before edits.
2. Run provider-free regression and parity checks; exercise rendered flows that are available.
3. Finish secure picker selection and bind only the authorized runtime secret.
4. Redeploy the exact saved source and verify actual manual inference and cache.
5. Fix observed defects narrowly and repeat affected checks.
6. Update acceptance, proof records and the existing draft PR. Keep unproven gates visible.

## Boundaries

No automatic paid calls, model substitution, example incidents, production promotion of unrelated work, new integrations, threshold changes, or security-policy relaxation. Functional checks are not confirmed attacks. Configuration, fixtures, API checks, rendered service access, and owner-browser acceptance are distinct evidence levels.

## Completed disposition

All six finite private-app acceptance gates above passed. The secure picker did not surface; the owner explicitly selected Personal / eidos_brain from connector-discovered targets. A dedicated key was bound only as the runtime secret. Sites source bb2bc77c7d9eaef67d6ff4833781e3c8517771d8 is deployed at environment revision 5.

The actual UI Sol request completed and repeats used its cache. A separate intentional provider attempt verified live duplicate 409 and cooldown 429 responses. Authenticated desktop/mobile verification exposed and rechecked a sensor-control focus defect and mobile word spacing. Initial failures remain in the receipts; the one Windows symlink-permission skip remains explicit. See acceptance_report.md, run_manifest.json and drive_manifest.json for the exact evidence, archive disposition and scientific limits.
