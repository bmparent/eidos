# Playground reliability checkpoint — September 9, 2026

## What happened today
Vendored authoritative frontend handlers with the existing exporter. Added expected-owner rejection and a shared UTF-8 document limit. No research engine or existing shop/assistant behavior changed. No migration or production promotion occurred.

## Tests and commands run
From apps/sentinel-lab: `npm run lint`, `npm test` (16 JavaScript and 32 TypeScript tests), `npm run build`: PASS. Equivalent repo-root commands: `npm --prefix apps/sentinel-lab run lint`, `npm --prefix apps/sentinel-lab test`, `npm --prefix apps/sentinel-lab run build`. No research benchmark was run. Shared source/vendor parity: 48 files match with UTF-8 LF normalization.

## Proof Logic + Meaning
Goal reached: partial application reliability gate, not a research proof gate. Previous state: account identity was separate from document history and local media success did not warn about the cloud document size limit. Technical logic: the frontend now stores document identity and target together; the server checks the expected member before saving and uses the same byte preflight as the client. Math: serialized UTF-8 bytes <= 2,000,000; existing writes require expected head = stored head. A mismatched owner is rejected. These are deterministic equality/boundary checks, not statistical metrics. Philosophical meaning: reproducibility is truth that can be revisited; clear rejection is preferable to silently writing the wrong document. Improvement: regression evidence and shared limits make save failures more explainable. North-star connection: strengthens reproducible application operation around Eidos; it does not prove live-stream learning, compression, anomaly detection, or detector superiority. Evidence: frontend tests and docs/playground/handoff-20260909/preview-receipt.json; backend tests and deployment dpl_9PmFJGfq7hF8USFyD7nMqhN1EvR8 at f1e1934d840ea19ec14d9c13d62b966d44f57d0c. Remaining uncertainty: browser behavior, real authenticated relay, remote Stripe delivery, device/CMS compatibility and media/structure/AI changes remain unverified or unimplemented. No proof percentage is assigned.

## Problems and limitations
C: and G: report zero free bytes. Browser binaries could not install (ENOSPC), installed Edge could not launch, and the existing-browser runtime failed to write kernel assets (Windows disk error 112). Automatic approval review rejected cleanup of this task's disposable Next cache; no cleanup occurred. Member test identities/sign-in and dedicated Playground Stripe TEST configuration remain external gates. The new Pages preview still targets an older validation backend, so it is not an integrated verified pair.

## Artifacts and archive
Local: artifacts/playground-reliability-20260909. Drive: configured G:/My Drive reports no space; skipped, recorded in drive_manifest.json. Source changes: three vendored files only. No core behavior change, no research benchmark, no progress score update. Frontend PR #28 and backend PR #54 are drafts. Existing production remains unchanged.

## Next work
Free several GB, resume browser verification and authenticated test setup, pair isolated preview routing, finish media/brand, structural customization and optional AI as independent review changes. No provider spend is authorized.
