# Playground test analysis - September 10

## What happened today
Implemented reviewable media, structural and optional AI changes using the existing Works platform. Frontend remains the shared source of truth. No research executor, scoring, model dynamics, shop checkout, mail service or existing assistant budget changed. No production release, live Playground sale or paid AI test was performed.

## Tests and commands run
From backend repo root: npm --prefix apps/sentinel-lab run lint; npm --prefix apps/sentinel-lab test (16 JavaScript + 35 TypeScript = 51 PASS); npm --prefix apps/sentinel-lab run build (PASS). Frontend: 24 Playground, 26 platform, 3 glass and 1 analytics tests PASS, build/typecheck/lint/Functions build PASS after recorded corrections. Local browser tests cover anonymous editing, media, versioned cards, actual ZIP exports and mocked AI. Real member identities, Stripe provider checkout, physical iPhone and authorized CMS verification remain blocked/not run separately.

## Proof Logic + Meaning
Goal reached: partial implementation/verification gate, with three additional reviewable frontend/backend pairs. Previous state had independent cloud identity and content history, uneven image limits, six fixed section IDs and no isolated Playground AI controls.

Technical logic: identity travels with undoable workspace frames; persistence acknowledgments track edit generations. Additive raster decoding and owner quota checks preserve historical data. Version-2 instance IDs distinguish block identity from type. AI admission uses the existing authenticated member and database adapters, but separate monetary and concurrency records. A single conditional insertion verifies account/global limits before dispatch. Unknown provider outcomes retain their reservation; retries replay known results or report pending.

Math: charged/reserved account sum + proposed reservation <= account budget, and the corresponding global sum <= global budget. Costs are integer micro-USD: ceil(sum(tokens by category * configured USD per million tokens)); reasoning is included in output, not double-counted. Text reservations use bounded UTF-8 input plus framing and capped output. Apply requires matching workspace identity and SHA-256 of the full base document. No research metric or readiness percentage is inferred.

Meaning: preserve identity before convenience; admit spending before dispatch; evidence before release. These changes strengthen reproducibility and human-readable receipts around the platform. They do not prove scientific quality or advance the Eidos streaming-codec research claims.

Evidence: source-vendor-parity.json, media-browser.json, structure/browser.json plus webkit-recheck.json, ai-browser.json, actual downloaded ZIP hashes and screenshots under C:/Users/bmpar/SystemDiagnostics/playground-20260910. Only controlled local fixture content was captured. All actual browser engine results remain distinct from physical Safari and authenticated production claims.

## Remaining uncertainty
Provider-generated Stripe delivery and immutable paid-download matching are not established by signed synthetic tests. Real AI evaluation is blocked pending an explicit total budget; the proposed USD 2 ceiling is not authorization. Long-running image delivery and unknown reservation reconciliation must be validated before image enablement. Existing standalone previews are not a verified integrated member/checkout pair. CMS/customer storefronts were not changed.

## Artifacts and Drive
Local receipts are preserved outside committed source; this journal and analysis are repo-local. A final manifest records the exact files and Drive copy outcome. No secrets or session state are included. The code is reviewable through stacked draft PRs; no merges or production promotion are authorized.
