# Works account and Playground release integration — October 19, 2026

This candidate combines recovered account backend #85 (`bc47de2c6eabe90c204a12643b88161a98a801d4`) with Playground readiness #83 (`f16bd3a94ae8a3a49065baac4c4d7ccfd5da1be7`). It preserves current main, password/Google identity handling, owner authorization, existing projects, receipts, archive pins and the original held previews.

The shared runtime is exported from the combined frontend (account `cd2068e93b2475ea1588be43d56f07ff5da0414a` plus tutorial/purchase fix `870c877fc6cffb5a7ea252cf7489ca2b0c48bcf0`). All 107 shared files match. The isolated delivery/proposal graph has 18 files and inventory SHA-256 `a173de046f9184dce51821b8c12682496196e1cca341f75f037fc8b2ec97646d`; both redirect adapters are checked by that graph's independent verifier.

The prior AI schema remains byte-for-byte in migration 0005. New scoped-AI accounting is additive migration 0009. The former legacy implementation has an explicit Legacy name to avoid a case-only filename collision. Its historical tests remain in place. Both complete migration passes and a retained unknown AI request are tested on the frontend. The combined webhook keeps legacy kit refunds and receipt downloads usable before template tables exist; paid template events still fail closed if their order is absent.

Local evidence: TypeScript passes; 20 JavaScript and 178 TypeScript backend tests pass; Next production build passes. General parity and the independent graph pass. These are source/local results, not hosted account/provider acceptance.

Automatic Vercel deployment of this release branch is disabled, following the existing held Playground branch guards. No credentials, environment bindings, database records, feature flags or production aliases are changed. Paid AI and checkout remain subject to their existing gates.

Before production: require exact-candidate CI, isolated paired deployment and migration verification, actual Google/password/reset delivery, two-account ownership and saved-project reopen, owner access, and the intended Stripe TEST webhook/recovery/refund lifecycle. The connected Stripe context currently exposes LIVE only; no LIVE payment was substituted. Preserve the public main baseline `34c3d21509dfa1017705244e0d42a6eb7bc7dff9` and the original immutable account/readiness previews as rollback identities; never undo additive schema or delete member/order/project rows.
