# Works accounts: plain-language test analysis

The tests show that separate local accounts keep separate cloud projects, password handlers work, and an old response cannot replace a design after another account signs in. A real local SQLite fixture and real password hashing were used. Delays were deliberately inserted around real local API responses. This is not hosted provider acceptance.

Backend tests passed: 16 JavaScript plus 48 TypeScript. Frontend validation includes 52 Playground cases including the account-operation allowlist test. Frontend production build and backend typecheck passed; backend webpack production build passed. Default Turbopack failed because the existing dependency junction is outside its filesystem root. Exact command to retry with a normal dependency layout: `npm --prefix apps/sentinel-lab run build` (expected production build success). No dependencies were reinstalled to mask that environment limitation.

Commands run from repository roots: frontend `npm run build`, `npm run test:playground`, `node --import tsx --test scripts/test-account-epoch.ts`; backend `npm --prefix apps/sentinel-lab run test -- --test-name-pattern=database`, `npm --prefix apps/sentinel-lab run lint`, `npm --prefix apps/sentinel-lab run build -- --webpack`. The test script still ran its full 16/48 suite; the appended argument did not select a smaller run.

Raw original failures remain under the external evidence directory. This package contains accepted results with source/vendor hashes. No remote migration, provider resource creation or production account release occurred. Next: verify existing authorized provider bindings and complete actual hosted two-account flows before rollout.

## Proof Logic + Meaning

### Goal reached
Partial: reproducible Works account reconciliation passes controlled owner-isolation, password/session, delayed-response and database-binding checks. Hosted acceptance remains blocked.

### Previous state
The held account branches could not safely be applied over the newer editor. Advanced document storage and delayed responses needed explicit acceptance boundaries.

### Technical logic utilized
Reused existing identity and session handlers. Owner equality and expected-head compare-and-swap protect cloud writes; transactional quotas limit admission. An account-change nonce invalidates pending UI work across tabs, while the server remains the authorization authority. Strict authoring schemas and real image decoding protect stored assets. Exported portable sources are byte-compared with their vendor copies.

### Math / scoring logic
Authorized save = authenticated owner equals stored owner AND expected owner equals session owner AND expected revision equals current head. Two concurrent writes against one head must produce one success and one conflict (200,409). No research readiness percentage is inferred from these software tests.

### Philosophical meaning
Reproducibility is truth that can be revisited; owner isolation makes receipts trustworthy.

### Why this is better
Before: incompatible held branches and possible stale account responses. After: separate reviewable stack, explicit default-off advanced cloud writes, local state preserved across actual account switches, and reproducible validation receipts.

### How this moves Eidos closer to the north-star goal
This improves the reliability of the Works hosting surface and evidence trail. It does not establish learning, compression, anomaly preservation or detector performance for the Eidos Brain north-star claim.

### Evidence
See run_manifest.json, source_vendor_parity.json, accounts-controlled-accepted.json, accounts-switch-final.json and the captured test/build logs in artifacts/works_accounts_20260912/.

### Remaining uncertainty
Hosted two-account recovery and email delivery, existing Google consent, Stripe TEST and physical-device checks remain unproven. Research/model/core behavior was not changed.
