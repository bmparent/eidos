# Account recovery — October 19, 2026

This isolated recovery merges held account backend `c18b580358dcfeb0b70ba7a045c8f506513f452f` with backend main `34c3d21509dfa1017705244e0d42a6eb7bc7dff9`. It pairs with the frontend `repair/account-acceptance-20261019` candidate, whose parents are held site `cc2808a62a6fe43e6c8779bcaf25ca6b6bdee97a` and public main `09cb0b0199c8e6b1f4df3689464cf0fee67e2ae3`.

The existing password, Google, owner-security and immutable-payment code is retained. Current frontend canonical sources are exported into this backend, including the local editor schema dependencies. Schema-5/studio cloud rejection precedes member/database access; the preview-only schema-4 authoring gate retains its existing explicit switch. No provider flag, credential, price, database binding, migration execution or production deployment changes.

The shared request wrapper now reads optional telemetry bindings only for monitored routes and inside its error containment. A failed telemetry binding cannot replace a prepared admission response. Regressions cover studio rejection before storage and a monitored account response when telemetry access throws.

Local TypeScript and webpack production build passed. Before the final added telemetry regression, 16 JavaScript and 69 TypeScript backend tests passed; the final focused suite is recorded on the draft PR. The frontend records its separate local editor/account checks and full-source parity. No hosted owner login, Google consent, inbox delivery, two real members or selected Stripe TEST lifecycle is claimed. New branch hosting/bindings must be explicitly verified before redirecting any existing preview. Production remains unaccepted.

Rollback source is the held backend above. Preserve all existing member/project/payment rows, additive schemas and immutable archives. Old PR67 and its deployed preview remain intact; this branch is a successor for review.
