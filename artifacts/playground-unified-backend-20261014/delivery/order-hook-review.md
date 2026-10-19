# Independent backend review of the approved-page order hook

Reviewed the frontend-authoritative source before mirroring. Scope is controlled TEST attachment and exact archive recovery, not real payment or hosted acceptance.

- A snapshot must belong to the signed-in owner. The chosen parent product/version/SHA256/byte count must match the server catalog and recomputed original archive bytes before any checkout request.
- The initial order transaction inserts the immutable attachment. The unique receipt reference serializes competing requests. A losing request reloads the winning order and must match owner/product/delivery ID before checkout can proceed. Retrying cannot add, remove or replace the attachment.
- Signed paid events require the exact stored session, price, owner email and delivery metadata. The configured merchant is verified against the actual account API response. Existing AI checkout merchant evidence remains separately enforced; no new balance or grant authority is introduced.
- Original archive downloads retain their default behavior. The approved-page selector requires a paid TEST order, ready fulfillment, the bound signed-in owner, no refund or independent payment revocation, and exact stored snapshot/archive hashes.
- Recovery's paid-order lookup and conditional claim already block ordinary refunds. The final source additionally rejects independent raw revocation before consuming the token and computes the returned state from a fresh order after receipt rotation.
- An approved stored archive does not depend on current sales flags, edition verification or the current original archive. The authoritative order suite proves retrieval with those changed; new checkout still requires current immutable parent acceptance.
- Additive migrations 0006/0007 and the isolated renderer graph do not replace legacy schema-1 cloud storage, schema-5 HTTP 409 protection, existing paid archive files or historical guest receipt data.

No material correctness finding remains after the recovery consistency test. Backend full regression: 100 passed / 1 explicit Windows symlink privilege skip. Authoritative frontend server suites: 37 passed with intercepted outbound traffic. See adjacent logs and hash receipts. Exact graph export and final build remain separate gates in the final source receipt.

Limits: no hosted database, actual checkout, real mail, customer identity or paid provider execution was tested. The next manual preview intentionally omits every paid archive and database binding. The original 32-archive TEST bundle remains retained and was checked again byte-for-byte against its prior receipt.
