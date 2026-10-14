# Studio cloud boundary: plain-language test analysis

The new creative studios save their full documents locally. The current backend accepts an older cloud format. This change refuses newer studio documents clearly before older validation can discard their data. The user gets a JSON-backup instruction and keeps the open local project.

All76backend tests and TypeScript passed. Three new tests prove no database access, no input mutation and no downgrade for newer/disguised documents. Existing schema1 concurrent saves and historical-purchase retrieval still passed. Seven actual Preview checks confirm the409 response, held account/AI/shop flags, invalid relay/origin denial and retained provider protection.

Artifacts are repo-local under artifacts/showcase-studio-boundary-20261013. The ignored scoped configuration supplies the same validation DB/relay/bypass to only the new branch Preview. No Production or old branch change, no email, paid transaction or model call occurred. Drive status and copied-file hashes are explicit in drive_manifest.json.

## Proof Logic + Meaning

The narrow format-safety gate passed locally and hosted. Before this work, the old contract did not name the new local-studio boundary. The predicate schema>=5 OR own studios field now yields409 before persistence. Tests compare untouched input and zero database reads; the unchanged legacy suite checks preservation. Honesty before convenience means an account-save button cannot mislabel a local-only document. This strengthens reproducibility and operator trust around auditable receipts. It does not prove the Brain learning/compression/anomaly north-star claims. Schema5 owner/reopen/asset isolation, actual providers and production activation remain unproven and gated.
