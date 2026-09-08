# Verified archive and handoff — 2026-09-05

Implementation commit: `8bcfbfa54e9a0a93b17ad38e066effe5e7d25fc6`.
[Draft PR #38](https://github.com/bmparent/eidos/pull/38) is stacked on [draft #36](https://github.com/bmparent/eidos/pull/36).

[Drive run folder](https://drive.google.com/drive/folders/1D7_g5HZ3uXUjhVbNtw2vkTKD2PFnBXot)

[Complete reproducibility ZIP](https://drive.google.com/file/d/1o4fured8Lxdw1P45SJcxQMzHHRBs7QI2/view)

[Decision report](https://drive.google.com/file/d/1dkDEeCBj-xVz0Xrx7Alo-GmG2m8cptir/view)

Local root: `artifacts/controlled_memory_utility_2026_09_05/`.
Drive root: `G:/My Drive`.
Drive folder: `Eidos_Brain_Proof_Phase/2026-09-04/quantized_memory_math/memory_utility_readiness_2026-09-05_201352`.

Archive size: **15,556,260 bytes**. SHA-256: `54cde9b130fcc7361d21e6a12522973c7acfeb2e2ed7c8edd030faaeb704b371`.
**389** artifact entries verified by size/SHA-256 inside a CRC-tested ZIP. **33** selected direct files (including the full archive) were copied and verified through the mounted Drive path, plus the post-copy receipt. The ZIP includes every pre-package artifact, the public source CSV, failed/blocked receipts, test fixtures and exact code snapshots. No artifact was silently skipped. Google Drive connector metadata independently confirmed ZIP/report identities, parent and byte sizes; it did not independently re-download cloud bytes for hashing.

The immutable drive_manifest.json was written before cloud metadata readback and therefore records cloud_metadata_verified=false. The subsequent cloud_readback.json supplies that successful additional check. Both are retained. These final archive/handoff addenda sit outside the ZIP to avoid circular hashes. The journal and analysis inside the ZIP reflect their pre-copy writing time; this receipt confirms completed archiving.

Validation remains: 32 memory/data checks passed (21 new); package 78 passed/4 skips; root 41 passed/5 known failures; full collection 271/6 known errors. Source and numerical model behavior are unchanged. Utility remains blocked/untested; the prospective runtime ceiling is 10%, not a measured candidate result.

The repository's automatic Vercel preview succeeded for the implementation commit. The user authorized deployment if necessary. No merge, production promotion, live Lab reconfiguration or core behavior change was performed.

## Proof Logic + Meaning

The artifact-delivery gate passed: local bytes match the mirrored bytes, all archive members match their manifest, and cloud metadata confirms the expected objects. This improves the ability to revisit and challenge the blocked utility finding. SHA-256 identity is evidence integrity, not detector correctness. The north-star contribution is reproducibility and honest self-reporting; detection utility, timestamp semantics and full-engine benefit remain unproven.
