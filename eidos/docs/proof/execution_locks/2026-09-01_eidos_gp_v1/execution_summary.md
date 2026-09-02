# Grand Proof v1 Execution Summary

## Final verdict

`BLOCKED_RESOURCE_BEFORE_HELDOUT`

The frozen design hashes, tests, 16-stream smoke matrix, A0-A7 smoke ablations, resource profiles, and execution lock completed. The resource protocol rejected every tested reservoir before calibration or held-out evaluation. No Grand Proof acceptance gate passed.

## Frozen document verification

| Document | Bytes | SHA-256 | Status |
|---|---:|---|---|
| `meaningful_surprise_v1_spec.md` | 20,838 | `3020b108ee947d00571a6bea6dc9d4839824b9cd2c8e2a583d82391b4172d101` | verified |
| `grand_proof_protocol_v1.md` | 25,447 | `a5d94d4a8e4910ea29fbe47dd8372b0990f78685841f390dfb5128bdb728f355` | verified |
| `local_codex_execution_brief_v1.md` | 14,881 | `2dfd1869f60cef8f93d4fd62a010f65d93369d2e68c66394ccf681030762e9bb` | verified |

## Validation

- Focused: 23 passed, 0 failed, 0 skipped.
- Compatible repository suites: 241 passed, 0 failed, 5 skipped.
- Smoke: 8 scenarios times 2 seeds; 16 complete live captures.
- Comparisons: 256 system/ablation metric rows and 522 paired interval rows.
- Execution lock: verified; held-out permission is false.

## Resource gate

| Reservoir | Frames/second | Projected synthetic seconds | Peak RSS bytes | Eligible |
|---:|---:|---:|---:|---|
| 128 | 38.426 | 39,972.904 | 379,121,664 | no |
| 256 | 32.570 | 47,159.584 | 420,675,584 | no |
| 512 | 32.455 | 47,327.227 | 449,929,216 | no |
| 1024 | 16.004 | 95,978.558 | 511,639,552 | no |

The locked time budget was 21,600 seconds. The projection covers only the synthetic suite; real and transfer work would add runtime.

## Blockers

- Data: the required CICIDS WebAttacks CSV was not found.
- Data/provenance: the Drive seizure CSV lacked discoverable license/README evidence and was not locally materialized for hashing.
- Identity: object `4a639cd693701fb764fe30ba672d4811bdbf5a75` is absent from fetched refs.
- Resources: no tested reservoir met the runtime budget.
- Independent review: not executed, and the implementer did not self-score.

## Artifact locations

- Local: `eidos/artifacts/grand_proof_v1_20260901T233145Z/`
- Google Drive: `G:\My Drive\Eidos_Brain_Proof_Phase\2026-09-01\grand_proof_v1_20260901T233145Z`

## Proof Logic + Meaning

The exact goal reached was a locked, causal, shadow-only implementation plus bounded smoke and resource evidence; the proof gate remains blocked. Previously there was no runnable full-engine common capture or auditable A0-A7 harness. The implementation records `e_t = x_t - xhat_t` after the live decision, derives past-only representations, and combines consequence value, quotient residual, persistence, disagreement, and raw escape without changing core behavior. Event quality retains precision, recall, and false-positive counts; registered comparisons use paired whole-seed resampling and Holm correction.

This represents restraint before alarm and reproducibility before claim. It is better because missing evidence, false positives, ablation outcomes, resource failure, and review ineligibility are preserved rather than hidden. It strengthens Eidos's ability to monitor internal state, preserve candidate anomalies, explain incidents, and rerun proof work. It does not prove superior compression, detection, generalization, production readiness, or cross-domain value.

Exactly one next experiment is registered: rerun the unchanged execution lock on a resource-eligible machine through calibration and held-out seeds before inspecting any held-out outcome.
