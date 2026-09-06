# Authenticated production audit supplement

The access boundary in the earlier audit is resolved: Brent entered the existing operator credential in the production password field. One bounded experiment launched through the application on release `31b1a895182b35a1c376a3e532ade9d3107aa13d` (PR #43).

- Job: `rd-8a14916b3ea7-696b7427`
- Diagnostic: `9ca66d3f-2ed2-427c-940d-0eb971a94b01`
- Production deployment: `dpl_DVvasVzqPNraYw67XXnQgvqmejgB`
- Lock: `8a14916b3ea7a0cf2ce766ba2a8c8e126745f8f7907b0ceccb2a7df8476cf0a5`
- Dispatch accepted at 23:33:01 UTC; completed at 23:33:24 UTC on September 6, 2026.
- Directly observed stages: runtime bootstrapping, completed engineering. Intermediate stages completed between browser observations; the UI progress rail is not a historical stage log.
- Engine: standard CPU profile, seed 0, 1,000 input rows; 200 calibration, 600 evaluation, 200 excluded holdout. Production Python was 3.14.4, distinct from the validation CI runtime.

Seven original artifacts were downloaded through the authenticated application. The manifest itself plus its six exposed immutable entries were received. Source hashes match the four corresponding Git objects at the execution commit. All 600 evaluation records match frozen engine predictions, including score, threshold, alert decision and source-row index. Dataset, lock, prediction trace, metrics and manifest hashes correspond.

The manifest declares 25 immutable artifacts. The new authenticated `artifact_verification.json` receipt checks all declared immutable files inside the existing completed snapshot, returning sizes and hashes only. It rejects paths outside the job directory, raw input/label files, missing files and mismatches. Limits are 512 entries and 128 MiB of verification reads. The existing retrieval wrapper stops resumed compute before provider metadata is read for the final receipt. This works with already-completed snapshots; it does not launch another experiment. The receipt is computed on retrieval and is not added to the immutable run manifest.

Reproduce verification from repository root after downloading receipts:

```powershell
npm.cmd test --prefix apps/sentinel-lab
npm.cmd run build --prefix apps/sentinel-lab
node apps/sentinel-lab/scripts/verify-live-evidence.mjs artifacts/sentinel-production-live-20260906
```

## Proof Logic + Meaning

### Goal reached
Authenticated production execution and seven original downloads passed. Full snapshot verification and reload/reconnect are being checked against this same job after the receipt release; their final results belong in the live evidence package, not an inferred success flag.

### Previous state
Only local real-data execution and controlled lifecycle tests had passed. Operator access was missing, and internal immutable hashes were not exposed for independent audit retrieval.

### Technical logic utilized
Production uses the pinned source-discovery launcher, a shared SQL admission transaction, stable retry intent, frozen label-free engine predictions, post-freeze evaluation, and explicit snapshot retrieval cleanup. The supplement adds an artifact verifier; core model behavior is unchanged.

### Math / scoring logic
For every immutable file, `SHA256(actual_bytes) = manifest.sha256` and byte count must match. All 600 evaluation rows correspond to frozen predictions. `FPR = FP / (FP + TN) = 301 / 600 = 0.5016666667`. There are zero positive examples, so recall and ROC AUC remain null. All seven proof gates remain locked.

### Philosophical meaning
Reproducibility is truth that can be revisited. Execution success must remain separate from useful detection quality.

### Why this is better
There is now an actual production job with downloaded source, dataset, prediction, evaluation and diagnostic receipts, instead of an access-blocked inference from local tests.

### North-star connection
This strengthens reproducible execution and inspection of internal state in the self-monitoring streaming intelligence codec. It does not prove compression advantage or anomaly-detection value.

### Evidence
PR #43, the live job and diagnostic above, `artifacts/sentinel-production-live-20260906/`, and `live-evidence-verification.json`. The audit package preserves earlier validation evidence separately.

### Remaining uncertainty
The evaluation slice is entirely benign and its false-positive rate is high. No held-out proof was opened. Active-download preservation, transient provider errors, crash receipts and expiry are covered by controlled regressions; the short live run completed before an active download was observed. Shared admission is qualified for production experiment jobs, not all simultaneous retrieval VMs or the optional external runner backend.
