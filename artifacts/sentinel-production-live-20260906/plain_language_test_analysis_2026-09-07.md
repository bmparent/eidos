# Sentinel Lab closeout — plain-language analysis

The production experiment worked yesterday, but two checks remained: reconnecting with operator access and checking every file retained in the stopped compute snapshot. Both passed today for the same job, `rd-8a14916b3ea7-696b7427`.

The completed result survived the original tab closing. Re-entering the existing credential and saved job ID brought back the same 600 scored predictions and engine diagnostics. The full verification receipt checked all 25 immutable files and found every hash and byte count matched. It also confirmed that the temporary retrieval compute was stopped afterward. No new experiment was launched and no core behavior changed.

The original seven downloads, full verification receipt, source comparisons, live logs, reports and progress files are saved under `artifacts/sentinel-production-live-20260906/`. The configured Google Drive archive is separate from previous evidence; `drive_manifest.json` gives its exact destination and file checksums. The earlier 36-file archive was checked again and had no mismatches.

## Proof Logic + Meaning

**Goal reached:** the remaining authenticated production reconnect and full snapshot-integrity checks passed. The earlier reload had already restored the same job and locked settings; today's fresh-tab reconnect recovered its terminal results.

**Previous state:** execution had succeeded, but access-dependent checks remained incomplete.

**Logic:** the app opened the stopped snapshot for verification, compared each file's actual bytes with the manifest, closed compute and read the provider's resulting state. Local checks tied those files back to the pinned source and all 600 frozen predictions. Authentication remained enforced.

**Math:** 25/25 file hashes and sizes matched. All 600 evaluation rows correspond to frozen predictions. However, 301 of 600 benign rows triggered false positives: `301/600 = 50.17%`. There were no attack examples, so recall and AUC are unavailable. Raw metrics stay visible; merged or calibrated event metrics were not computed. Zero scientific proof gates advanced and overall scientific readiness remains unknown.

**Meaning and improvement:** reproducibility is truth that can be revisited. Direct receipts now replace the previous incomplete checks, while preserving the poor false-positive result honestly. This strengthens reproducible operation and inspection of internal state in Eidos's streaming codec goal.

**Evidence:** the closeout JSON and full hash receipt in `apps/sentinel-lab/docs/`, plus `live-evidence-verification.json`, `live-lifecycle.json` and `reconnect-runtime-logs.txt` in the local evidence directory.

**Limits:** useful detection, compression advantage and held-out generalization remain unproven. Active downloads, provider outages, crashes and expiry were tested with controlled regressions rather than deliberately caused in production. No thresholds, labels, split or engine settings were changed to improve metrics. The appropriate next research step needs its own authorized protocol; it was not part of this closeout.
