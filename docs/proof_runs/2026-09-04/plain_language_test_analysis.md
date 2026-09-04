# Plain-language test analysis — 2026-09-04

The experiment asked whether rounding creates numerical memory, and whether saving its residual deserves more testing. All planned numerical comparisons finished. The answer is that rounding can materially change states in this small frozen reservoir, and residual carry improves numerical fidelity in the observed comparisons. It also costs more runtime, does not guarantee forgetting, and has not demonstrated operational utility.

The current 100-step rounding policy stays unchanged. This is an inconclusive adoption decision, not a failed numerical experiment. The fixture is synthetic and unlabeled for detection. No Sentinel alerts or false-positive claims were produced.

The main run took 24.52 minutes. All six full-horizon 50-digit checks passed, with maximum float64/reference difference 7.44e-15. All 192 trace metric rows were independently recomputed from archived arrays, and input/evaluator/raw-file hashes checked. Sampled carry subtraction had zero storage error across 196,608 exact checks; this is not a guarantee about every arithmetic operation.

The full original exact mathematical scope passed only after an explicitly documented time-budget extension. Both original-cap partial attempts remain available. The 11 targeted tests and 78 package tests passed. Four package skips are explained in validation.json. Remaining root-suite failures reproduce on the clean baseline and are not presented as passing gates.

Local artifacts and final Drive copy information are described in codex_journal.md, evidence_manifest.json and the post-package drive_manifest.json. The figure preserves zero values, absolute state units, seed ranges, task effects of both signs and observed runtime cost. Its underlying CSV is included.

## Proof Logic + Meaning


### Goal reached
Controlled numerical benchmark: **complete**; adoption utility gate: **missing / untested**. Source reproduction and repository test statuses are reported separately in the handoff receipts.

### Previous state
The research package supplied proofs and toy controls but had not measured the current Torch reservoir under matched rounding policies. The working checkout also contained unrelated residue. This work isolates the current fetched baseline and preserves that residue.

### Technical logic utilized
Freeze the real engine matrices; extract the listen recurrence and validate it against the real path; vary only state rounding. Use exact rational controls and certificates, two initializations, long zero-input tails, fixed exploration readouts, and disjoint suffix diagnostics.

### Math / scoring logic
r_next = (I-A)r + A*tanh(Wr + Bu). Pulse map: k_next = round_even((1-alpha)^m*k). Carry: z=F(r)+c, r_next=Q(z), c_next=z-r_next. Corrected-state error obeys |e_next| <= (I-A+AK)|e| + A(I+K)Delta/2; the invariant box gives |r-x| <= (I-K)^-1 Delta. MSE = mean((prediction-target)^2). RMS discrepancy = sqrt(mean((r-x)^2)). Runtime ratio = median(candidate replay time)/median(current replay time). No utility score is invented.

### Philosophical meaning
Reproducibility is truth that can be revisited. Apparent memory is trustworthy only after representation-induced persistence has been checked.

### Why this is better
The project now has a rerunnable isolated evaluator, frozen inputs, explicit precision limits, measured effects/costs, and retained negative controls. This improves auditability and operator trust; it does not establish useful predictive or detection improvement.

### How this moves Eidos closer to the north-star goal
For the self-monitoring streaming intelligence codec goal, this strengthens reproducibility and interpretation of internal-state memory. It prepares reliable evidence-backed detection work without claiming anomaly preservation, compression advantage, or incident explanation has been demonstrated here.

### Evidence
protocol.json, freeze.json, adapter_selected_size.json, calibration.json, raw/*.npz, results/*.json, precision/*.json, benchmark_summary.csv, temporal_blocks.csv, evidence_figure.png, and evidence_manifest.json. The archive preserves the delivered research ZIP and both original and rerun receipts.

### Remaining uncertainty
Default-size reservoir, GPU, adaptive training/thermostat feedback, exact forgetting of the actual coupled engine, detection labels, operational utility, acceptable overhead, full-model compression behavior, and theorem originality remain unproven.


## What should happen next

Define a labeled operational task and prespecified useful-effect/overhead requirements, then evaluate once on a chronological suffix with matched trials and temporal blocks. Default-size, GPU and adaptive training remain separate unproven scopes.

## Archive verification addendum

All 18 requested deliverables were copied to the new Drive research subfolder with matching local/mounted hashes. Connector metadata readback confirms the report and full-size ZIP. See archive_status.md.
