# Controlled memory benchmark decision

**Decision: inconclusive for adoption. Keep the current production policy pending a defensible utility test.**

Run status: **complete**. 192 trajectories across 24 matched configurations; 8 neurons, 3 input channels, 2 forcing/initialization seeds, four policies, two initializations. 7,927,584 unique state updates; three additional timing repeats for each. Main elapsed time: 1471.05 seconds.

The four state policies share identical native engine-generated matrices, binary coefficients, initial states, forcing, and frozen readouts. Default single leak and slow single leak preserve the native float32 coefficient values lifted to float64; the band alpha vector is natively float64. The implemented four bands are 0.2, 0.05, 0.005, 0.0005, rather than the roadmap example beginning at 0.5. The first rounding pulse follows update 100. Every-step input rounding is absent from every arm. Production files and defaults are untouched.

All streams include a 39,990-step zero-input tail (20 times the slowest scalar relaxation scale). This is a scalar diagnostic minimum, not a coupled mixing-time guarantee. The driven excitation has 4,097 frames; the recovered project fixture has 1,100 synthetic, unlabeled frames. The project fixture is not clinical or operational detection evidence.

## Measured effects and costs

Errors are measured against float64 without explicit grid rounding. Absolute values are in reservoir state units. The normalized state error divides by the corresponding exploration-prefix RMS, floored at 1e-6; it never divides by a vanishing tail reference. The table pools zero, impulse, driven and project streams descriptively; full per-stream/per-initialization values remain in CSV.

| Leak | Policy | Max absolute state discrepancy | Median state RMS discrepancy | Normalized median RMS | Task MSE delta vs current: min to max | Runtime/current | State bytes |
|---|---|---:|---:|---:|---:|---:|---:|
| default | none | 0 | 0 | 0 | -0.000387344 to 0.00120973 | 0.988 | 64 |
| default | every_step | 0.00282487 | 0.000711858 | 0.0136172 | -0.0225282 to 0.0023482 | 1.460 | 64 |
| default | pulse100 | 3.01509e-05 | 3.6293e-06 | 6.94302e-05 | 0 to 0 | 1.000 | 64 |
| default | carry | 1.13684e-05 | 1.2578e-06 | 2.45149e-05 | -0.000369719 to 0.0012328 | 1.744 | 128 |
| slow | none | 0 | 0 | 0 | -0.0133143 to 0.000804999 | 0.974 | 64 |
| slow | every_step | 0.0657184 | 0.00784385 | 2.78871 | -0.553585 to 0.0322458 | 1.455 | 64 |
| slow | pulse100 | 0.000516142 | 2.71722e-05 | 0.0172315 | 0 to 0 | 1.000 | 64 |
| slow | carry | 1.29331e-05 | 2.89282e-06 | 0.00105489 | -0.0134048 to 0.000806061 | 1.743 | 128 |
| bands | none | 0 | 0 | 0 | -0.000629733 to 1.81192e-05 | 1.135 | 64 |
| bands | every_step | 0.0501314 | 0.00819593 | 0.0601069 | -0.00282424 to 0.0116088 | 1.353 | 64 |
| bands | pulse100 | 0.000451075 | 7.99366e-05 | 0.000591805 | 0 to 0 | 1.000 | 64 |
| bands | carry | 2.85443e-05 | 3.9577e-06 | 2.75969e-05 | -0.000636736 to 2.08744e-05 | 1.750 | 128 |

Task MSE uses a shared exploration-only ridge readout; suffixes follow a 64-frame gap, with state carried continuously through the split. Readouts and normalization are frozen before candidate replay. Raw readout and engine-style rounded predictions are both saved. These are state-fidelity effects on a synthetic prediction diagnostic, not independently trained end-to-end systems. Contiguous block results remain visible; no independent-frame p values or confidence claims are made. No defensible smallest useful task effect or acceptable overhead requirement was supplied or found, so both gates remain null.

Runtime is the median of three full replays with rotated policy order, for both initializations, excluding exact carry auditing and reporting. The small-matrix result is dominated by Python/NumPy overhead and does not forecast 2,000-neuron or GPU cost. State memory counts actual persistent numeric arrays only: carry adds one float64 array. Common matrices, readout, inputs, Python objects and offline trace storage are separate.

## Precision, certification, and preserved failures

6 entire trajectories were checked at 50 decimal digits using the same frozen binary coefficients and forcing. Maximum float64/reference discrepancy: 7.435024818036595e-15; prespecified tolerance: 1e-09. See precision_summary.json for every status and high-precision policy discrepancy. Other trajectories remain float64 experiments, not exact ground truth.

Maximum sampled carry-storage error: 0; 196,608 exact binary-rational subtraction checks. This samples z-r storage, not all steps or errors in matrix multiplication, tanh, coefficient generation, or adding the previous carry. The exact-storage theorem is not asserted as a floating-point guarantee.

The actual sampled matrix is **outside the sufficient condition**: the exact minimum row sum of abs(W) exceeds one, which certifies rho(abs(W)) > 1. No rescaling was applied. This does not establish instability or invalidate the empirical comparison. A separate exact nonnormal clipped-linear control has K=[[0,2],[0,0]], h=(3,1), and h-Kh=(1,1)>0.

The ordinary-rounding scalar deadband, slow-leak pulse persistence, and signed residual-carry two-cycle remain explicit controls. Residual carry does not guarantee forgetting. Constant pulse-phase values in finite traces are described as observed plateaus, not proofs of infinite-time behavior. The report’s tanh counterexample is analytically separated from the clipped-linear exact computation.

No actual Sentinel alert path was exercised; alert counts, false positives, calibrated/raw detection precision and recall are **NA**. Geometry is descriptive only. No compression benefit, detection validity, general-domain benefit, new mechanism, or originality claim is made.

## Claim / evidence table

| Kind | Claim | Evidence | Limit |
|---|---|---|---|
| Mathematical deduction | Scalar forgetting iff (1-alpha)^m <= 1/2 for ties-to-even pulse map | Preserved report-source.md, Proposition 1 | Exact scalar zero-input restriction |
| Mathematical deduction | Carry envelope (I-K)^-1 Delta; ordinary bound (I-K)^-1 A^-1 Delta/2 | Preserved report-source.md, Theorem 2; explicit noncommuting matrix order | Common forcing, rho(K)<1, exact residual; not certified for actual W |
| Mathematical deduction | Uniform sharpness across the stated class | Theorem 3; alternating forcing construction | Not each fixed leak or the bounded tanh subclass; no priority claim |
| Finite exact checks | 21 slow pulse states; minimum intervals 1386 and 69; signed cycle; nonnormal bounds | calibration.json, evaluator tests, original and rerun research receipts | Finite checks support but do not replace universal proof |
| Floating-point experiment | Policy-dependent state/prediction effects and costs | benchmark_summary.csv, plot_data.csv, raw traces, precision_summary.json | Small frozen open-loop reservoir; selected higher-precision cases |
| Hypothesis | Better fidelity may improve useful anomaly detection | No utility evidence in this run | Requires labels, requirements and independently frozen utility protocol |
| Interpretation | Numerical persistence must be separated from useful memory | Preserved counterexamples and measured separation | Not a claim of intelligence, consciousness or physics |

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

## Next decisive step

Define a labeled operational task, a smallest useful improvement and overhead budget, then freeze a matched chronological utility experiment against pulse100. Revisit matrix size and the adaptive engine separately. Numerical fidelity is a reason to investigate, not to adopt.
