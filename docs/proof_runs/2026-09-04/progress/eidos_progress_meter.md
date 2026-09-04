# Controlled memory proof progress

Overall project readiness: **unknown**. Existing global gates were not reassessed; this is not a zero score.

| Local benchmark check | Status | Receipt |
|---|---|---|
| Frozen evaluator and inputs | passed | main/freeze.json |
| Actual listen adapter / reset | passed | main/adapter_selected_size.json |
| Exact scalar and nonnormal controls | passed | calibration/calibration.json |
| Declared numerical trajectories | passed | main/run_manifest.json |
| Full-horizon precision checks | passed | report/precision_summary.json |
| Operational utility gate | missing | main/protocol.json: null task/overhead thresholds |

Measured scope: 8 neurons, 192 trajectories, 6 precision checks.

## Proof Logic + Meaning

Matched frozen recurrence; exact controls; long zero tails; fixed shared readout; precision checks.

r_next=(I-A)r+A*tanh(Wr+b); carry z=F(r)+c, r_next=Q(z), c_next=z-r_next; RMS and MSE effects.

Recognition of numerical artifacts precedes claims about useful memory.

This strengthens reproducibility and interpretation of internal state, not validated detection or compression value.

Remaining uncertainty: Default-size reservoir, GPU, Adaptive engine feedback, Labeled detection utility, Compression value, Theorem originality.
