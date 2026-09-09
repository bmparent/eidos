# Implementation proof logic ledger

## Proof Logic + Meaning

### Goal reached
All four product milestones have working implementations and local integration receipts. Full hosted acceptance is partial: actual CSV/Torch and document retrieval passed, then Vercel blocked new workers with HTTP 402 payment_required. Hosted ingress, privacy, reviewed history and explicit failure handling passed on the final candidate. The remaining hosted URL confirmation/analysis and checkpointed telemetry processing require restored Sandbox capacity. Frozen operational alert-quality qualification failed. Production release approval is pending; this is a reviewable candidate with a documented external gate, not an operational detector qualification.

### Previous state
The main product was a research console and Kaggle workflow. General private uploads, named causal forecasts, owned monitors and document retrieval were absent. Legacy selection could choose the lower current residual after seeing the observation, and character prefixes did not represent document meaning.

### Technical logic utilized
M1: Byte checksums, explicit schema confirmation, original record references, owner-scoped libSQL records and idempotent jobs link the UI to actual isolated execution. M2: The actual canonical Torch RLS reservoir is called through a new named-feature adapter; a fixed 24-observation calibration prefix, historical-loss predictor selection, immutable issued forecasts and score-before-update preserve causal interpretation. Anomalies are skipped during adaptation by default. M3: Arrival/event timestamps, ingress deduplication, contiguous offsets, checkpoint compare-and-set and actual saved reservoir/RLS state preserve replay equivalence across batches. M4: Pinned MiniLM embeddings pool every token window; cosine retrieval returns original passages and content hashes, with no model-generated code or per-event LLM calls.

### Math / scoring logic
Reservoir: r_t = (1-alpha) r_(t-1) + alpha*tanh(W_in*x_t + W_rec*r_(t-1)). RLS: k=P*z/(lambda+z^T*P*z); W_out'=W_out+error*k^T. A forecast is committed before its target: prediction_t=f(state_before_t), residual_t=observation_t-prediction_t. Surprise uses the prior residual median and MAD scale; decision=score>=prior_threshold, then the recorded learning policy determines update. This robust score is not a probability. MAE=mean(abs(observed-issued)); coverage=count(target in issued band)/resolved intervals. Precision=TP_incidents/(TP_incidents+FP_incidents); recall=detected_truth_incidents/truth_incidents; false_alerts/day=FP_incidents/operating_seconds*86400. Matching horizon and maximum late window are explicit seconds. Pearson correlations are retrospective associations, with complete-pair counts; cosine similarity is association, not factual confidence. Unknown metrics stay null/NA.

### Philosophical meaning
Reproducibility is truth that can be revisited. Source drill-down is explanation before automation. Separating anomaly from consequence is restraint before alarm. Keeping failed qualification visible is honesty before optimization. Familiarity is recognition, never automatic permission to suppress a repeated harmful event.

### Why this is better
A user can now bring a supported source through confirmation to an actual saved result, inspect a named forecast and the records behind a finding, return after interruption, and revoke a source credential. The causal audit removes the retrospective-forecast ambiguity. The result is more inspectable; the measurements do not prove universal improvement or an acceptable operational alert burden.

### How this moves Eidos closer to the north-star goal
The north-star claim is: Eidos Brain is a self-monitoring streaming intelligence codec. It learns live streams, compresses predictable behavior, preserves meaningful anomalies, monitors its own internal state, and emits human-readable incident receipts. This release strengthens live-stream ingestion/learning, retained anomalies, visible internal state, source-linked incident explanation and reproducible execution. It does not establish new compression performance, broad domain generalization or value beyond all existing detectors/compressors.

### Evidence
The requirement-to-evidence table maps each milestone to browser, runner, provider, collector, replay, cancellation and semantic receipts. The immutable evaluation plan, acceptance freeze, final metrics, qualification decision, verified raw archive and frozen source archive retain the complete numerical result. Two-user access checks and real worker hashes support privacy/execution claims; hashes alone do not establish accuracy.

### Remaining uncertainty
All four Eidos variants failed operational precision/false-alert criteria; regulation also failed coverage. Synthetic periods do not establish natural-domain generalization. Vercel HTTP 402 payment_required blocks the remaining hosted compute matrix and a full final-SHA hosted CSV repeat; earlier actual Sandbox execution receipts identify their tested SHA separately. No sealed Grand Proof gate was opened or advanced. GPU, OCR/image PDFs, IPv6-only URLs, authenticated web imports, unrestricted stream rates, broad load testing and attributed provider cost are unqualified. No collector is installed for unrelated or live customer collection. Private sources require deliberate owner setup. Research readiness is unknown; no percentage is assigned.
