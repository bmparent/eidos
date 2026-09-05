# Memory utility data gate

This follow-up to the controlled numerical memory benchmark prepares a trustworthy data boundary for a later utility experiment. It does **not** run Sentinel, change rounding, fit a reservoir readout, or approve residual carry. The September 4 numerical evaluator and receipts remain unchanged.

## Run from the repository root

Use the existing `requirements-memory-benchmark.txt` environment. No new dependencies or PYTHONPATH edits are needed.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m pytest tests/test_memory_utility_data.py -q
python -m proof.memory_utility_data prepare --file path/to/source.csv --out artifacts/controlled_memory_utility_new/data_gate
python -m proof.memory_utility_data verify --out artifacts/controlled_memory_utility_new/data_gate
```

Use a new output directory. Exit **2** means data preparation is blocked and a receipt was saved; exit **0** means the requested preparation or integrity operation succeeded. A passing integrity check only verifies bytes, even for a blocked preparation. Unexpected errors retain a failed run manifest. Reusing an existing output directory is refused.

For a timestamped CICIDS file, explicitly supply its capture-local timestamp format, exploration cutoff and overlap gap:

```powershell
python -m proof.memory_utility_data prepare --file path/to/timestamped.csv --out artifacts/controlled_memory_utility_new/timestamped_gate --timestamp-column Timestamp --timestamp-format "%d/%m/%Y %H:%M:%S" --cutoff "2017-07-06T09:18:00" --gap-seconds 120
```

The times in this example are a **proposed schedule-based split**, not a validated split or frozen utility protocol. Verify the source's actual date format and capture convention. The [publisher's schedule](https://www.unb.ca/cic/datasets/ids-2017.html) puts Thursday WebAttacks after 09:20; this motivates exploration before that interval without selecting a split based on comparative detector outcomes. Longer flows require a larger gap. Missing timestamps, malformed rows, ambiguous required columns, unknown labels, negative/nonfinite durations, insufficient partitions or a single-class suffix block preparation. Known CIC export dash variants in the three Web Attack names are accepted; arbitrary non-benign names are not silently relabeled as attacks.

Complete-flow features are ordered by `Timestamp + Flow Duration` (microseconds), their earliest assumed availability, using source row as a stable tie-breaker. Equal completion times cannot cross the split. The gap must cover the longest observed flow duration. This prevents an evaluation flow from starting before the exploration cutoff; it is **not** proof of independent hosts, sessions, campaigns or labels. Capture timestamps alone do not certify upstream export semantics.

The three predetermined channels are flow duration, forward packet count and forward byte count. They keep the data interface compatible with the previous three-channel numerical harness; their detection sufficiency is unproven. The scaler sees exploration rows only, imputes nonfinite feature values using prefix means, then computes population standard deviations with floor `1e-6`. Transform is `clip((x - mean)/std, -3, 3)/3`. Nonfinite flow duration blocks timing rather than being imputed. An entirely missing exploration feature blocks the run. Constant features remain bounded. Labels, IP addresses and attack windows are excluded from `model_inputs.npz`; scoring labels have a separate file. A caller can still misuse these files, so future detector integration needs a label-invariance test too.

Replay must start once before the prefix and carry state continuously through the gap and suffix. Labels from the suffix are available for scoring only. One-step prediction targets must stay within a partition. None of this enforces a new model policy in production.

## Why the existing labeled runner is not reused unchanged

`eidos/tools/run_labeled_domain_proof.py:standardize_matrix` computes its mean and standard deviation over the whole selected sample. Its `load_labeled_dataset` calls this before any chronological training split. This is useful for a descriptive replay, but it is unsuitable for an unseen suffix utility claim.

`eidos/proof/sentinel_calibration_v1.py:_reason_for_suppression` explicitly retains events overlapping known attack windows and can suppress events in known benign context. It documents itself as a proof-stage accounting layer. This is not a label-blind deployable detector. Tests execute both existing mechanisms and demonstrate their dependence on suffix values and scoring labels. No legacy function or historical receipt was modified.

Future operational scoring must retain raw, merged, deduped and confirmed counts. Ground-truth-assisted calibration must remain a separate diagnostic view and cannot pass utility. Recall must count distinct detected attack windows divided by all attack windows; multiple alarms in one window must not manufacture recall. Record both FP per total 10,000 frames (legacy convention) and FP per 10,000 **benign** frames, naming denominators explicitly. No alert, precision or recall measurements were produced by this data gate.

## Remaining experiment requirements

1. Obtain timestamped labeled data with provenance. Reconcile or explicitly quarantine invalid-duration records, with class counts and retained source indices. Do not fabricate timestamps or patch the source CSV in place.
2. Validate the split and freeze source, preprocessing, matrices, readout, source code, noise, pulse phase and initial states. Residual carry is a candidate because of the prior numerical result, not because utility has been proved.
3. Specify a smallest useful task effect and acceptable runtime/memory cost **before** evaluating candidates. On September 5 the user supplied a **10% maximum extra per-frame runtime**: candidate/current runtime must be at most `1.10` under a frozen matched timing procedure. This is a prospective requirement, not a claim that the candidate meets it. Existing calibration acceptance requires lower FP and preserved recall, but supplies no operational effect magnitude. That magnitude and the memory budget remain null; do not choose them after seeing effects. The earlier immutable data-gate snapshot predates this answer; the dated `utility_requirements.json` records the amendment.
4. Exercise the real Sentinel event path with scoring labels unavailable to inference, postprocessing and calibration. Verify that changing suffix labels cannot change emitted events. Record any readout training, RLS, thermostat or feedback difference as a separate experimental scope.
5. Run matched current-policy/carry trials once on a disjoint suffix with a fixed readout. Freeze temporal blocks before replay; preserve all block effects and tried variants. A proposed 5-minute block is descriptive until validated against dependence and available block count. Do not use independent-frame p-values or infer production speed from an 8-neuron NumPy loop.

The data-preparation protocol is not a frozen utility evaluator. Data readiness, numerical fidelity, and adoption are separate statuses. Current adoption remains **inconclusive**.

## Proof Logic + Meaning

- **Goal reached:** the utility data gate is implemented and tested; the available operational replay is blocked by missing chronology and invalid durations.
- **Previous state:** a large labeled file existed, but source order could be mistaken for time order and label-assisted accounting could be mistaken for unseen detection performance.
- **Technical logic:** explicit flow availability, prefix-only fitting, separate model/scoring artifacts, strict rejection and immutable hash receipts.
- **Math:** `available_i = start_i + duration_i`; `max(duration) <= gap`; `x'_i = clip((x_i - mu_prefix)/sigma_prefix, -3, 3)/3`. Future event precision is `TP_events/(TP_events+FP_events)` and window recall is `detected_windows/all_windows`; neither is measured here.
- **Philosophical meaning:** honesty before optimization; a useful memory claim must survive a test that cannot see its answers.
- **Why better:** invalid evidence paths are reproducibly rejected before a candidate can receive a misleading utility pass.
- **North-star connection:** this strengthens reproducibility and the credibility of future evidence-backed anomaly detection. It does not yet prove better compression, preserved anomalies or useful memory.
- **Evidence:** dated dataset audit, immutable preparation receipts, leakage regression tests, pytest logs and journal.
- **Remaining uncertainty:** timestamped source semantics, corrupt durations, useful-effect and memory requirements, compliance with the 10% runtime ceiling, label-blind Sentinel integration, detection utility, default-size cost, GPU and end-to-end adaptive behavior remain unresolved.

## Change delivery boundary

Draft PR #36 remains open for the completed numerical benchmark. Its push triggered the repository's automatic Vercel preview integration. The user subsequently authorized deployment if necessary, covering the automatic preview when publishing this follow-up draft. This follow-up is stacked on #36 and does not request a merge or production promotion. No production code, deployment settings or original checkout residue is changed.
