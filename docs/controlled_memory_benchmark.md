# Controlled reservoir memory benchmark

This benchmark isolates **state rounding** in the actual packaged engine's frozen reservoir recurrence. It does not change production defaults or run Sentinel, adaptive training, or the live Lab.

From the git repository root, using Python 3.11+ (install the separate proof dependencies with `python -m pip install -r requirements-memory-benchmark.txt`):

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python -m pytest tests/test_memory_benchmark.py -q
python -m proof.memory_benchmark calibrate --out artifacts/controlled_memory_new/calibration
python -m proof.memory_benchmark prepare --out artifacts/controlled_memory_new/main --calibration artifacts/controlled_memory_new/calibration
python -m proof.memory_benchmark run --out artifacts/controlled_memory_new/main
python -m proof.memory_report --run artifacts/controlled_memory_new/main --out artifacts/controlled_memory_new/report
```

Use a **new output directory** each time. No manual PYTHONPATH edits are required. Calibration runs known controls, the real listen adapter replay, and timing only. Preparation freezes evaluator/source hashes, exact input files, matrices, leak vectors, initial norms, reference precision, metrics, thresholds, budgets, and readouts before policy comparison. Run rejects changes to frozen code, protocol or inputs. Incomplete runs remain intact; their manifest describes failure or the 60-minute cap. Report generation reads completed receipts without rerunning candidates.

The calibration tries 16 neurons and then 8 if the estimated cost of six complete mpmath trajectories exceeds 900 seconds. This is explicitly a smaller engine-generated configuration, not a default-size reservoir validation. Engine coefficient construction always seeds its weights with 42. Seeds 7 and 19 vary forcing and the second initialization; they are not independent weight trials.

The four policies are no explicit grid, every-step state rounding, current 100-step state rounding, and every-step residual carry. The old algorithm rounded inputs too; that algorithm is intentionally not an arm here. Main state/carry arithmetic is float64. Native float32 matrices and scalar leaks are lifted exactly; four-band leaks are generated natively in float64. NumPy implements the extracted recurrence; real Torch listen equivalence is checked at 99/100/101 and 199/200/201 with nonzero matched noise, plus repeated resets. Native float32 checks are also included for single bands. All comparisons use frozen noise zero, matrices, readout, and exogenous inputs.

The actual band implementation clips 0.5 to 0.2, so its four bands are 0.2, 0.05, 0.005, 0.0005. The slow single-band 0.0005 control does not substitute for this heterogeneous vector. Every trajectory includes at least 20 times the slowest scalar relaxation time in its zero-input tail. This is not a coupled mixing-time guarantee.

The project stream is the tracked 1,100-row `eidos/verify_data/incident_test_data.csv` synthetic fixture. Normalization uses its first 400 rows only. Synthetic driven readouts fit the first 2,048 frames. Both streams have a 64-frame split gap, one-step targets, and continuous state handoff. These suffix task effects remain descriptive: no defensible smallest useful task effect or overhead requirement exists for these data. **Numerical fidelity cannot pass an adoption utility gate.**

Outputs include protocol/freeze, environment, real-listen equivalence receipts, raw trajectories, metrics for both initializations, carry-storage samples using exact Fraction arithmetic, block-level task effects, timing repeats, increased-precision references, CSV plot data and evidence figures. Actual reservoir certification uses a floating candidate positive vector followed by exact rational checks against binary coefficients. A failed sufficient condition is not proof of instability. No matrices are rescaled.

The signed clipped-linear carry two-cycle and the analytical tanh counterexample remain visible. Exact research checks support mathematical claims only within their declared restrictions. Saving rounding residuals is classical error feedback; no originality claim is made.

Large inputs and traces are ignored by git. The dated evidence summary and Drive/local bundle manifest provide recoverable hashes. For research-package reproduction, retain the delivered ZIP and extract into a new copy before running, from this repository root:

```powershell
python artifacts/controlled_memory_new/research_rerun/verify.py --evaluate
python artifacts/controlled_memory_new/research_rerun/verify_sharpness.py
```

These source evaluators locate their files relative to their own paths. They overwrite results in that copy only. Never overwrite the delivered research folder or silently replace its protocol or freeze.

After writing the journal, analysis, progress and validation receipts into the run root, package evidence with:

```powershell
python -m proof.memory_bundle --run-root artifacts/controlled_memory_new
```

The bundler preserves exact source bytes, verifies the frozen evaluator against its manifest, CRC-checks the ZIP, and hashes each copied Drive file. It uses configured Drive roots when available, or records a skipped copy. `--drive-dir` can specify a **new** subfolder beneath a user-selected research folder. Existing destinations are never overwritten. The post-copy receipt is separate from the ZIP to avoid a self-referential checksum. Replay instructions are included in the bundle.
