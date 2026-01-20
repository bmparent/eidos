# Eidos Brain v0.4.7.02 — Trace-Seal + Diagonal-Cutter Surgical Patch

Use this as a **copy/paste task prompt** for Antigravity/Gemini to implement a positivity-preserving “projection layer” in **`EIDOS_BRAIN_UNIFIED_v0_4.7.02.py`**.

---

## Prompt 1 — Core Patch: Trace-Seal / Diagonal-Cutter (Surgical, Backwards-Compatible)

### Objective
Implement a **positivity-preserving anomaly score** that:
- **Diagonal-Cutter:** removes “trivial/baseline” directions in residual space (dominant predictable subspace)
- **Trace-Seal:** computes surprise energy as a **guaranteed non-negative quadratic form** (projected energy) to prevent sign/regularization “leaks”

Goal: reduce false positives caused by drift/seasonality/boundary effects while keeping current behavior available via config toggles.

### Constraints (must-follow)
1. **Surgical:** keep architecture intact; add one small module + minimal wiring where gating already happens.
2. **Backwards compatible:** if `trace_seal_enabled=False`, behavior must match current v0.4.7.02 gating.
3. **No new heavy deps:** use only numpy/torch already present (no SciPy).
4. **Logs/artifacts remain readable:** add a compact diagnostic payload only when enabled.

---

### Integration Point (exact spot)
Locate the per-frame gating section where the engine computes:
- `best_pred` (chosen between left/right)
- `best_err` (scalar)
- `ema_err`, `sigma` (MAD/std)
- `z_score`, `is_surprise`, `eff_z_thresh`

This is typically the block around:
- `err_R_t = torch.linalg.norm(frame - y_R)`
- `best_pred = ...`
- `best_err = float(best_err_t.item())`
- `z_score = abs(best_err - ema_err) / sigma`
- `is_surprise = bool(z_score >= eff_z_thresh)`

Patch by inserting a **vector residual** pathway and optionally using it to replace/augment `best_err`.

---

### Step 1 — Add Config Knobs
Add these keys to the central config dict (defaults chosen to be safe):

```python
"trace_seal_enabled": False,           # master toggle
"trace_seal_rank": 4,                  # how many top predictable eigen-directions to remove
"trace_seal_decay": 0.995,             # EMA update for residual covariance
"trace_seal_recalc_every": 250,         # recompute eigensystem periodically
"trace_seal_eig_floor": 1e-6,           # PSD safety floor
"trace_seal_mix": 1.0,                 # 1.0 = use TraceSeal error only; 0.0 = raw error only
"trace_seal_update_on_surprise": False, # update covariance on surprises? default False
"trace_seal_diag_every": 2000,          # log stats occasionally
```

Interpretation:
- **Diagonal-Cutter:** remove top-`rank` eigenvectors of the residual covariance (predictable subspace)
- **Trace-Seal:** anomaly energy computed from an orthogonal projector ensures **non-negativity**

---

### Step 2 — Implement `TraceSealProjector` (Standalone Sidecar Class)
Add a small class near other helper classes.

**Responsibilities**
- Maintain an EMA covariance estimate of residual vectors: `r_t = frame - best_pred`
- Periodically compute eigendecomposition
- Build orthogonal projector onto the complement of the top-rank subspace:
  - `P = I - U_k @ U_k.T` where `U_k` are top eigenvectors
- Provide:
  - `score(r)` returning `sqrt(r^T P r)` (guaranteed `>= 0`)
  - `update(r)` updates covariance
  - `diagnostics()` small dict: top eigenvalues, explained ratio, rank, etc.

**Implementation details**
- Use `torch` tensors on the same device as the engine.
- Keep covariance symmetric: `C = 0.5*(C + C.T)`.
- For the projector, use eigenvectors from `torch.linalg.eigh(C)` (since features are modest).
- Enforce numerical stability with `trace_seal_eig_floor` for diagnostic reconstruction (do NOT dump full matrices).

---

### Step 3 — Wire It Into the Run Loop (Minimal Disturbance)
1. Instantiate once before the frame loop:
   - `trace_seal = TraceSealProjector(features=FEATURES, device=device, dtype=DTYPE, cfg=CFG)`

2. After `best_pred` is chosen each frame, compute vector residual:
   - `resid_vec = (frame - best_pred).detach()`

3. Compute TraceSeal error:
   - `err_ts = trace_seal.score(resid_vec)`
   - keep existing scalar error (whatever v0.4.7.02 uses now): `err_raw = float(best_err_t.item())`

4. Define the scalar used for EMA/MAD/z-score:
   - `mix = CFG["trace_seal_mix"]`
   - `best_err = mix * err_ts + (1.0 - mix) * err_raw`

5. Update TraceSeal covariance:
   - compute `best_err` and decide `is_surprise` first
   - then:
     - if `trace_seal_update_on_surprise` is False: `if not is_surprise: trace_seal.update(resid_vec)`
     - else: always update

6. Diagnostics logging:
   - every `trace_seal_diag_every`, print a compact line:
     - rank, top eigvals (top 5), explained ratio, mix, `err_raw` vs `err_ts`

---

### Step 4 — Recordkeeping / Artifacts
When enabled, extend step + anomaly payloads with:
- `best_err_raw`
- `best_err_ts`
- `trace_seal_rank`
- `trace_seal_top_eigs` (top 5 eigenvalues)
- `trace_seal_explained` (energy ratio captured by removed subspace)

Keep it compact.

---

### Step 5 — Optional “Soft Cutoff” (No SciPy)
Add config:

```python
"trace_seal_sigma_taper": "none",   # "none" | "hann"
```

If `"hann"`:
- When computing `sigma` from residual history, compute a **weighted std** using a Hann window.
- Keep existing MAD logic unchanged when taper is `"none"`.

This is a pragmatic stand-in for PSWF/Slepian-style “optimal concentration” behavior without external libs.

---

### Step 6 — Acceptance Tests (must run)
Run these sanity checks:

1. **SYNTHETIC** mode with `trace_seal_enabled=False`
   - Must run and match prior behavior (no new logs, no failures).

2. **SYNTHETIC** mode with `trace_seal_enabled=True`, `trace_seal_mix=1.0`
   - Must run; log shows stable eigenvalues after warmup.
   - Confirm `best_err_ts >= 0` always.

3. A small **ARCHIVE**/**KAGGLE** run capped (e.g., 5k frames)
   - Must run with no device mismatches.
   - Surprise logs include raw vs ts errors.

---

### Deliverables
1. Updated single file: `EIDOS_BRAIN_UNIFIED_v0_4.7.02.py` (or bump to `0.4.7.03` in banner if desired).
2. A short patch note comment block near the new class explaining:
   - Diagonal-Cutter = remove top covariance eigenmodes
   - Trace-Seal = anomaly energy computed as `r^T P r` (PSD, non-negative)
3. Confirmation snippets from the 3 acceptance tests.

---

## Prompt 2 — Demo/Visualization Add-On (Lightweight, No New Deps)

### Objective
Add a **tiny demo dashboard** that makes Trace-Seal explainable during live runs (customer/investor demos) without changing core detection logic.

### Constraints
- No new dependencies beyond what’s already in the file (`matplotlib` is OK only if already present; otherwise use CSV logging only).
- Must be optional (guarded by config).
- Must not slow the main loop noticeably.

### Add Config Knobs
Add:

```python
"demo_enable": False,
"demo_every": 25,                  # update cadence
"demo_window": 400,                # number of recent frames to display
"demo_write_csv": True,
"demo_plot_matplotlib": False,      # only if matplotlib already included
"demo_out_csv": "demo_trace_seal.csv",
```

### What to Record Each Frame (Rolling Buffer)
Maintain a ring buffer (lists/collections.deque) for the last `demo_window` frames:
- `t_idx`
- `err_raw`
- `err_ts`
- `best_err` (mixed)
- `z_score`
- `eff_z_thresh`
- `is_surprise`
- `trace_seal_explained` (0–1)
- `top_eig_1` (largest eigenvalue, optional)

### Output Option A (Always Safe): CSV Logger
If `demo_write_csv=True`:
- append every `demo_every` frames to `demo_out_csv`
- keep header stable

CSV columns (minimum):
- `t, err_raw, err_ts, best_err, z, z_th, surprise, explained`

### Output Option B (If Matplotlib Already Present): Live Plot
If `demo_plot_matplotlib=True` and matplotlib exists:
- Use a single figure with 3 lines:
  1) `err_raw` vs `err_ts` (same axis)
  2) `z_score` with horizontal line `eff_z_thresh`
  3) `trace_seal_explained` (secondary axis or separate plot area)

Rules:
- No subplots requirement is flexible here **only if you must**—but prefer a single chart with multiple lines.
- Update figure every `demo_every` frames.
- Keep it non-blocking (`plt.pause(0.001)`), and guard for headless runs.

### Console “Demo Line” (Minimal, Very Useful)
Every `demo_every` frames print one concise line:
- `t=..., raw=..., ts=..., z=.../th=..., expl=..., SURPRISE={0|1}`

### Acceptance Criteria
- Running with `demo_enable=False` changes nothing.
- With `demo_enable=True`, CSV file updates and console line prints.
- No significant slowdown.

---

## Notes for the Implementer (Important)
- The projector approach is meant to be **robust and explainable**: it removes the top predictable residual modes so “true novelty” remains.
- Use **short lists** for eigen diagnostics (top 5 only). Never dump matrices.
- Keep everything behind toggles so production behavior is stable until explicitly enabled.

