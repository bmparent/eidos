# Normalization Spec

## Implementation
- Class: `OnlineVectorNormalizer` (`repo/src/eidos_brain/engine/eidos_v0_4_7_02.py`).
- Method: `update(x)` uses Welford per-dimension update.

## Definition (Welford)
Given vector `x_t`:
- `n_t = n_{t-1} + 1`
- `delta = x_t - mean_{t-1}`
- `mean_t = mean_{t-1} + delta / n_t`
- `M2_t = M2_{t-1} + delta * (x_t - mean_t)`
- `variance_t = M2_t / max(n_t-1, 1)`
- `std_t = sqrt(variance_t + eps)`
- Output: `z_t = (x_t - mean_t) / std_t`

## Behavior
- Per-dimension normalization.
- Warmup behavior: first sample sets mean and yields zero z-score.
- `eps` = 1e-8 guard to avoid divide-by-zero.

## Failure Modes
- NaN inputs propagate unless upstream sanitizes.
- Abrupt distribution shifts will spike z-score (expected).

## Notes
- Archive/tabular normalization uses dataset-level mean/std.
- Stream normalization is online via Welford.
