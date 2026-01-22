# Surprise Gate Spec

## Implementation
- Surprise score computed in `run_eidos_sentinel()` (engine v0.4.7.02).
- `ema_err` and `sigma` maintain baseline; `z_score` determines surprise.
- Adaptive quantile gate updates `z_thresh` each step to target a surprise rate.

## Definitions
- `err = ||x_t - \hat{x}_t||` (residual norm).
- `ema_err` updated as exponential moving average.
- `sigma` estimated from residuals (rolling std).
- `z = (err - ema_err) / (sigma + eps)`.
- Surprise if `z > z_thresh`.

## Gate Update
- `z_thresh = clip(z_thresh * exp(lr * (surprise_ema - target_surprise)))`.
- `surprise_ema` tracks recent surprise rate.

## Failure Modes
- Cold start: small sigma yields inflated z-scores.
- Extreme residuals can saturate z-threshold updates.
