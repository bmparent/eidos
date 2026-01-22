# Predictor Spec (Newtonian / Left Brain)

## Implementation
- Class: `NewtonianPredictor` in `repo/src/eidos_brain/engine/eidos_v0_4_7_02.py`.
- Uses a fixed-size window of past frames to compute a constant-velocity extrapolation.

## Definition
- Let `x_t` be the current frame.
- Approximate velocity `v_t = x_t - x_{t-1}`.
- Predict `x_{t+1} ≈ x_t + v_t`.

## Notes
- No explicit mass/inertia term is implemented; the "Newtonian" label refers to constant-velocity extrapolation.
- If history is insufficient, the predictor falls back to repeating the last frame.

## Failure Modes
- Rapidly changing signals produce high residuals (expected).
