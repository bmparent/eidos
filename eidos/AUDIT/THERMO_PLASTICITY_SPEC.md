# Thermodynamics / Plasticity Spec

## Implementation
- `update_thermodynamics()` in reservoir class updates forgetting/temperature.
- `controller_update()` scales learning rate based on surprise EMA + fatigue.

## Definition
- Energy proxy: surprise-driven residual metrics (`err`, `ema_err`).
- Temperature and rho modulate forgetting and learning rate scaling.
- Plasticity is clipped by `plasticity_min_scale` / `plasticity_max_scale`.

## Outputs
- `thermo_energy`, `thermo_rho`, `thermo_temp`, `thermo_lambda` logged per step.

## Failure Modes
- Aggressive plasticity scaling can stall learning (min clamp).
- High surprise rate drives cooldown (RED/AMBER states).
