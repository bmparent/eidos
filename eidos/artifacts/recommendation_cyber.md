# Domain tuning report: cyber

Best objective score: **0.8179**

## Summary metrics
- **surprise_rate**: 0.6923076923076923
- **z_var**: 0.7056205013440179
- **red_rate**: 0.0
- **max_red_burst**: 0
- **final_z_thresh**: 1.0767176902563804
- **final_sigma**: 1.8088721033608193
- **roc_auc**: 0.0
- **pr_auc**: 0.0
- **recall**: 0.0
- **fpr**: 0.0
- **mean_delay**: 0.0
- **delay_norm**: 0.0
- **score**: 0.8179290261000853
- **dataset**: incident_small

## Config deltas
- `sigma_k`: 1.5 → 1.1144624340609357
- `target_surprise`: 0.15 → 0.2515821342928252
- `ema_alpha`: 0.001 → 0.002472841541484021
- `warmup_cap`: 2000 → 2963
- `spectral_radius`: 1.27 → 1.775788463196263
- `leak_rate`: 0.01 → 0.3692479352052168
- `input_scaling`: 0.3 → 0.4507843245619566
- `forgetting`: 0.99 → 0.9532423010011892
- `weight_decay`: 0.0005 → 5.302842829460644e-05
- `reservoir`: 2000 → 1900
- `hippocampus_sim_theta`: 0.1 → 0.4396149980604702
- `hippocampus_write_z_thresh`: 4.0 → 3.009347590260939
- `hippocampus_freeze_strength`: 0.75 → 0.4857058498614038
- `hippocampus_compute_on_surprise_only`: True → False

## Parameter importance (ablation)
- `sigma_k`: score drop 0.0000
- `target_surprise`: score drop 0.0000
- `ema_alpha`: score drop 0.0000
- `warmup_cap`: score drop 0.0000
- `spectral_radius`: score drop 0.0000
- `leak_rate`: score drop 0.0000
- `input_scaling`: score drop 0.0000
- `forgetting`: score drop 0.0000