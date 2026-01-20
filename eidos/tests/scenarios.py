import numpy as np

def nominal(steps=100, features=64, seed=42):
    """Stable, low-noise sine wave pattern."""
    np.random.seed(seed)
    for i in range(steps):
        # Base signal: slow sine
        s = np.sin(i * 0.1) * 0.5
        # Low noise
        noise = np.random.normal(0, 0.05, features)
        yield (np.ones(features) * s + noise).astype(np.float32), {"idx": i, "kind": "nominal"}

def spike(steps=100, features=64, spike_idx=50, magnitude=100.0, seed=42):
    """Single extreme outlier at spike_idx."""
    np.random.seed(seed)
    for i in range(steps):
        vec = np.random.normal(0, 0.1, features)
        if i == spike_idx:
            vec += magnitude # Giant spike
        yield vec.astype(np.float32), {"idx": i, "kind": "spike" if i == spike_idx else "nominal"}

def shift(steps=100, features=64, shift_start=30, shift_val=2.0, seed=42):
    """Mean shift starting at shift_start."""
    np.random.seed(seed)
    for i in range(steps):
        vec = np.random.normal(0, 0.1, features)
        if i >= shift_start:
            vec += shift_val
        yield vec.astype(np.float32), {"idx": i, "kind": "shift" if i >= shift_start else "nominal"}

def burst(steps=100, features=64, burst_start=40, burst_len=20, var_mult=5.0, seed=42):
    """High variance window."""
    np.random.seed(seed)
    for i in range(steps):
        scale = 0.1
        if burst_start <= i < burst_start + burst_len:
            scale *= var_mult
        vec = np.random.normal(0, scale, features)
        yield vec.astype(np.float32), {"idx": i, "kind": "burst"}

def nan_inf(steps=100, features=64, bad_idx=50, type="nan", seed=42):
    """Injects NaN or Inf."""
    np.random.seed(seed)
    for i in range(steps):
        vec = np.random.normal(0, 0.1, features)
        if i == bad_idx:
            if type == "nan":
                vec[0] = np.nan
            else:
                vec[0] = np.inf
        yield vec.astype(np.float32), {"idx": i, "kind": "bad"}
