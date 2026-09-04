"""Benchmark-only frozen reservoir recurrence; production engine is unchanged."""
from __future__ import annotations

import copy
import importlib.util
import math
import os
from pathlib import Path
from fractions import Fraction as F
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "eidos/repo/src/eidos_brain/engine/eidos_v0_4_7_02.py"
POLICIES = ("none", "every_step", "pulse100", "carry")
CONFIGS = ("default", "slow", "bands")
DELTA = 1e-5


def quantize(x):
    """Match orch_or_collapse: nearest, binary half ties to even, no saturation."""
    return np.rint(x * 100000.0) / 100000.0


class Policy:
    def __init__(self, initial, name):
        if name not in POLICIES:
            raise ValueError(name)
        self.name = name
        self.reset(initial)

    def reset(self, initial):
        self.state = np.array(initial, copy=True)
        self.carry = np.zeros_like(self.state) if self.name == "carry" else None
        self.counter = 0

    def apply(self, proposed):
        self.counter += 1
        if self.name == "carry":
            z = proposed + self.carry
            self.state = quantize(z)
            self.carry = z - self.state  # signed residual, added at the next update
        elif self.name == "every_step" or (self.name == "pulse100" and self.counter % 100 == 0):
            self.state = quantize(proposed)
        else:
            self.state = proposed
        return self.state


def recurrence(state, forcing, W, alpha):
    return (1.0 - alpha) * state + alpha * np.tanh(forcing + W @ state)


def load_engine(artifact_root):
    """Load the real file, redirect its import-time artifacts, use CPU only here."""
    import torch
    Path(artifact_root).mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("controlled_memory_engine", ENGINE)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, {"EIDOS_ARTIFACT_ROOT": str(Path(artifact_root).resolve())}):
        spec.loader.exec_module(module)
    module._require_torch()
    module.device = torch.device("cpu")
    return module


def make_reservoir(module, config, n=16):
    import torch
    saved = copy.deepcopy(module.EIDOS_BRAIN_CONFIG)
    try:
        module.EIDOS_BRAIN_CONFIG.update(fractal_bands=4 if config == "bands" else 1,
                                        leak_rate_base=0.5, leak_q=10.0, thermo_enabled=False)
        # Preserve the engine's native float32 coefficient generation and fixed seed 42.
        with torch.random.fork_rng():
            r = module.RLS_Reservoir(3, n_reservoir=n, leak_rate=0.0005 if config == "slow" else 0.01)
    finally:
        module.EIDOS_BRAIN_CONFIG.clear()
        module.EIDOS_BRAIN_CONFIG.update(saved)
    return r


def frozen_arrays(r):
    return (r.W_res.detach().cpu().numpy().astype(np.float64),
            r.W_in.detach().cpu().numpy().astype(np.float64),
            np.broadcast_to(r.alpha.detach().cpu().numpy(), r.state.shape).astype(np.float64).copy())


def adapter_check(module, n=16):
    """Real listen path, including common nonzero noise, pulse timing, and reset."""
    import torch
    rows = []
    for config in CONFIGS:
        for dtype in (torch.float64, torch.float32) if config != "bands" else (torch.float64,):
            r = make_reservoir(module, config, n)
            r.W_res, r.W_in, r.state, r.alpha = [x.to(dtype=dtype) for x in (r.W_res, r.W_in, r.state, r.alpha)]
            W, B, alpha = [x.detach().numpy().copy() for x in (r.W_res, r.W_in, r.alpha)]
            rng = np.random.default_rng(20260904)
            inputs = rng.uniform(-0.8, 0.8, (202, 3)).astype(W.dtype)
            initial = rng.uniform(-0.1, 0.1, n).astype(W.dtype)
            draws = rng.normal(size=(202, n)).astype(W.dtype)
            r.thermo_enabled, r.temperature = True, 0.03
            paths = []
            for _ in range(2):
                r.state = torch.from_numpy(initial.copy())
                r._listen_step = 0
                p = Policy(initial, "pulse100")
                errors = []
                path = []
                for t, u in enumerate(inputs):
                    draw = torch.from_numpy(draws[t].copy())
                    with patch.object(module.torch, "randn_like", return_value=draw):
                        r.listen(torch.from_numpy(u.copy()))
                    noise = (r.temperature / math.sqrt(n)) * draws[t]
                    proposal = (1.0 - alpha) * p.state + alpha * np.tanh(B @ u + W @ p.state + noise)
                    p.apply(proposal)
                    errors.append(float(np.max(np.abs(r.state.numpy() - p.state))))
                    path.append(r.state.numpy().copy())
                paths.append(np.array(path))
                tol = 3e-6 if dtype == torch.float32 else 5e-13
                if max(errors) > tol:
                    raise AssertionError((config, str(dtype), max(errors), tol))
            assert np.array_equal(*paths)
            rows.append(dict(config=config, dtype=str(dtype), steps=202, repeats=2,
                             max_abs_error=max(errors), tolerance=tol,
                             boundary_errors={str(t): errors[t-1] for t in (99, 100, 101, 199, 200, 201)},
                             reset_bitwise_equal=True, noise="matched nonzero draws"))
    # Test exact torch operation wiring separately from NumPy backend differences.
    return rows


def round_fraction(x):
    q, rem = divmod(x.numerator, x.denominator)
    return q + int(2*rem > x.denominator or (2*rem == x.denominator and q % 2))


def exact_controls():
    ties = [round_fraction(F(k, 2)) for k in (-5, -3, -1, 1, 3, 5)]
    assert ties == [-2, -2, 0, 0, 2, 2]
    scalar = []
    for a, m, count, minimum in ((F(1, 2000), 100, 21, 1386), (F(1, 100), 100, 1, 69),
                               (F(1, 100), 1, 101, 69), (F(1, 2000), 1, 2001, 1386)):
        beta = (1-a)**m
        limit = math.ceil(1/(2*(1-beta))) + 2
        fixed = [k for k in range(-limit, limit+1) if round_fraction(beta*k) == k]
        assert len(fixed) == count
        assert (1-a)**minimum <= F(1, 2) < (1-a)**(minimum-1)
        scalar.append(dict(alpha=str(a), interval=m, fixed_count=count, minimum_interval=minimum,
                           fixed_min=min(fixed), fixed_max=max(fixed)))
    for k in range(-10, 11):
        assert round_fraction(F(k)) == k  # zero leak, excluded
        assert round_fraction(F(0)*k) == 0  # unit leak
    r, c = F(1), F(0)
    cycle = []
    for _ in range(4):
        cycle.append([str(r), str(c)])
        z = max(-1, min(1, -F(3, 4)*r)) + c
        new = F(round_fraction(z))
        c, r = z-new, new
    assert cycle == [["1", "0"], ["-1", "1/4"], ["1", "0"], ["-1", "1/4"]]
    # Exact nonnormal coupled clipped-linear control. K=[[0,2],[0,0]], h=(3,1).
    # (I-K)^-1 Delta = (3*Delta,Delta); heterogeneous A does not commute with K.
    alpha, delta = (F(1, 100), F(1, 2)), F(1, 100)
    bound = (3*delta, delta)
    ordinary_bound = (delta/(2*alpha[0])+delta/alpha[1], delta/(2*alpha[1]))
    x = r = p = (F(1, 10), F(-1, 10))
    c = (F(0), F(0))
    checks = 0
    for t in range(128):
        u = (F((-1)**t, 3), F((t % 3)-1, 4))
        def step(v):
            f = (max(-1, min(1, 2*v[1]+u[0])), max(-1, min(1, u[1])))
            return tuple((1-alpha[i])*v[i]+alpha[i]*f[i] for i in range(2))
        x = step(x)
        z = tuple(v+c[i] for i, v in enumerate(step(r)))
        r = tuple(delta*round_fraction(v/delta) for v in z)
        c = tuple(z[i]-r[i] for i in range(2))
        p = tuple(delta*round_fraction(v/delta) for v in step(p))
        for i in range(2):
            assert abs(r[i]-x[i]) <= bound[i]
            assert abs(p[i]-x[i]) <= ordinary_bound[i]
            assert abs(c[i]) <= delta/2
            checks += 3
    return dict(status="passed", scalar=scalar, half_ties=ties, signed_cycle=cycle,
                nonnormal=dict(K=[[0, 2], [0, 0]], h=[3, 1], margin=[1, 1],
                               exact_inequalities=checks),
                scope="Finite exact controls; analytic proofs are in the preserved source report.")


def certificate(W):
    """Float candidate h, then exact rational verification of binary coefficients."""
    K = np.abs(W)
    n = len(W)
    try:
        h = np.linalg.solve(np.eye(n)-K, np.ones(n))
    except np.linalg.LinAlgError:
        h = np.ones(n)
    if not np.all(np.isfinite(h)) or np.any(h <= 0):
        h = np.ones(n)
    hf = [F(float(v)) for v in h]
    margins = [hf[i] - sum((F(float(K[i,j]))*hf[j] for j in range(n)), F(0)) for i in range(n)]
    passed = all(v > 0 for v in margins)
    # A positive lower bound K*1>1 certifies failure of rho(K)<1, not engine instability.
    lower = min(sum((F(float(v)) for v in row), F(0)) for row in K)
    return dict(status="certified" if passed else "outside_sufficient_condition" if lower > 1 else "uncertified",
                h=[str(v) for v in hf], exact_margins=[str(v) for v in margins],
                exact_min_row_sum=str(lower), rho_abs_float_descriptive=float(max(abs(np.linalg.eigvals(K)))),
                rho_W_float_descriptive=float(max(abs(np.linalg.eigvals(W)))),
                note="Exact rational arithmetic certifies binary stored coefficients. Failure is not proof of instability.")
