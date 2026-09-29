import json
from pathlib import Path

import numpy as np
import pytest

from proof.memory_core import (Policy, adapter_check, certificate, exact_controls, load_engine,
                               quantize, recurrence)


def test_exact_controls():
    r = exact_controls()
    assert r["scalar"][0]["fixed_count"] == 21
    assert r["nonnormal"]["exact_inequalities"] == 768


def test_positive_negative_half_ties():
    np.testing.assert_array_equal(quantize(np.array([-2.5, -1.5, -.5, .5, 1.5, 2.5])*1e-5),
                                  np.array([-2, -2, 0, 0, 2, 2])/100000)


def test_pulse_boundary_and_reset():
    p = Policy([0.0], "pulse100")
    v = np.array([0.1234567])
    for t in range(1, 202):
        np.testing.assert_array_equal(p.apply(v), quantize(v) if t % 100 == 0 else v)
    p.reset([1.0])
    assert p.counter == 0 and p.carry is None


def test_signed_carry_and_reset():
    p = Policy(np.array([1.0]), "carry")
    # Delta=1e-5 version of the exact signed cycle, clipped linear control.
    p.reset(np.array([1e-5]))
    for expected in (-1e-5, 1e-5, -1e-5, 1e-5):
        p.apply(-.75*p.state)
        assert p.state[0] == expected
    p.reset(np.array([0.0]))
    assert p.counter == 0 and p.carry[0] == 0
    assert p.apply(np.array([0.0]))[0] == 0


def test_state_only_policy_wiring():
    W = np.array([[.2]])
    alpha = np.array([.01])
    policies = {name: Policy(np.array([.123456789]), name) for name in ("none", "every_step", "pulse100", "carry")}
    proposal = recurrence(policies['none'].state, np.array([.111111111]), W, alpha)
    for p in policies.values():
        p.apply(proposal.copy())
    assert policies['none'].state[0] == policies['pulse100'].state[0]
    assert policies['every_step'].state[0] == policies['carry'].state[0]
    assert policies['none'].state[0] != policies['every_step'].state[0]
    np.testing.assert_allclose(policies['carry'].state+policies['carry'].carry, proposal, rtol=0, atol=0)
    with pytest.raises(ValueError):
        Policy([0.0], 'wrong')


def test_certificate_nonnormal_and_rejection():
    assert certificate(np.array([[0., 2.], [0., 0.]]))['status'] == 'certified'
    assert certificate(np.array([[1.1, 0.], [0., 1.1]]))['status'] == 'outside_sufficient_condition'


def test_real_listen_adapter(tmp_path):
    rows = adapter_check(load_engine(tmp_path), n=8)
    assert len(rows) == 5
    assert all(row['reset_bitwise_equal'] for row in rows)


def test_freeze_rejects_modified_protocol_and_inputs(tmp_path, monkeypatch):
    from proof import memory_benchmark as b
    (tmp_path/'inputs').mkdir()
    (tmp_path/'inputs/a').write_text('original')
    b.save(tmp_path/'protocol.json', {'n':8})
    b.save(tmp_path/'freeze.json', {'protocol_sha256':b.sha(tmp_path/'protocol.json'),
                                   'source_hashes':{},'inputs':{'a':b.sha(tmp_path/'inputs/a')}})
    assert b.check_freeze(tmp_path)['n'] == 8
    (tmp_path/'inputs/a').write_text('tampered')
    with pytest.raises(ValueError, match='input changed'):
        b.check_freeze(tmp_path)
    (tmp_path/'protocol.json').write_text('{}')
    with pytest.raises(ValueError, match='protocol changed'):
        b.check_freeze(tmp_path)


def test_fidelity_metrics_and_precision(tmp_path):
    from proof.memory_benchmark import mp_replay, replay
    W, alpha = np.array([[0.]]), np.array([.0005])
    forcing, initial = np.zeros((300,1)), np.array([1e-4])
    exact = initial * (1-alpha[0])**np.arange(301)[:,None]
    ref,_ = replay(W,alpha,forcing,initial,'none')
    hp = mp_replay(W,alpha,forcing,initial)
    np.testing.assert_allclose(ref,hp,atol=1e-17,rtol=0)
    np.testing.assert_allclose(ref,exact,atol=1e-17,rtol=0)
    pulsed,_ = replay(W,alpha,forcing,initial,'pulse100')
    np.testing.assert_array_equal(pulsed[[100,200,300]],np.full((3,1),1e-4))
    carried, receipt = replay(W,alpha,forcing,initial,'carry',collect_storage=True)
    assert receipt['storage_samples'] > 0
    assert np.max(abs(carried-ref)) < 1e-5


def test_benchmark_preserves_deadline_failure():
    from proof.memory_benchmark import replay
    with pytest.raises(TimeoutError):
        replay(np.zeros((1,1)),np.ones(1)*.01,np.zeros((2,1)),np.ones(1),'none',deadline=1)
