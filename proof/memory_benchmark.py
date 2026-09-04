"""Controlled memory experiment. Run every command from the repository root.

python -m proof.memory_benchmark calibrate --out artifacts/memory_calibration
python -m proof.memory_benchmark prepare --out artifacts/memory_run --calibration artifacts/memory_calibration
python -m proof.memory_benchmark run --out artifacts/memory_run
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

from .memory_core import (CONFIGS, DELTA, ENGINE, POLICIES, ROOT, Policy, adapter_check,
                          certificate, exact_controls, frozen_arrays, load_engine,
                          make_reservoir, quantize, recurrence)

SOURCES = [ROOT / 'proof/memory_core.py', Path(__file__), ROOT / 'tests/test_memory_benchmark.py', ENGINE]
PROJECT_STREAM = ROOT / 'eidos/verify_data/incident_test_data.csv'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def mp_replay(W, alpha, forcing, initial, dps=50, deadline=None):
    """Increased precision for identical binary coefficients and frozen forcing."""
    import mpmath as mp
    with mp.workdps(dps):
        wm = [[mp.mpf(float(v)) for v in row] for row in W]
        am = [mp.mpf(float(v)) for v in alpha]
        state = [mp.mpf(float(v)) for v in initial]
        n = len(state)
        result = np.empty((len(forcing)+1, n))
        result[0] = initial
        for t, b in enumerate(forcing):
            state = [(1-am[i])*state[i] + am[i]*mp.tanh(mp.mpf(float(b[i])) +
                       mp.fsum(wm[i][j]*state[j] for j in range(n))) for i in range(n)]
            result[t+1] = [float(v) for v in state]
            if t % 256 == 0 and deadline and time.monotonic() > deadline:
                raise TimeoutError('main benchmark cap reached during increased precision')
    return result


def calibrate(out):
    out.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(1)
    module = load_engine(out/'engine_import')
    controls = exact_controls()
    equivalence = adapter_check(module)
    timings = []
    for n in (16, 8):
        W, B, a = frozen_arrays(make_reservoir(module, 'bands', n))
        rng = np.random.default_rng(7)
        forcing = rng.normal(0, .1, (1000, n))
        x = np.zeros(n)
        start = time.perf_counter()
        for f in forcing:
            x = recurrence(x, f, W, a)
        ordinary_seconds = time.perf_counter()-start
        start = time.perf_counter()
        mp_replay(W, a, forcing[:64], np.zeros(n))
        mp_seconds = time.perf_counter()-start
        estimated_hp_seconds = mp_seconds/64*44100*6
        timings.append(dict(n=n, float64_1000_seconds=ordinary_seconds,
                            mp50_64_seconds=mp_seconds, estimated_six_full_precision_checks_seconds=estimated_hp_seconds))
        if estimated_hp_seconds < 900:
            break
    chosen = timings[-1]['n']
    save(out/'calibration.json', dict(utc=stamp(), controls=controls, adapter=equivalence,
                                    timing=timings, chosen_n=chosen, decision_rule='Try 16 then 8; prefer estimated precision checks below 900 seconds.',
                                    scope='Timing and known controls only; no candidate comparative outcomes inspected.'))
    print(json.dumps(dict(calibration=str(out), chosen_n=chosen, timing=timings)), flush=True)


def stream_inputs(seed, tail):
    rng = np.random.default_rng(seed)
    t = np.arange(4097)
    driven = np.column_stack((.5*np.sin(t*.031), .4*np.cos(t*.017), .3*np.sin(t*.011+.4)))
    driven += rng.uniform(-.05, .05, driven.shape)
    project = np.loadtxt(PROJECT_STREAM, delimiter=',', skiprows=1)
    mean, scale = project[:400].mean(axis=0), project[:400].std(axis=0)
    scale = np.maximum(scale, 1e-6)
    project = np.clip((project-mean)/scale, -3, 3)/3
    return dict(zero=np.zeros((tail, 3)), impulse=np.vstack(([1., -.5, .25], np.zeros((tail, 3)))),
                driven=np.vstack((driven, np.zeros((tail, 3)))),
                project=np.vstack((project, np.zeros((tail, 3))))), dict(project_mean=mean.tolist(), project_scale=scale.tolist(),
                    project_transform='clip((u-mean)/std, -3,3)/3; fitted only on first 400 rows', project_rows=len(project))


def fit_readout(W, B, a, u, prefix):
    x = np.zeros(len(W))
    states = []
    for frame in u[:prefix-1]:
        x = recurrence(x, B@frame, W, a)
        states.append(x.copy())
    design = np.column_stack((states, np.ones(len(states))))
    ridge = 1e-4
    # Fixed ridge, no tuning. Predictions follow listen(u_t) and target u_(t+1).
    readout = np.linalg.solve(design.T@design + ridge*np.eye(design.shape[1]), design.T@u[1:prefix])
    return readout, float(np.sqrt(np.mean(np.square(states))))


def prepare(out, calibration):
    out.mkdir(parents=True, exist_ok=False)
    (out/'inputs').mkdir()
    calpath = calibration/'calibration.json'
    cal = json.loads(calpath.read_text())
    import torch
    torch.set_num_threads(1)
    module = load_engine(out/'engine_import')
    n = cal['chosen_n']
    save(out/'adapter_selected_size.json', adapter_check(module, n))
    matrices = {cfg: frozen_arrays(make_reservoir(module, cfg, n)) for cfg in CONFIGS}
    slowest_tau = max(-1/math.log1p(-float(a.min())) for _, _, a in matrices.values())
    tail = math.ceil(20*slowest_tau)
    configs = {}
    for cfg, (W, B, a) in matrices.items():
        np.savez(out/f'inputs/{cfg}.npz', W=W, B=B, alpha=a)
        configs[cfg] = dict(alpha=a.tolist(), certificate=certificate(W),
                           matrix_sha256=sha(out/f'inputs/{cfg}.npz'), native_coefficient_dtype='float32 lifted exactly to float64; band alpha native float64')
    trials = []
    for seed in (7, 19):
        streams, normalization = stream_inputs(seed, tail)
        save(out/f'inputs/normalization_{seed}.json', normalization)
        for cfg, (W, B, a) in matrices.items():
            initials = np.stack((np.zeros(n), np.random.default_rng(seed+1000).uniform(-.1, .1, n)))
            readouts = {}
            scales = {}
            for source, prefix in (('driven', 2048), ('project', 400)):
                readouts[source], scales[source] = fit_readout(W, B, a, streams[source], prefix)
            for stream, u in streams.items():
                key = f'{cfg}_{stream}_{seed}'
                source = 'project' if stream == 'project' else 'driven'
                R, scale = readouts[source], scales[source]
                forcing = u @ B.T  # frozen once, identical forcing for every policy/precision
                np.savez_compressed(out/f'inputs/{key}.npz', inputs=u, forcing=forcing, initials=initials, readout=R)
                end = 4096 if stream == 'driven' else normalization['project_rows']-1 if stream == 'project' else 0
                start = 2112 if stream == 'driven' else 464 if stream == 'project' else 0
                trials.append(dict(key=key, config=cfg, stream=stream, seed=seed, horizon=len(u),
                                   input_sha256=sha(out/f'inputs/{key}.npz'), initial_l2=np.linalg.norm(initials, axis=1).tolist(),
                                   state_scale=max(scale, 1e-6), state_scale_source=f'{source} no-rounding exploration-prefix RMS, floor 1e-6',
                                   task_start=start, task_stop=end, block_length=256 if stream == 'driven' else 64,
                                   readout_fit_prefix=400 if source == 'project' else 2048))
    protocol = dict(schema=1, frozen_utc=stamp(), starting_commit=git('rev-parse', 'HEAD'), historical_commit='6bb0b980349f94c55a6ca1ca9665737570c0fd01',
                    source_hashes={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in SOURCES},
                    project_stream=dict(path=str(PROJECT_STREAM.relative_to(ROOT)).replace('\\','/'), sha256=sha(PROJECT_STREAM),
                                        labels='unlabeled synthetic project fixture; no clinical/operational validity'),
                    calibration_sha256=sha(calpath), adapter_selected_size_sha256=sha(out/'adapter_selected_size.json'),
                    n_reservoir=n, n_inputs=3, weight_seed=42, forcing_initialization_seeds=[7,19],
                    policies=list(POLICIES), actual_current_policy='pulse100', configurations=configs, trials=trials,
                    delta=DELTA, state_dtype='float64', carry_dtype='float64', reference='float64 no explicit grid; selected entire trajectories checked at mpmath 50 decimal digits',
                    precision_cases=[f'{cfg}_{stream}_7' for cfg in CONFIGS for stream in ('zero','driven')],
                    precision_initialization={'zero':1,'driven':0}, precision_acceptance_max_abs=1e-9,
                    precision_scope='Same frozen binary matrices, alpha and precomputed forcing; checks recurrence arithmetic, not coefficient-generation truth.',
                    quantizer='numpy.rint(x*100000)/100000; nearest binary half ties to even; no saturation',
                    pulse_phase='counter zero at initialization; first pulse after update 100; all arms aligned',
                    carry_reset='zero at each trajectory and reset; residual z-r, added at next update',
                    feedback='No adapt, thermostat, hippocampus, Sentinel, or state-dependent input feedback. Frozen W and readout.',
                    noise='zero in numerical arms (thermodynamics off); nonzero matched noise tested in real-listen adapter calibration',
                    input_rounding='none in every arm; old input-rounding algorithm is not reproduced',
                    prediction='raw frozen affine readout and additional engine-style rounded dream; next-frame target',
                    readout='ridge 1e-4, no tuning; fitted only on exploration prefix of corresponding source, frozen before policy replay; same for both initializations',
                    chronology='Replay all inputs from declared initial state; training prefix then gap 64 then suffix. No resets at split. Targets only next frame. Gap exceeds one-frame target overlap.',
                    temporal_dependence='Contiguous prespecified blocks; report block effects and ranges, no independent-frame p values or confidence claims',
                    tail_steps=tail, largest_scalar_relaxation_time=slowest_tau, tail_multiple=20,
                    tail_note='Zero-input tail after excitation (or entire zero stream). Diagnostic minimum, not coupled mixing-time guarantee.',
                    metrics=['state RMS/Linf discrepancy', 'init L2 separation', 'tail pulse plateau/cycle', 'carry magnitude and exact binary subtraction storage error samples',
                             'raw and rounded prediction difference', 'suffix MSE and block deltas', 'runtime ratio', 'persistent array bytes', 'descriptive covariance geometry'],
                    storage_error_sampling='Every 100th step and first 101; exact Fraction(z)-Fraction(r)-Fraction(c); rounding in F and adding c are not included',
                    practical_task_effect_threshold=None, acceptable_overhead_threshold=None,
                    threshold_reason='No defensible project task-effect or overhead requirement exists for this unlabeled fixture / synthetic prediction task.',
                    adoption_decision_rule='Always inconclusive for adoption without a prespecified utility gate; numerical fidelity cannot pass it.',
                    utility_test_status='untested; suffix task errors are exploratory descriptive matched comparisons, not a validated utility experiment',
                    main_cap_seconds=3600, first_run_target_seconds=1800, partial_policy='Write each completed trial; retain failures and completed outputs at cap; never shorten horizon',
                    overhead='Three timing repeats, rotated deterministic policy order, per policy complete two-initialization trajectories; persistent arrays exclude common inputs/readout and offline trace storage')
    save(out/'protocol.json', protocol)
    input_hashes = {p.name:sha(p) for p in sorted((out/'inputs').iterdir())}
    save(out/'freeze.json', dict(utc=stamp(), protocol_sha256=sha(out/'protocol.json'), inputs=input_hashes,
                               source_hashes=protocol['source_hashes']))
    (out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True) +
                                     f'\nPython {sys.version}\nPlatform {platform.platform()}\nNumPy {np.__version__}\n',encoding='utf-8')
    (out/'git_commit.txt').write_text(git('rev-parse','HEAD')+'\n')
    save(out/'run_manifest.json', dict(status='frozen_not_evaluated', utc=stamp(), protocol_sha256=sha(out/'protocol.json')))
    print(json.dumps(dict(frozen=str(out), n=n, trials=len(trials), trajectories=len(trials)*8, tail_steps=tail,
                         state_updates=sum(t['horizon']*8 for t in trials))), flush=True)


def check_freeze(out):
    freeze = json.loads((out/'freeze.json').read_text())
    if sha(out/'protocol.json') != freeze['protocol_sha256']:
        raise ValueError('protocol changed after freeze')
    for name, digest in freeze['source_hashes'].items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'evaluator/source changed after freeze: {name}')
    for name, digest in freeze['inputs'].items():
        if sha(out/'inputs'/name) != digest:
            raise ValueError(f'input changed after freeze: {name}')
    return json.loads((out/'protocol.json').read_text())


def replay(W, a, forcing, initial, name, deadline=None, collect_storage=False):
    p = Policy(initial, name)
    states = np.empty((len(forcing)+1, len(initial)))
    states[0] = initial
    carry_max = 0.
    storage_max = Fraction(0)
    storage_samples = 0
    for t, f in enumerate(forcing):
        proposed = recurrence(p.state, f, W, a)
        z = proposed + p.carry if name == 'carry' else None
        states[t+1] = p.apply(proposed)
        if collect_storage and name == 'carry':
            carry_max = max(carry_max, float(np.max(np.abs(p.carry))))
            if t < 101 or (t+1) % 100 == 0:
                for zi, ri, ci in zip(z, p.state, p.carry):
                    err = abs(Fraction(float(zi))-Fraction(float(ri))-Fraction(float(ci)))
                    storage_max = max(storage_max, err)
                    storage_samples += 1
        if t % 256 == 0 and deadline and time.monotonic() > deadline:
            raise TimeoutError('main benchmark cap reached')
    return states, dict(max_carry_abs=carry_max, max_sampled_storage_error=float(storage_max),
                        exact_max_sampled_storage_error=str(storage_max), storage_samples=storage_samples)


def geometry(states):
    centered = states-states.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    power = singular**2
    if power.sum() < 1e-30:
        return dict(dominance=None, entropy=None, reason='centered window energy below 1e-30')
    p = power/power.sum()
    positive = p[p>0]
    return dict(dominance=float(p.max()), entropy=float(-np.sum(positive*np.log(positive))), reason='descriptive only')


def run_trial(out, trial, protocol, deadline):
    key, cfg = trial['key'], trial['config']
    matrices = np.load(out/f'inputs/{cfg}.npz')
    W, a = matrices['W'], matrices['alpha']
    data = np.load(out/f'inputs/{key}.npz')
    forcing, initials, u, readout = [data[v] for v in ('forcing','initials','inputs','readout')]
    traces, storage = {}, {}
    for name in POLICIES:
        paths, receipts = zip(*(replay(W,a,forcing,initial,name,deadline,True) for initial in initials))
        traces[name] = np.array(paths)
        storage[name] = receipts
    # Run-only overhead, no exact Fraction audit; the same full trajectory storage in every arm.
    timings = {name:[] for name in POLICIES}
    for repeat in range(3):
        order = list(POLICIES)[repeat:] + list(POLICIES)[:repeat]
        for name in order:
            start = time.perf_counter()
            for initial in initials:
                replay(W,a,forcing,initial,name,deadline)
            timings[name].append(time.perf_counter()-start)
    np.savez_compressed(out/f'raw/{key}.npz', **traces)
    metrics, curves, blocks = [], [], []
    reference = traces['none']
    for name in POLICIES:
        paths = traces[name]
        separation = np.linalg.norm(paths[1]-paths[0],axis=1)
        for init, states in enumerate(paths):
            errors = states-reference[init]
            refpred = reference[init,1:] @ readout[:-1]+readout[-1]
            pred = states[1:] @ readout[:-1]+readout[-1]
            start, stop = trial['task_start'], trial['task_stop']
            mse = mse_ref = rounded_mse = None
            if stop > start:
                target = u[start+1:stop+1]
                sq = np.mean((pred[start:stop]-target)**2,axis=1)
                sqref = np.mean((refpred[start:stop]-target)**2,axis=1)
                mse, mse_ref = float(sq.mean()), float(sqref.mean())
                rounded_mse = float(np.mean((quantize(pred[start:stop])-target)**2))
                for offset in range(0,stop-start,trial['block_length']):
                    sl = slice(offset,offset+trial['block_length'])
                    blocks.append(dict(trial=key,policy=name,initialization=init,block_start=start+offset,
                                       frames=len(sq[sl]),mse=float(sq[sl].mean()),reference_mse=float(sqref[sl].mean()),
                                       difference=float((sq[sl]-sqref[sl]).mean())))
            # Same phase of last ten pulses; intermediate values are preserved in curves/raw traces.
            pulse_indices = np.arange(100, len(states), 100)[-10:]
            pulse_states = states[pulse_indices]
            pulse_change = float(np.max(np.abs(np.diff(pulse_states,axis=0)))) if len(pulse_states)>1 else None
            row = dict(trial=key,config=cfg,stream=trial['stream'],seed=trial['seed'],policy=name,initialization=init,
                       frames=trial['horizon'],state_scale=trial['state_scale'],
                       discrepancy_rms=float(np.sqrt(np.mean(errors**2))),discrepancy_max_abs=float(np.max(np.abs(errors))),
                       discrepancy_normalized_rms=float(np.sqrt(np.mean(errors**2))/trial['state_scale']),
                       final_state_l2=float(np.linalg.norm(states[-1])), final_init_separation_l2=float(separation[-1]),
                       tail_pulse_max_change=pulse_change, pulse_phase_exactly_constant=bool(np.array_equal(pulse_states, np.broadcast_to(pulse_states[-1],pulse_states.shape))),
                       prediction_max_abs_difference=float(np.max(np.abs(pred-refpred))),
                       rounded_prediction_max_abs_difference=float(np.max(np.abs(quantize(pred)-quantize(refpred)))),
                       task_mse=mse, reference_task_mse=mse_ref, task_mse_delta_from_reference=mse-mse_ref if mse is not None else None,
                       rounded_task_mse=rounded_mse, seconds_median=float(np.median(timings[name])),
                       runtime_ratio_to_current=float(np.median(timings[name])/np.median(timings['pulse100'])),
                       persistent_state_bytes=int(len(a)*8*(2 if name=='carry' else 1)),
                       common_matrix_bytes=int(sum(matrices[v].nbytes for v in ('W','B','alpha'))),
                       geometry=geometry(states[-1024:]), **storage[name][init])
            metrics.append(row)
            indices = sorted(set([0,1,99,100,101,len(states)-1] + list(range(0,len(states),100)) + list(range(max(0,len(states)-201),len(states)))))
            for t in indices:
                curves.append(dict(trial=key,config=cfg,stream=trial['stream'],seed=trial['seed'],policy=name,initialization=init,step=t,
                                   state_l2=float(np.linalg.norm(states[t])),init_separation_l2=float(separation[t]),
                                   discrepancy_l2=float(np.linalg.norm(errors[t])),discrepancy_rms=float(np.sqrt(np.mean(errors[t]**2)))))
    save(out/f'results/{key}.json', dict(metrics=metrics,timing_repeats=timings,blocks=blocks,raw_sha256=sha(out/f'raw/{key}.npz')))
    save(out/f'curves/{key}.json', curves)
    return metrics


def run(out):
    p = check_freeze(out)
    for name in ('raw','results','curves','precision'):
        (out/name).mkdir(exist_ok=False)  # incomplete runs remain immutable; never silently overwrite
    start = time.monotonic()
    deadline = start+p['main_cap_seconds']
    manifest = dict(status='running',started_utc=stamp(),protocol_sha256=sha(out/'protocol.json'),completed_trials=[],completed_precision=[],failures=[])
    save(out/'run_manifest.json',manifest)
    try:
        for trial in p['trials']:
            run_trial(out,trial,p,deadline)
            manifest['completed_trials'].append(trial['key'])
            save(out/'run_manifest.json',manifest)
            print(json.dumps(dict(completed=trial['key'],seconds=round(time.monotonic()-start,2))),flush=True)
        for key in p['precision_cases']:
            trial = next(t for t in p['trials'] if t['key']==key)
            cfg = trial['config']
            m, data, raw = np.load(out/f'inputs/{cfg}.npz'), np.load(out/f'inputs/{key}.npz'), np.load(out/f'raw/{key}.npz')
            init = p['precision_initialization'][trial['stream']]
            before = time.perf_counter()
            high = mp_replay(m['W'],m['alpha'],data['forcing'],data['initials'][init],deadline=deadline)
            ref = raw['none'][init]
            maxerr = float(np.max(np.abs(high-ref)))
            metrics = {}
            for policy in POLICIES:
                e = raw[policy][init]-high
                metrics[policy] = dict(discrepancy_max_abs=float(np.max(np.abs(e))), discrepancy_rms=float(np.sqrt(np.mean(e**2))))
            receipt = dict(trial=key,initialization=init,dps=50,frames=len(high)-1,max_abs_float64_vs_mp50=maxerr,
                           tolerance=p['precision_acceptance_max_abs'],status='passed' if maxerr<=p['precision_acceptance_max_abs'] else 'reference_precision_sensitive',
                           policies=metrics,elapsed_seconds=time.perf_counter()-before)
            np.savez_compressed(out/f'precision/{key}.npz',reference=high)
            save(out/f'precision/{key}.json',receipt)
            manifest['completed_precision'].append(key)
            save(out/'run_manifest.json',manifest)
            print(json.dumps(receipt),flush=True)
        manifest['status']='complete'
    except Exception as exc:
        manifest['status']='partial'
        manifest['failures'].append(dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        manifest['finished_utc']=stamp()
        manifest['elapsed_seconds']=time.monotonic()-start
        save(out/'run_manifest.json',manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('calibrate','prepare','run'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--calibration',type=Path)
    args=parser.parse_args()
    if args.action=='calibrate':
        calibrate(args.out)
    elif args.action=='prepare':
        if not args.calibration:
            parser.error('--calibration required for prepare')
        prepare(args.out,args.calibration)
    else:
        run(args.out)


if __name__=='__main__':
    main()
