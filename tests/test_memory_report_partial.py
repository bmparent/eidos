"""A capped run must retain a readable report from its completed receipts."""

import json

from proof.memory_core import POLICIES
from proof.memory_report import report


def test_partial_zero_only_report_has_no_invented_task_effect(tmp_path):
    run = tmp_path / 'run'
    for folder in ('results', 'curves', 'precision'):
        (run / folder).mkdir(parents=True, exist_ok=True)
    (run / 'protocol.json').write_text(json.dumps({
        'configurations': ['single'], 'n_reservoir': 8, 'tail_steps': 40000,
        'precision_acceptance_max_abs': 1e-9, 'threshold_reason': 'No utility threshold',
    }), encoding='utf-8')
    (run / 'run_manifest.json').write_text(json.dumps({
        'status': 'partial', 'completed_trials': ['zero_7'],
        'failures': ['time cap'], 'elapsed_seconds': 10.0,
    }), encoding='utf-8')
    metrics = [dict(
        trial='zero_7', config='single', stream='zero', policy=policy,
        initialization=initialization, task_mse=None, discrepancy_max_abs=0.0,
        discrepancy_rms=0.0, discrepancy_normalized_rms=0.0,
        prediction_max_abs_difference=0.0, runtime_ratio_to_current=1.0,
        persistent_state_bytes=64, frames=40000,
        max_sampled_storage_error=0.0, storage_samples=0,
    ) for policy in POLICIES for initialization in (0, 1)]
    (run / 'results' / 'zero_7.json').write_text(
        json.dumps({'metrics': metrics, 'blocks': []}), encoding='utf-8')

    out = tmp_path / 'report'
    report(run, out)

    decision = json.loads((out / 'decision.json').read_text(encoding='utf-8'))
    assert decision['run_status'] == 'partial'
    assert all(row['task_mse_delta_current_min'] is None and
               row['task_mse_delta_current_max'] is None for row in decision['aggregates'])
    assert 'NA to NA' in (out / 'decision_report.md').read_text(encoding='utf-8')
    assert (out / 'evidence_figure.png').is_file()
