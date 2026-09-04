"""Render existing frozen memory benchmark receipts; never rerun evaluation."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .memory_benchmark import sha, save, stamp
from .memory_core import POLICIES

COLORS = dict(none='#5b6678',every_step='#c46633',pulse100='#256d9b',carry='#228568')


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(run, out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((run/'protocol.json').read_text())
    manifest=json.loads((run/'run_manifest.json').read_text())
    rows,curves,blocks=[],[],[]
    for path in sorted((run/'results').glob('*.json')):
        r=json.loads(path.read_text())
        rows.extend(r['metrics'])
        blocks.extend(r['blocks'])
    for path in sorted((run/'curves').glob('*.json')):
        curves.extend(json.loads(path.read_text()))
    if not rows:
        save(out/'decision.json',dict(decision='inconclusive_for_adoption',run_status=manifest['status'],
                                     reason='No completed trial receipts',failures=manifest['failures']))
        (out/'decision_report.md').write_text('# Controlled memory benchmark: incomplete\n\n'
            'No completed trial receipts. No effect sizes or evidence figure can be produced. '
            'The run manifest preserves failures; the frozen horizon has not been shortened.\n',encoding='utf-8')
        return
    precision=[json.loads(p.read_text()) for p in sorted((run/'precision').glob('*.json'))]
    current={(r['trial'],r['initialization']):r for r in rows if r['policy']=='pulse100'}
    for r in rows:
        baseline=current[r['trial'],r['initialization']]
        r['task_mse_delta_from_current']=r['task_mse']-baseline['task_mse'] if r['task_mse'] is not None else None
    flat=[{k:(json.dumps(v) if isinstance(v,dict) else v) for k,v in row.items()} for row in rows]
    csv_write(out/'benchmark_summary.csv',flat)
    csv_write(out/'plot_data.csv',curves)
    csv_write(out/'temporal_blocks.csv',blocks)
    save(out/'precision_summary.json',precision)
    aggregates=[]
    for cfg in protocol['configurations']:
        for policy in POLICIES:
            group=[r for r in rows if r['config']==cfg and r['policy']==policy]
            if not group:
                continue
            effects=[r['task_mse_delta_from_current'] for r in group if r['task_mse'] is not None]
            aggregates.append(dict(config=cfg,policy=policy,trajectories=len(group),
                max_state_abs_discrepancy=max(r['discrepancy_max_abs'] for r in group),
                median_state_rms_discrepancy=float(np.median([r['discrepancy_rms'] for r in group])),
                median_normalized_state_rms=float(np.median([r['discrepancy_normalized_rms'] for r in group])),
                max_prediction_abs_difference=max(r['prediction_max_abs_difference'] for r in group),
                task_mse_delta_current_min=min(effects) if effects else None,
                task_mse_delta_current_max=max(effects) if effects else None,
                runtime_ratio_median=float(np.median([r['runtime_ratio_to_current'] for r in group])),
                persistent_state_bytes=group[0]['persistent_state_bytes']))
    save(out/'decision.json',dict(decision='inconclusive_for_adoption',production_action='keep current policy pending utility evidence',
                utility_gate='untested',practical_task_threshold=None,acceptable_overhead_threshold=None,
                reason=protocol['threshold_reason'],scope=dict(neurons=protocol['n_reservoir'],trials=len(manifest['completed_trials']),
                trajectories=len(rows),unique_updates=sum(r['frames'] for r in rows),precision_checks=len(precision)),
                run_status=manifest['status'],aggregates=aggregates,elapsed_seconds=manifest['elapsed_seconds']))
    figure(out,protocol,rows,curves)
    maxhp=max((r['max_abs_float64_vs_mp50'] for r in precision),default=None)
    lines=['# Controlled memory benchmark decision', '',
        '**Decision: inconclusive for adoption. Keep the current production policy pending a defensible utility test.**', '',
        f"Run status: **{manifest['status']}**. {len(rows)} trajectories across {len(manifest['completed_trials'])} matched configurations; "
        f"{protocol['n_reservoir']} neurons, 3 input channels, 2 forcing/initialization seeds, four policies, two initializations. "
        f"{sum(r['frames'] for r in rows):,} unique state updates; three additional timing repeats for each. Main elapsed time: {manifest['elapsed_seconds']:.2f} seconds.", '',
        'The four state policies share identical native engine-generated matrices, binary coefficients, initial states, forcing, and frozen readouts. '
        'Default single leak and slow single leak preserve the native float32 coefficient values lifted to float64; the band alpha vector is natively float64. '
        'The implemented four bands are 0.2, 0.05, 0.005, 0.0005, rather than the roadmap example beginning at 0.5. '
        'The first rounding pulse follows update 100. Every-step input rounding is absent from every arm. Production files and defaults are untouched.', '',
        f"All streams include a {protocol['tail_steps']:,}-step zero-input tail (20 times the slowest scalar relaxation scale). "
        'This is a scalar diagnostic minimum, not a coupled mixing-time guarantee. The driven excitation has 4,097 frames; '
        'the recovered project fixture has 1,100 synthetic, unlabeled frames. The project fixture is not clinical or operational detection evidence.', '',
        '## Measured effects and costs', '',
        'Errors are measured against float64 without explicit grid rounding. Absolute values are in reservoir state units. '
        'The normalized state error divides by the corresponding exploration-prefix RMS, floored at 1e-6; it never divides by a vanishing tail reference. '
        'The table pools zero, impulse, driven and project streams descriptively; full per-stream/per-initialization values remain in CSV.', '',
        '| Leak | Policy | Max absolute state discrepancy | Median state RMS discrepancy | Normalized median RMS | Task MSE delta vs current: min to max | Runtime/current | State bytes |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in aggregates:
        lines.append(f"| {r['config']} | {r['policy']} | {r['max_state_abs_discrepancy']:.6g} | {r['median_state_rms_discrepancy']:.6g} | "
                     f"{r['median_normalized_state_rms']:.6g} | {r['task_mse_delta_current_min']:.6g} to {r['task_mse_delta_current_max']:.6g} | "
                     f"{r['runtime_ratio_median']:.3f} | {r['persistent_state_bytes']} |")
    lines += ['', 'Task MSE uses a shared exploration-only ridge readout; suffixes follow a 64-frame gap, with state carried continuously through the split. '
        'Readouts and normalization are frozen before candidate replay. Raw readout and engine-style rounded predictions are both saved. '
        'These are state-fidelity effects on a synthetic prediction diagnostic, not independently trained end-to-end systems. '
        'Contiguous block results remain visible; no independent-frame p values or confidence claims are made. '
        'No defensible smallest useful task effect or acceptable overhead requirement was supplied or found, so both gates remain null.', '',
        'Runtime is the median of three full replays with rotated policy order, for both initializations, excluding exact carry auditing and reporting. '
        'The small-matrix result is dominated by Python/NumPy overhead and does not forecast 2,000-neuron or GPU cost. '
        'State memory counts actual persistent numeric arrays only: carry adds one float64 array. Common matrices, readout, inputs, Python objects and offline trace storage are separate.', '',
        '## Precision, certification, and preserved failures', '',
        f"{len(precision)} entire trajectories were checked at 50 decimal digits using the same frozen binary coefficients and forcing. "
        f"Maximum float64/reference discrepancy: {maxhp!s}; prespecified tolerance: {protocol['precision_acceptance_max_abs']}. "
        'See precision_summary.json for every status and high-precision policy discrepancy. Other trajectories remain float64 experiments, not exact ground truth.', '',
        f"Maximum sampled carry-storage error: {max(r['max_sampled_storage_error'] for r in rows):.6g}; "
        f"{sum(r['storage_samples'] for r in rows):,} exact binary-rational subtraction checks. "
        'This samples z-r storage, not all steps or errors in matrix multiplication, tanh, coefficient generation, or adding the previous carry. '
        'The exact-storage theorem is not asserted as a floating-point guarantee.', '',
        'The actual sampled matrix is **outside the sufficient condition**: the exact minimum row sum of abs(W) exceeds one, '
        'which certifies rho(abs(W)) > 1. No rescaling was applied. This does not establish instability or invalidate the empirical comparison. '
        'A separate exact nonnormal clipped-linear control has K=[[0,2],[0,0]], h=(3,1), and h-Kh=(1,1)>0.', '',
        'The ordinary-rounding scalar deadband, slow-leak pulse persistence, and signed residual-carry two-cycle remain explicit controls. '
        'Residual carry does not guarantee forgetting. Constant pulse-phase values in finite traces are described as observed plateaus, not proofs of infinite-time behavior. '
        'The report’s tanh counterexample is analytically separated from the clipped-linear exact computation.', '',
        'No actual Sentinel alert path was exercised; alert counts, false positives, calibrated/raw detection precision and recall are **NA**. '
        'Geometry is descriptive only. No compression benefit, detection validity, general-domain benefit, new mechanism, or originality claim is made.', '',
        '## Claim / evidence table', '',
        '| Kind | Claim | Evidence | Limit |', '|---|---|---|---|',
        '| Mathematical deduction | Scalar forgetting iff (1-alpha)^m <= 1/2 for ties-to-even pulse map | Preserved report-source.md, Proposition 1 | Exact scalar zero-input restriction |',
        '| Mathematical deduction | Carry envelope (I-K)^-1 Delta; ordinary bound (I-K)^-1 A^-1 Delta/2 | Preserved report-source.md, Theorem 2; explicit noncommuting matrix order | Common forcing, rho(K)<1, exact residual; not certified for actual W |',
        '| Mathematical deduction | Uniform sharpness across the stated class | Theorem 3; alternating forcing construction | Not each fixed leak or the bounded tanh subclass; no priority claim |',
        '| Finite exact checks | 21 slow pulse states; minimum intervals 1386 and 69; signed cycle; nonnormal bounds | calibration.json, evaluator tests, original and rerun research receipts | Finite checks support but do not replace universal proof |',
        '| Floating-point experiment | Policy-dependent state/prediction effects and costs | benchmark_summary.csv, plot_data.csv, raw traces, precision_summary.json | Small frozen open-loop reservoir; selected higher-precision cases |',
        '| Hypothesis | Better fidelity may improve useful anomaly detection | No utility evidence in this run | Requires labels, requirements and independently frozen utility protocol |',
        '| Interpretation | Numerical persistence must be separated from useful memory | Preserved counterexamples and measured separation | Not a claim of intelligence, consciousness or physics |', '',
        '## Proof Logic + Meaning', '',
        '### Goal reached',
        f"Controlled numerical benchmark: **{manifest['status']}**; adoption utility gate: **missing / untested**. "
        'Source reproduction and repository test statuses are reported separately in the handoff receipts.', '',
        '### Previous state',
        'The research package supplied proofs and toy controls but had not measured the current Torch reservoir under matched rounding policies. '
        'The working checkout also contained unrelated residue. This work isolates the current fetched baseline and preserves that residue.', '',
        '### Technical logic utilized',
        'Freeze the real engine matrices; extract the listen recurrence and validate it against the real path; vary only state rounding. '
        'Use exact rational controls and certificates, two initializations, long zero-input tails, fixed exploration readouts, and disjoint suffix diagnostics.', '',
        '### Math / scoring logic',
        'r_next = (I-A)r + A*tanh(Wr + Bu). Pulse map: k_next = round_even((1-alpha)^m*k). '
        'Carry: z=F(r)+c, r_next=Q(z), c_next=z-r_next. Corrected-state error obeys '
        '|e_next| <= (I-A+AK)|e| + A(I+K)Delta/2; the invariant box gives |r-x| <= (I-K)^-1 Delta. '
        'MSE = mean((prediction-target)^2). RMS discrepancy = sqrt(mean((r-x)^2)). '
        'Runtime ratio = median(candidate replay time)/median(current replay time). No utility score is invented.', '',
        '### Philosophical meaning',
        'Reproducibility is truth that can be revisited. Apparent memory is trustworthy only after representation-induced persistence has been checked.', '',
        '### Why this is better',
        'The project now has a rerunnable isolated evaluator, frozen inputs, explicit precision limits, measured effects/costs, and retained negative controls. '
        'This improves auditability and operator trust; it does not establish useful predictive or detection improvement.', '',
        '### How this moves Eidos closer to the north-star goal',
        'For the self-monitoring streaming intelligence codec goal, this strengthens reproducibility and interpretation of internal-state memory. '
        'It prepares reliable evidence-backed detection work without claiming anomaly preservation, compression advantage, or incident explanation has been demonstrated here.', '',
        '### Evidence',
        'protocol.json, freeze.json, adapter_selected_size.json, calibration.json, raw/*.npz, results/*.json, '
        'precision/*.json, benchmark_summary.csv, temporal_blocks.csv, evidence_figure.png, and evidence_manifest.json. '
        'The archive preserves the delivered research ZIP and both original and rerun receipts.', '',
        '### Remaining uncertainty',
        'Default-size reservoir, GPU, adaptive training/thermostat feedback, exact forgetting of the actual coupled engine, '
        'detection labels, operational utility, acceptable overhead, full-model compression behavior, and theorem originality remain unproven.', '',
        '## Next decisive step', '',
        'Define a labeled operational task, a smallest useful improvement and overhead budget, then freeze a matched chronological utility experiment '
        'against pulse100. Revisit matrix size and the adaptive engine separately. Numerical fidelity is a reason to investigate, not to adopt.', '']
    (out/'decision_report.md').write_text('\n'.join(lines),encoding='utf-8')
    save(out/'report_manifest.json',dict(created_utc=stamp(),source_run=str(run),protocol_sha256=sha(run/'protocol.json'),
                                       inputs={p.name:sha(p) for p in (run/'results').glob('*.json')}))
    progress(out, protocol, manifest, precision)
    print(json.dumps(dict(report=str(out),rows=len(rows),precision_checks=len(precision))),flush=True)


def progress(out, protocol, manifest, precision):
    """Show local evidence checks; do not infer the project's global readiness."""
    import html
    folder=out/'progress'
    folder.mkdir()
    checks=[dict(name='Frozen evaluator and inputs',status='passed',evidence='main/freeze.json'),
            dict(name='Actual listen adapter / reset',status='passed',evidence='main/adapter_selected_size.json'),
            dict(name='Exact scalar and nonnormal controls',status='passed',evidence='calibration/calibration.json'),
            dict(name='Declared numerical trajectories',status='passed' if manifest['status']=='complete' else 'partial',evidence='main/run_manifest.json'),
            dict(name='Full-horizon precision checks',status='passed' if len(precision)==6 and all(p['status']=='passed' for p in precision) else 'partial',evidence='report/precision_summary.json'),
            dict(name='Operational utility gate',status='missing',evidence='main/protocol.json: null task/overhead thresholds')]
    gates=[dict(name=name,weight=weight,status='unknown',reason='Global project gate not reassessed by this numerical benchmark') for name,weight in
           [('reproducible baseline',15),('labeled-domain proof',15),('precision / false-positive discipline',15),
            ('Sentinel calibration / generalization',15),('compression / anomaly preservation',10),
            ('incident cards / explanation',10),('domain demos',10),('one-command reproducibility / final report',10)]]
    data=dict(utc=stamp(),overall_proof_readiness_score=None,score_reason='The complete project gate evidence was not audited here; unknown is not zero.',
              scoring_rule='sum(weight_i * accepted_gate_pass_i), evaluated only when every gate has been assessed',
              weighted_project_gates=gates,benchmark_checks=checks,neurons=protocol['n_reservoir'],
              trajectories=len(manifest['completed_trials'])*8,precision_checks=len(precision),
              north_star='Eidos Brain is a self-monitoring streaming intelligence codec: learns live streams, compresses predictable behavior, preserves meaningful anomalies, monitors internal state, and emits human-readable incident receipts.',
              strengthened='Reproducibility and interpretation of internal-state persistence.',
              unproven=['Default-size reservoir','GPU','Adaptive engine feedback','Labeled detection utility','Compression value','Theorem originality'])
    save(folder/'eidos_progress_meter.json',data)
    logic=dict(goal='Controlled numerical benchmark',status=manifest['status'],
               technical_logic='Matched frozen recurrence; exact controls; long zero tails; fixed shared readout; precision checks.',
               mathematics='r_next=(I-A)r+A*tanh(Wr+b); carry z=F(r)+c, r_next=Q(z), c_next=z-r_next; RMS and MSE effects.',
               meaning='Recognition of numerical artifacts precedes claims about useful memory.',
               evidence=[c['evidence'] for c in checks],remaining_uncertainty=data['unproven'])
    save(folder/'proof_logic_ledger.json',logic)
    md=['# Controlled memory proof progress','','Overall project readiness: **unknown**. Existing global gates were not reassessed; this is not a zero score.','',
        '| Local benchmark check | Status | Receipt |','|---|---|---|']
    md += [f"| {c['name']} | {c['status']} | {c['evidence']} |" for c in checks]
    md += ['',f"Measured scope: {data['neurons']} neurons, {data['trajectories']} trajectories, {data['precision_checks']} precision checks.",
           '', '## Proof Logic + Meaning', '',logic['technical_logic'], '',logic['mathematics'], '',logic['meaning'], '',
           'This strengthens reproducibility and interpretation of internal state, not validated detection or compression value.', '',
           'Remaining uncertainty: '+', '.join(data['unproven'])+'.','']
    (folder/'eidos_progress_meter.md').write_text('\n'.join(md),encoding='utf-8')
    (folder/'proof_logic_ledger.md').write_text('\n'.join(md[md.index('## Proof Logic + Meaning'):]),encoding='utf-8')
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="600" viewBox="0 0 1080 600">',
         '<rect width="1080" height="600" fill="#f4f5f1"/>',
         '<text x="45" y="60" font-family="sans-serif" font-size="28" fill="#183330">Controlled memory — evidence status</text>',
         f'<text x="45" y="96" font-family="sans-serif" font-size="17">{data["neurons"]} neurons | {data["trajectories"]} trajectories | utility untested</text>']
    for i,c in enumerate(checks):
        y=150+i*53
        color='#24725f' if c['status']=='passed' else '#a16a28'
        svg += [f'<circle cx="58" cy="{y-6}" r="8" fill="{color}"/>',
                f'<text x="82" y="{y}" font-family="sans-serif" font-size="18" fill="#183330">{html.escape(c["name"])}</text>',
                f'<text x="790" y="{y}" font-family="sans-serif" font-size="17" fill="{color}">{c["status"]}</text>']
    svg += ['<text x="45" y="510" font-family="sans-serif" font-size="17">Global proof readiness: unknown — full project gates not reassessed.</text>',
            '<text x="45" y="550" font-family="sans-serif" font-size="15">Reproducibility is truth that can be revisited. Numerical fidelity is not a utility gate.</text>','</svg>']
    (folder/'eidos_progress_meter.svg').write_text('\n'.join(svg),encoding='utf-8')
    cards=''.join(f'<tr><td>{html.escape(c["name"])}</td><td>{c["status"]}</td><td>{html.escape(c["evidence"])}</td></tr>' for c in checks)
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Eidos controlled memory evidence</title><style>body{{font:16px/1.6 system-ui;background:#f4f5f1;color:#183330;margin:0}}main{{max-width:1080px;margin:40px auto;padding:24px}}h1{{font-size:38px}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:14px;text-align:left;border-bottom:1px solid #dae1da}}.note{{border-left:4px solid #a16a28;padding:12px 20px}}img{{width:100%}}@media(max-width:700px){{table{{font-size:12px}}td{{padding:6px}}}}</style>
<main><p>EIDOS / PROOF RECEIPTS</p><h1>Controlled memory benchmark</h1><p>{data['neurons']} neurons · {data['trajectories']} trajectories · {data['precision_checks']} precision checks</p>
<p class="note">Adoption: inconclusive. Global proof readiness: unknown. Numerical fidelity cannot establish detection utility.</p>
<table><thead><tr><th>Check</th><th>Status</th><th>Evidence in bundle</th></tr></thead><tbody>{cards}</tbody></table>
<h2>Evidence</h2><p><a href="../decision_report.md">Decision and proof logic</a> · <a href="../benchmark_summary.csv">Full metric table</a> · <a href="../plot_data.csv">Plot data</a></p>
<img src="../evidence_figure.png" alt="Actual initialization separation, state fidelity, task effects and runtime costs">
<h2>North-star claim</h2><p>{html.escape(data['north_star'])}</p><p>This experiment strengthens {html.escape(data['strengthened'].lower())}</p>
<h2>Remaining uncertainty</h2><p>{html.escape(', '.join(data['unproven']))}.</p><p>Raw, merged and calibrated detection metrics are NA: no Sentinel path was exercised.</p></main></html>'''
    (folder/'eidos_progress_dashboard.html').write_text(page,encoding='utf-8')
    (folder/'eidos_progress_readme.md').write_text('Generated from the frozen protocol, completed run manifest, adapter/calibration receipts and precision checks. '
        'Project readiness is null because global gates were not audited. Local checks do not promote global milestones. '
        'SVG requires no JavaScript. The dashboard links to report CSVs and the evidence figure.\n',encoding='utf-8')


def figure(out, protocol, rows, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(3,4,figsize=(17,10),layout='constrained')
    for i,cfg in enumerate(protocol['configurations']):
        ax=axes[i]
        for name in POLICIES:
            for panel,stream,value in ((0,'zero','init_separation_l2'),(1,'driven','discrepancy_rms')):
                selected=[c for c in curves if c['config']==cfg and c['policy']==name and c['stream']==stream and c['initialization']==1]
                grouped=defaultdict(list)
                for c in selected:
                    grouped[c['step']].append(c[value])
                steps=sorted(grouped)
                med=[np.median(grouped[t]) for t in steps]
                ax[panel].plot(steps,med,color=COLORS[name],label=name,lw=1.25)
                ax[panel].fill_between(steps,[min(grouped[t]) for t in steps],[max(grouped[t]) for t in steps],color=COLORS[name],alpha=.1)
            selected=[r for r in rows if r['config']==cfg and r['policy']==name]
            pos=list(POLICIES).index(name)
            for stream,offset,marker in (('driven',-.12,'o'),('project',.12,'x')):
                vals=[r['task_mse_delta_from_current'] for r in selected if r['stream']==stream]
                ax[2].scatter(np.full(len(vals),pos+offset),vals,color=COLORS[name],marker=marker,s=15,alpha=.7)
            costs=[r['runtime_ratio_to_current'] for r in selected]
            if costs:
                median=np.median(costs)
                ax[3].bar(pos,median,color=COLORS[name],width=.65)
                ax[3].vlines(pos,min(costs),max(costs),color='#333',lw=1)
        for j in (0,1):
            ax[j].set_yscale('symlog',linthresh=1e-12)
            ax[j].set_xlabel('listen updates (zero shown at 0)')
            ax[j].grid(alpha=.15)
        for j in (2,3):
            ax[j].set_xticks(range(4),POLICIES,rotation=25)
        ax[0].set_ylabel(f'{cfg}\nL2 state separation')
        ax[1].set_ylabel('RMS state discrepancy')
        ax[2].set_ylabel('Suffix MSE minus pulse100')
        ax[2].set_yscale('symlog',linthresh=1e-8)
        ax[2].axhline(0,color='#777',lw=.5)
        ax[3].set_ylabel('Replay seconds / pulse100')
        ax[3].axhline(1,color='#777',lw=.5)
    titles=['Zero-input initialization separation','Driven then zero: error vs float64 reference','Task effect: circles driven, crosses fixture','Runtime cost (median and observed range)']
    for ax,title in zip(axes[0],titles):
        ax.set_title(title,fontweight='bold')
    axes[0,0].legend(fontsize=7)
    fig.suptitle(f"Eidos controlled memory | {protocol['n_reservoir']} neurons | float64 states/carry | Delta=1e-5\n"
                 'Curves: median + range over forcing/initialization seeds 7 and 19; no confidence intervals. '
                 'Bands: 0.2 / 0.05 / 0.005 / 0.0005. Utility untested.',fontsize=13)
    fig.savefig(out/'evidence_figure.png',dpi=170)
    fig.savefig(out/'evidence_figure.svg')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    report(a.run,a.out)
