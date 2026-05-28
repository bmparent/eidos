from __future__ import annotations
import argparse, json
from pathlib import Path
from .baselines import baseline_probability
from .config import stable_hash
from .eidos_forecaster import forecast_probability
from .feature_builder import build_world_features
from .ledger import append_prediction, deterministic_prediction_id, utc_now
from .reports import write_report
from .sources import fixture_world_events
from .world_telemetry import make_targets

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--fixture',action='store_true'); p.add_argument('--out',required=True); args=p.parse_args()
    run_id=utc_now().replace(':','').replace('-','')
    out=Path(args.out)/run_id; out.mkdir(parents=True,exist_ok=True)
    ledger=Path(args.out).parent/'ledger'; events=fixture_world_events(); feats=build_world_features(events)
    cfg={"horizons":["3d","7d","30d","365d"]}; cfg_hash=stable_hash(cfg); data_hash=stable_hash(feats)
    preds=[]
    for t in make_targets([f['topic_hash'] for f in feats],cfg['horizons']):
        signal=sum(x['urgency_proxy'] for x in feats)/max(len(feats),1)-0.5
        prob=forecast_probability(signal); pid=deterministic_prediction_id('world_telemetry',run_id,t['horizon'],t['target_definition'],cfg_hash,data_hash)
        row={"prediction_id":pid,"experiment_type":"world_telemetry","created_at_utc":utc_now(),"run_id":run_id,"git_commit":"unknown","config_hash":cfg_hash,"data_snapshot_hash":data_hash,"source_window_start_utc":feats[0]['timestamp'],"source_window_end_utc":feats[-1]['timestamp'],"horizon":t['horizon'],"target_window":t['horizon'],"target_definition":t['target_definition'],"prediction_type":"probability","prediction":prob,"confidence":abs(prob-0.5)*2,"baseline_prediction":baseline_probability(),"eidos_state_summary":"deterministic lightweight forecaster","sentinel_regime_summary":"n/a","status":"pending","evaluation_due_at_utc":"9999-12-31T00:00:00Z","notes":"fixture"}
        append_prediction(ledger,row); preds.append(row)
    (out/'predictions.jsonl').write_text('\n'.join(json.dumps(x,sort_keys=True) for x in preds)+'\n',encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps({"run_id":run_id,"experiment_type":"world_telemetry","google_drive_sync":"skipped_missing_secret"},indent=2),encoding='utf-8')
    (out/'summary.csv').write_text('prediction_id,horizon,prediction\n'+'\n'.join(f"{x['prediction_id']},{x['horizon']},{x['prediction']}" for x in preds),encoding='utf-8')
    write_report(out/'experiment_report.md','World Telemetry Experiment',[f'Sources read: {len(events)}','Pending predictions recorded.', 'What to watch next: elevated topic velocity clusters.'])
    write_report(out/'experiment_status.md','Experiment Status',[f'pending={len(preds)} evaluated=0'])

if __name__=='__main__': main()
