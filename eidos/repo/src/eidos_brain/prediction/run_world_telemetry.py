from __future__ import annotations
import argparse, json
from pathlib import Path
from .baselines import baseline_probability
from .config import load_yaml, stable_hash
from .eidos_forecaster import forecast_probability
from .feature_builder import build_world_features
from .ledger import append_prediction, deterministic_prediction_id, utc_now
from .reports import write_report
from .sources import default_world_sources, ingest_world_sources, source_manifest
from .world_telemetry import make_targets

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--fixture',action='store_true'); p.add_argument('--out',required=True); p.add_argument('--max-events',type=int,default=75); p.add_argument('--timeout-seconds',type=int,default=20); args=p.parse_args()
    run_id=utc_now().replace(':','').replace('-','')
    out=Path(args.out)/run_id; out.mkdir(parents=True,exist_ok=True)
    ledger=Path(args.out).parent/'ledger'
    loaded_cfg=load_yaml(args.config)
    cfg={"horizons":loaded_cfg.get("horizons",["3d","7d","30d","365d"]),"sources":loaded_cfg.get("sources",default_world_sources())}
    sources=[{"id":"fixture_world","type":"fixture"}] if args.fixture else cfg["sources"]
    events=ingest_world_sources(sources,observed_at_utc=utc_now(),base_dir=Path.cwd(),timeout_seconds=args.timeout_seconds,max_events=args.max_events)
    usable_events=[event for event in events if event.get("ingest_status")=="ok" and (event.get("title") or event.get("text"))]
    feats=build_world_features(usable_events)
    cfg_hash=stable_hash({"horizons":cfg["horizons"],"sources":sources,"fixture":args.fixture,"corpus_version":"real_world_corpus_v0"}); data_hash=stable_hash(feats)
    timestamps=sorted(f['timestamp'] for f in feats if f.get('timestamp'))
    source_window_start=timestamps[0] if timestamps else utc_now()
    source_window_end=timestamps[-1] if timestamps else utc_now()
    preds=[]
    for t in make_targets([f['topic_hash'] for f in feats],cfg['horizons']):
        signal=sum(x['urgency_proxy'] for x in feats)/max(len(feats),1)-0.5
        prob=forecast_probability(signal); pid=deterministic_prediction_id('world_telemetry',run_id,t['horizon'],t['target_definition'],cfg_hash,data_hash)
        row={"prediction_id":pid,"experiment_type":"world_telemetry","created_at_utc":utc_now(),"run_id":run_id,"git_commit":"unknown","config_hash":cfg_hash,"data_snapshot_hash":data_hash,"source_window_start_utc":source_window_start,"source_window_end_utc":source_window_end,"horizon":t['horizon'],"target_window":t['horizon'],"target_definition":t['target_definition'],"prediction_type":"probability","prediction":prob,"confidence":abs(prob-0.5)*2,"baseline_prediction":baseline_probability(),"eidos_state_summary":"deterministic lightweight forecaster","sentinel_regime_summary":"n/a","status":"pending","evaluation_due_at_utc":"9999-12-31T00:00:00Z","notes":"fixture" if args.fixture else "real_world_corpus_v0"}
        append_prediction(ledger,row); preds.append(row)
    (out/'source_events.jsonl').write_text(('\n'.join(json.dumps(x,sort_keys=True) for x in events)+'\n') if events else '',encoding='utf-8')
    (out/'predictions.jsonl').write_text(('\n'.join(json.dumps(x,sort_keys=True) for x in preds)+'\n') if preds else '',encoding='utf-8')
    manifest={"run_id":run_id,"experiment_type":"world_telemetry","fixture_mode":args.fixture,"corpus":source_manifest(events,sources),"config_hash":cfg_hash,"data_snapshot_hash":data_hash,"source_window_start_utc":source_window_start,"source_window_end_utc":source_window_end,"google_drive_sync":"skipped_missing_secret"}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding='utf-8')
    (out/'summary.csv').write_text('prediction_id,horizon,prediction\n'+'\n'.join(f"{x['prediction_id']},{x['horizon']},{x['prediction']}" for x in preds),encoding='utf-8')
    corpus=manifest["corpus"]
    write_report(out/'experiment_report.md','World Telemetry Experiment',[f"Corpus version: {corpus['corpus_version']}",f"Sources configured: {corpus['source_count']}",f"Source events read: {corpus['event_count']} total, {corpus['ok_event_count']} usable",f"Statuses: {json.dumps(corpus['status_counts'],sort_keys=True)}",f"Pending predictions recorded: {len(preds)}", 'What to watch next: elevated topic velocity clusters.'])
    write_report(out/'experiment_status.md','Experiment Status',[f'pending={len(preds)} evaluated=0'])

if __name__=='__main__': main()
