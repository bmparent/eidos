from __future__ import annotations
import argparse, json
from pathlib import Path
from .baselines import baseline_probability
from .config import market_symbols, stable_hash
from .eidos_forecaster import forecast_probability
from .ledger import append_prediction, deterministic_prediction_id, utc_now
from .market_data import fixture_market_series, returns
from .reports import write_report

HORIZONS=['30m','1h','eod','1w','1m','1y']

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--fixture',action='store_true'); p.add_argument('--out',required=True); args=p.parse_args()
    run_id=utc_now().replace(':','').replace('-','')
    out=Path(args.out)/run_id; out.mkdir(parents=True,exist_ok=True)
    ledger=Path(args.out).parent/'ledger'; cfg={"market":{"symbols":["SPY","QQQ","AAPL","MSFT","NVDA","BTC-USD"]}}; syms=market_symbols(cfg)
    data=fixture_market_series(syms); cfg_hash=stable_hash(cfg); data_hash=stable_hash(data); preds=[]
    for s,series in data.items():
        r=returns(series)
        for h in HORIZONS:
            prob=forecast_probability(r*10)
            td=f'{s}:{h}:direction_up'; pid=deterministic_prediction_id('market_forecast',run_id,h,td,cfg_hash,data_hash)
            row={"prediction_id":pid,"experiment_type":"market_forecast","created_at_utc":utc_now(),"run_id":run_id,"git_commit":"unknown","config_hash":cfg_hash,"data_snapshot_hash":data_hash,"source_window_start_utc":utc_now(),"source_window_end_utc":utc_now(),"horizon":h,"target_time_utc":"9999-12-31T00:00:00Z","target_definition":td,"prediction_type":"direction","prediction":{"p_up":prob,"expected_return":r},"confidence":abs(prob-0.5)*2,"baseline_prediction":baseline_probability(),"eidos_state_summary":"deterministic lightweight forecaster","sentinel_regime_summary":"n/a","status":"pending","evaluation_due_at_utc":"9999-12-31T00:00:00Z","notes":"Research experiment only. Not financial advice. No trading execution."}
            append_prediction(ledger,row); preds.append(row)
    (out/'predictions.jsonl').write_text('\n'.join(json.dumps(x,sort_keys=True) for x in preds)+'\n',encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps({"run_id":run_id,"experiment_type":"market_forecast","google_drive_sync":"skipped_missing_secret"},indent=2),encoding='utf-8')
    (out/'summary.csv').write_text('prediction_id,symbol,horizon,p_up\n'+'\n'.join(f"{x['prediction_id']},{x['target_definition'].split(':')[0]},{x['horizon']},{x['prediction']['p_up']}" for x in preds),encoding='utf-8')
    write_report(out/'experiment_report.md','Market Forecast Experiment',[f'Symbols: {", ".join(syms)}','Research experiment only. Not financial advice. No trading execution.'])
    write_report(out/'experiment_status.md','Experiment Status',[f'pending={len(preds)} evaluated=0'])

if __name__=='__main__': main()
