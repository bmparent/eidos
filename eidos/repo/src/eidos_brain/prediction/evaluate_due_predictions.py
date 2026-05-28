from __future__ import annotations
import argparse, json
from pathlib import Path
from .ledger import append_evaluation, due_predictions, utc_now
from .scoring import brier

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument('--ledger',required=True); p.add_argument('--out',required=True); args=p.parse_args()
    ledger=Path(args.ledger); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    due=due_predictions(ledger,utc_now()); rows=[]
    for pred in due:
        actual=0
        score=brier(pred['prediction']['p_up'] if isinstance(pred['prediction'],dict) else pred['prediction'],actual)
        base=brier(pred['baseline_prediction'],actual)
        row={"prediction_id":pred['prediction_id'],"evaluated_at_utc":utc_now(),"actual_observation":actual,"scoring_method":"brier","score":score,"baseline_score":base,"eidos_vs_baseline_delta":base-score,"status":"evaluated","reason_if_skipped":""}
        append_evaluation(ledger,row); rows.append(row)
    (out/'evaluations.jsonl').write_text('\n'.join(json.dumps(x,sort_keys=True) for x in rows)+'\n' if rows else '',encoding='utf-8')

if __name__=='__main__': main()
