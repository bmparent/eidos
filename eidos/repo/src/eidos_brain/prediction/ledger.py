from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def deterministic_prediction_id(experiment_type:str,created_at_run:str,horizon:str,target_definition:str,config_hash:str,data_snapshot_hash:str)->str:
    raw='|'.join([experiment_type,created_at_run,horizon,target_definition,config_hash,data_snapshot_hash])
    return hashlib.sha256(raw.encode()).hexdigest()[:20]

def _append_jsonl(path:Path,row:dict[str,Any])->None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(row, sort_keys=True)+"\n")

def append_prediction(ledger_dir:Path,row:dict[str,Any])->None:
    _append_jsonl(ledger_dir/'predictions.jsonl',row)

def append_evaluation(ledger_dir:Path,row:dict[str,Any])->None:
    _append_jsonl(ledger_dir/'evaluations.jsonl',row)

def load_jsonl(path:Path)->list[dict[str,Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def due_predictions(ledger_dir:Path, now_utc:str)->list[dict[str,Any]]:
    preds=load_jsonl(ledger_dir/'predictions.jsonl'); evald={e['prediction_id'] for e in load_jsonl(ledger_dir/'evaluations.jsonl')}
    return [p for p in preds if p.get('status') in {'pending','due'} and p.get('evaluation_due_at_utc','')<=now_utc and p['prediction_id'] not in evald]
