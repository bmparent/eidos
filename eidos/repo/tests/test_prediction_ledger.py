from pathlib import Path
from eidos_brain.prediction.ledger import deterministic_prediction_id, append_prediction, load_jsonl, due_predictions

def test_deterministic_prediction_id():
    a=deterministic_prediction_id('world','run','3d','target','c','d')
    b=deterministic_prediction_id('world','run','3d','target','c','d')
    assert a==b

def test_ledger_append_reload(tmp_path:Path):
    row={"prediction_id":"x","status":"pending","evaluation_due_at_utc":"2020-01-01T00:00:00Z"}
    append_prediction(tmp_path,row)
    assert load_jsonl(tmp_path/'predictions.jsonl')[0]['prediction_id']=='x'
    assert due_predictions(tmp_path,'2026-01-01T00:00:00Z')
