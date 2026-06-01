import json, subprocess, sys

def test_world_smoke(tmp_path):
    out=tmp_path/'world'; subprocess.check_call([sys.executable,'-m','eidos_brain.prediction.run_world_telemetry','--fixture','--out',str(out)])
    assert any(out.iterdir())

def test_world_local_jsonl_corpus_smoke(tmp_path):
    data=tmp_path/'events.jsonl'
    data.write_text(
        '\n'.join([
            json.dumps({"title":"Grid outage response expands","text":"grid outage response", "published_at_utc":"2026-05-22T10:00:00Z"}),
            json.dumps({"title":"AI chip supply normalizes","text":"ai chips supply", "published_at_utc":"2026-05-22T11:00:00Z"}),
        ])+'\n',
        encoding='utf-8',
    )
    config=tmp_path/'world_sources.yaml'
    config.write_text(f"sources:\n  - id: local_smoke\n    type: local_jsonl\n    path: {str(data).replace(chr(92), '/')}\nhorizons:\n  - 3d\n",encoding='utf-8')
    out=tmp_path/'world_real'
    subprocess.check_call([sys.executable,'-m','eidos_brain.prediction.run_world_telemetry','--config',str(config),'--out',str(out),'--max-events','2'])
    run_dir=next(out.iterdir())
    manifest=json.loads((run_dir/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['fixture_mode'] is False
    assert manifest['corpus']['ok_event_count']==2
    assert (run_dir/'source_events.jsonl').exists()
