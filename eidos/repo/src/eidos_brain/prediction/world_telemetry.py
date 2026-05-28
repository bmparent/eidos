from __future__ import annotations

def make_targets(topic_hashes:list[str], horizons:list[str])->list[dict]:
    return [{"horizon":h,"target_definition":f"topic:{th}:velocity_up_20pct"} for th in sorted(set(topic_hashes))[:3] for h in horizons]
