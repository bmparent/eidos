from __future__ import annotations
from datetime import datetime, timezone

def fixture_world_events() -> list[dict]:
    now=datetime.now(timezone.utc)
    return [
        {"timestamp":now.isoformat(),"source_id":"fixture_news","title":"Energy prices rise amid storms","text":"energy storm supply chain"},
        {"timestamp":now.isoformat(),"source_id":"fixture_forum","title":"AI chips demand surges","text":"ai chips demand market"},
    ]
