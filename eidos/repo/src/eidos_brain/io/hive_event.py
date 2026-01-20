"""
hive_event.py

Defines the standard HiveEventV1 schema for all Eidos Brain events.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

@dataclass
class HiveEventV1:
    event_type: str  # e.g. "eidos.surprise.v1"
    ts: str          # UTC ISO 8601
    session_id: str
    engine_version: str
    engine_hash: str
    source: str      # LOCAL/DRIVE/KAGGLE/STREAM
    payload: Dict[str, Any]
    artifact_refs: List[str] = field(default_factory=list)

    @classmethod
    def create(cls, event_type: str, session_id: str, payload: Dict[str, Any], 
               source: str = "UNKNOWN", artifacts: List[str] = None):
        return cls(
            event_type=event_type,
            ts=datetime.utcnow().isoformat() + "Z",
            session_id=session_id,
            engine_version="0.4.7.02", # In real app, import VERSION
            engine_hash="UNKNOWN", # Should be passed in or resolved
            source=source,
            payload=payload,
            artifact_refs=artifacts or []
        )

    def to_jsonl_line(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_jsonl_line(cls, line: str) -> "HiveEventV1":
        data = json.loads(line)
        return cls(**data)
