from __future__ import annotations
import hashlib, json, os
from pathlib import Path
from typing import Any
import yaml

def load_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}

def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def market_symbols(cfg: dict[str, Any]) -> list[str]:
    env = os.getenv("EIDOS_MARKET_SYMBOLS", "").strip()
    if env:
        return [s.strip() for s in env.split(',') if s.strip()]
    return cfg.get('market', {}).get('symbols', ["SPY","QQQ","AAPL","MSFT","NVDA","BTC-USD"])
