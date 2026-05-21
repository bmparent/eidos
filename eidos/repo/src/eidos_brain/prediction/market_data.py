from __future__ import annotations
import csv
from pathlib import Path

def fixture_market_series(symbols:list[str])->dict[str,list[float]]:
    return {s:[100.0,101.0,100.5,102.0,103.0] for s in symbols}

def returns(series:list[float])->float:
    return (series[-1]-series[-2])/series[-2] if len(series)>1 else 0.0
