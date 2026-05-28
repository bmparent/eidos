from __future__ import annotations

def forecast_probability(signal:float)->float:
    p=0.5+max(min(signal,1.0),-1.0)*0.2
    return round(max(0.01,min(0.99,p)),4)
