from __future__ import annotations
import math

def brier(prob:float, actual:int)->float:
    return (prob-actual)**2

def finite_or_zero(x:float)->float:
    return x if math.isfinite(x) else 0.0
