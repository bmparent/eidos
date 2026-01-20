from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import deque
import time
import math
import json
import numpy as np

@dataclass
class EpisodeRecord:
    step: int
    ts: float
    regime: str
    z: float
    err: float
    signature: Any  # hypervector (list or bytes)
    entities: Dict[str, Any]
    exemplars: List[Any]
    top_drivers: List[Dict[str, Any]]

class EpisodeIndex:
    def __init__(self, maxlen: int = 5000):
        self.buf = deque(maxlen=maxlen)

    def add(self, rec: EpisodeRecord):
        self.buf.append(rec)

    def topk(self, signature_vec, regime: str, k: int = 3) -> List[Dict[str, Any]]:
        # Simple cosine/mean similarity
        if not self.buf:
            return []
            
        scores = []
        # Support numpy or tensor
        sig_arr = np.array(signature_vec)
        if hasattr(signature_vec, "cpu"):
             sig_arr = signature_vec.cpu().numpy()
             
        norm_sig = np.linalg.norm(sig_arr)
        if norm_sig < 1e-9: norm_sig = 1.0
        
        for rec in self.buf:
            # We can optionally filter by regime if we want specific regime matching
            # if rec.regime != regime: continue
            
            other_sig = rec.signature
            if hasattr(other_sig, "cpu"): other_sig = other_sig.cpu().numpy()
            other_sig = np.array(other_sig)
            
            norm_other = np.linalg.norm(other_sig)
            if norm_other < 1e-9: norm_other = 1.0
            
            sim = np.dot(sig_arr, other_sig) / (norm_sig * norm_other)
            scores.append((sim, rec))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:k]
        
        results = []
        for sim, rec in top:
            results.append({
                "step": rec.step,
                "ts": rec.ts,
                "regime": rec.regime,
                "sim": float(sim),
                "summary": f"Prior {rec.regime} event at step {rec.step}",
                "drivers": rec.top_drivers,
                "entities": rec.entities
            })
        return results

@dataclass
class IncidentCard:
    incident_id: str
    domain: str
    regime: str
    severity: str
    step: int
    ts: float

    summary: str
    hypotheses: List[Dict[str, float]]
    evidence: Dict[str, Any]    # {"drivers":..., "entities":..., "exemplars":..., "baseline":...}

    similar_episodes: List[Dict[str, Any]]
    invariant: Dict[str, Any]         # shared drivers/entities

    forecast: Dict[str, Any]          # filled by ForecastEngine
    actions: List[Dict[str, Any]]     # filled by ProceduralMemory

    confidence: float

class IncidentCardEmitter:
    def __init__(self, enabled: bool, min_gap_steps: int):
        self.enabled = enabled
        self.min_gap_steps = min_gap_steps
        self.last_emit_step = -9999
        self.active_id = None
        
    def should_emit(self, regime: str, step: int) -> bool:
        if not self.enabled:
            return False
        # Emit on AMBER or RED
        if regime not in ["AMBER", "RED"]:
            return False
        if (step - self.last_emit_step) < self.min_gap_steps:
            return False
        return True
        
    def emit(self, card: IncidentCard) -> None:
        self.last_emit_step = card.step
        # The actual writing to jsonl happens via the caller (Sentinel main loop)
        # Here we just mark internal state if necessary
        pass
