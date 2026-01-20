import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Deque
from collections import deque
import json

@dataclass
class TrajectoryRecord:
    domain: str
    outcome: str  # e.g., RED, GREEN_RECOVERED, OUTAGE
    horizon: int
    sig_seq: List[List[float]]  # list of packed signature vectors
    z_seq: List[float]
    err_seq: List[float]

class ForecastEngine:
    def __init__(self, window: int = 30, horizons: List[int] = [10, 30, 100], temp: float = 6.0, enabled: bool = False):
        self.window = window
        self.horizons = horizons
        self.temp = temp
        self.enabled = enabled
        
        self.trajectories: List[TrajectoryRecord] = []
        
        # Live rolling window
        self.live_sigs: Deque[List[float]] = deque(maxlen=window)
        self.live_zs: Deque[float] = deque(maxlen=window)
        self.live_errs: Deque[float] = deque(maxlen=window)
        
        # Add some mock trajectories for demo purposes if enabled
        if enabled:
            self._seed_mock_trajectories()

    def _seed_mock_trajectories(self):
        # Create a few synthetic trajectories for the demo
        # 1. Rising risk trajectory
        sig_ramp = [ [0.1 * i] * 10 for i in range(10) ] # dummy
        self.trajectories.append(TrajectoryRecord(
            domain="generic",
            outcome="RED",
            horizon=30,
            sig_seq=sig_ramp,
            z_seq=[1.0, 2.0, 3.0, 4.0],
            err_seq=[0.1, 0.2, 0.4, 0.8]
        ))

    def update(self, signature_vec: Any, z: float, err: float, regime: str, domain: str):
        if not self.enabled:
            return
            
        # Convert torch/numpy to list for storage
        sig_val = signature_vec
        if hasattr(sig_val, "cpu"): sig_val = sig_val.cpu().numpy()
        if hasattr(sig_val, "tolist"): sig_val = sig_val.tolist()
        elif isinstance(sig_val, np.ndarray): sig_val = sig_val.tolist()
        
        self.live_sigs.append(sig_val)
        self.live_zs.append(float(z))
        self.live_errs.append(float(err))

    def risk(self, domain: str, regime: str) -> Dict[str, Any]:
        """
        Compare live window to stored trajectories.
        risk[outcome] = softmax(sim/temp)
        """
        if not self.enabled or not self.trajectories or len(self.live_sigs) < 1:
            return {
                "risk_by_horizon": {h: {"UNKNOWN": 1.0} for h in self.horizons},
                "likely_mode": "unknown",
                "confidence": 0.0,
                "evidence": []
            }
            
        # Compare with end of trajectory (assuming trajectory encodes the "ramp up")
        live_curr = np.array(self.live_sigs[-1])
        norm_live = np.linalg.norm(live_curr)
        if norm_live < 1e-9: norm_live = 1.0
        
        sims = []
        for traj in self.trajectories:
            if not traj.sig_seq: continue
            
            # Simple endpoint matching for v1 speed
            # Real version would use DTW on live_sigs vs traj.sig_seq
            traj_end = np.array(traj.sig_seq[-1])
            # Pad if dims mismatch (hack for demo robustness)
            if len(traj_end) != len(live_curr):
                 # re-init to match dimension
                 traj_end = np.zeros_like(live_curr)
            
            norm_traj = np.linalg.norm(traj_end)
            if norm_traj < 1e-9: norm_traj = 1.0
            
            sim = np.dot(live_curr, traj_end) / (norm_live * norm_traj)
            sims.append((sim, traj))
            
        sims.sort(key=lambda x: x[0], reverse=True)
        top_k = sims[:5]
        
        # Aggregate outcomes
        outcomes = {}
        total_score = 0.0
        
        for sim, traj in top_k:
            # Temperature scaling for softmax-like weight
            # Filter negative sims
            s_val = max(0.0, sim)
            score = np.exp(s_val * self.temp)
            outcomes[traj.outcome] = outcomes.get(traj.outcome, 0.0) + score
            total_score += score
            
        risk_dist = {}
        if total_score > 0:
            for out, sc in outcomes.items():
                risk_dist[out] = sc / total_score
        else:
            # Fallback
            risk_dist["UNKNOWN"] = 1.0
            
        # Horizon distribution 
        risk_by_horizon = {}
        for h in self.horizons:
            risk_by_horizon[h] = risk_dist
            
        best_outcome = max(risk_dist.items(), key=lambda x: x[1])[0] if risk_dist else "UNKNOWN"
        confidence = risk_dist.get(best_outcome, 0.0)
        
        return {
            "risk_by_horizon": risk_by_horizon,
            "likely_mode": best_outcome,
            "confidence": confidence,
            "evidence": [{"traj_outcome": t.outcome, "sim": float(s)} for s, t in top_k]
        }
