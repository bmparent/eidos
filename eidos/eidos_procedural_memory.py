import json
import os
import numpy as np
from typing import Dict, List, Any, Optional

class ProceduralMemory:
    def __init__(self, domain: str, policy: str = "recommend", min_similarity: float = 0.35, enabled: bool = True):
        self.domain = domain
        self.policy = policy # recommend | auto
        self.min_similarity = min_similarity
        self.enabled = enabled
        
        self.proto: Dict[str, Optional[np.ndarray]] = {}   # signature prototypes
        self.Q: Dict[str, float] = {}            # learned success
        self.N: Dict[str, int] = {}              # counts
        
        self._load_defaults()

    def _load_defaults(self):
        # Initialize with standard actions for the domain
        defaults = {
            "cyber": ["RATE_LIMIT", "BLOCK_TOP_IPS", "MFA_CHALLENGE", "LOCK_TARGETED_ACCOUNTS", "ISOLATE_HOST", "ROTATE_CREDS"],
            "web": ["ROLLBACK_DEPLOY", "DISABLE_FEATURE_FLAG", "SCALE_UP", "SHED_LOAD", "PURGE_CACHE", "BOT_THROTTLE"],
            "dataset": ["QUARANTINE_PARTITION", "REVERT_SCHEMA", "BACKFILL", "RETRAIN_GUARDRAIL"],
            "flight": ["FLAG_MAINTENANCE_INSPECTION", "COMPARE_FLEET_COHORT", "SCHEDULE_COMPONENT_CHECK"],
            "healthcare": ["CLINICIAN_ALERT", "RECHECK_LABS", "VERIFY_DEVICE_ARTIFACT", "CHECKLIST_PROMPT"],
            "generic": ["ALERT_HUMAN", "LOG_DEBUG", "INCREASE_SAMPLING"]
        }
        
        actions = defaults.get(self.domain, defaults["generic"])
        for act in actions:
            if act not in self.proto:
                self.proto[act] = None
                self.Q[act] = 0.5 # start neutral
                self.N[act] = 0

    def rank_actions(self, signature_vec: Any, regime: str) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
            
        scores = []
        # Handle conversion
        sig_arr = np.array(signature_vec)
        if hasattr(signature_vec, "cpu"): sig_arr = signature_vec.cpu().numpy()
        
        sig_norm = np.linalg.norm(sig_arr)
        
        for act, proto_vec in self.proto.items():
            sim = 0.0
            if proto_vec is not None:
                p_norm = np.linalg.norm(proto_vec)
                if sig_norm > 1e-6 and p_norm > 1e-6:
                    sim = float(np.dot(sig_arr, proto_vec) / (sig_norm * p_norm))
            else:
                # If no prototype, we give a moderate score to encourage trying it initially
                # provided we are in a regime that needs it.
                sim = 0.2

            # Composite score: Similarity * Quality
            # score = sim * (0.5 + 0.5 * sigmoid(Q))
            # Just using linear Q for now clipped 0..1
            q_val = max(0.0, min(1.0, self.Q.get(act, 0.5)))
            
            # If we have no prototype, similarity is weak, but Q helps
            score = sim * 0.5 + q_val * 0.5
            
            scores.append({
                "action": act,
                "score": score,
                "sim": sim,
                "q": q_val,
                "policy": self.policy
            })
            
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores

    def update_reward(self, action: str, reward: float):
        # EMA update
        if action not in self.Q: return
        alpha = 0.1
        curr = self.Q.get(action, 0.5)
        new_val = (1.0 - alpha) * curr + alpha * reward
        self.Q[action] = new_val
        self.N[action] = self.N.get(action, 0) + 1

    def update_prototype(self, action: str, signature_vec: Any, eta: float = 0.1):
        if action not in self.proto:
            return
            
        sig_arr = np.array(signature_vec)
        if hasattr(signature_vec, "cpu"): sig_arr = signature_vec.cpu().numpy()
            
        curr_proto = self.proto[action]
        if curr_proto is None:
            self.proto[action] = sig_arr.copy()
        else:
            # Prototype learning: move towards new instance
            # p_new = (1-eta)*p + eta*s
            self.proto[action] = (1.0 - eta) * curr_proto + eta * sig_arr

    def save_bank(self, path: str):
        out = {
            "domain": self.domain,
            "actions": []
        }
        for act in self.Q:
            out["actions"].append({
                "name": act,
                "q": self.Q[act],
                "n": self.N[act],
                "has_proto": self.proto[act] is not None
                # Prototypes are not serialized in this v1 for simplicity
            })
        try:
            with open(path, 'w') as f:
                json.dump(out, f, indent=2)
        except Exception as e:
            # Silently fail or log in real app
            pass

    def load_bank(self, path: str):
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            if data.get("domain") != self.domain:
                return # Mismatch
            
            for item in data.get("actions", []):
                nm = item["name"]
                if nm in self.Q:
                    self.Q[nm] = item["q"]
                    self.N[nm] = item.get("n", 0)
        except Exception:
            pass
