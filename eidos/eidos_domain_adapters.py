import numpy as np
import torch
from typing import Dict, Any, List, Optional, Union
import json
import logging

class DomainAdapter:
    domain = "generic"
    
    def extract(self, raw_event: Any) -> Dict[str, Any]:
        """
        Return dict with:
        - x: np.ndarray or torch.Tensor shape (FEATURES,)
        - entities: dict[str, Any] (top actors: ip, user, endpoint, sensor_id, etc.)
        - exemplars: list[str] or list[dict] (small evidence snippets)
        - feature_names: optional list[str]
        - privacy: {"contains_phi": bool, "redacted": bool}
        """
        raise NotImplementedError

class GenericVectorAdapter(DomainAdapter):
    domain = "generic"
    
    def extract(self, raw_event: Any) -> Dict[str, Any]:
        # Expects raw_event to be a vector-like object or a dict with 'x'
        if isinstance(raw_event, dict) and 'x' in raw_event:
            x = raw_event['x']
        else:
            x = raw_event
            
        # Ensure x is array
        if not isinstance(x, (np.ndarray, list, torch.Tensor)):
            try:
                x = np.array(x)
            except:
                x = np.zeros(10) # Fallback

        return {
            "x": x,
            "entities": {},
            "exemplars": [],
            "feature_names": None,
            "privacy": {"contains_phi": False, "redacted": False}
        }

class CyberSecurityAdapter(DomainAdapter):
    domain = "cyber"
    
    def extract(self, raw_event: Any) -> Dict[str, Any]:
        # Handle dict inputs (e.g. from JSON logs)
        x_vec = []
        entities = {}
        exemplars = []
        
        if isinstance(raw_event, dict):
            # 1. Extract vector
            if 'vector' in raw_event:
                x_vec = raw_event['vector']
            elif 'x' in raw_event:
                x_vec = raw_event['x']
            else:
                # Fallback: create numerical features from known keys
                # This is a stub. Real impl would normalize/embed.
                # Just usage of ports or simple counts
                try:
                    port = float(raw_event.get('dest_port', 0))
                    bytes_in = float(raw_event.get('bytes_in', 0))
                    bytes_out = float(raw_event.get('bytes_out', 0))
                    x_vec = np.array([port, bytes_in, bytes_out])
                except:
                    x_vec = np.zeros(10)
            
            # 2. Extract Entities
            if 'src_ip' in raw_event: entities['src_ip'] = raw_event['src_ip']
            if 'dest_ip' in raw_event: entities['dest_ip'] = raw_event['dest_ip']
            if 'user' in raw_event: entities['user'] = raw_event['user']
            
            # 3. Exemplars
            exemplars.append(str(raw_event)[:150])

        else:
            x_vec = raw_event

        return {
            "x": x_vec,
            "entities": entities,
            "exemplars": exemplars,
            "feature_names": ["port", "bytes_in", "bytes_out"] if len(x_vec) == 3 else None,
            "privacy": {"contains_phi": False, "redacted": False}
        }

class WebsiteMetricsAdapter(DomainAdapter):
    domain = "web"
    
    def extract(self, raw_event: Any) -> Dict[str, Any]:
        # Expects dict with rps, latency, errors, etc.
        x_vec = []
        feature_names = ["rps", "latency_p95", "error_rate", "cpu_load", "mem_usage"]
        entities = {}
        
        if isinstance(raw_event, dict):
            x_vec = [
                float(raw_event.get("rps", 0.0)),
                float(raw_event.get("latency_p95", 0.0)),
                float(raw_event.get("error_rate", 0.0)),
                float(raw_event.get("cpu_load", 0.0)),
                float(raw_event.get("mem_usage", 0.0))
            ]
            if "endpoint" in raw_event: entities["endpoint"] = raw_event["endpoint"]
            if "server_id" in raw_event: entities["server_id"] = raw_event["server_id"]
            if "status" in raw_event: entities["status"] = raw_event["status"]
            
        else:
            x_vec = raw_event

        return {
            "x": np.array(x_vec, dtype=np.float32),
            "entities": entities,
            "exemplars": [],
            "feature_names": feature_names,
            "privacy": {"contains_phi": False, "redacted": False}
        }

class DatasetRowAdapter(DomainAdapter):
    domain = "dataset"
    
    def extract(self, raw_event: Any) -> Dict[str, Any]:
        # Expects list/array or dict
        x_vec = raw_event
        entities = {}
        if isinstance(raw_event, dict):
             # Try to find numeric vector if 'x' present
             if 'x' in raw_event:
                 x_vec = raw_event['x']
             else:
                 # Naive: try to convert all values
                 vals = []
                 for k, v in raw_event.items():
                     if isinstance(v, (int, float)):
                         vals.append(v)
                 x_vec = vals
             
        return {
            "x": x_vec,
            "entities": {},
            "exemplars": [],
            "feature_names": None,
            "privacy": {"contains_phi": False, "redacted": False}
        }

class FlightDataAdapter(DomainAdapter):
    domain = "flight"
    
    def extract(self, raw_event: Any) -> Dict[str, Any]:
        # Pass through assuming raw_event is numeric vector from CSV
        # Entities could be sensor IDs if provided in a wrapper
        x_vec = raw_event
        entities = {}
        if isinstance(raw_event, dict):
            if 'x' in raw_event: x_vec = raw_event['x']
            if 'sensor_id' in raw_event: entities['sensor_id'] = raw_event['sensor_id']
            
        return {
            "x": x_vec,
            "entities": entities,
            "exemplars": [],
            "feature_names": None, # Could map if columns known
            "privacy": {"contains_phi": False, "redacted": False}
        }

class HealthcareAdapter(DomainAdapter):
    domain = "healthcare"
    
    def extract(self, raw_event: Any) -> Dict[str, Any]:
        # Strict redaction
        x_vec = raw_event
        if isinstance(raw_event, dict):
            # Extract only vitals, ignore PII
            vitals = []
            # Example whitelist
            for k in ["hr", "spo2", "bp_sys", "bp_dia", "resp_rate"]:
                vitals.append(float(raw_event.get(k, 0.0)))
            x_vec = vitals
            
        return {
            "x": x_vec,
            "entities": {}, # Don't pass patient IDs
            "exemplars": [],
            "feature_names": ["hr", "spo2", "bp_sys", "bp_dia", "resp_rate"],
            "privacy": {"contains_phi": True, "redacted": True}
        }

def get_domain_adapter(domain: str) -> DomainAdapter:
    mapping = {
        "generic": GenericVectorAdapter,
        "cyber": CyberSecurityAdapter,
        "web": WebsiteMetricsAdapter,
        "dataset": DatasetRowAdapter,
        "flight": FlightDataAdapter,
        "healthcare": HealthcareAdapter
    }
    return mapping.get(domain, GenericVectorAdapter)()
