from __future__ import annotations
import hashlib

def build_world_features(events:list[dict])->list[dict]:
    out=[]
    for e in events:
        text=(e.get('title','')+' '+e.get('text','')).lower()
        tokens=[t for t in text.split() if len(t)>2]
        timestamp=e.get('timestamp') or e.get('published_at_utc') or e.get('observed_at_utc')
        out.append({**e,'timestamp':timestamp,'entities':tokens[:5],'topic_hash':hashlib.md5(' '.join(sorted(set(tokens))).encode()).hexdigest()[:8],'urgency_proxy':1.0 if any(k in text for k in ['surge','storm','crisis']) else 0.2})
    return out
