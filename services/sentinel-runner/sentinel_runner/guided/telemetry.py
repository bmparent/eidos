"""Replay-equivalent causal state transitions; offsets and event-time policy are explicit."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .causal import baseline, hash_json, initialize_engine, make_reservoir, restore, snapshot


def process(request: dict, directory: Path) -> dict:
    events = request["events"]
    checkpoint = request.get("checkpoint") or {}
    state = {"version": "eidos.telemetry.v1", "warmup": [], "errors": [], "lastTime": None,
             "lastValue": None, "issued": None, "processedOffset": 0, "accepted": 0, "late": 0, "gaps": 0,
             "resetId": request.get("resetId", "initial"), **checkpoint}
    engine, engine_path = initialize_engine(directory)
    reservoir = make_reservoir(engine, 1)
    if state.get("model"):
        restore(reservoir, state["model"])
    outcomes = []
    for event in events:
        if event["offset"] <= state["processedOffset"]:
            continue
        if event["offset"] != state["processedOffset"] + 1:
            raise ValueError("OFFSET_GAP: do not silently skip a stored event.")
        event_time = pd.Timestamp(event["eventTime"]).timestamp()
        outcome = {"offset": event["offset"], "recordId": event["id"], "eventTime": event["eventTime"],
                   "arrivalTime": event["arrivalTime"], "value": event["value"], "status": "warmup", "anomaly": False}
        if state["lastTime"] is not None and event_time <= state["lastTime"]:
            state["late"] += 1
            outcome["status"] = "late_or_out_of_order_retained_not_trained"
        else:
            state["accepted"] += 1
            if state["lastTime"] is not None and event_time - state["lastTime"] > request.get("staleSeconds", 300):
                state["gaps"] += 1; outcome["gapSeconds"] = event_time - state["lastTime"]
            if len(state["warmup"]) < 24:
                state["warmup"].append(event["value"])
                if len(state["warmup"]) == 24:
                    state["center"] = float(np.mean(state["warmup"]))
                    state["scale"] = max(float(np.std(state["warmup"])), 1e-6)
                    for previous, current in zip(state["warmup"][:-1], state["warmup"][1:]):
                        reservoir.listen(torch.tensor([(previous - state["center"]) / state["scale"]], dtype=torch.float32))
                        reservoir.adapt(torch.tensor([(current - state["center"]) / state["scale"]], dtype=torch.float32))
                        state["errors"].append(abs(current - previous) / state["scale"])
            else:
                issue = state["issued"]
                if issue:
                    residual = abs(event["value"] - issue["prediction"]) / state["scale"]
                    score = max(0., (residual - issue["median"]) / issue["scale"])
                    outcome.update(status="evaluated", score=score, threshold=issue["threshold"], anomaly=score >= issue["threshold"],
                                   forecast=issue, consequence="unreviewed", familiar=False)
                    if not outcome["anomaly"]:
                        reservoir.adapt(torch.tensor([(event["value"] - state["center"]) / state["scale"]], dtype=torch.float32))
                        state["errors"] = (state["errors"] + [residual])[-128:]
                value = event["value"] if not outcome["anomaly"] else state["lastValue"]
                reservoir.listen(torch.tensor([(value - state["center"]) / state["scale"]], dtype=torch.float32))
            state["lastTime"], state["lastValue"] = event_time, event["value"]
            if len(state["warmup"]) == 24:
                median, scale = baseline(state["errors"])
                state["issued"] = {"issuedAt": event["eventTime"], "afterOffset": event["offset"], "target": request["target"],
                                   "unit": request["unit"], "horizon": "next accepted event; event time unknown at issuance",
                                   "prediction": float((reservoir.W_out @ reservoir.state).item() * state["scale"] + state["center"]),
                                   "median": median, "scale": scale, "threshold": 5., "method": "canonical Torch RLS causal profile"}
                state["issued"]["commitment"] = hash_json(state["issued"])
        state["processedOffset"] = event["offset"]
        outcomes.append(outcome)
    state["model"] = snapshot(reservoir)
    return {"schema": "eidos.telemetry-result.v1", "checkpoint": state, "outcomes": outcomes,
            "processedOffset": state["processedOffset"], "inputOffsets": [e["offset"] for e in events],
            "engineSha256": __import__("hashlib").sha256(engine_path.read_bytes()).hexdigest(),
            "stateSha256": hash_json(state), "limits": ["Late events remain evidence but do not rewind trained state.",
              "First 24 accepted events are warmup; gaps and resets are separate from normal observations.",
              "Next-event forecast has no promised elapsed-time horizon; named time-horizon forecasts use dataset analysis."]}
