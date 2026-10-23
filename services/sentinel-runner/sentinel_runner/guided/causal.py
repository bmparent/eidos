"""Causal product profile using the canonical Torch RLS_Reservoir, never legacy best_pred.

Predictions and thresholds are committed from pre-observation state. The adapter is
separately versioned; neither historical engine defaults nor proof receipts change.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..engine_bridge import discover_engine_path, load_engine
from .ingestion import confirm, parse_time, plain

POLICY = "eidos.causal-rls.v1"
THRESHOLD = 5.0
MAX_ENTITIES = 8


def hash_json(value) -> str:
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def baseline(errors: list[float], floor: float = 0.02) -> tuple[float, float]:
    if not errors:
        return 0.0, 1.0
    values = np.asarray(errors[-128:])
    median = float(np.median(values))
    return median, max(floor, float(1.4826 * np.median(np.abs(values - median))))


def initialize_engine(artifact_dir: Path, bands: int = 1):
    engine_path = discover_engine_path()
    engine = load_engine(engine_path, artifact_dir)
    engine.device = torch.device("cpu")
    engine.DTYPE = torch.float32
    engine.EIDOS_BRAIN_CONFIG.update({"fractal_bands": bands, "thermo_enabled": False,
                                      "rls_rejuvenate_every": 5000})
    torch.set_num_threads(1)
    return engine, engine_path


def make_reservoir(engine, dimensions: int):
    return engine.RLS_Reservoir(dimensions, n_reservoir=64, spectral_radius=0.9,
                                forgetting=0.995, leak_rate=0.1, input_scaling=0.3)


def snapshot(model) -> dict:
    state = {name: getattr(model, name).detach().cpu().tolist() for name in ["state", "W_out", "P"]} | {"adaptStep": model._adapt_step}
    if model.thermo_enabled:
        state["regulation"] = {"W_res": model.W_res.detach().cpu().tolist(), "currentRho": model.current_rho,
                               "temperature": model.temperature, "forgetting": model.forgetting, "energyEMA": model.energy_ema}
    return state


def restore(model, state: dict):
    for name in ["state", "W_out", "P"]:
        expected = getattr(model, name)
        value = torch.tensor(state[name], dtype=expected.dtype)
        if value.shape != expected.shape or not torch.isfinite(value).all():
            raise ValueError("INVALID_CHECKPOINT")
        setattr(model, name, value)
    model._adapt_step = int(state["adaptStep"])


def prepare_groups(dataset: dict, mapping: dict) -> list[tuple[str, list[dict]]]:
    groups = {}
    for index, row in enumerate(dataset["records"]):
        entity = str(row.get(mapping.get("entity"), "all"))
        session = str(row.get(mapping.get("session"), "all"))
        key = json.dumps([entity, session], separators=(",", ":"))
        groups.setdefault(key, []).append({"id": dataset["recordIds"][index], "row": row,
                                           "time": parse_time(row[mapping["timestamp"]], mapping.get("timezone"))})
    if len(groups) > MAX_ENTITIES:
        raise ValueError("ENTITY_LIMIT: this profile supports at most eight entity/session groups.")
    return [(key, sorted(rows, key=lambda r: r["time"])) for key, rows in groups.items()]


def temporal(dataset: dict, mapping: dict, options: dict, artifact_dir: Path) -> dict:
    started = time.monotonic()
    mapping = confirm(dataset, mapping)
    if mapping["mode"] != "temporal":
        raise ValueError("CHRONOLOGY_REQUIRED: forecasts need confirmed timestamps, not file order.")
    features = mapping["features"]
    target = options.get("target", features[0])
    if target not in features:
        raise ValueError("NAMED_TARGET_REQUIRED")
    target_index = features.index(target)
    horizon = options.get("horizonSeconds", 60)
    window = options.get("windowSeconds", horizon)
    if not isinstance(horizon, (float, int)) or not 1 <= horizon <= 86400 or not isinstance(window, (float, int)) or not 1 <= window <= 86400:
        raise ValueError("HORIZON_LIMIT: horizon and matching window must be 1–86,400 seconds.")
    bands = 4 if options.get("mechanism") == "multiscale" else 1
    engine, engine_path = initialize_engine(artifact_dir, bands)
    issued, observations, findings, group_receipts, omitted = [], [], [], [], []
    evaluation_errors, interval_hits, widths, persistence_errors = [], [], [], []
    with (artifact_dir / "issued_forecasts.jsonl").open("w", encoding="utf-8") as audit:
        for key, rows in prepare_groups(dataset, mapping):
            if len(rows) < 48:
                raise ValueError(f"INSUFFICIENT_HISTORY: group {key} needs at least 48 observations; received {len(rows)}.")
            # Fixed upfront; appending future rows cannot move the calibration boundary.
            calibration_count = 24
            raw = np.asarray([[pd.to_numeric(row["row"].get(f), errors="coerce") for f in features] for row in rows], dtype=float)
            calibration = raw[:calibration_count]
            median = np.nanmedian(calibration, axis=0)
            if not np.isfinite(median).all():
                raise ValueError("EMPTY_CALIBRATION_FEATURE: no numeric calibration value.")
            imputed = np.where(np.isfinite(calibration), calibration, median)
            center, scale = imputed.mean(axis=0), imputed.std(axis=0)
            scale = np.maximum(scale, 1e-6)
            model = make_reservoir(engine, len(features))
            model.thermo_enabled = options.get("mechanism") == "regulation"
            errors, absolute_errors, prior_losses, pending = [], [], [[], []], []
            last = None
            last_time = None
            seen = 0
            contract_hash = hash_json({"center": center, "scale": scale, "median": median,
                                      "calibrationIds": [r["id"] for r in rows[:calibration_count]],
                                      "mapping": {k: mapping.get(k) for k in ["features", "timestamp", "timezone", "entity", "session", "units", "missing", "reference", "version"]}})
            chain = "0" * 64
            for index, row in enumerate(rows):
                current = raw[index]
                if not np.isfinite(current).all():
                    if mapping["missing"] == "exclude":
                        omitted.append({"recordId": row["id"], "reason": "missing measurement"})
                        continue
                    current = np.where(np.isfinite(current), current, median)
                value = torch.tensor((current - center) / scale, dtype=torch.float32)
                instant = pd.Timestamp(row["time"]).timestamp()
                warmup = index < calibration_count
                if index < calibration_count - 1:
                    # Calibration is retrospective preparation, never an issued
                    # forecast. All prefix observations are known before the first
                    # issue at the end of the fixed calibration period.
                    if last is not None:
                        errors.append(float(torch.linalg.norm(value - last)))
                        absolute_errors.append(float(abs((value[target_index] - last[target_index]).item() * scale[target_index])))
                        model.adapt(value)
                    model.listen(value)
                    last = value
                    seen += 1
                    continue
                # The most recent eligible issuance is used once. Other overlapping
                # issues are explicitly accounted as superseded, never extra events.
                eligible = [p for p in pending if p["targetEpoch"] <= instant]
                pending = [p for p in pending if p["targetEpoch"] > instant]
                committed = eligible[-1] if eligible else None
                for old in eligible[:-1]:
                    old["forecast"]["resolution"] = "superseded within matching window"
                is_alert = False
                if committed:
                    issue = committed["forecast"]
                    if instant - committed["targetEpoch"] > window:
                        issue["resolution"] = "expired: no observation in declared matching window"
                    else:
                        residual = float(np.linalg.norm((current - committed["prediction"]) / scale))
                        score = max(0.0, (residual - issue["baselineMedian"]) / issue["baselineScale"])
                        is_alert = not warmup and score >= issue["threshold"]
                        error = abs(float(current[target_index]) - issue["prediction"])
                        hit = issue["lower"] is not None and issue["lower"] <= float(current[target_index]) <= issue["upper"]
                        observation = {"recordId": row["id"], "entity": key, "time": row["time"], "forecastId": issue["id"],
                                       "actual": float(current[target_index]), "expected": issue["prediction"], "score": score,
                                       "threshold": issue["threshold"], "anomaly": is_alert, "warmup": warmup,
                                       "consequence": "unknown; no reviewed outcome or operational rule", "lower": issue["lower"], "upper": issue["upper"],
                                       "target": target, "unit": mapping["units"][target], "delaySeconds": instant - committed["targetEpoch"],
                                       "referenceRecords": issue["referenceRecords"], "drivers": [
                                           {"feature": f, "observed": float(current[j]), "expected": float(committed["prediction"][j]),
                                            "absoluteStandardizedResidual": float(abs((current[j] - committed["prediction"][j]) / scale[j]))}
                                           for j, f in enumerate(features)]}
                        observations.append(observation)
                        issue["resolution"] = "matched"
                        if not warmup:
                            evaluation_errors.append(error)
                            persistence_errors.append(abs(float(current[target_index]) - committed["persistence"][target_index]))
                            if issue["lower"] is not None:
                                interval_hits.append(bool(hit)); widths.append(issue["upper"] - issue["lower"])
                        # Historical loss updates occur only AFTER the observation has
                        # resolved an already issued prediction. No oracle selection.
                        for predictor in range(2):
                            prior_losses[predictor].append(float(np.mean(np.abs(current - committed["candidates"][predictor]) / scale)))
                            prior_losses[predictor] = prior_losses[predictor][-128:]
                        if is_alert:
                            findings.append(observation)
                        if warmup or not is_alert or options.get("mechanism") == "adapt_all":
                            errors.append(residual); absolute_errors.append(error)
                            errors, absolute_errors = errors[-128:], absolute_errors[-128:]
                            # Learn the observed target from its issuance state, then
                            # restore the current recurrent state before consuming x_t.
                            current_state = model.state
                            model.state = committed["state"]
                            model.adapt(value)
                            model.state = current_state
                        if model.thermo_enabled:
                            model.update_thermodynamics({"error_rms": residual / np.sqrt(len(features)), "surprise_score": score})
                # Freeze learning on anomalous frames. Familiarity never suppresses alerts.
                model.listen(value if not is_alert or last is None else last)
                last = value
                last_time = instant
                seen += 1
                # These are named feature forecasts; no projection is inverted.
                reservoir = (model.W_out @ model.state).detach().numpy() * scale + center
                persistence = current.copy()
                losses = [float(np.mean(loss)) if loss else None for loss in prior_losses]
                choice = 0 if losses[0] is not None and losses[0] < losses[1] else 1
                prediction = [reservoir, persistence][choice]
                med, sigma = baseline(errors)
                radius = float(np.quantile(absolute_errors, 0.9, method="higher")) if len(absolute_errors) >= 20 else None
                prediction_scalar = float(prediction[target_index])
                issue = {"id": f"forecast-{len(issued) + 1}", "sourceRecordId": row["id"], "entity": key,
                         "issuedAt": row["time"], "targetTime": pd.Timestamp(instant + horizon, unit="s", tz="UTC").isoformat(),
                         "horizonSeconds": horizon, "matchingWindowSeconds": window, "target": target, "unit": mapping["units"][target],
                         "prediction": prediction_scalar, "lower": prediction_scalar - radius if radius is not None else None,
                         "upper": prediction_scalar + radius if radius is not None else None, "nominalCoverage": 0.9,
                         "uncertaintyMethod": "past 128 accepted absolute errors; empirical 90th percentile; no drift guarantee",
                         "predictor": "eidos_rls" if choice == 0 else "persistence", "historicalLoss": losses,
                         "threshold": THRESHOLD, "baselineMedian": med, "baselineScale": sigma,
                         "preprocessingHash": contract_hash, "stateHash": hash_json(snapshot(model)),
                         "referenceRecords": [r["id"] for r in rows[:calibration_count]], "warmup": warmup}
                commitment = hash_json({"previous": chain, "forecast": issue})
                issue["commitment"] = commitment
                issue["previousCommitment"] = chain
                chain = commitment
                # Flush immutable issuance BEFORE the loop can inspect the target row.
                audit.write(json.dumps(issue, allow_nan=False) + "\n"); audit.flush()
                issued.append(issue)
                pending.append({"forecast": issue, "targetEpoch": instant + horizon, "prediction": prediction,
                                "persistence": persistence, "candidates": [reservoir, persistence], "state": model.state.clone()})
                if len(pending) > 5000:
                    raise ValueError("HORIZON_BUFFER_LIMIT")
            for item in pending:
                item["forecast"]["resolution"] = "future target not observed"
            group_receipts.append({"entity": key, "calibrationRows": calibration_count, "processedRows": seen,
                                   "calibrationRecordIds": [r["id"] for r in rows[:calibration_count]],
                                   "normalization": {"center": plain(center), "scale": plain(scale), "median": plain(median)},
                                   "preprocessingHash": contract_hash, "checkpoint": snapshot(model), "lastEventTime": last_time,
                                   "chainHead": chain})
    elapsed = time.monotonic() - started
    return {"schema": POLICY, "method": "Causal canonical Torch RLS reservoir with historically selected persistence baseline",
            "evidenceClass": "PRODUCT_ENGINEERING", "gatesAdvanced": 0, "sourceBadge": "Your data · actual Torch Eidos + identified baseline",
            "forecasts": issued, "observations": observations, "findings": group_findings(findings, window),
            "groups": group_receipts, "excludedRecords": omitted, "mapping": mapping,
            "metrics": {"forecastMAE": float(np.mean(evaluation_errors)) if evaluation_errors else None,
                        "persistenceMAE": float(np.mean(persistence_errors)) if persistence_errors else None,
                        "intervalCoverage": float(np.mean(interval_hits)) if interval_hits else None,
                        "intervalMeanWidth": float(np.mean(widths)) if widths else None, "evaluatedForecasts": len(evaluation_errors),
                        "intervalCount": len(interval_hits), "precision": None, "recall": None, "elapsedSeconds": elapsed,
                        "rowsPerSecond": len(dataset["records"]) / elapsed, "llmCalls": 0, "llmCostUSD": 0},
            "engine": {"codeSha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(), "torchVersion": torch.__version__,
                       "class": "RLS_Reservoir", "policy": POLICY, "processIsolation": True,
                       "config": {"reservoir": 64, "features": len(features), "spectralRadius": 0.9, "leak": 0.1,
                                  "forgetting": 0.995, "seed": 42, "fractalBands": bands, "domain": "confirmed_named_measurements", "projection": None,
                                  "learning": "all observations" if options.get("mechanism") == "adapt_all" else "skip anomalous observations",
                                  "experimentalMechanism": options.get("mechanism", "none"), "threshold": THRESHOLD}},
            "limitations": ["Engineering profile; useful detection and operational severity are not established.",
                            "Unlabeled input has no measured precision or recall. Empty findings do not imply safety.",
                            "Rolling error bands use accepted past errors; drift and contamination can reduce coverage.",
                            "Forecast selection may use persistence; its outputs are identified separately from Eidos.",
                            "Calibration prefix is a declared reference, not independently verified benign data.",
                            "Duplicate entity timestamps are rejected. Irregular times use the explicit horizon and matching window."]}


def group_findings(events: list[dict], seconds: float) -> list[dict]:
    groups = []
    for event in events:
        prior = next((g for g in reversed(groups) if g["entity"] == event["entity"] and
                      0 <= (pd.Timestamp(event["time"]) - pd.Timestamp(g["end"])).total_seconds() <= seconds), None)
        if prior:
            prior["members"].append(event); prior["end"] = event["time"]
            prior["score"] = max(prior["score"], event["score"])
            continue
        groups.append({"schema": "eidos.finding.v1", "id": f"finding-{len(groups) + 1}", "entity": event["entity"],
                       "start": event["time"], "end": event["time"], "score": event["score"], "members": [event],
                       "what": f"{event['target']} deviated from its committed forecast.",
                       "whyItMatters": "Unusual relative to the confirmed reference; consequence is unreviewed.",
                       "basis": "normalized residual above a threshold committed before this observation",
                       "uncertainty": ["A regime change, collection problem or rare valid event can also explain this difference."],
                       "nextAction": "Inspect original records and compare the operating context before assigning severity.",
                       "grouping": f"Same entity/session, consecutive anomalies separated by at most {seconds} seconds.",
                       "ranking": "maximum member anomaly score; not a severity probability", "detector": POLICY})
    return sorted(groups, key=lambda g: (-g["score"], g["id"]))


def unordered(dataset: dict, mapping: dict, artifact_dir: Path) -> dict:
    from sklearn.ensemble import IsolationForest
    mapping = confirm(dataset, mapping)
    if len(dataset["records"]) < 10:
        raise ValueError("INSUFFICIENT_ROWS: unordered analysis needs at least ten records.")
    features = mapping["features"]
    values = np.asarray([[pd.to_numeric(row.get(f), errors="coerce") for f in features] for row in dataset["records"]], dtype=float)
    # Unordered mode describes the whole supplied reference, not future performance.
    valid = np.isfinite(values).all(axis=1)
    if mapping["missing"] == "prefix_median":
        raise ValueError("UNORDERED_MISSING_POLICY: select exclusion; a calibration prefix has no chronological meaning here.")
    if valid.sum() < 10:
        raise ValueError("INSUFFICIENT_COMPLETE_ROWS")
    x = values[valid]
    med = np.median(x, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(x - med), axis=0), 1e-6)
    robust_scores = np.max(np.abs(x - med) / scale, axis=1)
    forest = IsolationForest(n_estimators=100, random_state=42, n_jobs=1, contamination="auto").fit(x)
    scores = -forest.score_samples(x)
    cutoff = float(np.quantile(scores, .95, method="higher"))
    findings, observations = [], []
    for offset, index in enumerate(np.flatnonzero(valid)):
        record_id = dataset["recordIds"][int(index)]
        event = {"recordId": record_id, "score": float(scores[offset]), "robustScore": float(robust_scores[offset]),
                 "anomaly": bool(scores[offset] > cutoff), "entity": str(dataset["records"][int(index)].get(mapping.get("entity"), "all")),
                 "threshold": cutoff}
        observations.append(event)
        if event["anomaly"]:
            findings.append({"schema": "eidos.finding.v1", "id": f"finding-{record_id}", "members": [event], "entity": event["entity"],
                             "score": event["score"], "what": "Unusual combination of measurements within this dataset.",
                             "whyItMatters": "Relative rarity only; operational consequence is unknown.",
                             "basis": "Isolation Forest score above the declared whole-reference 95th percentile",
                             "uncertainty": ["In-sample ranking; not a calibrated anomaly probability or measured false-alert rate."],
                             "nextAction": "Compare this record with peers and review the source values.", "detector": "sklearn.IsolationForest",
                             "grouping": "One finding per unordered record; no invented time sequence.", "ranking": "Isolation Forest score"})
    return {"schema": "eidos.unordered.v1", "method": "Isolation Forest (identified baseline), with robust median/MAD comparison",
            "sourceBadge": "Your data · Isolation Forest baseline", "engine": None, "mapping": mapping,
            "observations": observations, "findings": findings, "forecasts": [], "gatesAdvanced": 0,
            "metrics": {"precision": None, "recall": None, "rows": len(x), "referencePercentile": .95},
            "excludedRecords": [{"recordId": dataset["recordIds"][i], "reason": "missing measurement"} for i in np.flatnonzero(~valid)],
            "limitations": ["Unordered in-sample exploration; no Torch temporal-engine claim, forecast or chronology.",
                            "Reference percentile is a review budget, not an operational false-alert guarantee."]}
