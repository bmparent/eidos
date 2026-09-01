"""Full-engine capture and paired shadow evaluation for EIDOS-GP-v1."""

from __future__ import annotations

from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .frame_observer import FrameObserver, canonical_json, canonical_sha256, read_capture
from .grand_proof_metrics import (
    MetricResult,
    anomaly_preservation,
    compression_reference,
    detector_metrics,
    evidence_completeness,
    eidos_value_scalar,
    holm_adjust,
    paired_bootstrap,
    pareto_front,
    registered_sensitivity_grid,
    split_rmse,
)
from .grand_proof_scenarios import SCENARIO_IDS, ScenarioConfig, ScenarioStream, generate_scenario
from .meaningful_surprise import (
    CausalConsequenceMemory,
    DomainContract,
    MeaningfulSurprisePolicy,
    MemoryConsequence,
    PolicyConfig,
    verify_canonical_decision,
)
from .representations import RepresentationPipeline


SYSTEM_IDS = (
    "eidos_ms_full",
    "eidos_live_current",
    "eidos_minimal",
    "rolling_z",
    "ewma",
    "cusum",
    "isolation_forest",
    "knn_episode",
)
ABLATION_IDS = (
    "A0_full",
    "A1_no_hdc",
    "A2_no_geometry",
    "A3_no_multiscale",
    "A4_no_thermo",
    "A5_no_scout",
    "A6_no_raw_escape",
    "A7_no_voi",
)
BYTE_OPERATING_POINTS = (0.10, 0.25, 0.50, 1.00)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def synthetic_domain_contract() -> DomainContract:
    return DomainContract.from_dict(
        {
            "contract_id": "EIDOS-GP-v1-synthetic-bounded-review",
            "actions": ["observe", "review", "escalate"],
            "outcomes": ["benign", "harmful"],
            "loss": {
                "observe": {"benign": 0.0, "harmful": 1.0},
                "review": {"benign": 0.20, "harmful": 0.18},
                "escalate": {"benign": 0.65, "harmful": 0.05},
            },
            "horizon": 64,
            "costs": {"bytes": 0.20, "latency": 0.10, "false_alert": 0.25, "human_review": 0.35},
            "provenance": "EIDOS-GP-v1 locked synthetic domain contract",
        }
    )


@dataclass(frozen=True)
class RunnerConfig:
    artifact_root: Path
    repo_root: Path
    reservoir: int
    scenario_config: ScenarioConfig
    code_commit: str
    thresholds: Mapping[str, float]
    byte_operating_points: tuple[float, ...] = BYTE_OPERATING_POINTS

    @property
    def hash(self) -> str:
        return "sha256:" + canonical_sha256(
            {
                "reservoir": self.reservoir,
                "scenario_config": asdict(self.scenario_config),
                "code_commit": self.code_commit,
                "thresholds": dict(self.thresholds),
                "byte_operating_points": self.byte_operating_points,
            }
        )


def _engine_module() -> Any:
    from eidos_brain.engine import eidos_v0_4_7_02 as engine

    return engine


def capture_live_scenario(
    scenario: ScenarioStream,
    *,
    out_dir: Path,
    reservoir: int,
    code_commit: str,
    replay_command: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the actual live engine once and capture only completed decisions."""

    engine = _engine_module()
    capture_path = out_dir / "live_frame_observer.jsonl"
    engine_artifacts = out_dir / "live_engine_artifacts"
    engine_config = {
        "steps": scenario.config.total_frames,
        "warmup_cap": scenario.config.warmup_frames,
        "reservoir": int(reservoir),
        "residual_codec_enabled": True,
        "residual_codec_store_prediction": True,
        "residual_codec_store_jsonl": True,
    }
    config_hash = "sha256:" + canonical_sha256(engine_config)
    observer = FrameObserver(
        capture_path,
        run_id=f"{scenario.scenario_id}-seed-{scenario.seed}",
        config_hash=config_hash,
        code_commit=code_commit,
        replay_command=replay_command,
    )
    started = time.perf_counter()
    engine_log = out_dir / "live_engine.log"
    try:
        engine.reset_runtime_state()
        engine._apply_runtime_config(
            {
                "artifact_root": str(engine_artifacts),
                "profile_label": f"gp_{scenario.scenario_id}_{scenario.seed}",
                "engine_config": engine_config,
            }
        )
        with engine_log.open("w", encoding="utf-8", newline="\n") as log_handle:
            with redirect_stdout(log_handle), redirect_stderr(log_handle):
                engine.run_sentinel_stream(
                    gen_factory=lambda: scenario.online_frames(),
                    est_frames=scenario.config.total_frames,
                    features=scenario.config.features,
                    profile_label=f"gp_{scenario.scenario_id}_{scenario.seed}",
                    session_label="grand_proof_v1",
                    warmup=scenario.config.warmup_frames,
                    sample_geometry=False,
                    save_surprise_artifacts=False,
                    proof_observer=observer,
                )
        status = observer.finalize()
    except Exception as exc:
        observer.close_partial(f"{type(exc).__name__}: {exc}")
        raise
    elapsed = time.perf_counter() - started
    records = read_capture(capture_path)
    receipt = {
        "status": status["status"],
        "capture_path": capture_path.as_posix(),
        "engine_log_path": engine_log.as_posix(),
        "records": len(records),
        "runtime_seconds": elapsed,
        "frames_per_second": len(records) / max(elapsed, 1e-12),
        "reservoir": reservoir,
        "engine_config": engine_config,
        "engine_config_hash": config_hash,
        "prediction_source": "live_eidos_consensus_best_pred",
        "sentinel_source": "live_sentinel_analyze",
        "hdc_source": "live_hippocampus_metrics",
    }
    return records, receipt


def _causal_baseline_scores(records: Sequence[Mapping[str, Any]], name: str) -> np.ndarray:
    frames = [np.asarray(record["frame"], dtype=np.float64) for record in records]
    residual = np.asarray([float(record["normalized_error"]) for record in records], dtype=np.float64)
    scores = np.zeros(len(records), dtype=np.float64)
    if name == "eidos_live_current":
        return np.asarray([float(record["surprise_score"]) for record in records], dtype=np.float64)
    if name == "rolling_z":
        history: deque[float] = deque(maxlen=128)
        for index, value in enumerate(residual):
            mean = float(np.mean(history)) if history else value
            std = float(np.std(history)) if len(history) > 1 else 1.0
            scores[index] = abs(value - mean) / max(std, 1e-6)
            history.append(float(value))
        return scores
    if name == "ewma":
        mean = float(residual[0]) if residual.size else 0.0
        deviation = 1.0
        for index, value in enumerate(residual):
            scores[index] = abs(value - mean) / max(deviation, 1e-6)
            error = abs(value - mean)
            mean = 0.95 * mean + 0.05 * value
            deviation = 0.95 * deviation + 0.05 * error
        return scores
    if name == "cusum":
        mean = float(residual[0]) if residual.size else 0.0
        positive = negative = 0.0
        for index, value in enumerate(residual):
            delta = value - mean
            positive = max(0.0, positive + delta - 0.05)
            negative = min(0.0, negative + delta + 0.05)
            scores[index] = max(positive, -negative)
            mean = 0.99 * mean + 0.01 * value
        return scores
    if name == "eidos_minimal":
        raw = _causal_baseline_scores(records, "rolling_z")
        scalar_history: deque[float] = deque(maxlen=64)
        spectral = np.zeros(len(records), dtype=np.float64)
        spectral_history: deque[float] = deque(maxlen=256)
        for index, frame in enumerate(frames):
            scalar_history.append(float(frame.mean()))
            if len(scalar_history) >= 8:
                values = np.asarray(scalar_history)
                energy = float(np.abs(np.fft.rfft((values - values.mean()) * np.hanning(values.size)))[1:8].mean())
            else:
                energy = 0.0
            center = float(np.mean(spectral_history)) if spectral_history else energy
            scale = float(np.std(spectral_history)) if len(spectral_history) > 1 else 1.0
            spectral[index] = abs(energy - center) / max(scale, 1e-6)
            spectral_history.append(energy)
        return np.maximum(raw, spectral)
    if name == "knn_episode":
        history: deque[np.ndarray] = deque(maxlen=512)
        for index, frame in enumerate(frames):
            if len(history) < 5:
                scores[index] = 0.0
            else:
                distances = sorted(float(np.linalg.norm(frame - candidate)) for candidate in history)
                scores[index] = float(np.mean(distances[:5]))
            history.append(frame)
        return scores
    if name == "isolation_forest":
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError as exc:
            raise RuntimeError("scikit-learn dependency unavailable") from exc
        if len(frames) < 64:
            raise RuntimeError("insufficient causal calibration rows for Isolation Forest")
        split = min(max(32, len(frames) // 4), len(frames) - 1)
        model = IsolationForest(n_estimators=100, random_state=20260901, contamination="auto")
        model.fit(np.vstack(frames[:split]))
        scores[split:] = -model.score_samples(np.vstack(frames[split:]))
        return scores
    raise ValueError(f"unknown baseline: {name}")


def _threshold_for(name: str, thresholds: Mapping[str, float]) -> float:
    defaults = {
        "eidos_live_current": 1.5,
        "eidos_minimal": 3.0,
        "rolling_z": 3.0,
        "ewma": 3.0,
        "cusum": 8.0,
        "isolation_forest": 0.65,
        "knn_episode": 0.45,
    }
    return float(thresholds.get(name, defaults[name]))


def _common_wrapper(
    *,
    record: Mapping[str, Any],
    score: float,
    threshold: float,
    selected_representation: str,
    action: str,
    uncertainty: float,
    raw_retained: bool,
) -> dict[str, Any]:
    return {
        "observation": {"frame_id": record["frame_id"], "source_id": record["source_id"]},
        "score": float(score),
        "threshold": float(threshold),
        "source_range": record["source_range"],
        "selected_representation": selected_representation,
        "uncertainty": float(uncertainty),
        "next_action": action,
        "raw_retained": bool(raw_retained),
        "raw_references": record.get("source_refs") if raw_retained else [],
        "config_hash": record["config_hash"],
        "code_commit": record["code_commit"],
        "replay_command": record["replay_command"],
    }


def _reconstruct(records: Sequence[Mapping[str, Any]], fidelities: Sequence[str]) -> np.ndarray:
    output: list[np.ndarray] = []
    for record, fidelity in zip(records, fidelities):
        frame = np.asarray(record["frame"], dtype=np.float64)
        pred = np.asarray(record["best_pred"], dtype=np.float64)
        residual = frame - pred
        if fidelity == "raw_frame_plus_full_context" or fidelity == "structured_residual":
            decoded = frame
        elif fidelity == "quantized_residual":
            decoded = pred + np.rint(residual / 0.025) * 0.025
        else:
            decoded = pred
        output.append(decoded)
    return np.vstack(output)


def _byte_accounting(
    records: Sequence[Mapping[str, Any]],
    fidelities: Sequence[str],
    wrappers: Sequence[Mapping[str, Any]],
) -> dict[str, int | float]:
    payload_cost = {
        "reference_or_null": 1,
        "quantized_residual": 2,
        "structured_residual": 4,
        "raw_frame_plus_full_context": 8,
    }
    features = len(records[0]["frame"]) if records else 0
    payload_bytes = sum(payload_cost[value] * features for value in fidelities)
    index_bytes = len(records) * 8
    card_bytes = sum(len(canonical_json(wrapper).encode("utf-8")) for wrapper in wrappers)
    model_state_bytes = 0
    manifest_bytes = len(canonical_json({"frames": len(records), "features": features}).encode("utf-8"))
    total = payload_bytes + index_bytes + card_bytes + model_state_bytes + manifest_bytes
    raw = len(records) * features * 8
    return {
        "payload_bytes": payload_bytes,
        "index_bytes": index_bytes,
        "card_bytes": card_bytes,
        "model_state_bytes": model_state_bytes,
        "manifest_bytes": manifest_bytes,
        "total_bytes": total,
        "raw_bytes": raw,
        "nbc": total / max(raw, 1),
    }


def _metric_for_system(
    records: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    alerts: np.ndarray,
    fidelities: Sequence[str],
    wrappers: Sequence[Mapping[str, Any]],
    *,
    replay_success: bool,
    uncertainty: float,
    runtime_seconds: float,
) -> tuple[MetricResult, dict[str, Any]]:
    detection = detector_metrics(alerts, labels)
    reconstructed = _reconstruct(records, fidelities)
    original = np.vstack([np.asarray(record["frame"], dtype=np.float64) for record in records])
    normal_rmse, anomaly_rmse = split_rmse(original, reconstructed, labels)
    accounting = _byte_accounting(records, fidelities, wrappers)
    metric = MetricResult(
        **detection,
        apr=anomaly_preservation(fidelities, labels),
        normal_rmse=normal_rmse,
        anomaly_rmse=anomaly_rmse,
        nbc=float(accounting["nbc"]),
        evidence_completeness=float(np.mean([evidence_completeness(row) for row in wrappers])),
        replay_success=1.0 if replay_success else 0.0,
        memory_utility=None,
        uncertainty=float(uncertainty),
        runtime_seconds=float(runtime_seconds),
    )
    return metric, accounting


def _metric_from_row(row: Mapping[str, Any]) -> MetricResult:
    integer_fields = {"false_negatives", "false_positives", "peak_memory_bytes", "crash_count", "nonfinite_count"}
    text_fields = {"status", "skip_reason"}
    values: dict[str, Any] = {}
    for key, field_definition in MetricResult.__dataclass_fields__.items():
        value = row.get(key, field_definition.default)
        if value in (None, "", "None"):
            values[key] = None
        elif key in text_fields:
            values[key] = str(value)
        elif key in integer_fields:
            values[key] = int(value)
        else:
            values[key] = float(value)
    return MetricResult(**values)


def shadow_evaluate(
    scenario: ScenarioStream,
    records: Sequence[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    labels = scenario.labels[[int(record["frame_id"]) for record in records]]
    contract = synthetic_domain_contract()
    pipeline = RepresentationPipeline(
        min_calibration=16,
        raw_nuisance_projector=scenario.nuisance_projector,
    )
    policies = {
        ablation: MeaningfulSurprisePolicy(contract, config=PolicyConfig())
        for ablation in ABLATION_IDS
    }
    memories = {
        ablation: CausalConsequenceMemory(min_support=0.5, min_confidence=0.6)
        for ablation in ABLATION_IDS
    }
    decisions: dict[str, list[dict[str, Any]]] = {ablation: [] for ablation in ABLATION_IDS}
    first_event = next((event for event in scenario.events if event.event_id == "repeat_first"), None)
    feedback_added = {ablation: False for ablation in ABLATION_IDS}
    voi_realizations: list[dict[str, Any]] = []

    started = time.perf_counter()
    for record in records:
        rep = pipeline.observe(record)
        frame_id = int(record["frame_id"])
        for ablation in ABLATION_IDS:
            memory = memories[ablation]
            if first_event and first_event.feedback_at is not None and frame_id >= first_event.feedback_at and not feedback_added[ablation]:
                vector = scenario.frames[first_event.start : first_event.end + 1].mean(axis=0)
                memory.add(
                    vector,
                    risk=1.0 if first_event.outcome == "harmful" else 0.0,
                    confidence=1.0,
                    available_at_frame=first_event.feedback_at,
                    provenance=f"{scenario.scenario_id}:{first_event.event_id}:delayed_outcome",
                )
                outcome = first_event.outcome
                baseline_loss = contract.bounded_loss("observe", outcome)
                prior = [
                    row
                    for row in decisions[ablation]
                    if first_event.start <= int(row["frame_id"]) <= first_event.end
                ]
                by_lift: dict[str, list[float]] = {}
                for row in prior:
                    action = str(row["decision"]["action"])
                    realized = policies[ablation].voi_tracker.record_realized(
                        str(row["selected_lift"]),
                        risk_without=baseline_loss,
                        risk_with=contract.bounded_loss(action, outcome),
                    )
                    by_lift.setdefault(str(row["selected_lift"]), []).append(realized)
                voi_realizations.append(
                    {
                        "ablation": ablation,
                        "event_id": first_event.event_id,
                        "outcome": outcome,
                        "available_at_frame": first_event.feedback_at,
                        "samples_by_lift": {key: len(value) for key, value in sorted(by_lift.items())},
                        "mean_realized_by_lift": {
                            key: float(np.mean(value)) for key, value in sorted(by_lift.items())
                        },
                        "provenance": f"{scenario.scenario_id}:{first_event.event_id}:delayed_outcome",
                    }
                )
                feedback_added[ablation] = True
            consequence = memory.recall(record["frame"], frame_id=frame_id)
            decision = policies[ablation].decide(
                record,
                rep,
                memory_consequence=consequence,
                ablation=ablation,
            )
            decisions[ablation].append(decision)
    shadow_runtime = time.perf_counter() - started

    systems: dict[str, Any] = {}
    for ablation, rows in decisions.items():
        scores = np.asarray([float(row["structural_evidence"]) for row in rows])
        alert_threshold = float(thresholds.get("eidos_ms_full", PolicyConfig().review_threshold))
        alerts = np.asarray(
            [
                bool(row["safety"]["raw_escape_triggered"])
                or (
                    row["decision"]["action"] != "observe"
                    and float(row["structural_evidence"]) >= alert_threshold
                )
                for row in rows
            ],
            dtype=bool,
        )
        fidelities = [str(row["decision"]["fidelity"]) for row in rows]
        wrappers = [
            _common_wrapper(
                record=record,
                score=float(row["structural_evidence"]),
                threshold=alert_threshold,
                selected_representation=str(row["selected_lift"]),
                action=str(row["decision"]["action"]),
                uncertainty=float(row["uncertainty"]),
                raw_retained=str(row["decision"]["fidelity"]) == "raw_frame_plus_full_context",
            )
            for record, row in zip(records, rows)
        ]
        replay_success = all(verify_canonical_decision(row) for row in rows)
        metric, accounting = _metric_for_system(
            records,
            labels,
            alerts,
            fidelities,
            wrappers,
            replay_success=replay_success,
            uncertainty=float(np.mean([row["uncertainty"] for row in rows])),
            runtime_seconds=shadow_runtime,
        )
        systems[ablation] = {
            "metrics": metric.as_dict(),
            "value_vector": metric.value_vector(),
            "byte_accounting": accounting,
            "alerts": alerts.astype(int).tolist(),
            "scores": scores.astype(float).tolist(),
            "fidelities": fidelities,
            "wrappers": wrappers,
        }
    systems["eidos_ms_full"] = systems["A0_full"]

    for system_id in SYSTEM_IDS[1:]:
        if system_id == "isolation_forest":
            try:
                scores = _causal_baseline_scores(records, system_id)
                status = "OK"
                reason = None
            except RuntimeError as exc:
                systems[system_id] = {"status": "SKIPPED", "skip_reason": str(exc), "metrics": None}
                continue
        else:
            scores = _causal_baseline_scores(records, system_id)
            status = "OK"
            reason = None
        threshold = _threshold_for(system_id, thresholds)
        if system_id == "eidos_live_current":
            alerts = np.asarray(
                [bool((record.get("sentinel_metrics") or {}).get("is_surprise", False)) for record in records]
            )
        else:
            alerts = scores >= threshold
        fidelities = ["structured_residual" if alert else "reference_or_null" for alert in alerts]
        wrappers = [
            _common_wrapper(
                record=record,
                score=float(score),
                threshold=threshold,
                selected_representation="live_eidos" if system_id == "eidos_live_current" else "raw",
                action="review" if alert else "observe",
                uncertainty=1.0,
                raw_retained=False,
            )
            for record, score, alert in zip(records, scores, alerts)
        ]
        metric, accounting = _metric_for_system(
            records,
            labels,
            alerts,
            fidelities,
            wrappers,
            replay_success=True,
            uncertainty=1.0,
            runtime_seconds=shadow_runtime,
        )
        systems[system_id] = {
            "status": status,
            "skip_reason": reason,
            "metrics": metric.as_dict(),
            "value_vector": metric.value_vector(),
            "byte_accounting": accounting,
            "alerts": alerts.astype(int).tolist(),
            "scores": scores.astype(float).tolist(),
            "fidelities": fidelities,
            "wrappers": wrappers,
        }

    compression = [compression_reference(scenario.frames, method) for method in ("raw", "gzip", "lzma", "zstd")]
    discovery_policy = MeaningfulSurprisePolicy(contract, config=PolicyConfig())
    discovery_cards = [
        discovery_policy.discovery_card(row)
        for row in decisions["A0_full"]
        if row["decision"]["action"] != "observe"
    ]
    return {
        "scenario": scenario.score_receipt(),
        "systems": systems,
        "decisions": decisions,
        "compression_references": compression,
        "discovery_cards": discovery_cards,
        "voi_realizations": voi_realizations,
    }


class GrandProofRunner:
    def __init__(
        self,
        config: RunnerConfig,
        *,
        capture_fn: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]] = capture_live_scenario,
    ) -> None:
        self.config = config
        self.capture_fn = capture_fn
        self.failures: list[dict[str, Any]] = []

    def run(self, *, stage: str, seeds: Sequence[int], scenarios: Sequence[str] = SCENARIO_IDS) -> dict[str, Any]:
        root = self.config.artifact_root
        root.mkdir(parents=True, exist_ok=True)
        all_rows: list[dict[str, Any]] = []
        for scenario_id in scenarios:
            for seed in seeds:
                scenario = generate_scenario(scenario_id, seed=int(seed), config=self.config.scenario_config)
                scenario_dir = root / "scenarios" / scenario_id / str(seed)
                replay_command = (
                    f"python eidos/tools/run_grand_proof_v1.py run --stage {stage} "
                    f"--seeds {seed} --scenarios {scenario_id} --out {root.as_posix()}"
                )
                try:
                    records, capture_receipt = self.capture_fn(
                        scenario,
                        out_dir=scenario_dir / "live_capture",
                        reservoir=self.config.reservoir,
                        code_commit=self.config.code_commit,
                        replay_command=replay_command,
                    )
                    evaluated = shadow_evaluate(scenario, records, thresholds=self.config.thresholds)
                    write_json(scenario_dir / "capture_receipt.json", capture_receipt)
                    write_json(scenario_dir / "scenario_receipt.json", evaluated["scenario"])
                    write_jsonl(
                        scenario_dir / "shadow_decisions.jsonl",
                        [
                            {"ablation": ablation, **decision}
                            for ablation, decisions in evaluated["decisions"].items()
                            for decision in decisions
                        ],
                    )
                    append_jsonl(
                        root / "captures" / "live_frame_observer.jsonl",
                        [
                            {"stage": stage, "scenario": scenario_id, "seed": int(seed), **record}
                            for record in records
                        ],
                    )
                    append_jsonl(
                        root / "captures" / "shadow_decisions.jsonl",
                        [
                            {"stage": stage, "scenario": scenario_id, "seed": int(seed), "ablation": ablation, **decision}
                            for ablation, ablation_decisions in evaluated["decisions"].items()
                            for decision in ablation_decisions
                        ],
                    )
                    append_jsonl(
                        root / "captures" / "shadow_tokens.jsonl",
                        [
                            {
                                "stage": stage,
                                "scenario": scenario_id,
                                "seed": int(seed),
                                "frame_id": decision["frame_id"],
                                "decision_sha256": decision["decision_sha256"],
                                "selected_lift": decision["selected_lift"],
                                "action": decision["decision"]["action"],
                                "fidelity": decision["decision"]["fidelity"],
                                "meaning_status": decision["meaning_status"],
                            }
                            for decision in evaluated["decisions"]["A0_full"]
                        ],
                    )
                    for card in evaluated["discovery_cards"]:
                        write_json(
                            scenario_dir
                            / "eidos_ms_full"
                            / "discovery_cards"
                            / f"frame_{int(card['frame_id']):08d}.json",
                            card,
                        )
                    for system_id, result in evaluated["systems"].items():
                        system_dir = scenario_dir / system_id
                        if result.get("metrics") is None:
                            write_json(system_dir / "metrics.json", result)
                            continue
                        write_json(system_dir / "metrics.json", result["metrics"])
                        write_jsonl(system_dir / "events.jsonl", result["wrappers"])
                        write_json(system_dir / "byte_accounting.json", result["byte_accounting"])
                        all_rows.append(
                            {
                                "stage": stage,
                                "scenario": scenario_id,
                                "seed": int(seed),
                                "system": system_id,
                                **result["metrics"],
                            }
                        )
                    write_json(scenario_dir / "compression_references.json", evaluated["compression_references"])
                    write_json(scenario_dir / "voi_realizations.json", evaluated["voi_realizations"])
                except Exception as exc:
                    failure = {
                        "timestamp_utc": utc_now(),
                        "stage": stage,
                        "scenario": scenario_id,
                        "seed": int(seed),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    self.failures.append(failure)
                    write_json(scenario_dir / "failure.json", failure)
                    write_jsonl(root / "failures" / "failure_ledger.jsonl", self.failures)
                    raise
        self._write_tables(all_rows)
        return {"stage": stage, "seeds": list(seeds), "rows": all_rows, "failures": self.failures}

    def _write_tables(self, rows: Sequence[Mapping[str, Any]]) -> None:
        path = self.config.artifact_root / "ablations" / "paired_results.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        merged: dict[tuple[Any, ...], dict[str, Any]] = {}
        if path.is_file():
            with path.open("r", newline="", encoding="utf-8") as handle:
                for prior in csv.DictReader(handle):
                    key = (prior.get("stage"), prior.get("scenario"), prior.get("seed"), prior.get("system"))
                    merged[key] = dict(prior)
        for row in rows:
            material = dict(row)
            key = (material.get("stage"), material.get("scenario"), str(material.get("seed")), material.get("system"))
            merged[key] = material
        table_rows = list(merged.values())
        if table_rows:
            fields = list(table_rows[0])
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(table_rows)
        vectors = [
            {
                "index": index,
                "scenario": row["scenario"],
                "seed": row["seed"],
                "system": row["system"],
                "value_vector": MetricResult(
                    **_metric_from_row(row).as_dict()
                ).value_vector(),
            }
            for index, row in enumerate(table_rows)
        ]
        front = pareto_front(vectors) if vectors else []
        write_json(self.config.artifact_root / "statistics" / "pareto_points.json", {"front_indices": front, "rows": vectors})
        pareto_path = self.config.artifact_root / "statistics" / "pareto_points.csv"
        with pareto_path.open("w", newline="", encoding="utf-8") as handle:
            fields = ["index", "scenario", "seed", "system", "on_front", "F2", "APR", "EC", "REP", "MU", "-NBC", "-NDL", "-FPR", "-U"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in vectors:
                writer.writerow(
                    {
                        "index": row["index"],
                        "scenario": row["scenario"],
                        "seed": row["seed"],
                        "system": row["system"],
                        "on_front": row["index"] in front,
                        **row["value_vector"],
                    }
                )
        sensitivity: list[dict[str, Any]] = []
        for row in table_rows:
            metric = _metric_from_row(row)
            for grid_index, grid in enumerate(registered_sensitivity_grid()):
                sensitivity.append(
                    {
                        "scenario": row["scenario"],
                        "seed": row["seed"],
                        "system": row["system"],
                        "grid_index": grid_index,
                        "evm": eidos_value_scalar(metric, grid["weights"], grid["penalties"]),
                    }
                )
        write_json(self.config.artifact_root / "statistics" / "weight_sensitivity.json", sensitivity)
        sensitivity_path = self.config.artifact_root / "statistics" / "weight_sensitivity.csv"
        if sensitivity:
            with sensitivity_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(sensitivity[0]))
                writer.writeheader()
                writer.writerows(sensitivity)

        primary_components = ("f2", "apr", "nbc", "normalized_delay", "fp_per_10k")
        comparisons = sorted({str(row["system"]) for row in table_rows if row["system"] != "eidos_ms_full"})
        interval_rows: list[dict[str, Any]] = []
        families: dict[str, dict[str, float]] = {}
        for stage in sorted({str(row["stage"]) for row in table_rows}):
            for scenario in sorted({str(row["scenario"]) for row in table_rows if str(row["stage"]) == stage}):
                subset = [row for row in table_rows if str(row["stage"]) == stage and str(row["scenario"]) == scenario]
                full_by_seed = {str(row["seed"]): row for row in subset if row["system"] == "eidos_ms_full"}
                for comparison in comparisons:
                    other_by_seed = {str(row["seed"]): row for row in subset if row["system"] == comparison}
                    common = sorted(set(full_by_seed) & set(other_by_seed), key=int)
                    for component in primary_components:
                        pairs = [
                            (full_by_seed[seed].get(component), other_by_seed[seed].get(component))
                            for seed in common
                        ]
                        pairs = [(left, right) for left, right in pairs if left not in (None, "") and right not in (None, "")]
                        if not pairs:
                            continue
                        left_values = [float(left) for left, _right in pairs]
                        right_values = [float(right) for _left, right in pairs]
                        # Lower byte cost, delay, and false positives are benefits;
                        # orient every interval so positive means full is better.
                        if component in {"nbc", "normalized_delay", "fp_per_10k"}:
                            left_values, right_values = ([-value for value in left_values], [-value for value in right_values])
                        interval = paired_bootstrap(left_values, right_values)
                        comparison_id = f"{stage}:{scenario}:{comparison}:{component}"
                        family_id = f"{stage}:{scenario}:{component}"
                        row = {
                            "comparison_id": comparison_id,
                            "family_id": family_id,
                            "stage": stage,
                            "scenario": scenario,
                            "comparison": comparison,
                            "component": component,
                            **interval,
                        }
                        interval_rows.append(row)
                        families.setdefault(family_id, {})[comparison_id] = float(interval["p_value_two_sided"])
        adjusted = {
            comparison_id: value
            for family in families.values()
            for comparison_id, value in holm_adjust(family).items()
        }
        for row in interval_rows:
            row["holm_adjusted_p"] = adjusted[row["comparison_id"]]
        interval_path = self.config.artifact_root / "statistics" / "paired_intervals.csv"
        if interval_rows:
            with interval_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(interval_rows[0]))
                writer.writeheader()
                writer.writerows(interval_rows)
        write_json(self.config.artifact_root / "statistics" / "paired_intervals.json", interval_rows)


def build_run_lock(
    *,
    repo_root: Path,
    artifact_root: Path,
    reservoir: int,
    scenario_config: ScenarioConfig,
    thresholds: Mapping[str, float],
    datasets: Mapping[str, Any],
    resource_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    code_commit = git_output(repo_root, "rev-parse", "HEAD")
    implementation_paths = ("eidos/repo/src", "eidos/repo/tests", "eidos/tools")
    implementation_trees = {
        path: git_output(repo_root, "rev-parse", f"HEAD:{path}")
        for path in implementation_paths
    }
    design_dir = repo_root / "eidos" / "docs" / "proof" / "design_freeze"
    locked_files = {}
    for name in (
        "meaningful_surprise_v1_spec.md",
        "grand_proof_protocol_v1.md",
        "local_codex_execution_brief_v1.md",
        "design_freeze_manifest_v1.json",
    ):
        path = design_dir / name
        locked_files[name] = {
            "path": path.relative_to(repo_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    lock = {
        "protocol_id": "EIDOS-GP-v1-2026-09-01",
        "created_at_utc": utc_now(),
        "code_commit": code_commit,
        "implementation_trees": implementation_trees,
        "branch": git_output(repo_root, "branch", "--show-current"),
        "git_status_short": git_output(repo_root, "status", "--short"),
        "locked_files": locked_files,
        "scenario_config": asdict(scenario_config),
        "scenario_config_hash": scenario_config.hash,
        "domain_contract": asdict(synthetic_domain_contract()),
        "domain_contract_hash": synthetic_domain_contract().hash,
        "calibration_seeds": list(range(10, 20)),
        "heldout_seeds": list(range(100, 120)),
        "smoke_seeds": [0, 1],
        "reservoir": int(reservoir),
        "thresholds": dict(thresholds),
        "byte_operating_points": list(BYTE_OPERATING_POINTS),
        "systems": list(SYSTEM_IDS),
        "ablations": list(ABLATION_IDS),
        "resource_receipt": dict(resource_receipt),
        "heldout_allowed": resource_receipt.get("selection_status") == "SELECTED",
        "datasets": dict(datasets),
        "artifact_root": artifact_root.as_posix(),
        "commands": {
            "tests": "python -m pytest",
            "smoke": "python eidos/tools/run_grand_proof_v1.py run --stage smoke --seeds 0,1",
            "calibration": "python eidos/tools/run_grand_proof_v1.py run --stage calibration --seeds 10-19",
            "heldout": "python eidos/tools/run_grand_proof_v1.py run --stage heldout --seeds 100-119",
        },
        "environment": {"python": sys.version, "platform": platform.platform()},
    }
    lock["run_lock_sha256"] = canonical_sha256(lock)
    return lock


def verify_run_lock(lock: Mapping[str, Any], *, repo_root: Path) -> list[str]:
    failures: list[str] = []
    material = dict(lock)
    stored = material.pop("run_lock_sha256", None)
    if stored != canonical_sha256(material):
        failures.append("run_lock hash mismatch")
    head = git_output(repo_root, "rev-parse", "HEAD")
    code_commit = str(lock.get("code_commit"))
    if head != code_commit:
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", code_commit, head],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if ancestry.returncode != 0:
            failures.append("locked code commit is not an ancestor of HEAD")
    for path, expected_tree in lock.get("implementation_trees", {}).items():
        try:
            actual_tree = git_output(repo_root, "rev-parse", f"HEAD:{path}")
        except subprocess.CalledProcessError:
            failures.append(f"missing implementation tree: {path}")
            continue
        if actual_tree != expected_tree:
            failures.append(f"implementation tree mismatch: {path}")
    dirty_implementation = git_output(
        repo_root,
        "status",
        "--short",
        "--",
        "eidos/repo/src",
        "eidos/repo/tests",
        "eidos/tools",
    )
    if dirty_implementation:
        failures.append("uncommitted implementation changes")
    for name, receipt in lock.get("locked_files", {}).items():
        path = repo_root / receipt["path"]
        if not path.is_file():
            failures.append(f"missing locked file: {name}")
            continue
        if path.stat().st_size != int(receipt["bytes"]):
            failures.append(f"byte mismatch: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
            failures.append(f"hash mismatch: {name}")
    return failures
