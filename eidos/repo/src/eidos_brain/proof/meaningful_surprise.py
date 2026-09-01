"""EIDOS-MS-v1 shadow policy.

This module never mutates the live engine. It turns a completed live observation
and causal representation evidence into a versioned shadow decision.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .frame_observer import canonical_sha256


MEANINGFUL_SURPRISE_VERSION = "EIDOS-MS-v1"
MEANING_STATES = ("STRUCTURAL_HYPOTHESIS", "OUTCOME_ESTIMATED", "OUTCOME_VALIDATED")
ACTION_ORDER = ("observe", "review", "escalate")
FIDELITY_ORDER = ("reference_or_null", "quantized_residual", "structured_residual", "raw_frame_plus_full_context")


@dataclass(frozen=True)
class DomainContract:
    contract_id: str
    actions: tuple[str, ...]
    outcomes: tuple[str, ...]
    loss: Mapping[str, Mapping[str, float]]
    horizon: int
    costs: Mapping[str, float]
    provenance: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DomainContract":
        contract = cls(
            contract_id=str(value["contract_id"]),
            actions=tuple(str(item) for item in value["actions"]),
            outcomes=tuple(str(item) for item in value["outcomes"]),
            loss={
                str(action): {str(outcome): float(score) for outcome, score in scores.items()}
                for action, scores in value["loss"].items()
            },
            horizon=int(value["horizon"]),
            costs={str(key): float(cost) for key, cost in value.get("costs", {}).items()},
            provenance=str(value.get("provenance", "unknown")),
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        if not self.contract_id or not self.actions or not self.outcomes:
            raise ValueError("domain contract requires id, actions, and outcomes")
        if self.horizon <= 0:
            raise ValueError("domain contract horizon must be positive")
        for action in self.actions:
            if action not in self.loss:
                raise ValueError(f"missing loss row for action: {action}")
            missing = set(self.outcomes) - set(self.loss[action])
            if missing:
                raise ValueError(f"missing outcomes for action {action}: {sorted(missing)}")
            for outcome in self.outcomes:
                score = self.loss[action][outcome]
                if not np.isfinite(score) or score < 0.0:
                    raise ValueError("domain losses must be finite and nonnegative")

    @property
    def hash(self) -> str:
        return "sha256:" + canonical_sha256(
            {
                "contract_id": self.contract_id,
                "actions": self.actions,
                "outcomes": self.outcomes,
                "loss": self.loss,
                "horizon": self.horizon,
                "costs": self.costs,
                "provenance": self.provenance,
            }
        )

    def bounded_loss(self, action: str, outcome: str) -> float:
        return float(self.loss[action][outcome])


@dataclass(frozen=True)
class MemoryConsequence:
    state: str
    risk: float | None
    effective_support: float
    confidence: float
    provenance: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "risk": self.risk,
            "effective_support": self.effective_support,
            "confidence": self.confidence,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class ConsequenceTrace:
    vector: tuple[float, ...]
    risk: float
    confidence: float
    available_at_frame: int
    provenance: str


class CausalConsequenceMemory:
    """Delayed consequence annotations, unavailable until their horizon closes."""

    def __init__(self, *, min_support: float = 1.5, min_confidence: float = 0.6, max_traces: int = 4096) -> None:
        self.min_support = float(min_support)
        self.min_confidence = float(min_confidence)
        self.traces: deque[ConsequenceTrace] = deque(maxlen=int(max_traces))

    def add(
        self,
        vector: Sequence[float],
        *,
        risk: float,
        confidence: float,
        available_at_frame: int,
        provenance: str,
    ) -> None:
        vec = np.asarray(vector, dtype=np.float64).reshape(-1)
        if vec.size == 0 or not np.all(np.isfinite(vec)):
            raise ValueError("consequence vector must be finite and non-empty")
        if not 0.0 <= risk <= 1.0 or not 0.0 <= confidence <= 1.0:
            raise ValueError("risk and confidence must be in [0, 1]")
        self.traces.append(
            ConsequenceTrace(
                vector=tuple(float(item) for item in vec),
                risk=float(risk),
                confidence=float(confidence),
                available_at_frame=int(available_at_frame),
                provenance=str(provenance),
            )
        )

    def recall(self, vector: Sequence[float], *, frame_id: int, neighbors: int = 5) -> MemoryConsequence:
        query = np.asarray(vector, dtype=np.float64).reshape(-1)
        eligible = [trace for trace in self.traces if trace.available_at_frame <= frame_id]
        if not eligible:
            return MemoryConsequence("UNKNOWN", None, 0.0, 0.0, [])
        ranked: list[tuple[float, ConsequenceTrace]] = []
        for trace in eligible:
            candidate = np.asarray(trace.vector, dtype=np.float64)
            candidate = np.resize(candidate, query.size)
            denom = max(float(np.linalg.norm(query) * np.linalg.norm(candidate)), 1e-12)
            similarity = float(np.clip(np.dot(query, candidate) / denom, 0.0, 1.0))
            ranked.append((similarity * trace.confidence, trace))
        selected = sorted(ranked, key=lambda item: item[0], reverse=True)[: int(neighbors)]
        support = float(sum(weight for weight, _trace in selected))
        if support <= 0.0:
            return MemoryConsequence("UNKNOWN", None, 0.0, 0.0, [])
        risk = float(sum(weight * trace.risk for weight, trace in selected) / support)
        confidence = float(sum(weight * trace.confidence for weight, trace in selected) / support)
        provenance = [trace.provenance for _weight, trace in selected]
        if support < self.min_support or confidence < self.min_confidence:
            return MemoryConsequence("UNKNOWN", None, support, confidence, provenance)
        return MemoryConsequence("VALIDATED", risk, support, confidence, provenance)


class CausalVOITracker:
    """Past-only lower-confidence estimates and later realized-loss receipts."""

    def __init__(self, *, max_history: int = 2048, kappa: float = 1.0) -> None:
        self.history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=int(max_history)))
        self.kappa = float(kappa)

    def estimate(self, lift_id: str) -> dict[str, float | None]:
        values = np.asarray(self.history[lift_id], dtype=np.float64)
        if values.size == 0:
            return {"estimate": None, "uncertainty": 1.0, "lcb": None, "realized": None}
        estimate = float(values.mean())
        uncertainty = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 1.0
        return {
            "estimate": estimate,
            "uncertainty": uncertainty,
            "lcb": estimate - self.kappa * uncertainty,
            "realized": None,
        }

    def record_realized(self, lift_id: str, *, risk_without: float, risk_with: float) -> float:
        realized = float(risk_without - risk_with)
        if not np.isfinite(realized):
            raise ValueError("realized VOI must be finite")
        self.history[lift_id].append(realized)
        return realized


@dataclass(frozen=True)
class PolicyConfig:
    structural_thresholds: Mapping[str, float] = field(
        default_factory=lambda: {
            "raw": 0.65,
            "spectral": 0.65,
            "multiscale": 0.65,
            "geometry": 0.65,
            "memory": 0.65,
            "consensus": 0.65,
        }
    )
    persistence_threshold: float = 0.55
    raw_escape_threshold: float = 0.85
    memory_risk_threshold: float = 0.65
    unknown_risk_floor: float = 0.10
    review_threshold: float = 0.25
    escalate_threshold: float = 0.65
    coefficients: Mapping[str, float] = field(
        default_factory=lambda: {
            "memory_risk": 0.7,
            "persistence": 0.35,
            "disagreement": 0.2,
            "bytes": 0.2,
            "false_positive": 0.25,
            "latency": 0.1,
            "uncertainty": 0.2,
            "thermodynamic": 0.15,
        }
    )
    fidelity_costs: Mapping[str, float] = field(
        default_factory=lambda: {
            "reference_or_null": 0.0,
            "quantized_residual": 0.25,
            "structured_residual": 0.55,
            "raw_frame_plus_full_context": 1.0,
        }
    )
    action_costs: Mapping[str, float] = field(
        default_factory=lambda: {"observe": 0.0, "review": 0.35, "escalate": 0.8}
    )
    raw_escape_fidelity: str = "raw_frame_plus_full_context"


def _ranked_at_least(value: str, minimum: str, order: tuple[str, ...]) -> str:
    return order[max(order.index(value), order.index(minimum))]


class MeaningfulSurprisePolicy:
    def __init__(
        self,
        domain_contract: DomainContract,
        *,
        config: PolicyConfig | None = None,
        voi_tracker: CausalVOITracker | None = None,
    ) -> None:
        self.domain_contract = domain_contract
        self.config = config or PolicyConfig()
        self.voi_tracker = voi_tracker or CausalVOITracker()

    def _risk_value(self, memory: MemoryConsequence) -> float:
        if memory.state == "VALIDATED" and memory.risk is not None:
            return float(memory.risk)
        return float(self.config.unknown_risk_floor)

    def decide(
        self,
        observation: Mapping[str, Any],
        representation_result: Mapping[str, Any],
        *,
        memory_consequence: MemoryConsequence | None = None,
        ablation: str = "A0_full",
    ) -> dict[str, Any]:
        lifts = dict(representation_result["lifts"])
        memory = memory_consequence or MemoryConsequence("UNKNOWN", None, 0.0, 0.0, [])
        if ablation == "A1_no_hdc":
            memory = MemoryConsequence("UNKNOWN", None, 0.0, 0.0, [])
        disagreement = 0.0 if ablation == "A5_no_scout" else float(
            representation_result["representation_disagreement"]
        )
        thermodynamic_evidence = (
            0.0
            if ablation == "A4_no_thermo"
            else float(representation_result.get("thermodynamic_evidence", 0.0))
        )
        raw_score = float(lifts["raw"]["structural_evidence"])
        raw_escape = raw_score >= self.config.raw_escape_threshold and ablation != "A6_no_raw_escape"
        risk = self._risk_value(memory)

        def policy_structural(lift_id: str) -> float:
            evidence = lifts[lift_id]
            score = float(evidence["structural_evidence"])
            quotient = evidence.get("quotient_residual")
            if quotient is not None:
                quotient_score = float(quotient) / (1.0 + float(quotient))
                return min(score, quotient_score)
            return score

        allowed_lifts = list(lifts)
        if ablation == "A2_no_geometry":
            allowed_lifts.remove("geometry")
        elif ablation == "A3_no_multiscale":
            allowed_lifts.remove("multiscale")
        elif ablation == "A5_no_scout":
            allowed_lifts = ["raw"]
        candidates = [
            lift_id
            for lift_id in allowed_lifts
            if lift_id == "raw"
            or policy_structural(lift_id)
            >= float(self.config.structural_thresholds.get(lift_id, 1.0))
            or float(lifts[lift_id]["persistence"]) >= self.config.persistence_threshold
            or risk >= self.config.memory_risk_threshold
        ]

        rows: list[dict[str, Any]] = []
        for lift_id in candidates:
            evidence = lifts[lift_id]
            voi = self.voi_tracker.estimate(lift_id)
            lcb = 0.0 if ablation == "A7_no_voi" or voi["lcb"] is None else float(voi["lcb"])
            uncertainty = float(voi["uncertainty"])
            for fidelity in FIDELITY_ORDER:
                for action in ACTION_ORDER:
                    score = (
                        lcb
                        + self.config.coefficients["memory_risk"] * risk
                        + self.config.coefficients["persistence"] * float(evidence["persistence"])
                        + self.config.coefficients["disagreement"] * disagreement
                        + self.config.coefficients["thermodynamic"] * thermodynamic_evidence
                        - self.config.coefficients["bytes"] * self.config.fidelity_costs[fidelity]
                        - self.config.coefficients["false_positive"] * self.config.action_costs[action]
                        - self.config.coefficients["latency"] * self.config.action_costs[action]
                        - self.config.coefficients["uncertainty"] * uncertainty
                    )
                    rows.append(
                        {
                            "lift": lift_id,
                            "fidelity": fidelity,
                            "action": action,
                            "net_value": float(score),
                            "voi": voi,
                        }
                    )
        selected = max(
            rows,
            key=lambda row: (
                row["net_value"],
                -allowed_lifts.index(row["lift"]),
                -FIDELITY_ORDER.index(row["fidelity"]),
                -ACTION_ORDER.index(row["action"]),
            ),
        )
        selected_evidence = lifts[selected["lift"]]
        urgency_signal = max(
            policy_structural(selected["lift"]),
            float(selected_evidence["persistence"]),
            risk,
        )
        action = "observe"
        if urgency_signal >= self.config.review_threshold:
            action = "review"
        if urgency_signal >= self.config.escalate_threshold and memory.state == "VALIDATED":
            action = "escalate"
        fidelity = selected["fidelity"]
        if urgency_signal >= self.config.review_threshold:
            fidelity = _ranked_at_least(fidelity, "structured_residual", FIDELITY_ORDER)
        overrides: list[str] = []
        if memory.state == "VALIDATED" and memory.risk is not None:
            if memory.risk >= self.config.memory_risk_threshold:
                action = _ranked_at_least(action, "review", ACTION_ORDER)
                fidelity = _ranked_at_least(fidelity, "structured_residual", FIDELITY_ORDER)
                overrides.append("validated_memory_risk_monotonicity")
            if memory.risk >= 0.85:
                action = _ranked_at_least(action, "escalate", ACTION_ORDER)
                fidelity = _ranked_at_least(fidelity, "raw_frame_plus_full_context", FIDELITY_ORDER)
        if risk >= self.config.memory_risk_threshold and float(selected["voi"]["uncertainty"]) >= 0.5:
            fidelity = _ranked_at_least(fidelity, "structured_residual", FIDELITY_ORDER)
            action = _ranked_at_least(action, "review", ACTION_ORDER)
            overrides.append("high_risk_uncertainty_retention")
        if raw_escape:
            action = _ranked_at_least(action, "review", ACTION_ORDER)
            fidelity = _ranked_at_least(fidelity, self.config.raw_escape_fidelity, FIDELITY_ORDER)
            overrides.append("raw_residual_escape")

        meaning_status = "STRUCTURAL_HYPOTHESIS"
        if selected["voi"]["estimate"] is not None:
            meaning_status = "OUTCOME_ESTIMATED"
        if memory.state == "VALIDATED":
            meaning_status = "OUTCOME_VALIDATED"
        candidate_rows = [dict(lifts[lift_id]) for lift_id in candidates]
        decision = {
            "meaningful_surprise_version": MEANINGFUL_SURPRISE_VERSION,
            "frame_id": int(observation["frame_id"]),
            "source_id": str(observation.get("source_id", "unknown")),
            "meaning_status": meaning_status,
            "domain_contract_hash": self.domain_contract.hash,
            "candidate_lifts": candidate_rows,
            "selected_lift": selected["lift"],
            "structural_evidence": float(selected_evidence["structural_evidence"]),
            "quotient_residual": selected_evidence["quotient_residual"],
            "persistence": float(selected_evidence["persistence"]),
            "representation_disagreement": disagreement,
            "representation_disagreement_definition": representation_result[
                "representation_disagreement_definition"
            ],
            "phase_coherence": representation_result.get("phase_coherence"),
            "thermodynamic_evidence": thermodynamic_evidence,
            "familiarity": float((observation.get("hdc_metrics") or {}).get("familiarity") or 0.0),
            "memory_consequence": {**memory.as_dict(), "effective_risk": risk},
            "voi": selected["voi"],
            "uncertainty": float(selected["voi"]["uncertainty"]),
            "decision": {"action": action, "fidelity": fidelity, "net_value": selected["net_value"]},
            "safety": {"raw_escape_triggered": raw_escape, "overrides": overrides},
            "ablation": ablation,
            "source_refs": list(observation.get("source_refs") or []),
            "config_hash": str(observation.get("config_hash", "UNKNOWN")),
            "code_commit": str(observation.get("code_commit", "UNKNOWN")),
            "replay_command": str(observation.get("replay_command", "")),
        }
        decision["decision_sha256"] = canonical_sha256(decision)
        return decision

    def discovery_card(self, decision: Mapping[str, Any]) -> dict[str, Any]:
        card = {
            "card_version": "EIDOS-MS-DISCOVERY-v1",
            "frame_id": decision["frame_id"],
            "source_id": decision["source_id"],
            "meaning_status": decision["meaning_status"],
            "what_happened": (
                f"Live evidence selected the {decision['selected_lift']} representation "
                f"with structural evidence {decision['structural_evidence']:.3f}."
            ),
            "why_it_matters": (
                "This is a bounded shadow recommendation. It does not establish attack, failure, "
                "seizure, compromise, root cause, or universal meaning."
            ),
            "evidence": {
                "selected_lift": decision["selected_lift"],
                "raw_escape_triggered": decision["safety"]["raw_escape_triggered"],
                "source_refs": decision["source_refs"],
                "decision_sha256": decision["decision_sha256"],
            },
            "uncertainty": decision["uncertainty"],
            "next_action": decision["decision"]["action"],
            "replay_command": decision["replay_command"],
        }
        card["card_sha256"] = canonical_sha256(card)
        return card


def verify_canonical_decision(decision: Mapping[str, Any]) -> bool:
    material = dict(decision)
    stored = material.pop("decision_sha256", None)
    return stored == canonical_sha256(material)
