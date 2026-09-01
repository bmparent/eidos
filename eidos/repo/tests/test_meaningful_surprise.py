import copy

from eidos_brain.proof.meaningful_surprise import (
    DomainContract,
    MeaningfulSurprisePolicy,
    MemoryConsequence,
    PolicyConfig,
    verify_canonical_decision,
)


def contract() -> DomainContract:
    return DomainContract.from_dict(
        {
            "contract_id": "test",
            "actions": ["observe", "review", "escalate"],
            "outcomes": ["benign", "harmful"],
            "loss": {
                "observe": {"benign": 0.0, "harmful": 1.0},
                "review": {"benign": 0.2, "harmful": 0.2},
                "escalate": {"benign": 0.7, "harmful": 0.0},
            },
            "horizon": 4,
            "costs": {},
            "provenance": "test",
        }
    )


def representation(raw: float = 0.2) -> dict:
    lifts = {}
    for name in ("raw", "spectral", "multiscale", "geometry", "memory", "consensus"):
        lifts[name] = {
            "lift_id": name,
            "structural_evidence": raw if name == "raw" else 0.1,
            "persistence": 0.1,
            "quotient_residual": 0.0 if name == "raw" else None,
            "calibration_status": "READY",
        }
    return {
        "lifts": lifts,
        "representation_disagreement": 0.1,
        "representation_disagreement_definition": "normalized_weighted_evidence_variance",
        "phase_coherence": 0.5,
        "thermodynamic_evidence": 0.1,
    }


def observation(familiarity: float = 0.0) -> dict:
    return {
        "frame_id": 10,
        "source_id": "source",
        "hdc_metrics": {"familiarity": familiarity},
        "source_refs": [{"frame_id": 10}],
        "config_hash": "cfg",
        "code_commit": "commit",
        "replay_command": "cmd",
    }


def test_raw_escape_cannot_be_suppressed_by_cost_familiarity_or_negative_voi():
    policy = MeaningfulSurprisePolicy(contract(), config=PolicyConfig(raw_escape_threshold=0.8))
    decision = policy.decide(
        observation(familiarity=1.0),
        representation(raw=0.95),
        memory_consequence=MemoryConsequence("VALIDATED", 0.0, 2.0, 1.0, ["benign"]),
    )
    assert decision["safety"]["raw_escape_triggered"] is True
    assert decision["decision"]["action"] in {"review", "escalate"}
    assert decision["decision"]["fidelity"] == "raw_frame_plus_full_context"


def test_increasing_validated_danger_cannot_reduce_action_or_fidelity():
    policy = MeaningfulSurprisePolicy(contract())
    low = policy.decide(
        observation(),
        representation(),
        memory_consequence=MemoryConsequence("VALIDATED", 0.2, 2.0, 1.0, ["low"]),
    )
    high = policy.decide(
        observation(),
        representation(),
        memory_consequence=MemoryConsequence("VALIDATED", 0.95, 2.0, 1.0, ["high"]),
    )
    action_order = ["observe", "review", "escalate"]
    fidelity_order = ["reference_or_null", "quantized_residual", "structured_residual", "raw_frame_plus_full_context"]
    assert action_order.index(high["decision"]["action"]) >= action_order.index(low["decision"]["action"])
    assert fidelity_order.index(high["decision"]["fidelity"]) >= fidelity_order.index(low["decision"]["fidelity"])


def test_familiarity_alone_does_not_certify_safety_and_unknown_is_not_zero():
    policy = MeaningfulSurprisePolicy(contract())
    decision = policy.decide(
        observation(familiarity=1.0),
        representation(),
        memory_consequence=MemoryConsequence("UNKNOWN", None, 0.0, 0.0, []),
    )
    assert decision["memory_consequence"]["state"] == "UNKNOWN"
    assert decision["memory_consequence"]["effective_risk"] > 0.0
    assert decision["meaning_status"] == "STRUCTURAL_HYPOTHESIS"


def test_higher_uncertainty_at_high_risk_cannot_lower_retention():
    class Tracker:
        def __init__(self, uncertainty):
            self.uncertainty = uncertainty

        def estimate(self, _lift):
            return {"estimate": 0.2, "lcb": -0.8, "uncertainty": self.uncertainty, "realized": None}

    low_policy = MeaningfulSurprisePolicy(contract(), voi_tracker=Tracker(0.1))
    high_policy = MeaningfulSurprisePolicy(contract(), voi_tracker=Tracker(0.9))
    memory = MemoryConsequence("VALIDATED", 0.9, 2.0, 1.0, ["danger"])
    low = low_policy.decide(observation(), representation(), memory_consequence=memory)
    high = high_policy.decide(observation(), representation(), memory_consequence=memory)
    order = ["reference_or_null", "quantized_residual", "structured_residual", "raw_frame_plus_full_context"]
    assert order.index(high["decision"]["fidelity"]) >= order.index(low["decision"]["fidelity"])


def test_canonical_decision_replay_is_deterministic():
    left = MeaningfulSurprisePolicy(contract()).decide(observation(), representation())
    right = MeaningfulSurprisePolicy(contract()).decide(observation(), representation())
    assert left == right
    assert verify_canonical_decision(left)

