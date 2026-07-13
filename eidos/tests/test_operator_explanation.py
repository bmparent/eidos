from operator_explanation import (
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
    enrich_incident_card,
    render_operator_narrative,
)
from sentinel.event_merge import ConfirmedEvent, event_to_incident_card


def test_confirmed_event_gets_operator_explanation_without_claiming_attack():
    event = ConfirmedEvent(
        event_id="evt_anomaly_12_17",
        start_frame=12,
        end_frame=17,
        duration=6,
        peak_score=4.4,
        event_count=6,
        severity="RED",
        confidence=0.86,
        why_flagged=["evidence persisted for six frames"],
        top_drivers=[{"name": "geometry_change", "score": 0.7}],
        raw_evidence_refs=[f"steps.csv:{frame}" for frame in range(12, 18)],
    )

    card = event_to_incident_card(event)
    explanation = card["operator_explanation"]

    assert explanation["schema_version"] == SCHEMA_VERSION
    assert all(field in explanation for field in REQUIRED_FIELDS)
    assert "frames), triaged as RED" in explanation["what_happened"]["summary"]
    assert explanation["evidence"]["raw_references"] == {
        "items": ["steps.csv:12", "steps.csv:13", "steps.csv:14"],
        "total_count": 6,
        "omitted_count": 3,
    }
    assert explanation["uncertainty"]["confidence_channels"]["detection"]["value"] == 0.86
    assert "not an attack probability" in explanation["uncertainty"]["confidence_channels"]["detection"]["meaning"]
    asserted_text = " ".join(
        explanation[field]["summary"] for field in ("what_happened", "why_it_matters", "evidence")
    ).lower()
    assert "confirmed attack" not in asserted_text
    assert card["operator_narrative"]["source"] == "deterministic"
    assert card["operator_narrative"]["authoritative"] is False
    assert card["peak_score"] == 4.4


def test_engine_card_separates_confidence_channels_and_explains_projected_feature():
    legacy_card = {
        "incident_id": "inc_42",
        "domain": "cyber",
        "regime": "AMBER",
        "severity": "AMBER",
        "step": 42,
        "confidence": 0.85,
        "hypotheses": [{"label": "anomaly", "confidence": 1.0}],
        "evidence": {
            "drivers": [{"name": "cicids_projected_41", "value": 3.2}],
            "exemplars": ["residuals.jsonl:42"],
            "baseline": {"z": 4.1},
        },
        "forecast": {"likely_mode": "unknown", "confidence": 0.0},
        "similar_episodes": [
            {
                "step": 10,
                "regime": "AMBER",
                "sim": 0.82,
                "summary": "Prior AMBER event at step 10",
                "drivers": [{"name": "do_not_copy", "value": 99}],
                "entities": {"large": "payload"},
            }
        ],
        "actions": [{"action": "ISOLATE_HOST", "score": 0.35, "policy": "recommend"}],
    }

    first = enrich_incident_card(legacy_card)
    second = enrich_incident_card(legacy_card)
    explanation = first["operator_explanation"]

    assert first == second
    driver = explanation["evidence"]["top_drivers"][0]
    assert driver["feature"] == "projected traffic component 41"
    assert "cannot be attributed" in driver["limitation"]
    channels = explanation["uncertainty"]["confidence_channels"]
    assert channels["detection"]["value"] == 0.85
    assert channels["hypothesis"]["value"] == 1.0
    assert channels["forecast"]["value"] == 0.0
    assert explanation["evidence"]["nearest_similar_event"] == {
        "step": 10,
        "regime": "AMBER",
        "sim": 0.82,
        "summary": "Prior AMBER event at step 10",
    }
    assert explanation["next_action"]["machine_action_authority"] == "recommendation_only"
    assert "ISOLATE_HOST" not in explanation["next_action"]["summary"]


def test_optional_renderer_is_constrained_and_falls_back_on_unsafe_claim():
    card = enrich_incident_card({"severity": "AMBER", "step": 9, "confidence": 0.8})
    explanation = card["operator_explanation"]

    def safe_renderer(request):
        assert request["constraints"]["return_exactly"] == list(REQUIRED_FIELDS)
        return {field: f"Operator wording for {field}." for field in REQUIRED_FIELDS}

    safe = render_operator_narrative(explanation, safe_renderer)
    assert safe["source"] == "constrained_renderer"

    def unsafe_renderer(_request):
        result = {field: f"Operator wording for {field}." for field in REQUIRED_FIELDS}
        result["what_happened"] = "A confirmed attack compromised the network."
        return result

    fallback = render_operator_narrative(explanation, unsafe_renderer)
    assert fallback["source"] == "deterministic_fallback"
    assert fallback["fallback_reason"] == "renderer_failed_validation"
    assert fallback["fields"]["what_happened"] == explanation["what_happened"]["summary"]


def test_renderer_rejects_a_tampered_authoritative_explanation():
    card = enrich_incident_card({"severity": "AMBER", "step": 9, "confidence": 0.8})
    explanation = card["operator_explanation"]
    explanation["what_happened"]["summary"] = "Tampered after composition."

    try:
        render_operator_narrative(explanation)
    except ValueError as exc:
        assert "digest" in str(exc)
    else:
        raise AssertionError("tampered structured facts should not render")
