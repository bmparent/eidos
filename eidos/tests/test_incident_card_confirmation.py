from sentinel import EvidenceFrame, process_stream


def confirmed_burst_stream():
    for frame in range(50):
        if 12 <= frame <= 17:
            yield EvidenceFrame(
                frame=frame,
                residual_score=4.4,
                geometry_change=0.7,
                novelty=0.7,
                top_drivers=[{"name": "geometry_change", "score": 0.7}],
                similar_past_events=[{"event_id": "prior_1", "similarity": 0.82}],
                raw_evidence_ref=f"steps.csv:{frame}",
            )
        else:
            yield EvidenceFrame(frame=frame, residual_score=0.5, geometry_change=0.02, novelty=0.02)


def test_confirmed_event_emits_one_incident_card_compatible_record():
    result = process_stream(confirmed_burst_stream(), mode="balanced")

    assert len(result.confirmed_events) == 1
    assert len(result.incident_cards) == 1
    card = result.incident_cards[0]
    for key in (
        "incident_id",
        "event_id",
        "start_frame",
        "end_frame",
        "severity",
        "confidence",
        "why_flagged",
        "top_drivers",
        "similar_past_events",
        "raw_evidence_refs",
        "operator_explanation",
        "operator_narrative",
    ):
        assert key in card
    assert card["severity"] in {"AMBER", "RED"}
    assert card["start_frame"] == 12
    assert card["end_frame"] == 17


def eidos_life_lifecycle_stream():
    for generation in range(120):
        if 63 <= generation <= 70:
            yield EvidenceFrame(
                frame=generation,
                residual_score=3.4,
                geometry_change=0.85,
                novelty=0.65,
                lifecycle_phase="collapse",
                top_drivers=[{"name": "alive_ratio_collapse", "score": 0.96}],
                raw_evidence_ref=f"eidos-life:generation:{generation}",
            )
        elif 90 <= generation <= 96:
            yield EvidenceFrame(
                frame=generation,
                residual_score=2.9,
                geometry_change=0.55,
                novelty=0.5,
                lifecycle_phase="recovery",
                top_drivers=[{"name": "post_extinction_reseed", "score": 0.88}],
                raw_evidence_ref=f"eidos-life:generation:{generation}",
            )
        else:
            yield EvidenceFrame(frame=generation, residual_score=0.7, geometry_change=0.04, novelty=0.04)


def test_eidos_life_lifecycle_events_confirm_without_post_recovery_spam():
    result = process_stream(eidos_life_lifecycle_stream(), mode="balanced")

    event_types = [event.event_type for event in result.confirmed_events]
    assert "lifecycle_collapse" in event_types
    assert "lifecycle_recovery" in event_types
    assert result.red_count == 1
    assert any(event.severity == "RECOVERY" for event in result.confirmed_events)
    assert all(event.start_frame < 100 for event in result.confirmed_events)
