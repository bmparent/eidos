from sentinel import EvidenceFrame, process_stream


def mixed_recall_stream():
    for frame in range(140):
        if frame == 20:
            yield EvidenceFrame(frame=frame, residual_score=2.2, geometry_change=0.25, novelty=0.25)
        elif 60 <= frame <= 61:
            yield EvidenceFrame(frame=frame, residual_score=2.8, geometry_change=0.3, novelty=0.3)
        elif 100 <= frame <= 104:
            yield EvidenceFrame(frame=frame, residual_score=4.7, geometry_change=0.7, novelty=0.7)
        else:
            yield EvidenceFrame(frame=frame, residual_score=0.6, geometry_change=0.03, novelty=0.03)


def test_modes_order_false_positive_control_vs_recall():
    low_noise = process_stream(list(mixed_recall_stream()), mode="low_noise")
    balanced = process_stream(list(mixed_recall_stream()), mode="balanced")
    high_recall = process_stream(list(mixed_recall_stream()), mode="high_recall")

    assert len(low_noise.confirmed_events) < len(balanced.confirmed_events)
    assert len(balanced.confirmed_events) <= len(high_recall.confirmed_events)
    assert any(event.start_frame <= 100 <= event.end_frame for event in high_recall.confirmed_events)
