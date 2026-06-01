from sentinel import EvidenceFrame, process_stream


def burst_stream(start=20, end=29, frames=80):
    for frame in range(frames):
        if start <= frame <= end:
            yield EvidenceFrame(
                frame=frame,
                residual_score=4.6,
                geometry_change=0.7,
                novelty=0.65,
                top_drivers=[{"name": "residual_energy", "score": 4.6}],
                raw_evidence_ref=f"burst:{frame}",
            )
        else:
            yield EvidenceFrame(frame=frame, residual_score=0.6, geometry_change=0.03, novelty=0.03)


def test_sustained_anomaly_burst_confirms_as_one_event():
    result = process_stream(burst_stream(), mode="balanced")

    assert len(result.confirmed_events) == 1
    event = result.confirmed_events[0]
    assert event.start_frame == 20
    assert event.end_frame == 29
    assert event.duration == 10
    assert event.event_count == 10
    assert event.peak_score == 4.6
    assert len(result.incident_cards) == 1


def test_repeated_nearby_spikes_merge_without_alert_spam():
    spike_frames = {10, 14, 18}
    stream = [
        EvidenceFrame(
            frame=frame,
            residual_score=4.8 if frame in spike_frames else 0.6,
            geometry_change=0.65 if frame in spike_frames else 0.03,
            novelty=0.7 if frame in spike_frames else 0.03,
            raw_evidence_ref=f"nearby:{frame}" if frame in spike_frames else None,
        )
        for frame in range(50)
    ]

    result = process_stream(stream, mode="balanced")

    assert len(result.confirmed_events) == 1
    event = result.confirmed_events[0]
    assert event.start_frame == 10
    assert event.end_frame == 18
    assert event.event_count == 3
    assert result.cooldown_suppressions == 0


def test_cooldown_suppresses_same_cluster_alert_spam():
    hot_frames = set(range(10, 15)) | set(range(28, 33))
    stream = [
        EvidenceFrame(
            frame=frame,
            residual_score=4.7 if frame in hot_frames else 0.6,
            geometry_change=0.7 if frame in hot_frames else 0.03,
            novelty=0.7 if frame in hot_frames else 0.03,
        )
        for frame in range(60)
    ]

    result = process_stream(stream, mode="balanced")

    assert len(result.confirmed_events) == 1
    assert result.cooldown_suppressions == 1
