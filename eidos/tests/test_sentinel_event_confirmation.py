from sentinel import EvidenceFrame, process_stream
from sentinel.calibration import get_mode_config
from sentinel.normal_suppression import evidence_weight, is_stable_normal_context


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


def test_isolated_spike_stays_candidate_only():
    stream = [
        EvidenceFrame(
            frame=frame,
            residual_score=5.2 if frame == 12 else 0.5,
            geometry_change=0.7 if frame == 12 else 0.02,
            novelty=0.7 if frame == 12 else 0.02,
        )
        for frame in range(40)
    ]

    result = process_stream(stream, mode="balanced")

    assert result.candidate_events == 1
    assert result.suppressed_candidates == 1
    assert result.confirmed_events == []


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


def test_normal_suppression_only_stable_periods_raise_bar():
    config = get_mode_config("balanced")

    stable_weight = evidence_weight(
        residual_score=2.6,
        geometry_change=0.01,
        novelty=0.01,
        config=config,
    )
    active_weight = evidence_weight(
        residual_score=2.6,
        geometry_change=0.8,
        novelty=0.8,
        config=config,
    )

    assert is_stable_normal_context(
        residual_score=1.0,
        geometry_change=0.02,
        novelty=0.02,
        config=config,
    )
    assert active_weight > stable_weight


def test_high_recall_confirms_earlier_than_low_noise():
    stream = [
        EvidenceFrame(
            frame=frame,
            residual_score=4.3 if frame in {10, 11} else 0.5,
            geometry_change=0.5 if frame in {10, 11} else 0.02,
            novelty=0.5 if frame in {10, 11} else 0.02,
        )
        for frame in range(35)
    ]

    high_recall = process_stream(stream, mode="high_recall")
    low_noise = process_stream(stream, mode="low_noise")

    assert len(high_recall.confirmed_events) == 1
    assert low_noise.confirmed_events == []
