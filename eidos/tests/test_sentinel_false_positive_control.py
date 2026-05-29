from sentinel import EvidenceFrame, process_stream


def normal_only_stream(frames=10000):
    harmless_spikes = {1000, 3000, 6000, 8500}
    for frame in range(frames):
        if frame in harmless_spikes:
            yield EvidenceFrame(
                frame=frame,
                residual_score=3.4,
                geometry_change=0.04,
                novelty=0.04,
                raw_evidence_ref=f"normal_spike:{frame}",
            )
        else:
            yield EvidenceFrame(frame=frame, residual_score=0.7, geometry_change=0.03, novelty=0.03)


def test_low_noise_normal_only_false_positive_target():
    result = process_stream(normal_only_stream(), mode="low_noise")

    assert len(result.confirmed_events) <= 5
    assert result.red_count == 0
    assert result.suppressed_candidates >= 4


def test_single_isolated_spike_stays_candidate_in_low_noise():
    stream = [
        EvidenceFrame(frame=idx, residual_score=0.5, geometry_change=0.02, novelty=0.02)
        for idx in range(80)
    ]
    stream[40] = EvidenceFrame(
        frame=40,
        residual_score=5.2,
        geometry_change=0.8,
        novelty=0.8,
        raw_evidence_ref="isolated_spike:40",
    )

    result = process_stream(stream, mode="low_noise")

    assert result.candidate_events == 1
    assert result.suppressed_candidates == 1
    assert result.confirmed_events == []
