import numpy as np

from eidos_brain.proof.grand_proof_scenarios import SCENARIO_IDS, ScenarioConfig, generate_scenario


def test_all_scenarios_are_deterministic_and_finite():
    config = ScenarioConfig.smoke()
    for scenario_id in SCENARIO_IDS:
        left = generate_scenario(scenario_id, seed=7, config=config)
        right = generate_scenario(scenario_id, seed=7, config=config)
        assert np.array_equal(left.frames, right.frames)
        assert np.array_equal(left.labels, right.labels)
        assert np.isfinite(left.frames).all()


def test_s7_s8_are_matched_except_consequence():
    config = ScenarioConfig.smoke()
    harmless = generate_scenario("S7_harmless_repeat", seed=3, config=config)
    dangerous = generate_scenario("S8_dangerous_repeat", seed=3, config=config)
    assert np.array_equal(harmless.frames, dangerous.frames)
    assert [event.start for event in harmless.events] == [event.start for event in dangerous.events]
    assert harmless.events[0].outcome == "benign"
    assert dangerous.events[0].outcome == "harmful"
    assert harmless.events[0].feedback_at < harmless.events[1].start


def test_online_stream_seals_truth_and_outcomes():
    scenario = generate_scenario("S8_dangerous_repeat", seed=1, config=ScenarioConfig.smoke())
    _frame, metadata = next(scenario.online_frames())
    assert "label" not in metadata
    assert "outcome" not in metadata
    assert "consequential" not in metadata

