from eidos_brain.prediction.sources import fixture_world_events

def test_fixture_sources():
    assert len(fixture_world_events())>0
