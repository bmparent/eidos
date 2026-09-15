import pytest

from eidos_agents.router import cheapest_capable_route, route_question


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Find the last three experiments involving hippocampus familiarity.", "archivist"),
        ("Design an experiment with negative controls.", "curie"),
        ("Why are false positives high?", "sentry"),
        ("Derive the stability condition for leak dynamics.", "gauss"),
        ("Implement the approved patch.", "forge"),
        ("Run benchmark seed sweep.", "bench"),
        ("Audit result independently.", "auditor"),
        ("Resolve specialist disagreement.", "council"),
    ],
)
def test_expected_routes(question, expected):
    assert route_question(question) == expected


def test_astra_not_used_for_clerical_summary():
    assert cheapest_capable_route("Run Astra to summarize this log") == "deterministic_python"


def test_recall_tradeoff_routes_to_sentry():
    assert route_question("False positives fell from 30 to 2 but recall fell from .98 to .41") == "sentry"
