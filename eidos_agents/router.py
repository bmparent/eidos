"""Cheap deterministic routing before any LLM judgment."""

from __future__ import annotations

ROUTES: list[tuple[tuple[str, ...], str]] = [
    (("find", "previous", "last experiment", "retrieve", "history", "artifact"), "archivist"),
    (("design an experiment", "experiment design", "negative control", "ablation", "falsifiable"), "curie"),
    (("false positive", "fp", "over-alert", "precision", "recall", "sentinel", "alert pressure"), "sentry"),
    (("derive", "stability", "spectral radius", "power-law", "mathemat", "rls", "reservoir"), "gauss"),
    (("implement", "patch", "write code", "approved change"), "forge"),
    (("run benchmark", "pytest", "measure", "seed sweep", "benchmark"), "bench"),
    (("audit", "verify independently", "review result", "certify"), "auditor"),
    (("disagreement", "adjudicate", "new mathematics", "novel mathematics"), "council"),
]


def route_question(question: str) -> str:
    lowered = question.lower()
    scores: dict[str, int] = {}
    for keywords, agent in ROUTES:
        scores[agent] = sum(1 for keyword in keywords if keyword in lowered)
    best = max(scores, key=scores.get)
    return best if scores[best] else "director"


def cheapest_capable_route(question: str) -> str:
    if "summarize" in question.lower() and "log" in question.lower():
        return "deterministic_python"
    return route_question(question)
