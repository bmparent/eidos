"""Deterministic, operator-facing explanations for Eidos incident cards.

The structured explanation is authoritative.  An optional renderer may improve
wording, but it cannot add facts, change the five-field schema, or replace the
deterministic fallback.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any, Callable, Dict, List, Mapping, Optional


SCHEMA_VERSION = "eidos.operator_explanation.v1"
REQUIRED_FIELDS = (
    "what_happened",
    "why_it_matters",
    "evidence",
    "uncertainty",
    "next_action",
)
_UNSAFE_RENDERED_CLAIMS = (
    re.compile(r"\bconfirmed (?:cyber ?)?attack\b", re.IGNORECASE),
    re.compile(r"\bmalicious activity (?:was |has been )?detected\b", re.IGNORECASE),
    re.compile(r"\b(?:host|account|system|network) (?:is|was) compromised\b", re.IGNORECASE),
    re.compile(r"\bcontain(?:ment)? (?:is|required|immediately)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:block|isolate|disable|delete|quarantine|contain) (?:the )?"
        r"(?:host|account|asset|ip|system|network)\b",
        re.IGNORECASE,
    ),
)


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _display_number(value: Any) -> Optional[str]:
    number = _finite_number(value)
    if number is None:
        return None
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _humanize_feature(name: Any) -> Dict[str, Any]:
    raw = str(name or "unnamed_feature")
    projected = re.fullmatch(r"(?:cicids_)?projected_(\d+)", raw, re.IGNORECASE)
    generic = re.fullmatch(r"(?:feat|feature|f)_?(\d+)", raw, re.IGNORECASE)
    if projected:
        label = f"projected traffic component {projected.group(1)}"
        limitation = "This projected component cannot be attributed to one raw input feature from this card alone."
    elif generic:
        label = f"model feature {generic.group(1)}"
        limitation = "This model index has no registered human-readable source feature in this card."
    else:
        label = raw.replace("_", " ").strip()
        limitation = None
    return {"raw_name": raw, "label": label, "limitation": limitation}


def _driver_items(card: Mapping[str, Any]) -> List[Dict[str, Any]]:
    legacy_evidence = card.get("evidence") if isinstance(card.get("evidence"), Mapping) else {}
    drivers = card.get("top_drivers") or legacy_evidence.get("drivers") or []
    result: List[Dict[str, Any]] = []
    for driver in list(drivers)[:3]:
        if not isinstance(driver, Mapping):
            continue
        name = driver.get("name", driver.get("feature", driver.get("idx", driver.get("index"))))
        feature = _humanize_feature(name)
        value_field = next(
            (key for key in ("score", "value", "abs_residual", "residual") if driver.get(key) is not None),
            None,
        )
        item: Dict[str, Any] = {
            "feature": feature["label"],
            "raw_feature": feature["raw_name"],
            "observed_value": _finite_number(driver.get(value_field)) if value_field else None,
            "value_kind": value_field or "not_provided",
        }
        if feature["limitation"]:
            item["limitation"] = feature["limitation"]
        result.append(item)
    return result


def _reference_items(card: Mapping[str, Any]) -> Dict[str, Any]:
    refs = [str(item) for item in card.get("raw_evidence_refs", []) if item is not None]
    exemplars: List[str] = []
    legacy_evidence = card.get("evidence") if isinstance(card.get("evidence"), Mapping) else {}
    for item in legacy_evidence.get("exemplars", []) or []:
        exemplars.append(str(item))
    combined = list(dict.fromkeys(refs + exemplars))
    return {
        "items": combined[:3],
        "total_count": len(combined),
        "omitted_count": max(0, len(combined) - 3),
    }


def _similar_episode(card: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    episodes = card.get("similar_past_events") or card.get("similar_episodes") or []
    if not episodes:
        return None
    episode = episodes[0]
    if not isinstance(episode, Mapping):
        return {"reference": str(episode)}
    return {
        key: episode[key]
        for key in ("event_id", "step", "regime", "similarity", "sim", "summary")
        if key in episode
    }


def _window(card: Mapping[str, Any]) -> Optional[Dict[str, int]]:
    start = card.get("start_frame")
    end = card.get("end_frame")
    if isinstance(start, int) and isinstance(end, int):
        return {"start_frame": start, "end_frame": end, "duration_frames": max(0, end - start + 1)}
    step = card.get("step")
    if isinstance(step, int):
        return {"step": step}
    return None


def _what_happened(card: Mapping[str, Any]) -> Dict[str, Any]:
    severity = str(card.get("severity") or card.get("regime") or "UNKNOWN").upper()
    event_type = str(card.get("event_type") or "anomaly")
    window = _window(card)
    if window and "start_frame" in window:
        summary = (
            f"Eidos confirmed a {event_type.replace('_', ' ')} evidence window from frame "
            f"{window['start_frame']} through {window['end_frame']} "
            f"({window['duration_frames']} frames), triaged as {severity}."
        )
    elif window and "step" in window:
        summary = f"Eidos emitted a {severity} anomaly card at step {window['step']}."
    else:
        summary = f"Eidos emitted a {severity} {event_type.replace('_', ' ')} card."
    observations: List[str] = []
    peak = _display_number(card.get("peak_score"))
    if peak is not None:
        observations.append(f"Peak evidence score: {peak}.")
    legacy_evidence = card.get("evidence") if isinstance(card.get("evidence"), Mapping) else {}
    baseline = legacy_evidence.get("baseline") if isinstance(legacy_evidence.get("baseline"), Mapping) else {}
    z_value = _display_number(baseline.get("z"))
    if z_value is not None:
        observations.append(f"Observed anomaly z-score: {z_value}.")
    for reason in list(card.get("why_flagged") or [])[:3]:
        observations.append(str(reason))
    return {
        "summary": summary,
        "observations": observations,
        "window": window,
        "event_type": event_type,
        "severity": severity,
    }


def _why_it_matters(card: Mapping[str, Any]) -> Dict[str, Any]:
    severity = str(card.get("severity") or card.get("regime") or "UNKNOWN").upper()
    window = _window(card)
    is_confirmed_window = window is not None and "start_frame" in window
    if severity == "RED":
        priority = "prompt"
        qualifier = "sustained " if is_confirmed_window else ""
        summary = f"The {qualifier}evidence crossed the RED triage threshold and warrants prompt investigation."
    elif severity == "AMBER":
        priority = "timely"
        summary = "The evidence crossed the AMBER triage threshold and warrants timely investigation."
    elif severity == "RECOVERY":
        priority = "monitor"
        summary = "The recovery transition is operationally relevant and should be checked against the preceding event."
    else:
        priority = "routine"
        summary = "The observation is available for routine review; this card alone does not establish impact."
    return {
        "summary": summary,
        "triage_priority": priority,
        "bounded_meaning": "Severity prioritizes review; it is not proof of malicious activity, root cause, or business impact.",
    }


def _evidence(card: Mapping[str, Any]) -> Dict[str, Any]:
    drivers = _driver_items(card)
    references = _reference_items(card)
    similar = _similar_episode(card)
    if drivers:
        driver_names = ", ".join(item["feature"] for item in drivers)
        summary = f"The strongest recorded contributors were {driver_names}."
    else:
        summary = "No ranked feature contributors were preserved in this card."
    return {
        "summary": summary,
        "top_drivers": drivers,
        "raw_references": references,
        "nearest_similar_event": similar,
        "source_priority": "Recorded observations and raw references outrank narrative interpretation.",
    }


def _confidence_channels(card: Mapping[str, Any]) -> Dict[str, Any]:
    hypotheses = card.get("hypotheses") or []
    hypothesis = hypotheses[0] if hypotheses and isinstance(hypotheses[0], Mapping) else {}
    forecast = card.get("forecast") if isinstance(card.get("forecast"), Mapping) else {}
    return {
        "detection": {
            "value": _finite_number(card.get("confidence")),
            "meaning": "Card or confirmation confidence; not an attack probability.",
        },
        "hypothesis": {
            "label": hypothesis.get("label"),
            "value": _finite_number(hypothesis.get("confidence")),
            "meaning": "Confidence in the named hypothesis only; a generic anomaly label is not a cause classification.",
        },
        "forecast": {
            "likely_mode": forecast.get("likely_mode"),
            "value": _finite_number(forecast.get("confidence")),
            "meaning": "Forecast confidence, separate from detection and cause classification.",
        },
    }


def _uncertainty(card: Mapping[str, Any]) -> Dict[str, Any]:
    channels = _confidence_channels(card)
    unknown = [
        "Whether the activity is malicious or benign.",
        "Root cause and affected asset or identity, unless corroborated outside this card.",
        "Business impact and whether a response action is necessary.",
    ]
    forecast = channels["forecast"]
    if forecast["value"] in (None, 0.0) or str(forecast["likely_mode"] or "unknown").lower() == "unknown":
        unknown.append("Future incident mode; no informative forecast is available.")
    limitations = [
        item["limitation"]
        for item in _driver_items(card)
        if item.get("limitation")
    ]
    return {
        "summary": "The anomaly observation is stronger than any causal or maliciousness conclusion.",
        "known": ["The recorded signal met the configured criteria for this emitted card."],
        "inferred": ["The observation is unusual enough to prioritize contextual investigation."],
        "unknown": unknown,
        "confidence_channels": channels,
        "limitations": list(dict.fromkeys(limitations)),
    }


def _next_action(card: Mapping[str, Any]) -> Dict[str, Any]:
    severity = str(card.get("severity") or card.get("regime") or "UNKNOWN").upper()
    domain = str(card.get("domain") or "generic").lower()
    window = _window(card)
    location = (
        f"frames {window['start_frame']}-{window['end_frame']}"
        if window and "start_frame" in window
        else f"step {window['step']}" if window and "step" in window else "the recorded event window"
    )
    references = _reference_items(card)
    if references["total_count"]:
        steps = [f"Inspect the ranked drivers and raw references for {location}."]
    else:
        steps = [
            f"Inspect the ranked drivers for {location} and retrieve the underlying telemetry; "
            "this card preserves no raw evidence reference."
        ]
    if domain in {"cyber", "web"}:
        steps.append("Correlate the window with authentication, network, endpoint, and change logs for the named assets or identities.")
    else:
        steps.append("Correlate the window with source-system logs, recent changes, and the responsible asset or process owner.")
    steps.append("Record corroborating and disconfirming observations before changing incident status.")
    if severity == "RED":
        urgency = "prompt"
    elif severity == "AMBER":
        urgency = "timely"
    else:
        urgency = "routine"
    return {
        "summary": f"A security operator should investigate {location} and seek independent corroboration.",
        "actor": "security_operator",
        "urgency": urgency,
        "steps": steps,
        "escalate_if": [
            "Independent telemetry shows unauthorized access, exploitation, harmful execution, or material impact.",
            "The signal persists, recurs, or expands across assets after benign explanations are checked.",
        ],
        "not_justified_by_this_card": [
            "Declaring a confirmed attack or compromise.",
            "Blocking, isolating, or containing an asset without corroborating evidence or an approved response policy.",
        ],
        "machine_action_suggestions": list(card.get("actions") or [])[:3],
        "machine_action_authority": "recommendation_only",
    }


def _facts_digest(explanation: Mapping[str, Any]) -> str:
    facts = {
        "schema_version": explanation.get("schema_version"),
        **{field: explanation.get(field) for field in REQUIRED_FIELDS},
    }
    canonical = json.dumps(facts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compose_operator_explanation(card: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the authoritative five-field explanation from preserved card facts."""
    explanation: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "what_happened": _what_happened(card),
        "why_it_matters": _why_it_matters(card),
        "evidence": _evidence(card),
        "uncertainty": _uncertainty(card),
        "next_action": _next_action(card),
    }
    explanation["facts_digest_sha256"] = _facts_digest(explanation)
    validate_operator_explanation(explanation)
    return explanation


def validate_operator_explanation(explanation: Mapping[str, Any]) -> None:
    """Raise ValueError if an authoritative explanation violates its contract."""
    if explanation.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported operator explanation schema: {explanation.get('schema_version')!r}")
    for field in REQUIRED_FIELDS:
        value = explanation.get(field)
        if not isinstance(value, Mapping) or not str(value.get("summary") or "").strip():
            raise ValueError(f"operator explanation field {field!r} requires a non-empty summary")
    if explanation.get("facts_digest_sha256") != _facts_digest(explanation):
        raise ValueError("operator explanation facts digest does not match its structured fields")


def _deterministic_narrative(explanation: Mapping[str, Any]) -> Dict[str, str]:
    return {field: str(explanation[field]["summary"]) for field in REQUIRED_FIELDS}


def _validate_rendered_narrative(rendered: Any) -> Dict[str, str]:
    if not isinstance(rendered, Mapping) or set(rendered) != set(REQUIRED_FIELDS):
        raise ValueError("renderer must return exactly the five operator narrative fields")
    result: Dict[str, str] = {}
    for field in REQUIRED_FIELDS:
        text = rendered.get(field)
        if not isinstance(text, str) or not text.strip() or len(text) > 1000:
            raise ValueError(f"renderer field {field!r} must be 1-1000 characters")
        if any(pattern.search(text) for pattern in _UNSAFE_RENDERED_CLAIMS):
            raise ValueError(f"renderer field {field!r} contains an unsupported certainty or response claim")
        result[field] = text.strip()
    return result


def render_operator_narrative(
    explanation: Mapping[str, Any],
    renderer: Optional[Callable[[Dict[str, Any]], Mapping[str, str]]] = None,
) -> Dict[str, Any]:
    """Render wording with deterministic fallback; never mutate authoritative facts.

    A renderer receives a deep copy containing the five structured fields plus
    explicit constraints.  Invalid output and renderer failures are recorded
    as fallback reasons without exposing exception details.
    """
    validate_operator_explanation(explanation)
    fallback = _deterministic_narrative(explanation)
    if renderer is None:
        return {
            "source": "deterministic",
            "authoritative": False,
            "facts_digest_sha256": explanation.get("facts_digest_sha256"),
            "fields": fallback,
        }
    request = {
        "schema_version": SCHEMA_VERSION,
        "facts_digest_sha256": explanation.get("facts_digest_sha256"),
        "facts": {field: copy.deepcopy(explanation[field]) for field in REQUIRED_FIELDS},
        "constraints": {
            "return_exactly": list(REQUIRED_FIELDS),
            "task": "Reword only. Do not add facts, diagnoses, impact, certainty, or response authority.",
        },
    }
    try:
        rendered = _validate_rendered_narrative(renderer(request))
    except Exception:
        return {
            "source": "deterministic_fallback",
            "authoritative": False,
            "facts_digest_sha256": explanation.get("facts_digest_sha256"),
            "fallback_reason": "renderer_failed_validation",
            "fields": fallback,
        }
    return {
        "source": "constrained_renderer",
        "authoritative": False,
        "facts_digest_sha256": explanation.get("facts_digest_sha256"),
        "fields": rendered,
    }


def enrich_incident_card(
    card: Mapping[str, Any],
    renderer: Optional[Callable[[Dict[str, Any]], Mapping[str, str]]] = None,
) -> Dict[str, Any]:
    """Return a compatible card enriched with structured facts and narrative."""
    enriched = copy.deepcopy(dict(card))
    explanation = compose_operator_explanation(enriched)
    enriched["operator_explanation"] = explanation
    enriched["operator_narrative"] = render_operator_narrative(explanation, renderer)
    return enriched
