from __future__ import annotations

import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

SUPPORTED_SOURCE_TYPES = {"rss", "atom", "gdelt_doc_api", "arxiv_api", "local_jsonl", "fixture"}
EVENT_SCHEMA_KEYS = (
    "event_id",
    "source_id",
    "source_type",
    "url",
    "title",
    "summary",
    "text",
    "published_at_utc",
    "observed_at_utc",
    "language",
    "domain",
    "raw_hash",
    "license_note",
    "ingest_status",
    "error",
)

FetchBytes = Callable[[str, int], bytes]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_world_sources() -> list[dict[str, Any]]:
    return [
        {
            "id": "bbc_world_rss",
            "type": "rss",
            "url": "https://feeds.bbci.co.uk/news/world/rss.xml",
            "max_items": 20,
            "license_note": "Public BBC RSS metadata; retain short feed metadata only.",
        },
        {
            "id": "gdelt_disruption_watch",
            "type": "gdelt_doc_api",
            "query": '("supply chain" OR outage OR flood OR wildfire OR earthquake)',
            "timespan": "1d",
            "max_records": 25,
            "license_note": "GDELT DOC API article metadata; source article licenses vary by publisher.",
        },
        {
            "id": "arxiv_ai_systems",
            "type": "arxiv_api",
            "search_query": "cat:cs.AI OR cat:cs.LG",
            "max_results": 10,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "license_note": "arXiv API metadata and abstracts; article licenses vary by record.",
        },
    ]


def fixture_world_events() -> list[dict]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [
        {
            "source_id": "fixture_news",
            "source_type": "fixture",
            "title": "Energy prices rise amid storms",
            "summary": "Fixture event for deterministic world telemetry smoke tests.",
            "text": "energy storm supply chain",
            "published_at_utc": now,
            "url": "fixture://world/energy-prices-rise-amid-storms",
        },
        {
            "source_id": "fixture_forum",
            "source_type": "fixture",
            "title": "AI chips demand surges",
            "summary": "Fixture event for deterministic world telemetry smoke tests.",
            "text": "ai chips demand market",
            "published_at_utc": now,
            "url": "fixture://world/ai-chips-demand-surges",
        },
    ]
    return [
        normalize_event(
            raw=row,
            source_id=row["source_id"],
            source_type="fixture",
            observed_at_utc=now,
            url=row["url"],
            title=row["title"],
            summary=row["summary"],
            text=row["text"],
            published_at_utc=row["published_at_utc"],
            license_note="Local deterministic fixture for CI and smoke tests.",
        )
        for row in rows
    ]


def ingest_world_sources(
    sources: Iterable[dict[str, Any]] | None = None,
    *,
    observed_at_utc: str | None = None,
    base_dir: Path | None = None,
    fetcher: FetchBytes | None = None,
    timeout_seconds: int = 20,
    max_events: int | None = None,
) -> list[dict[str, Any]]:
    observed = observed_at_utc or utc_now()
    events: list[dict[str, Any]] = []
    for source in list(sources or default_world_sources()):
        source_id = str(source.get("id") or source.get("source_id") or "unnamed_source")
        source_type = str(source.get("type") or "").strip().lower()
        if not source.get("enabled", True):
            events.append(_status_event(source, observed, "skipped", "source disabled"))
            continue
        if source_type not in SUPPORTED_SOURCE_TYPES:
            events.append(_status_event(source, observed, "error", f"unsupported source type: {source_type}"))
            continue
        try:
            if source_type == "fixture":
                events.extend(fixture_world_events())
            elif source_type == "rss":
                events.extend(_ingest_rss(source, observed, fetcher or _fetch_url, timeout_seconds))
            elif source_type == "atom":
                events.extend(_ingest_atom(source, observed, fetcher or _fetch_url, timeout_seconds, "atom"))
            elif source_type == "arxiv_api":
                events.extend(_ingest_arxiv(source, observed, fetcher or _fetch_url, timeout_seconds))
            elif source_type == "gdelt_doc_api":
                events.extend(_ingest_gdelt_doc(source, observed, fetcher or _fetch_url, timeout_seconds))
            elif source_type == "local_jsonl":
                events.extend(_ingest_local_jsonl(source, observed, base_dir or Path.cwd()))
        except Exception as exc:  # pragma: no cover - exercised through error-path tests
            events.append(_status_event(source, observed, "error", str(exc)))
        if max_events is not None and len([e for e in events if e.get("ingest_status") == "ok"]) >= max_events:
            break
    if max_events is not None:
        ok_seen = 0
        limited: list[dict[str, Any]] = []
        for event in events:
            if event.get("ingest_status") != "ok":
                limited.append(event)
                continue
            if ok_seen < max_events:
                limited.append(event)
                ok_seen += 1
        return limited
    return events


def source_manifest(events: list[dict[str, Any]], sources: Iterable[dict[str, Any]]) -> dict[str, Any]:
    source_list = list(sources)
    status_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    ok_events = [event for event in events if event.get("ingest_status") == "ok"]
    for event in events:
        status = str(event.get("ingest_status") or "unknown")
        source_type = str(event.get("source_type") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        type_counts[source_type] = type_counts.get(source_type, 0) + 1
    return {
        "corpus_version": "real_world_corpus_v0",
        "source_count": len(source_list),
        "event_count": len(events),
        "ok_event_count": len(ok_events),
        "status_counts": status_counts,
        "source_type_counts": type_counts,
        "sources": [
            {
                "id": str(source.get("id") or source.get("source_id") or "unnamed_source"),
                "type": str(source.get("type") or "").strip().lower(),
                "enabled": bool(source.get("enabled", True)),
            }
            for source in source_list
        ],
    }


def normalize_event(
    *,
    raw: Any,
    source_id: str,
    source_type: str,
    observed_at_utc: str,
    url: str | None = None,
    title: str | None = None,
    summary: str | None = None,
    text: str | None = None,
    published_at_utc: str | None = None,
    language: str | None = None,
    domain: str | None = None,
    license_note: str | None = None,
    ingest_status: str = "ok",
    error: str | None = None,
    event_id: str | None = None,
) -> dict[str, Any]:
    clean_url = _clean(url)
    clean_title = _clean(title)
    clean_summary = _clean(summary)
    clean_text = _clean(text) or _clean(" ".join(x for x in [clean_title, clean_summary] if x))
    published = _parse_datetime(published_at_utc)
    observed = _parse_datetime(observed_at_utc) or observed_at_utc
    raw_hash = _raw_hash(raw)
    identity = clean_url or clean_title or raw_hash
    normalized = {
        "event_id": event_id
        or _event_id(
            source_id=source_id,
            source_type=source_type,
            identity=identity,
            published_at_utc=published,
        ),
        "source_id": source_id,
        "source_type": source_type,
        "url": clean_url,
        "title": clean_title,
        "summary": clean_summary,
        "text": clean_text,
        "published_at_utc": published,
        "observed_at_utc": observed,
        "language": _clean(language),
        "domain": _clean(domain) or _domain(clean_url),
        "raw_hash": raw_hash,
        "license_note": _clean(license_note),
        "ingest_status": ingest_status,
        "error": error,
    }
    normalized["timestamp"] = normalized["published_at_utc"] or normalized["observed_at_utc"]
    return normalized


def _ingest_rss(source: dict[str, Any], observed_at_utc: str, fetcher: FetchBytes, timeout_seconds: int) -> list[dict[str, Any]]:
    url = _required(source, "url")
    raw_feed = fetcher(url, timeout_seconds)
    root = ET.fromstring(raw_feed)
    channel = root.find("channel")
    feed_language = _child_text(channel, "language") if channel is not None else None
    items = root.findall(".//item")[: _limit(source, "max_items", 25)]
    license_note = source.get("license_note") or "Public RSS metadata; source item licenses vary."
    events: list[dict[str, Any]] = []
    for item in items:
        raw_item = ET.tostring(item, encoding="unicode")
        title = _child_text(item, "title")
        link = _child_text(item, "link") or _child_text(item, "guid")
        summary = _strip_markup(_child_text(item, "description"))
        published = _child_text(item, "pubDate") or _child_text(item, "published") or _child_text(item, "updated")
        language = _child_text(item, "language") or feed_language
        events.append(
            normalize_event(
                raw=raw_item,
                source_id=str(source.get("id")),
                source_type="rss",
                observed_at_utc=observed_at_utc,
                url=link,
                title=title,
                summary=summary,
                text=" ".join(x for x in [title, summary] if x),
                published_at_utc=published,
                language=language,
                license_note=license_note,
            )
        )
    return events or [_status_event(source, observed_at_utc, "skipped", "rss feed contained no item elements")]


def _ingest_atom(
    source: dict[str, Any],
    observed_at_utc: str,
    fetcher: FetchBytes,
    timeout_seconds: int,
    source_type: str,
) -> list[dict[str, Any]]:
    url = _required(source, "url")
    raw_feed = fetcher(url, timeout_seconds)
    root = ET.fromstring(raw_feed)
    return _atom_entries_to_events(
        source=source,
        source_type=source_type,
        root=root,
        observed_at_utc=observed_at_utc,
        license_note=source.get("license_note") or "Public Atom metadata; source item licenses vary.",
    )


def _ingest_arxiv(
    source: dict[str, Any],
    observed_at_utc: str,
    fetcher: FetchBytes,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    url = _arxiv_url(source)
    raw_feed = fetcher(url, timeout_seconds)
    root = ET.fromstring(raw_feed)
    arxiv_source = {**source, "url": url}
    return _atom_entries_to_events(
        source=arxiv_source,
        source_type="arxiv_api",
        root=root,
        observed_at_utc=observed_at_utc,
        license_note=source.get("license_note") or "arXiv API metadata and abstracts; article licenses vary by record.",
    )


def _ingest_gdelt_doc(
    source: dict[str, Any], observed_at_utc: str, fetcher: FetchBytes, timeout_seconds: int
) -> list[dict[str, Any]]:
    url = _gdelt_url(source)
    raw_payload = fetcher(url, timeout_seconds)
    payload = json.loads(raw_payload.decode("utf-8-sig"))
    articles = payload.get("articles") or payload.get("items") or []
    events: list[dict[str, Any]] = []
    for article in articles[: _limit(source, "max_records", 25)]:
        title = article.get("title")
        summary = article.get("description") or article.get("snippet") or article.get("summary")
        article_url = article.get("url") or article.get("url_mobile")
        text = " ".join(str(x) for x in [title, summary, article.get("domain")] if x)
        events.append(
            normalize_event(
                raw=article,
                source_id=str(source.get("id")),
                source_type="gdelt_doc_api",
                observed_at_utc=observed_at_utc,
                url=article_url,
                title=title,
                summary=summary,
                text=text,
                published_at_utc=article.get("seendate") or article.get("date"),
                language=article.get("language"),
                domain=article.get("domain"),
                license_note=source.get("license_note")
                or "GDELT DOC API article metadata; source article licenses vary by publisher.",
            )
        )
    return events or [_status_event(source, observed_at_utc, "skipped", "gdelt response contained no articles")]


def _ingest_local_jsonl(source: dict[str, Any], observed_at_utc: str, base_dir: Path) -> list[dict[str, Any]]:
    raw_path = Path(_required(source, "path"))
    path = raw_path if raw_path.is_absolute() else base_dir / raw_path
    max_items = _limit(source, "max_items", 1000)
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= max_items:
                break
            if not line.strip():
                continue
            raw = json.loads(line)
            source_id = str(raw.get("source_id") or source.get("id"))
            source_type = str(raw.get("source_type") or "local_jsonl")
            status = str(raw.get("ingest_status") or "ok")
            error = raw.get("error")
            if status == "ok" and not (raw.get("title") or raw.get("text") or raw.get("summary")):
                status = "skipped"
                error = "local_jsonl record missing title, text, and summary"
            events.append(
                normalize_event(
                    raw=raw,
                    source_id=source_id,
                    source_type=source_type if source_type in SUPPORTED_SOURCE_TYPES else "local_jsonl",
                    observed_at_utc=str(raw.get("observed_at_utc") or observed_at_utc),
                    url=raw.get("url"),
                    title=raw.get("title"),
                    summary=raw.get("summary") or raw.get("abstract"),
                    text=raw.get("text"),
                    published_at_utc=raw.get("published_at_utc") or raw.get("timestamp"),
                    language=raw.get("language"),
                    domain=raw.get("domain"),
                    license_note=raw.get("license_note") or source.get("license_note") or "Local JSONL corpus record.",
                    ingest_status=status,
                    error=error,
                    event_id=raw.get("event_id"),
                )
            )
    return events or [_status_event(source, observed_at_utc, "skipped", "local_jsonl file contained no records")]


def _atom_entries_to_events(
    *,
    source: dict[str, Any],
    source_type: str,
    root: ET.Element,
    observed_at_utc: str,
    license_note: str,
) -> list[dict[str, Any]]:
    entries = root.findall("{http://www.w3.org/2005/Atom}entry") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
    feed_language = root.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
    events: list[dict[str, Any]] = []
    for entry in entries[: _limit(source, "max_items", _limit(source, "max_results", 25))]:
        raw_entry = ET.tostring(entry, encoding="unicode")
        title = _atom_text(entry, "title")
        summary = _strip_markup(_atom_text(entry, "summary") or _atom_text(entry, "content"))
        entry_id = _atom_text(entry, "id")
        link = _atom_link(entry) or entry_id
        events.append(
            normalize_event(
                raw=raw_entry,
                source_id=str(source.get("id")),
                source_type=source_type,
                observed_at_utc=observed_at_utc,
                url=link,
                title=title,
                summary=summary,
                text=" ".join(x for x in [title, summary] if x),
                published_at_utc=_atom_text(entry, "published") or _atom_text(entry, "updated"),
                language=entry.attrib.get("{http://www.w3.org/XML/1998/namespace}lang") or feed_language,
                license_note=license_note,
            )
        )
    return events or [_status_event(source, observed_at_utc, "skipped", f"{source_type} feed contained no entry elements")]


def _status_event(source: dict[str, Any], observed_at_utc: str, status: str, error: str | None) -> dict[str, Any]:
    source_id = str(source.get("id") or source.get("source_id") or "unnamed_source")
    source_type = str(source.get("type") or "unknown").strip().lower() or "unknown"
    raw = {"source": source, "status": status, "error": error}
    return normalize_event(
        raw=raw,
        source_id=source_id,
        source_type=source_type if source_type in SUPPORTED_SOURCE_TYPES else "local_jsonl",
        observed_at_utc=observed_at_utc,
        url=source.get("url"),
        title=f"{source_id} ingest {status}",
        summary=error,
        text=error,
        published_at_utc=None,
        language=source.get("language"),
        domain=source.get("domain") or _domain(source.get("url")),
        license_note=source.get("license_note"),
        ingest_status=status,
        error=error,
    )


def _gdelt_url(source: dict[str, Any]) -> str:
    base_url = str(source.get("url") or "https://api.gdeltproject.org/api/v2/doc/doc")
    params = {
        "query": _required(source, "query"),
        "mode": source.get("mode", "ArtList"),
        "format": source.get("format", "json"),
        "maxrecords": str(_limit(source, "max_records", 25)),
        "sort": source.get("sort", "DateDesc"),
    }
    if source.get("timespan"):
        params["timespan"] = str(source["timespan"])
    if source.get("startdatetime"):
        params["startdatetime"] = str(source["startdatetime"])
    if source.get("enddatetime"):
        params["enddatetime"] = str(source["enddatetime"])
    return f"{base_url}?{urlencode(params)}"


def _arxiv_url(source: dict[str, Any]) -> str:
    base_url = str(source.get("url") or "https://export.arxiv.org/api/query")
    params = {
        "search_query": source.get("search_query", "cat:cs.AI"),
        "start": str(int(source.get("start", 0))),
        "max_results": str(_limit(source, "max_results", 10)),
    }
    for key in ("sortBy", "sortOrder"):
        if source.get(key):
            params[key] = str(source[key])
    return f"{base_url}?{urlencode(params)}"


def _fetch_url(url: str, timeout_seconds: int) -> bytes:
    request = Request(url, headers={"User-Agent": "eidos-brain-prediction-corpus/0.1"})
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


def _event_id(*, source_id: str, source_type: str, identity: str, published_at_utc: str | None) -> str:
    raw = "|".join([source_id, source_type, identity or "", published_at_utc or ""])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _raw_hash(raw: Any) -> str:
    if isinstance(raw, bytes):
        payload = raw
    elif isinstance(raw, str):
        payload = raw.encode("utf-8")
    else:
        payload = json.dumps(raw, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_datetime(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d{14}", text):
        dt = datetime.strptime(text, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if re.fullmatch(r"\d{8}", text):
        dt = datetime.strptime(text, "%Y%m%d").replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        dt = parsedate_to_datetime(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError, IndexError, OverflowError):
        pass
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return text


def _child_text(parent: ET.Element | None, tag: str) -> str | None:
    if parent is None:
        return None
    child = parent.find(tag)
    if child is None or child.text is None:
        return None
    return _clean(child.text)


def _atom_text(entry: ET.Element, tag: str) -> str | None:
    child = entry.find(f"{{http://www.w3.org/2005/Atom}}{tag}")
    if child is None:
        child = entry.find(tag)
    if child is None:
        return None
    return _clean("".join(child.itertext()))


def _atom_link(entry: ET.Element) -> str | None:
    links = entry.findall("{http://www.w3.org/2005/Atom}link") or entry.findall("link")
    fallback = None
    for link in links:
        href = link.attrib.get("href")
        if not href:
            continue
        rel = link.attrib.get("rel", "alternate")
        if rel == "alternate":
            return href
        fallback = fallback or href
    return fallback


def _strip_markup(value: str | None) -> str | None:
    if value is None:
        return None
    return _clean(re.sub(r"<[^>]+>", " ", html.unescape(value)))


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = html.unescape(str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _domain(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(str(url))
    return parsed.netloc.lower() or None


def _required(source: dict[str, Any], key: str) -> str:
    value = source.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"{source.get('id', 'source')} missing required field: {key}")
    return str(value)


def _limit(source: dict[str, Any], key: str, default: int) -> int:
    value = int(source.get(key, default))
    return max(value, 0)
