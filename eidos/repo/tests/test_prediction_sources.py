import json
from pathlib import Path

from eidos_brain.prediction.sources import EVENT_SCHEMA_KEYS, fixture_world_events, ingest_world_sources


def _assert_event_schema(event: dict) -> None:
    for key in EVENT_SCHEMA_KEYS:
        assert key in event


def test_fixture_sources_keep_smoke_events():
    events = fixture_world_events()
    assert len(events) == 2
    assert {event["source_type"] for event in events} == {"fixture"}
    assert [event["title"] for event in events] == [
        "Energy prices rise amid storms",
        "AI chips demand surges",
    ]
    for event in events:
        _assert_event_schema(event)
        assert event["ingest_status"] == "ok"
        assert event["timestamp"] == event["published_at_utc"]


def test_rss_source_normalizes_items_with_stable_ids():
    rss = b"""<?xml version="1.0"?>
    <rss><channel><language>en</language><item>
      <title>Storm supply disruption</title>
      <link>https://example.com/storm</link>
      <description><![CDATA[<p>Ports slowed by storms.</p>]]></description>
      <pubDate>Fri, 22 May 2026 10:00:00 GMT</pubDate>
    </item></channel></rss>"""

    def fetcher(url: str, timeout: int) -> bytes:
        assert url == "https://example.com/rss.xml"
        assert timeout == 5
        return rss

    source = [{"id": "rss_unit", "type": "rss", "url": "https://example.com/rss.xml"}]
    first = ingest_world_sources(source, observed_at_utc="2026-05-22T10:01:00Z", fetcher=fetcher, timeout_seconds=5)
    second = ingest_world_sources(source, observed_at_utc="2026-05-22T11:01:00Z", fetcher=fetcher, timeout_seconds=5)
    assert first[0]["event_id"] == second[0]["event_id"]
    assert first[0]["source_type"] == "rss"
    assert first[0]["published_at_utc"] == "2026-05-22T10:00:00Z"
    assert first[0]["domain"] == "example.com"
    assert "Ports slowed" in first[0]["summary"]


def test_atom_source_normalizes_entries():
    atom = b"""<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xml:lang="en">
      <entry>
        <id>tag:example.com,2026:entry</id>
        <title>New release shipped</title>
        <link rel="alternate" href="https://example.com/releases/1" />
        <summary>Release notes for a public software project.</summary>
        <updated>2026-05-22T12:00:00Z</updated>
      </entry>
    </feed>"""

    events = ingest_world_sources(
        [{"id": "atom_unit", "type": "atom", "url": "https://example.com/feed.atom"}],
        observed_at_utc="2026-05-22T12:01:00Z",
        fetcher=lambda _url, _timeout: atom,
    )
    assert events[0]["source_type"] == "atom"
    assert events[0]["url"] == "https://example.com/releases/1"
    assert events[0]["language"] == "en"


def test_gdelt_source_normalizes_articles():
    payload = {
        "articles": [
            {
                "url": "https://news.example/a",
                "title": "Flood response expands",
                "seendate": "20260522123000",
                "domain": "news.example",
                "language": "English",
            }
        ]
    }
    seen_urls = []

    def fetcher(url: str, _timeout: int) -> bytes:
        seen_urls.append(url)
        return json.dumps(payload).encode("utf-8")

    events = ingest_world_sources(
        [{"id": "gdelt_unit", "type": "gdelt_doc_api", "query": "flood", "max_records": 1}],
        observed_at_utc="2026-05-22T12:31:00Z",
        fetcher=fetcher,
    )
    assert "api.gdeltproject.org/api/v2/doc/doc" in seen_urls[0]
    assert "mode=ArtList" in seen_urls[0]
    assert events[0]["source_type"] == "gdelt_doc_api"
    assert events[0]["published_at_utc"] == "2026-05-22T12:30:00Z"


def test_arxiv_source_uses_atom_api_shape():
    atom = b"""<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2605.00001v1</id>
        <title>Auditable forecasting systems</title>
        <summary>Abstract text.</summary>
        <published>2026-05-22T13:00:00Z</published>
      </entry>
    </feed>"""
    seen_urls = []

    def fetcher(url: str, _timeout: int) -> bytes:
        seen_urls.append(url)
        return atom

    events = ingest_world_sources(
        [{"id": "arxiv_unit", "type": "arxiv_api", "search_query": "cat:cs.AI", "max_results": 1}],
        observed_at_utc="2026-05-22T13:01:00Z",
        fetcher=fetcher,
    )
    assert "export.arxiv.org/api/query" in seen_urls[0]
    assert "search_query=cat%3Acs.AI" in seen_urls[0]
    assert events[0]["source_type"] == "arxiv_api"
    assert events[0]["url"] == "http://arxiv.org/abs/2605.00001v1"


def test_local_jsonl_source_reads_normalized_records(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps(
            {
                "url": "https://local.example/event",
                "title": "Local corpus event",
                "text": "local corpus signal",
                "published_at_utc": "2026-05-22T14:00:00Z",
                "license_note": "test fixture",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    events = ingest_world_sources(
        [{"id": "local_unit", "type": "local_jsonl", "path": "events.jsonl"}],
        observed_at_utc="2026-05-22T14:01:00Z",
        base_dir=tmp_path,
    )
    assert events[0]["source_id"] == "local_unit"
    assert events[0]["source_type"] == "local_jsonl"
    assert events[0]["license_note"] == "test fixture"


def test_source_errors_are_audited_without_raising():
    def bad_fetcher(_url: str, _timeout: int) -> bytes:
        raise RuntimeError("network unavailable")

    events = ingest_world_sources(
        [{"id": "rss_error", "type": "rss", "url": "https://example.com/rss.xml"}],
        observed_at_utc="2026-05-22T15:00:00Z",
        fetcher=bad_fetcher,
    )
    assert events[0]["ingest_status"] == "error"
    assert events[0]["error"] == "network unavailable"
