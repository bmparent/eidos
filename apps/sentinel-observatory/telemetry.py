"""Read-only, bounded public HTTP telemetry and a verifiable local evidence journal.

This measures this process's HEAD requests, not visitor traffic or origin health.
Only fixed Eidos Works endpoints can be requested; bodies and credentials are
never collected. SHA256 checksums detect accidental alteration, not an attacker
who can rewrite the database and recompute every checksum.
"""
from __future__ import annotations

import hashlib
import json
import math
import queue
import socket
import sqlite3
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

SOURCES = {
    "home": {"label": "Home", "url": "https://eidos-works.com/"},
    "account": {"label": "Account", "url": "https://eidos-works.com/account/"},
    "community": {"label": "Community", "url": "https://eidos-works.com/community/"},
    "playground": {"label": "Playground", "url": "https://eidos-works.com/playground/"},
    "probe": {"label": "Controlled probe", "url": "https://eidos-works.com/__sentinel_observatory_probe__"},
}
SAFE_HEADERS = ("cf-cache-status", "cache-status", "server", "content-type", "date")
GENESIS = "0" * 64
SECURITY_HEADER_KEYS = ("hsts_present", "csp_present", "x_content_type_options", "frame_options", "referrer_policy", "cache_private", "cache_no_store")


def security_header_observations(headers):
    """Presence and bounded enum observations only; never a security grade."""
    if headers is None:
        return {key: None for key in SECURITY_HEADER_KEYS}
    def enum(name, allowed):
        value = headers.get(name)
        if value is None:
            return "absent"
        normalized = value.strip().lower()
        return normalized if normalized in allowed else "other"
    cache = {part.strip().split("=", 1)[0].lower() for part in headers.get("Cache-Control", "").split(",")}
    return {"hsts_present": bool(headers.get("Strict-Transport-Security")), "csp_present": bool(headers.get("Content-Security-Policy")), "x_content_type_options": enum("X-Content-Type-Options", {"nosniff"}), "frame_options": enum("X-Frame-Options", {"deny", "sameorigin"}), "referrer_policy": enum("Referrer-Policy", {"no-referrer", "no-referrer-when-downgrade", "origin", "origin-when-cross-origin", "same-origin", "strict-origin", "strict-origin-when-cross-origin", "unsafe-url"}), "cache_private": "private" in cache, "cache_no_store": "no-store" in cache}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def checksum(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def deterministic_analysis(event):
    """Evidence-linked description; never pretend a model has interpreted it."""
    eid, status = event["id"], event.get("status")
    facts = [{"text": f"A real HEAD request sampled the {SOURCES[event['source_id']]['label']} route from this runtime.", "evidence_ids": [eid]}]
    hypotheses = []
    unknowns = ["Visitor traffic, origin logs, request bodies, and internal systems were not observed."]
    if status is None:
        summary = "The observation path failed before an HTTP response was available. Site availability is unknown."
        facts.append({"text": "No HTTP response status was received; this is an observation-vantage failure, not proof the website is down.", "evidence_ids": [eid]})
        hypotheses.append({"text": "The outbound proxy, DNS, TLS, connectivity, or remote endpoint may have prevented this observation.", "support": [eid], "against": [], "next_observation": "Repeat a bounded HEAD check from this vantage and compare with an independently authorized vantage."})
    else:
        summary = f"The {SOURCES[event['source_id']]['label']} route returned HTTP {status} in {event['latency_ms']:.0f} ms from this runtime."
        facts.append({"text": f"The observed response status was {status}; elapsed client request time was {event['latency_ms']:.1f} ms.", "evidence_ids": [eid]})
        if event.get("error"):
            facts.append({"text": "An HTTP response was received, but its redirect chain was not completed within the permitted observation boundary.", "evidence_ids": [eid]})
            unknowns.append("The final redirect destination's behavior is unknown; it was not requested.")
        if status >= 400:
            hypotheses.append({"text": "This response could reflect edge policy, route behavior, or an upstream response; the measurement does not identify which layer caused it.", "support": [eid], "against": [], "next_observation": "Compare the next same-route observation and obtain authorized edge or origin logs before attributing a cause."})
        if event.get("kind") == "controlled_probe":
            facts.append({"text": "This was an explicitly requested diagnostic HEAD to a fixed nonexistent-path candidate. A 404 was expected but the actual status above is authoritative; applications may return a fallback page.", "evidence_ids": [eid]})
    if abs(event.get("source_clock_offset_seconds") or 0) > 60:
        facts.append({"text": "The response Date and runtime clock differ by more than 60 seconds. Local freshness uses runtime observation time, not the response Date.", "evidence_ids": [eid]})
        unknowns.append("The response Date may be cached or reflect clock skew; its cause is unproven.")
    return {"kind": "deterministic", "summary": summary, "facts": facts, "hypotheses": hypotheses, "unknowns": unknowns}


class NoAutomaticRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def safe_redirect(current, location):
    target = urllib.parse.urljoin(current, location)
    parsed = urllib.parse.urlsplit(target)
    if parsed.scheme != "https" or parsed.hostname != "eidos-works.com" or parsed.port not in (None, 443) or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("redirect_outside_boundary")
    return target


def safe_error(code):
    messages = {
        "timeout": "The observation timed out before a usable response was received.",
        "tls_error": "The observation encountered a TLS validation or connection failure.",
        "transport_error": "The runtime could not obtain an HTTP response through its observation path.",
        "redirect_outside_boundary": "The response redirected outside the permitted same-host HTTPS boundary, or included a query or fragment; it was not followed.",
        "redirect_limit": "The response exceeded the permitted three same-host HTTPS redirects.",
        "collector_error": "The local collector could not complete this observation.",
    }
    return {"code": code, "message": messages[code], "scope": "observation_boundary" if code.startswith("redirect_") else "observation_vantage"}


def collect_head(source_id, *, timeout=8.0, opener=None):
    """One real HEAD, at most three same-host redirects; response body never read."""
    if source_id not in SOURCES:
        raise ValueError("Unknown source")
    begin, observed = time.monotonic(), utc_now()
    event = {"id": "evt_" + uuid.uuid4().hex, "source_id": source_id, "url": SOURCES[source_id]["url"], "observed_at": observed, "kind": "controlled_probe" if source_id == "probe" else "measurement", "method": "HEAD", "status": None, "error": None, "headers": {}, "security_headers": security_header_observations(None), "redirect_count": 0, "response_chain_complete": False, "timing_scope": "client_head_elapsed", "source_clock_offset_seconds": None}
    opener = opener or urllib.request.build_opener(NoAutomaticRedirect())
    url = event["url"]
    try:
        for hop in range(4):
            request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Eidos-Sentinel-Observatory/0.1 (read-only HEAD monitor)", "Accept": "*/*"})
            try:
                response = opener.open(request, timeout=timeout)
            except urllib.error.HTTPError as exc:
                response = exc  # HTTP errors are measured responses, never bodies.
            try:
                status = response.code
                event["status"] = int(status)
                event["headers"] = {key: str(response.headers[key]).replace("\r", " ").replace("\n", " ")[:256] for key in SAFE_HEADERS if response.headers.get(key)}
                event["security_headers"] = security_header_observations(response.headers)
                if status in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    if location:
                        if hop >= 3:
                            raise ValueError("redirect_limit")
                        url = safe_redirect(url, location)
                        event["redirect_count"] += 1
                        continue
                event["response_chain_complete"] = True
                break
            finally:
                response.close()
    except (TimeoutError, socket.timeout):
        event["error"] = safe_error("timeout")
    except ssl.SSLError:
        event["error"] = safe_error("tls_error")
    except urllib.error.URLError as exc:
        reason = exc.reason
        code = "timeout" if isinstance(reason, (TimeoutError, socket.timeout)) else "tls_error" if isinstance(reason, ssl.SSLError) else "transport_error"
        event["error"] = safe_error(code)
    except ValueError as exc:
        code = str(exc) if str(exc) in ("redirect_outside_boundary", "redirect_limit") else "transport_error"
        event["error"] = safe_error(code)
    except (OSError, UnicodeError):
        event["error"] = safe_error("transport_error")
    event["latency_ms"] = round((time.monotonic() - begin) * 1000, 3)
    event["completed_at"] = utc_now()
    if event["headers"].get("date"):
        try:
            server_time = parsedate_to_datetime(event["headers"]["date"])
            event["source_clock_offset_seconds"] = round((server_time - parse_time(event["completed_at"])).total_seconds(), 1)
        except (TypeError, ValueError, OverflowError):
            pass
    return event


class EventStore:
    def __init__(self, path, retention=5000):
        if retention < 1:
            raise ValueError("Retention must be positive")
        self.path, self.retention, self.lock = str(path), retention, threading.RLock()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT UNIQUE NOT NULL,observed_at TEXT NOT NULL,source_id TEXT NOT NULL,payload TEXT NOT NULL,previous_hash TEXT NOT NULL,evidence_hash TEXT NOT NULL,analysis TEXT NOT NULL,analysis_hash TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS events_source ON events(source_id,seq);
        """)
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('anchor',?)", (GENESIS,))
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('total_recorded','0')")
        self.db.commit()
        self.verified_at = utc_now()
        self.verify()

    def verify(self):
        with self.lock:
            previous = self.db.execute("SELECT value FROM metadata WHERE key='anchor'").fetchone()[0]
            checked = 0
            for row in self.db.execute("SELECT * FROM events ORDER BY seq"):
                if row["previous_hash"] != previous or checksum({"previous_hash": previous, "event": json.loads(row["payload"])}) != row["evidence_hash"]:
                    raise ValueError("Evidence journal integrity check failed; collection must not continue")
                if checksum(json.loads(row["analysis"])) != row["analysis_hash"]:
                    raise ValueError("Analysis checksum failed; collection must not continue")
                previous, checked = row["evidence_hash"], checked + 1
            self.verified_at = utc_now()
            return {"state": "verified", "verified_at": self.verified_at, "records_checked": checked, "algorithm": "SHA-256 append chain", "authenticated": False, "retention_limit": self.retention}

    def append(self, event):
        payload = {key: value for key, value in event.items() if key not in ("analysis", "analysis_hash", "evidence_hash", "previous_hash")}
        analysis = event.get("analysis") or deterministic_analysis(payload)
        with self.lock, self.db:
            row = self.db.execute("SELECT evidence_hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            previous = row[0] if row else self.db.execute("SELECT value FROM metadata WHERE key='anchor'").fetchone()[0]
            digest = checksum({"previous_hash": previous, "event": payload})
            self.db.execute("INSERT INTO events(id,observed_at,source_id,payload,previous_hash,evidence_hash,analysis,analysis_hash) VALUES (?,?,?,?,?,?,?,?)", (payload["id"], payload["observed_at"], payload["source_id"], canonical(payload), previous, digest, canonical(analysis), checksum(analysis)))
            self.db.execute("UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='total_recorded'")
            cutoff = self.db.execute("SELECT seq,evidence_hash FROM events ORDER BY seq DESC LIMIT 1 OFFSET ?", (self.retention,)).fetchone()
            if cutoff:
                self.db.execute("UPDATE metadata SET value=? WHERE key='anchor'", (cutoff["evidence_hash"],))
                self.db.execute("DELETE FROM events WHERE seq<=?", (cutoff["seq"],))
        return self.get(payload["id"])

    @staticmethod
    def unpack(row):
        if row is None:
            return None
        return {**json.loads(row["payload"]), "previous_hash": row["previous_hash"], "evidence_hash": row["evidence_hash"], "analysis": json.loads(row["analysis"]), "analysis_hash": row["analysis_hash"]}

    def get(self, eid):
        with self.lock:
            return self.unpack(self.db.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone())

    def recent(self, limit=200):
        with self.lock:
            return [self.unpack(row) for row in self.db.execute("SELECT * FROM events ORDER BY seq DESC LIMIT ?", (max(1, min(limit, self.retention)),))]

    def update_analysis(self, eid, analysis):
        with self.lock, self.db:
            self.db.execute("UPDATE events SET analysis=?,analysis_hash=? WHERE id=?", (canonical(analysis), checksum(analysis), eid))

    def stats(self):
        with self.lock:
            return {"retained": self.db.execute("SELECT COUNT(*) FROM events").fetchone()[0], "total_recorded": int(self.db.execute("SELECT value FROM metadata WHERE key='total_recorded'").fetchone()[0]), "per_source": {r["source_id"]: r["n"] for r in self.db.execute("SELECT source_id,COUNT(*) n FROM events GROUP BY source_id")}}

    def close(self):
        with self.lock:
            self.db.close()


class Observatory:
    def __init__(self, db_path, interval=30.0, *, collect=True, monitor=None, interpreter=None):
        if not math.isfinite(interval) or interval < 5:
            raise ValueError("Collection interval must be at least five seconds")
        from brain import BrainMonitor
        self.interval, self.collect_enabled = interval, collect
        self.started_at, self.store = utc_now(), EventStore(db_path)
        self.session_id = "session_" + uuid.uuid4().hex
        self.brain = monitor or BrainMonitor()
        self.brain_lock, self.probe_lock = threading.RLock(), threading.Lock()
        self.stop_event, self.wake = threading.Event(), threading.Event()
        self.analysis_queue, self.probe_queue = queue.PriorityQueue(maxsize=32), queue.Queue(maxsize=1)
        self.interpreter = interpreter
        self.model_state = {"state": "unavailable", "name": None, "reason": "No local model interpreter has initialized; deterministic evidence summaries remain available."}
        self.last_probe, self.analysis_failures, self.dropped_analysis = -float("inf"), 0, 0
        self.collector_errors = 0
        self.model_schedule = {}
        self.model_job_sequence = 0
        # A fresh process starts a fresh live model epoch. Re-observing retained
        # evidence here would manufacture retrospectively issued forecasts.
        # Historical per-event Brain outputs remain inspectable in the journal.
        self.journal_status = self.store.verify()
        self.threads = []

    def _brain_observe(self, event):
        with self.brain_lock:
            return self.brain.observe(event["source_id"], event["id"], event.get("completed_at", event["observed_at"]), event["latency_ms"] if event.get("status") is not None and not event.get("error") else None, event["status"])

    def start(self):
        for name, target in (("telemetry-collector", self._collect_loop), ("evidence-interpreter", self._analysis_loop)):
            thread = threading.Thread(name=name, target=target, daemon=True)
            self.threads.append(thread)
            thread.start()

    def stop(self):
        self.stop_event.set()
        self.wake.set()
        for thread in self.threads:
            thread.join(timeout=1)

    def request_probe(self):
        with self.probe_lock:
            remaining = 30 - (time.monotonic() - self.last_probe)
            if remaining > 0 or self.probe_queue.full():
                return {"accepted": False, "retry_after_seconds": max(1, math.ceil(remaining))}
            self.last_probe = time.monotonic()
            self.probe_queue.put_nowait("probe")
            self.wake.set()
            return {"accepted": True, "source_id": "probe", "expected_status": 404, "message": "One fixed-path real diagnostic HEAD was queued; the actual response will appear as evidence."}

    def observe_once(self, source_id):
        event = collect_head(source_id)
        event["collection_session_id"] = self.session_id
        event["brain"] = self._brain_observe(event)
        event["brain"]["observation_timestamp_scope"] = "response_completed_at"
        event["analysis"] = deterministic_analysis(event)
        saved = self.store.append(event)
        self.journal_status = self.store.verify()
        previous = self.model_schedule.get(source_id)
        signature = (event["status"], (event.get("error") or {}).get("code"), event["brain"].get("state"))
        now = time.monotonic()
        # First observation, meaningful state/status changes, and explicit probes
        # get interpretation. Stable routes are sampled at most once per 120s.
        if source_id == "probe" or previous is None or signature != previous[1] or now - previous[0] >= 120:
            priority = 0 if source_id == "probe" else 1 if previous and signature != previous[1] else 10
            try:
                self.model_job_sequence += 1
                self.analysis_queue.put_nowait((priority, self.model_job_sequence, saved["id"]))
                self.model_schedule[source_id] = (now, signature)
            except queue.Full:
                self.dropped_analysis += 1
        return saved

    def _collect_loop(self):
        regular = tuple(key for key in SOURCES if key != "probe")
        due = {key: time.monotonic() + index * self.interval / len(regular) for index, key in enumerate(regular)}
        while not self.stop_event.is_set():
            self.wake.clear()
            try:
                source = self.probe_queue.get_nowait()
            except queue.Empty:
                source = min(due, key=due.get) if self.collect_enabled else None
                if source and due[source] > time.monotonic():
                    source = None
            if source:
                # No catch-up burst: at least the configured interval between each route's starts.
                if source != "probe":
                    due[source] = time.monotonic() + self.interval
                try:
                    self.observe_once(source)
                except Exception:
                    # Do not log exception strings: transport errors can contain private proxy details.
                    self.collector_errors += 1
                continue
            delay = max(.05, min(due.values()) - time.monotonic()) if self.collect_enabled else 30
            self.wake.wait(min(delay, 30))

    def _analysis_loop(self):
        if self.interpreter is None:
            try:
                from interpreter import Interpreter
                self.interpreter = Interpreter()
            except Exception:
                self.model_state = {"state": "unavailable", "name": None, "reason": "The optional local interpreter could not initialize. Deterministic summaries are active."}
        if self.interpreter is not None and hasattr(self.interpreter, "status"):
            self.model_state = self.interpreter.status()
        while not self.stop_event.is_set():
            try:
                _, _, eid = self.analysis_queue.get(timeout=.5)
            except queue.Empty:
                continue
            try:
                event = self.store.get(eid)
                if self.interpreter is not None and event:
                    result = self.interpreter.analyze(event, self.store.recent(30))
                    if isinstance(result, dict) and result.get("kind") in ("model", "deterministic"):
                        self.store.update_analysis(eid, result)
                    if hasattr(self.interpreter, "status"):
                        self.model_state = self.interpreter.status()
            except Exception:
                self.analysis_failures += 1
            finally:
                self.analysis_queue.task_done()

    def snapshot(self):
        captured = utc_now()
        events, stats = self.store.recent(), self.store.stats()
        sources = []
        for source_id, config in SOURCES.items():
            latest = next((event for event in events if event["source_id"] == source_id), None)
            age = max(0, (parse_time(captured) - parse_time(latest["observed_at"])).total_seconds()) if latest else None
            state = "waiting" if latest is None else "stale" if age > max(2 * self.interval + 10, 30) else "unavailable" if latest["status"] is None or latest.get("error") else "fresh"
            sources.append({"id": source_id, **config, "state": state, "latest": latest, "age_seconds": round(age, 1) if age is not None else None, "event_count": stats["per_source"].get(source_id, 0)})
        with self.brain_lock:
            brain_summary = self.brain.summary()
            brain_summary["session_id"] = self.session_id
            brain_summary["session_started_at"] = self.started_at
            brain_summary["restart_policy"] = "Fresh live calibration per process; retained events are not replayed as live-issued forecasts. Historical per-event outputs remain inspectable."
        return {"mode": "live", "captured_at": captured, "started_at": self.started_at, "cadence_seconds": self.interval, "collection_enabled": self.collect_enabled, "boundary": {"title": "Eidos Works · public HTTP edge", "description": "Fixed-route, read-only HEAD observations from this runtime. These are active measurements of the public HTTP response path, not user traffic or a network packet capture.", "missing": ["Cloudflare visitor traffic", "Origin logs", "Internal network", "TLS certificate inspection"], "vantage": "This runtime via outbound proxy", "sampling": f"One HEAD per route every {self.interval:g}s; not visitor traffic", "clock": "Observation timestamps use runtime UTC. Response Date and source_clock_offset_seconds expose upstream clock or cache differences; freshness uses runtime time."}, "sources": sources, "events": events, "brain": brain_summary, "model": self.model_state, "coverage": {"configured_routes": 4, "routes_observed": sum(1 for source in sources if source["id"] != "probe" and source["latest"]), "fresh_routes": sum(1 for source in sources if source["id"] != "probe" and source["state"] == "fresh"), "successful_http_observations": sum(event["status"] is not None for event in events), "transport_failures": sum(event["status"] is None for event in events), "snapshot_events": len(events), "retained_events": stats["retained"], "total_recorded": stats["total_recorded"], "retention_limit": 5000, "analysis_queue": self.analysis_queue.qsize(), "analysis_skipped": self.dropped_analysis, "analysis_failures": self.analysis_failures, "collector_errors": self.collector_errors, "journal": self.journal_status}}
