"""Bounded read-only tail of an explicitly configured Suricata/Zeek JSON log.

OBSERVATORY_SENSOR_LOG is the only source of the path. HTTP parameters never
select files. This module does not start sensors, sniff traffic or capture packets.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

MAX_BYTES = 2 * 1024 * 1024
MAX_RECORDS = 2000
MAX_LINE_BYTES = 65536
PRIVACY = ("Local sensor metadata remains on this computer and its Observatory browser. "
           "Internal addresses and DNS names may be visible. Payloads, headers, URLs, "
           "certificates and notice messages are removed. No capture or model upload is started.")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _configured():
    value = os.environ.get("OBSERVATORY_SENSOR_LOG", "")
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("OBSERVATORY_SENSOR_LOG must be an absolute path to a JSON log")
    return path


def _open(path):
    # Reject FIFOs, devices and symlinks before reading. A replaced file is checked
    # again on the open descriptor, so no streaming special file can block a request.
    initial = path.lstat()
    if not stat.S_ISREG(initial.st_mode):
        raise ValueError("Configured sensor path must be a regular file, not a symlink or device")
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Configured sensor path is not a regular file")
        return os.fdopen(fd, "rb"), info
    except BaseException:
        os.close(fd)
        raise


def _base(path=None):
    return {"mode": "local_log_tail", "source_id": "local_sensor_" + hashlib.sha256(str(path).encode()).hexdigest()[:16] if path else None,
            "source_name": "Configured local sensor", "format": "auto", "formats": ["suricata_eve", "zeek_json"],
            "privacy": PRIVACY, "max_bytes": MAX_BYTES, "max_records": MAX_RECORDS,
            "interval_ms": 5000, "starts_capture": False}


def sensor_capabilities():
    try:
        path = _configured()
        if path is None:
            return {**_base(), "connected": False, "reason": "Set OBSERVATORY_SENSOR_LOG to an existing Suricata EVE or Zeek JSON log and restart the local companion"}
        handle, info = _open(path)
        handle.close()
        return {**_base(path), "connected": True, "reason": "Read-only sensor log configured; visibility follows your sensor placement",
                "file_updated_at": datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat()}
    except (OSError, ValueError):
        return {**_base(), "connected": False, "reason": "Configured sensor log is unavailable or not an absolute regular file"}


def _text(value, limit=240):
    if not isinstance(value, str):
        return None
    return "".join(ch for ch in value if ord(ch) >= 32 and ord(ch) != 127)[:limit]


def _scalar(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value if 0 <= value <= 10**20 else None
    return _text(value)


def _pick(record, fields):
    if not isinstance(record, dict):
        return {}
    return {field: value for field in fields if (value := _scalar(record.get(field))) is not None}


def sanitize_sensor_record(record):
    """Keep the source's schema for the browser's strict normalizer to verify."""
    if not isinstance(record, dict):
        return None
    kind = record.get("event_type")
    if kind in {"alert", "flow", "dns", "tls"}:
        out = _pick(record, ("timestamp", "event_type", "flow_id", "src_ip", "dest_ip", "src_port", "dest_port", "proto", "app_proto", "tx_id", "pcap_cnt"))
        fields = {"alert": ("signature_id", "signature", "category", "action", "severity"),
                  "flow": ("bytes_toserver", "bytes_toclient", "pkts_toserver", "pkts_toclient", "age", "state"),
                  "dns": ("rrname", "rrtype", "rcode"), "tls": ("sni", "version")}
        inner = record.get(kind)
        out[kind] = _pick(inner, fields[kind])
        if kind == "dns" and isinstance(inner, dict) and isinstance(inner.get("queries"), list):
            out[kind]["queries"] = [_pick(row, ("rrname", "rrtype")) for row in inner["queries"][:8]]
        return out
    if record.get("ts") is not None and (record.get("uid") is not None or isinstance(record.get("note"), str)):
        out = _pick(record, ("ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto", "src", "dst",
                             "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts", "duration", "conn_state", "service", "note",
                             "query", "rcode_name", "qtype_name", "version", "server_name"))
        if isinstance(record.get("id"), dict):
            out["id"] = _pick(record["id"], ("orig_h", "orig_p", "resp_h", "resp_p"))
        if isinstance(record.get("actions"), list):
            out["actions"] = [_text(value, 80) for value in record["actions"][:8] if isinstance(value, str)]
        return out
    return None


def sensor_snapshot():
    observed_at = _now()
    try:
        path = _configured()
        if path is None:
            return {**sensor_capabilities(), "observed_at": observed_at, "text": "", "coverage": {"state": "disconnected", "missing": ["No sensor source configured"]}}
        handle, info = _open(path)
        with handle:
            truncated = info.st_size > MAX_BYTES
            handle.seek(max(0, info.st_size - MAX_BYTES))
            raw = handle.read(MAX_BYTES)
        if truncated:
            # The byte window may start inside a JSON row; never parse that fragment.
            raw = raw.split(b"\n", 1)[1] if b"\n" in raw else b""
        lines = [line for line in raw.splitlines() if line.strip()]
        capped = len(lines) > MAX_RECORDS
        lines = lines[-MAX_RECORDS:]
        rows, malformed, unsupported, oversized = [], 0, 0, 0
        for line in lines:
            if len(line) > MAX_LINE_BYTES:
                oversized += 1
                continue
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError, RecursionError):
                malformed += 1
                continue
            sanitized = sanitize_sensor_record(record)
            if sanitized is None:
                unsupported += 1
            else:
                rows.append(json.dumps(sanitized, separators=(",", ":"), ensure_ascii=True, allow_nan=False))
        missing = ["Only an existing sensor's exported metadata; Observatory is not capturing packets", "No coverage outside sensor placement"]
        if truncated or capped:
            missing.append("Bounded tail only; earlier records are outside this snapshot")
        if malformed or oversized:
            missing.append("Malformed, oversized or currently incomplete rows excluded")
        if unsupported:
            missing.append("Unsupported sensor record kinds excluded")
        if not rows:
            missing.append("No supported records in current tail; this is not proof of no security events")
        return {**_base(path), "connected": True, "observed_at": observed_at, "text": "\n".join(rows),
                "file_updated_at": datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(),
                "coverage": {"state": "live_log_tail", "truncated": truncated or capped, "missing": missing},
                "counts": {"records": len(rows), "malformed": malformed, "unsupported": unsupported, "oversized": oversized}}
    except (OSError, ValueError, OverflowError):
        return {**_base(), "connected": False, "observed_at": observed_at, "text": "", "error": "Configured sensor log cannot be read",
                "coverage": {"state": "unavailable", "missing": ["Current sensor evidence unavailable; no security verdict"]}}
