"""Versioned, append-only proof observations from the live Eidos engine.

The observer is deliberately outside the engine's decision path. It accepts a
completed live decision, validates it, and writes one canonical JSON object per
line. The feature factory returns None while disabled and therefore creates no
files or directories.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any


OBSERVER_SCHEMA_VERSION = "EIDOS-LIVE-FRAME-v1"
OBSERVER_STATUS_VERSION = "EIDOS-LIVE-FRAME-STATUS-v1"


def _plain(value: Any) -> Any:
    """Convert common numeric/container values without silently fixing NaN."""

    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def validate_finite(value: Any, *, path: str = "$") -> None:
    """Fail visibly when a proof record contains a non-finite number."""

    value = _plain(value)
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite numeric value at {path}: {value!r}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            validate_finite(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            validate_finite(item, path=f"{path}[{index}]")
        return
    raise TypeError(f"unsupported proof value at {path}: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    plain = _plain(value)
    validate_finite(plain)
    return json.dumps(plain, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def serialized_jsonl_bytes(value: Any) -> int:
    return len((canonical_json(value) + "\n").encode("utf-8"))


@dataclass(frozen=True)
class ObserverPaths:
    capture: Path
    status: Path


class FrameObserver:
    """Incremental JSONL writer with explicit partial/completed state."""

    def __init__(
        self,
        capture_path: str | Path,
        *,
        run_id: str,
        config_hash: str,
        code_commit: str,
        replay_command: str,
        resume: bool = False,
    ) -> None:
        capture = Path(capture_path)
        self.paths = ObserverPaths(capture=capture, status=capture.with_suffix(capture.suffix + ".status.json"))
        self.run_id = str(run_id)
        self.config_hash = str(config_hash)
        self.code_commit = str(code_commit)
        self.replay_command = str(replay_command)
        self._handle = None
        self._next_sequence = 0
        self._last_frame_id: int | None = None
        self._status = "INITIALIZING"

        capture.parent.mkdir(parents=True, exist_ok=True)
        if capture.exists() and not resume:
            raise FileExistsError(f"capture already exists; choose a new run_id or resume: {capture}")
        if resume:
            self._recover_existing()
        self._handle = capture.open("a", encoding="utf-8", newline="\n")
        self._write_status("RUNNING", reason=None)

    @property
    def records_written(self) -> int:
        return self._next_sequence

    @property
    def last_frame_id(self) -> int | None:
        return self._last_frame_id

    def _recover_existing(self) -> None:
        if not self.paths.capture.exists():
            return
        with self.paths.capture.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if record.get("observer_schema_version") != OBSERVER_SCHEMA_VERSION:
                        raise ValueError("schema mismatch")
                    if record.get("run_id") != self.run_id:
                        raise ValueError("run_id mismatch")
                    stored_hash = record.get("record_sha256")
                    material = dict(record)
                    material.pop("record_sha256", None)
                    if stored_hash != canonical_sha256(material):
                        raise ValueError("record hash mismatch")
                except Exception as exc:
                    raise ValueError(
                        f"cannot resume capture: invalid line {line_number}: {exc}"
                    ) from exc
                self._next_sequence = int(record["sequence"]) + 1
                self._last_frame_id = int(record["frame_id"])

    def observe(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        if self._handle is None:
            raise RuntimeError("observer is closed")
        record = _plain(dict(observation))
        frame_id = record.get("frame_id")
        if not isinstance(frame_id, int):
            self.mark_partial(f"invalid frame_id: {frame_id!r}")
            raise ValueError("frame_id must be an integer")
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            self.mark_partial(f"non-monotonic frame_id: {frame_id} <= {self._last_frame_id}")
            raise ValueError("frame_id must increase monotonically")

        material = {
            "observer_schema_version": OBSERVER_SCHEMA_VERSION,
            "run_id": self.run_id,
            "sequence": self._next_sequence,
            **record,
            "config_hash": record.get("config_hash", self.config_hash),
            "code_commit": record.get("code_commit", self.code_commit),
            "replay_command": record.get("replay_command", self.replay_command),
        }
        try:
            validate_finite(material)
            material["record_sha256"] = canonical_sha256(material)
            line = canonical_json(material) + "\n"
            self._handle.write(line)
            self._handle.flush()
            os.fsync(self._handle.fileno())
        except Exception as exc:
            self.mark_partial(f"observation write failed: {type(exc).__name__}: {exc}")
            raise

        self._next_sequence += 1
        self._last_frame_id = frame_id
        return material

    def _write_status(self, status: str, *, reason: str | None) -> None:
        payload = {
            "status_schema_version": OBSERVER_STATUS_VERSION,
            "run_id": self.run_id,
            "status": status,
            "reason": reason,
            "capture_path": self.paths.capture.as_posix(),
            "records_written": self._next_sequence,
            "last_frame_id": self._last_frame_id,
            "config_hash": self.config_hash,
            "code_commit": self.code_commit,
            "replay_command": self.replay_command,
        }
        payload["status_sha256"] = canonical_sha256(payload)
        tmp = self.paths.status.with_suffix(self.paths.status.suffix + ".tmp")
        tmp.write_text(canonical_json(payload) + "\n", encoding="utf-8", newline="\n")
        os.replace(tmp, self.paths.status)
        self._status = status

    def mark_partial(self, reason: str) -> None:
        self._write_status("PARTIAL", reason=str(reason))

    def finalize(self) -> dict[str, Any]:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None
        self._write_status("COMPLETE", reason=None)
        return json.loads(self.paths.status.read_text(encoding="utf-8"))

    def close_partial(self, reason: str) -> dict[str, Any]:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None
        self._write_status("PARTIAL", reason=str(reason))
        return json.loads(self.paths.status.read_text(encoding="utf-8"))

    def __enter__(self) -> "FrameObserver":
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, _tb: Any) -> bool:
        if exc is None:
            self.finalize()
        else:
            self.close_partial(f"{exc_type.__name__}: {exc}")
        return False


def observer_from_config(
    config: Mapping[str, Any],
    *,
    run_id: str,
    config_hash: str,
    code_commit: str,
    replay_command: str,
) -> FrameObserver | None:
    """Create no artifact unless the explicitly default-off feature is enabled."""

    if not bool(config.get("meaningful_surprise_enabled", False)):
        return None
    capture_path = config.get("meaningful_surprise_observer_path")
    if not capture_path:
        raise ValueError("meaningful_surprise_observer_path is required while enabled")
    return FrameObserver(
        capture_path,
        run_id=run_id,
        config_hash=config_hash,
        code_commit=code_commit,
        replay_command=replay_command,
        resume=bool(config.get("meaningful_surprise_observer_resume", False)),
    )


def read_capture(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            stored_hash = record.pop("record_sha256", None)
            if stored_hash != canonical_sha256(record):
                raise ValueError(f"capture hash mismatch at line {line_number}")
            record["record_sha256"] = stored_hash
            records.append(record)
    return records
