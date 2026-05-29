import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def default_artifact_root() -> Path:
    env_root = os.environ.get("EIDOS_LIFE_LAB_ARTIFACT_DIR")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "artifacts" / "eidos_life_lab"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ArtifactStore:
    def __init__(self, root: Optional[Path] = None):
        self.root = (root or default_artifact_root()).resolve()
        self.exports_dir = self.root / "exports"
        self.checkpoints_dir = self.root / "checkpoints"
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def save_export(self, full_state: Dict[str, Any]) -> Dict[str, Any]:
        return self._save_json("export", self.exports_dir, full_state)

    def save_checkpoint(self, full_state: Dict[str, Any]) -> Dict[str, Any]:
        return self._save_json("checkpoint", self.checkpoints_dir, full_state)

    def _save_json(self, kind: str, directory: Path, full_state: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = utc_now_iso()
        stamp = timestamp.replace(":", "").replace("-", "").replace(".", "_").replace("Z", "Z")
        run_id = f"{kind}_{stamp}"
        path = directory / f"{run_id}.json"
        payload = dict(full_state)
        payload["artifact_metadata"] = {
            "kind": kind,
            "run_id": run_id,
            "timestamp_utc": timestamp,
            "path": str(path),
        }
        self._write_json(path, payload)
        drive_manifest = self._mirror_files(run_id, [path])
        drive_manifest_path = directory / f"{run_id}_drive_manifest.json"
        self._write_json(drive_manifest_path, drive_manifest)
        return {
            "kind": kind,
            "run_id": run_id,
            "path": str(path),
            "drive_manifest_path": str(drive_manifest_path),
            "drive": drive_manifest,
        }

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def _mirror_files(self, run_id: str, files: Iterable[Path]) -> Dict[str, Any]:
        files = [Path(path) for path in files]
        timestamp = utc_now_iso()
        root, reason = self._discover_drive_root()
        manifest: Dict[str, Any] = {
            "drive_copy_attempted": root is not None,
            "drive_copy_success": False,
            "drive_root": str(root) if root else "unknown",
            "drive_run_dir": "unknown",
            "reason": reason,
            "files_considered": [],
            "files_copied": [],
            "files_skipped": [],
            "timestamp_utc": timestamp,
        }
        for path in files:
            manifest["files_considered"].append(
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size if path.exists() else None,
                    "sha256": sha256_file(path) if path.exists() else None,
                }
            )
        if root is None:
            return manifest

        drive_run_dir = root / "Eidos_Brain_Proof_Phase" / timestamp[:10] / run_id
        manifest["drive_run_dir"] = str(drive_run_dir)
        try:
            drive_run_dir.mkdir(parents=True, exist_ok=True)
            for path in files:
                if not path.exists():
                    manifest["files_skipped"].append({"path": str(path), "reason": "missing"})
                    continue
                target = drive_run_dir / path.name
                shutil.copy2(path, target)
                manifest["files_copied"].append(
                    {
                        "source": str(path),
                        "target": str(target),
                        "size_bytes": target.stat().st_size,
                        "sha256": sha256_file(target),
                    }
                )
            manifest["drive_copy_success"] = len(manifest["files_copied"]) == len(files)
            if manifest["drive_copy_success"]:
                manifest["reason"] = "copied to configured Drive artifact root"
        except Exception as exc:  # pragma: no cover - environment-specific.
            manifest["reason"] = f"Drive copy failed: {exc}"
        return manifest

    def _discover_drive_root(self) -> tuple[Optional[Path], str]:
        candidates: List[tuple[str, Optional[str]]] = [
            ("EIDOS_PROOF_DRIVE_DIR", os.environ.get("EIDOS_PROOF_DRIVE_DIR")),
            ("EIDOS_ARTIFACT_ROOT", os.environ.get("EIDOS_ARTIFACT_ROOT")),
        ]
        for name, value in candidates:
            if not value:
                continue
            path = Path(value).expanduser()
            if path.exists() and path.is_dir() and os.access(path, os.W_OK):
                return path.resolve(), f"using writable {name}"
            return None, f"{name} is set but is not a writable directory"

        colab_drive = Path("/content/drive/MyDrive")
        if colab_drive.exists() and colab_drive.is_dir() and os.access(colab_drive, os.W_OK):
            return colab_drive.resolve(), "using mounted Colab Drive"
        return None, "EIDOS_PROOF_DRIVE_DIR not set, EIDOS_ARTIFACT_ROOT not set, and no mounted Colab Drive path found"
