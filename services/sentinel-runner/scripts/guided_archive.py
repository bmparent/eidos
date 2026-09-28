"""Sanitize, inventory and mirror only this task's new evidence; never delete archives."""
import hashlib
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / "artifacts/sentinel-guided-20260908"
PRIVATE = ROOT / "artifacts/sentinel-guided-private"


def digest(data): return hashlib.sha256(data).hexdigest()


def load(path): return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value): path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    now = datetime.now(timezone.utc).isoformat()
    secrets = list(load(PRIVATE / "local-access.json").values())
    source_key = PRIVATE / "preview-source-key.json"
    if source_key.exists(): secrets.extend(load(source_key).values())
    for path in PRIVATE.glob("preview*access.json"):
        value = load(path).get("url", "")
        if value: secrets += [value, value.split("?")[-1]]
    for path in [ROOT / ".env.local", ROOT / "apps/sentinel-lab/.env.local"]:
        if not path.exists(): continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if "=" not in line or line.lstrip().startswith("#"): continue
            key, value = line.split("=", 1)
            if any(k in key for k in ["TOKEN", "SECRET", "PASSWORD", "API_KEY"]):
                value = value.strip().strip('"').strip("'")
                if len(value) >= 16: secrets.append(value)
    needles = [value.encode() for value in secrets if len(value) >= 16]
    def scan(data, name):
        if any(value in data for value in needles): raise RuntimeError(f"Private value found; no mirror performed: {name}")
    scanned = 0
    for base in [OUT, ROOT / "apps/sentinel-lab/.next/static"]:
        for path in base.rglob("*"):
            if not path.is_file(): continue
            scan(path.read_bytes(), str(path.relative_to(ROOT))); scanned += 1
            if path.suffix == ".zip":
                with zipfile.ZipFile(path) as archive:
                    for name in archive.namelist(): scan(archive.read(name), f"{path.name}:{name}")
    write(OUT / "secret-scan.json", {"status": "passed", "checkedAt": now, "filesScanned": scanned,
        "scope": "exact known QA credentials, protected preview links and local environment secret values; public evidence including ZIP contents and browser static bundle", "valuesPrinted": False,
        "limitation": "exact-value scan plus source review; not a universal secret detector"})
    before, after = load(PRIVATE / "provider-current.json"), load(PRIVATE / "provider-final.json")
    def production(project):
        return {e["key"]: {k: e.get(k) for k in ["id", "type", "value", "target", "gitBranch", "updatedAt"]}
                for e in project.get("env", []) if "production" in e.get("target", [])}
    old, new = production(before), production(after)
    changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
    write(OUT / "provider-comparison.json", {"checkedAt": now, "project": after["id"], "rootDirectory": after.get("rootDirectory"),
        "productionEnvironmentUnchangedSincePrePreviewSnapshot": old == new, "changedProductionKeys": changed,
        "beforeFingerprint": digest(json.dumps(old, sort_keys=True).encode()), "afterFingerprint": digest(json.dumps(new, sort_keys=True).encode()),
        "taskBranchPreviewVariables": [{k: e.get(k) for k in ["key", "target", "gitBranch", "type"]} for e in after.get("env", []) if e.get("gitBranch") == "codex/sentinel-guided-analysis-20260908"],
        "productionDeployment": "233a1d8404d6b9be806cdf9c45a5ac861123e871 from separate upstream member task; no production deployment by guided implementation",
        "limits": "configuration equality is not proof of runtime behavior; actual preview workers have separate receipts"})
    entries = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "drive_manifest.json"}: continue
        entries.append({"path": path.relative_to(OUT).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path.read_bytes())})
    write(OUT / "manifest.json", {"schema": "eidos.implementation-evidence.v1", "createdAt": now, "files": entries,
        "selfHashExcluded": ["manifest.json", "drive_manifest.json"], "privateDataExcluded": ["artifacts/sentinel-guided-private", "artifacts/guided-jobs", ".env*", "local libSQL database"],
        "rawEvaluation": "420 original files preserved in a lossless ZIP with per-entry checksum manifest", "researchGatesAdvanced": 0})
    if "--inventory-only" in sys.argv:
        print(json.dumps({"secretScan": "passed", "artifactFiles": len(entries), "driveCopy": "not requested; existing status preserved for connector upload"}))
        return
    drive_root = None
    for key in ["EIDOS_PROOF_DRIVE_DIR", "EIDOS_ARTIFACT_ROOT"]:
        value = os.environ.get(key)
        if value and Path(value).is_dir() and os.access(value, os.W_OK): drive_root = Path(value); break
    if drive_root is None and os.environ.get("COLAB_RELEASE_TAG") and Path("/content/drive/MyDrive").is_dir(): drive_root = Path("/content/drive/MyDrive")
    status = {"drive_copy_attempted": drive_root is not None, "drive_copy_success": False, "drive_root": str(drive_root) if drive_root else "unknown",
        "drive_run_dir": "unknown", "reason": "No configured writable Drive path", "files_considered": [e["path"] for e in entries] + ["manifest.json", "drive_manifest.json"],
        "files_copied": [], "files_skipped": [], "timestamp_utc": now, "verification": "SHA-256 readback of mounted mirror; remote background sync not independently established"}
    if drive_root:
        destination = drive_root / "Eidos_Brain_Proof_Phase/2026-09-08/sentinel-guided-20260908"
        if not destination.resolve().is_relative_to(drive_root.resolve()): raise RuntimeError("Mirror path escaped configured Drive root")
        if destination.exists() and any(destination.iterdir()) and not (destination / "manifest.json").exists():
            raise RuntimeError("Existing non-task archive at mirror destination; refusing to overwrite")
        destination.mkdir(parents=True, exist_ok=True)
        status["drive_run_dir"] = str(destination)
        try:
            for name in status["files_considered"]:
                if name == "drive_manifest.json": continue
                source, target = OUT / name, destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.read_bytes() == source.read_bytes():
                    status["files_copied"].append({"path": name, "sha256": digest(source.read_bytes()), "action": "already identical task artifact"}); continue
                # Only this task's named folder is eligible for a report refresh.
                shutil.copy2(source, target)
                if digest(source.read_bytes()) != digest(target.read_bytes()): raise IOError(f"Mirror checksum mismatch: {name}")
                status["files_copied"].append({"path": name, "sha256": digest(source.read_bytes()), "action": "copied and readback verified"})
            status.update(drive_copy_success=True, reason="New task artifacts copied and verified; historical Drive archives untouched")
        except OSError as error:
            status["reason"] = str(error)
    write(OUT / "drive_manifest.json", status)
    if drive_root and status["drive_copy_success"]:
        shutil.copy2(OUT / "drive_manifest.json", Path(status["drive_run_dir"]) / "drive_manifest.json")
        assert (OUT / "drive_manifest.json").read_bytes() == (Path(status["drive_run_dir"]) / "drive_manifest.json").read_bytes()
    print(json.dumps({"secretScan": "passed", "artifactFiles": len(entries), "driveCopySuccess": status["drive_copy_success"], "copiedAndVerified": len(status["files_copied"]), "productionEnvironmentUnchanged": old == new}))


if __name__ == "__main__": main()
