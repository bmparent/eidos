// Executed with the Sandbox's existing Python standard library. No dataset or
// model code is imported, no files are changed, and only hashes leave the VM.
export const ARTIFACT_VERIFIER = String.raw`
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve(strict=True)
manifest_path = root / "run_manifest.json"
if manifest_path.is_symlink() or manifest_path.stat().st_size > 2000000:
    raise ValueError("Invalid manifest")
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
artifacts = manifest["artifacts"]
if not isinstance(artifacts, dict) or not 0 < len(artifacts) <= 512:
    raise ValueError("Invalid artifact count")
allowed = {"request.json", "source_receipt.json", "dataset_receipt.json", "metrics.json", "engine_trace.jsonl", "evaluation_trace.jsonl", "engine_diagnostics.json"}
results = []
budget = 128 * 1024 * 1024
for name, expected in sorted(artifacts.items()):
    entry = {"path": name, "expected": expected, "matched": False}
    try:
        relative = pathlib.PurePosixPath(name)
        if "\\" in name or relative.is_absolute() or ".." in relative.parts or not (name in allowed or (name.startswith("engine_artifacts/") and len(relative.parts) > 1)):
            raise ValueError("Path outside immutable artifact allowlist")
        path = root.joinpath(*relative.parts).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Artifact escapes job directory or is not a file")
        size = path.stat().st_size
        if size > budget:
            raise ValueError("Verification byte budget exceeded")
        budget -= size
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        entry["actual"] = {"bytes": size, "sha256": digest.hexdigest()}
        entry["matched"] = entry["actual"] == expected
    except (OSError, ValueError) as error:
        entry["error"] = type(error).__name__ + ": " + str(error)
    results.append(entry)
print(json.dumps({"schema": "eidos.sentinel-lab.artifact-verification.v0.1", "jobId": root.name, "manifestSha256": hashlib.sha256(manifest_bytes).hexdigest(), "declaredCount": len(results), "matchedCount": sum(item["matched"] for item in results), "allMatched": all(item["matched"] for item in results), "files": results}))
`;
