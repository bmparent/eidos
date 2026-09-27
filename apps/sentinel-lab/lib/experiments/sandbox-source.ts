import type { Session } from "@vercel/sandbox";
import { isAbsolute } from "node:path";
import { redactDispatchDiagnostic } from "./dispatch-diagnostics.js";

// Probe from the provider's actual cwd. Managed images and source layouts can
// differ: never append a repository name to an assumed absolute directory.
export const SOURCE_PROBE = `
import hashlib, json, subprocess, sys
from pathlib import Path
workspace = Path.cwd().resolve()
required = [
    "services/sentinel-runner/sentinel_runner/sandbox_launcher.py",
    "services/sentinel-runner/pyproject.toml",
    "eidos/EIDOS_BRAIN_UNIFIED_v0_4.7.02.py",
    "eidos/repo/src/eidos_brain/engine/eidos_v0_4_7_02.py",
]
candidates = [workspace] + sorted(p for p in workspace.iterdir() if p.is_dir() and not p.name.startswith("."))
roots = [p for p in candidates if all((p / name).is_file() for name in required)]
if len(roots) != 1:
    raise RuntimeError(f"Expected one complete Eidos checkout in {workspace}; found {len(roots)}")
root = roots[0].resolve()
def git(*args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, timeout=10).strip()
if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
    raise RuntimeError("The runner is not at the verified repository root")
revision = git("rev-parse", "HEAD").lower()
if revision != sys.argv[1].lower():
    raise RuntimeError(f"Source commit mismatch: expected {sys.argv[1]}, received {revision}")
if git("status", "--porcelain", "--untracked-files=no", "--", *required):
    raise RuntimeError("Required source files differ from the pinned commit")
print(json.dumps({"repositoryRoot": str(root), "workingDirectory": str(workspace),
    "commit": revision, "interpreter": sys.executable,
    "launcherPath": str(root / required[0]),
    "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in required}}))
`;

export type SourceReceipt = {
  repositoryRoot: string;
  workingDirectory: string;
  commit: string;
  interpreter: string;
  launcherPath: string;
  files: Record<string, string>;
};

export async function verifySandboxSource(session: Pick<Session, "runCommand">, revision: string): Promise<SourceReceipt> {
  const attempts: string[] = [];
  for (const interpreter of ["python", "python3"]) {
    try {
      // Omitting cwd is intentional: the SDK uses the session's real cwd.
      const result = await session.runCommand({ cmd: interpreter, args: ["-c", SOURCE_PROBE, revision], timeoutMs: 30_000 });
      if (result.exitCode !== 0) throw new Error(await result.output("both"));
      const receipt = JSON.parse(await result.stdout()) as SourceReceipt;
      if (receipt.commit !== revision.toLowerCase() || typeof receipt.repositoryRoot !== "string" || !isAbsolute(receipt.repositoryRoot) || typeof receipt.interpreter !== "string" || !isAbsolute(receipt.interpreter)) {
        throw new Error("Invalid source verification receipt");
      }
      return receipt;
    } catch (error) {
      attempts.push(`${interpreter}: ${redactDispatchDiagnostic(error instanceof Error ? error.message : error)}`);
    }
  }
  throw Object.assign(new Error(`Sandbox source verification failed. ${attempts.join(" | ")}`), { code: "SANDBOX_SOURCE_VERIFICATION_FAILED" });
}
