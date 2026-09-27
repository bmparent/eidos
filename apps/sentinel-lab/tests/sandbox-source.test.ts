import assert from "node:assert/strict";
import test from "node:test";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { mkdtemp, mkdir, writeFile, rm, cp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import type { Session } from "@vercel/sandbox";
import { verifySandboxSource } from "../lib/experiments/sandbox-source";

const exec = promisify(execFile);
const files = [
  "services/sentinel-runner/sentinel_runner/sandbox_launcher.py",
  "services/sentinel-runner/pyproject.toml",
  "eidos/EIDOS_BRAIN_UNIFIED_v0_4.7.02.py",
  "eidos/repo/src/eidos_brain/engine/eidos_v0_4_7_02.py",
];

async function fixture(nested: boolean) {
  const workspace = await mkdtemp(join(tmpdir(), "eidos-source-"));
  const root = nested ? join(workspace, "repository-with-a-different-name") : workspace;
  for (const file of files) {
    await mkdir(dirname(join(root, file)), { recursive: true });
    await writeFile(join(root, file), "# pinned fixture source\n");
  }
  await exec("git", ["init", "-q", root]);
  await exec("git", ["-C", root, "add", "."]);
  await exec("git", ["-C", root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"]);
  const revision = (await exec("git", ["-C", root, "rev-parse", "HEAD"])).stdout.trim();
  const session = { async runCommand(options: { cmd: string; args: string[]; cwd?: string }) {
    assert.equal(options.cwd, undefined, "the first probe must use the provider cwd");
    try {
      const result = await exec(options.cmd, options.args, { cwd: workspace });
      return { exitCode: 0, stdout: async () => result.stdout, output: async () => result.stdout + result.stderr };
    } catch (error) {
      const result = error as { stderr?: string };
      return { exitCode: 1, output: async () => result.stderr || String(error) };
    }
  } } as unknown as Pick<Session, "runCommand">;
  return { workspace, root, revision, session };
}

for (const nested of [false, true]) test(`source probe verifies a real Git checkout in ${nested ? "a child directory" : "the provider cwd"}`, async () => {
  const f = await fixture(nested);
  try {
    const receipt = await verifySandboxSource(f.session, f.revision);
    assert.equal(receipt.repositoryRoot, f.root);
    assert.equal(receipt.commit, f.revision);
    assert.equal(receipt.launcherPath, join(f.root, files[0]));
    assert.equal(Object.keys(receipt.files).length, 4);
    for (const hash of Object.values(receipt.files)) assert.match(hash, /^[a-f0-9]{64}$/);
  } finally { await rm(f.workspace, { recursive: true, force: true }); }
});

for (const fault of ["wrong commit", "changed source", "missing launcher", "ambiguous checkout"]) test(`source verification blocks ${fault}`, async () => {
  const f = await fixture(true);
  try {
    if (fault === "changed source") await writeFile(join(f.root, files[0]), "# altered source\n");
    if (fault === "missing launcher") await rm(join(f.root, files[0]));
    if (fault === "ambiguous checkout") await cp(f.root, join(f.workspace, "duplicate"), { recursive: true });
    await assert.rejects(verifySandboxSource(f.session, fault === "wrong commit" ? "0".repeat(40) : f.revision), /Sandbox source verification failed/);
  } finally { await rm(f.workspace, { recursive: true, force: true }); }
});
